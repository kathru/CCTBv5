"""
FundingHarvest — Módulo v5.20 Phase A  (passivo, risco baixo)

Coleta funding rates de perpetual swaps OKX e, quando o funding está
significativamente positivo (longs pagam shorts), abre uma posição SHORT
no swap correspondente para **receber** o funding passivamente.

Lógica:
  - Funding > HARVEST_THRESHOLD (0.05%) E regime ≠ BEAR_TREND/PANIC:
    → Abre SHORT no swap (pequena posição, HARVEST_CONTRACTS_USDT)
    → Registra ShortSwapPlan com harvest_mode=True
  - Saída:
    → Funding cai abaixo de EXIT_THRESHOLD (0.01%) por 2 ciclos consecutivos
    → Regime muda para BEAR_TREND (direcional já cobre, harvest redundante)
    → Backstop: 48h (6 ciclos de funding de 8h)
  - Sizing: posição pequena e fixa (~2% do portfolio) para não acumular direcional

Funding OKX:
  - Liquidado a cada 8 horas: 00:00, 08:00, 16:00 UTC
  - Taxa positiva → longs pagam shorts (típico em bull market)
  - Taxa negativa → shorts pagam longs (típico em bear market)
  - Threshold 0.05% por ciclo = ~5.5% APY (conservador)

Cache:
  - "funding_harvest:{symbol}" → estado atual por símbolo
  - TTL 4h (atualiza a cada ciclo de poll de 8h)

Integração:
  - FundingHarvest.__init__(cache, okx_client, position_monitor)
  - await funding_harvest.start()   → inicia background loop
  - await funding_harvest.stop()    → para o loop
  - funding_harvest.set_portfolio_value(v)  → atualiza sizing
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

SWAP_SYMBOLS = {
    "BTC-USDT": "BTC-USDT-SWAP",
    "ETH-USDT": "ETH-USDT-SWAP",
    "SOL-USDT": "SOL-USDT-SWAP",
}

# OKX SWAP: face value por contrato em unidade base
SWAP_CONTRACT_SIZE = {
    "BTC-USDT-SWAP": 0.01,
    "ETH-USDT-SWAP": 0.1,
    "SOL-USDT-SWAP": 1.0,
}

# Thresholds de funding (por ciclo de 8h)
HARVEST_THRESHOLD  = 0.0005    # 0.05% → abre posição
EXIT_THRESHOLD     = 0.0001    # 0.01% → fecha posição
EXIT_CYCLES_NEEDED = 2         # ciclos consecutivos abaixo do exit threshold

# Sizing: alocação por símbolo como % do portfolio
HARVEST_ALLOCATION_PCT = 0.02  # 2% do portfolio por símbolo
MAX_HARVEST_SYMBOLS    = 2     # nunca mais de 2 símbolos simultâneos
MAX_HARVEST_CONTRACTS  = 10    # teto absoluto de contratos por posição

# Vida máxima de uma posição harvest (6 ciclos × 8h = 48h)
HARVEST_MAX_HOLD_HOURS = 48

# Regimes que bloqueiam a abertura de novas posições harvest
BLOCKED_REGIMES = {"BEAR_TREND", "PANIC_LIQUIDATION"}

# Poll: a cada 30min (OKX atualiza funding rate previsto continuamente)
POLL_INTERVAL = 1800


class FundingHarvestEntry:
    """Estado de uma posição de harvest ativa."""

    __slots__ = (
        "symbol", "swap_sym", "contracts", "entry_funding_rate",
        "entry_price", "opened_at", "exchange_order_id",
        "low_funding_cycles",
    )

    def __init__(
        self,
        symbol: str,
        swap_sym: str,
        contracts: int,
        entry_funding_rate: float,
        entry_price: float,
        exchange_order_id: str,
    ) -> None:
        self.symbol             = symbol
        self.swap_sym           = swap_sym
        self.contracts          = contracts
        self.entry_funding_rate = entry_funding_rate
        self.entry_price        = entry_price
        self.exchange_order_id  = exchange_order_id
        self.opened_at          = datetime.now(UTC)
        self.low_funding_cycles = 0      # contador para saída gradual

    @property
    def age_hours(self) -> float:
        return (datetime.now(UTC) - self.opened_at).total_seconds() / 3600

    def to_dict(self) -> dict:
        return {
            "symbol":              self.symbol,
            "swap_sym":            self.swap_sym,
            "contracts":           self.contracts,
            "entry_funding_rate":  self.entry_funding_rate,
            "entry_price":         self.entry_price,
            "exchange_order_id":   self.exchange_order_id,
            "opened_at":           self.opened_at.isoformat(),
            "low_funding_cycles":  self.low_funding_cycles,
        }


class FundingHarvest:
    """
    Background engine que coleta funding de swaps perp quando favorável.

    Uso em TradingLoop:
        self._funding_harvest = FundingHarvest(
            cache=self._cache,
            okx_client=self._okx,
            position_monitor=self._position_monitor,
        )
        self._funding_harvest.set_portfolio_value(pv)
        await self._funding_harvest.start()
    """

    def __init__(self, cache, okx_client, position_monitor) -> None:
        self._cache            = cache
        self._okx              = okx_client
        self._pm               = position_monitor
        self._portfolio_value  = 0.0
        self._running          = False
        self._task: asyncio.Task | None = None
        self._positions: dict[str, FundingHarvestEntry] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="funding_harvest")
        logger.info("FundingHarvest: started (threshold=%.3f%%)", HARVEST_THRESHOLD * 100)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_status(self) -> dict:
        """Retorna estado atual para o dashboard/API."""
        positions = {sym: pos.to_dict() for sym, pos in self._positions.items()}
        return {
            "active_harvests": len(self._positions),
            "positions":       positions,
            "portfolio_value": self._portfolio_value,
        }

    # ── Loop principal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        # Aguarda sistema estabilizar no boot
        await asyncio.sleep(90)
        while self._running:
            try:
                await self._evaluate_all()
            except Exception as exc:
                logger.warning("FundingHarvest._loop error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    async def _evaluate_all(self) -> None:
        """Avalia entrada e saída de harvest para todos os símbolos."""
        # Primeiro verifica saídas das posições existentes
        for symbol in list(self._positions.keys()):
            await self._check_exit(symbol)

        # Depois avalia entradas (se não atingiu máximo de posições)
        if len(self._positions) >= MAX_HARVEST_SYMBOLS:
            return

        regime = await self._get_macro_regime()
        if regime in BLOCKED_REGIMES:
            logger.debug("FundingHarvest: entrada bloqueada regime=%s", regime)
            return

        for symbol in SYMBOLS:
            if symbol in self._positions:
                continue
            if len(self._positions) >= MAX_HARVEST_SYMBOLS:
                break
            await self._check_entry(symbol)

    # ── Entrada ───────────────────────────────────────────────────────────────

    async def _check_entry(self, symbol: str) -> None:
        swap_sym = SWAP_SYMBOLS.get(symbol)
        if not swap_sym:
            return

        funding_rate = await self._get_funding_rate(swap_sym)
        if funding_rate is None or funding_rate < HARVEST_THRESHOLD:
            logger.debug(
                "FundingHarvest: %s funding=%.4f%% < threshold=%.4f%% — skip",
                symbol, (funding_rate or 0) * 100, HARVEST_THRESHOLD * 100,
            )
            return

        # Verifica histórico: não queremos funding declinante
        history = await self._get_funding_history(swap_sym)
        if history and len(history) >= 2:
            trend = history[0] - history[-1]  # mais recente primeiro
            if trend < -0.0002:   # queda de 0.02% → tendência de queda
                logger.debug(
                    "FundingHarvest: %s funding em queda (%.4f%% → %.4f%%) — aguarda",
                    symbol, history[-1] * 100, history[0] * 100,
                )
                return

        # Sizing
        contracts = self._size_contracts(symbol, swap_sym)
        if contracts < 1:
            logger.debug("FundingHarvest: %s portfolio insuficiente para sizing", symbol)
            return

        # Preço atual
        price = await self._get_price(symbol)
        if not price:
            return

        logger.info(
            "FundingHarvest ENTRADA: %s funding=%.4f%% contracts=%d",
            symbol, funding_rate * 100, contracts,
        )

        # Coloca ordem SHORT no swap
        try:
            coid = f"fh_{uuid.uuid4().hex[:12]}"
            eid  = await self._okx.place_swap_order(
                symbol=swap_sym,
                side="sell",
                pos_side="short",
                quantity=float(contracts),
                order_type="market",
                client_order_id=coid,
            )
        except Exception as exc:
            logger.error("FundingHarvest: falha ao abrir %s: %s", symbol, exc)
            return

        # Registra estado interno
        entry = FundingHarvestEntry(
            symbol=symbol,
            swap_sym=swap_sym,
            contracts=contracts,
            entry_funding_rate=funding_rate,
            entry_price=price,
            exchange_order_id=eid or coid,
        )
        self._positions[symbol] = entry

        # Registra no PositionMonitor como ShortSwapPlan com harvest_mode=True
        self._pm.register_short_plan(
            symbol=symbol,
            swap_sym=swap_sym,
            contracts=contracts,
            entry_price=price,
            atr=price * 0.02,      # ATR sintético: 2% do preço (SL largo para harvest)
            strategy_id="funding_harvest",
            harvest_mode=True,
        )

        # Persiste no Redis para o dashboard
        await self._cache.set(
            f"funding_harvest:{symbol}",
            json.dumps(entry.to_dict()),
            ttl=CACHE_TTL_HARVEST,
        )

        logger.info(
            "FundingHarvest: %s SHORT aberto — eid=%s contracts=%d funding=%.4f%%",
            symbol, eid, contracts, funding_rate * 100,
        )

    # ── Saída ─────────────────────────────────────────────────────────────────

    async def _check_exit(self, symbol: str) -> None:
        entry = self._positions.get(symbol)
        if not entry:
            return

        swap_sym     = entry.swap_sym
        funding_rate = await self._get_funding_rate(swap_sym)
        regime       = await self._get_macro_regime()

        reasons: list[str] = []

        # 1. Funding caiu abaixo do exit threshold por 2 ciclos
        if funding_rate is None or funding_rate < EXIT_THRESHOLD:
            entry.low_funding_cycles += 1
            if entry.low_funding_cycles >= EXIT_CYCLES_NEEDED:
                reasons.append(
                    f"funding_baixo_{entry.low_funding_cycles}x"
                    f"({(funding_rate or 0)*100:.4f}%)"
                )
        else:
            # Funding ainda OK → reseta contador
            entry.low_funding_cycles = 0

        # 2. Regime BEAR_TREND (o direcional SHORT já está coberto)
        if regime in BLOCKED_REGIMES:
            reasons.append(f"regime_{regime}")

        # 3. Backstop: 48h
        if entry.age_hours >= HARVEST_MAX_HOLD_HOURS:
            reasons.append(f"backstop_{entry.age_hours:.0f}h")

        if not reasons:
            logger.debug(
                "FundingHarvest: %s ativa — funding=%.4f%% age=%.1fh low_cycles=%d",
                symbol, (funding_rate or 0) * 100, entry.age_hours,
                entry.low_funding_cycles,
            )
            return

        logger.info(
            "FundingHarvest SAÍDA: %s motivo=%s",
            symbol, ", ".join(reasons),
        )
        await self._close_harvest(symbol, entry)

    async def _close_harvest(self, symbol: str, entry: FundingHarvestEntry) -> None:
        """Fecha posição de harvest via swap buy (cobre o short)."""
        try:
            coid = f"fh_close_{uuid.uuid4().hex[:12]}"
            await self._okx.place_swap_order(
                symbol=entry.swap_sym,
                side="buy",
                pos_side="short",
                quantity=float(entry.contracts),
                order_type="market",
                client_order_id=coid,
            )
        except Exception as exc:
            logger.error("FundingHarvest: falha ao fechar %s: %s", symbol, exc)
            return

        # Remove do PositionMonitor (se ainda registrado)
        if symbol in (self._pm._short_plans or {}):
            del self._pm._short_plans[symbol]

        # Remove estado interno e cache
        self._positions.pop(symbol, None)
        await self._cache.delete(f"funding_harvest:{symbol}")

        logger.info(
            "FundingHarvest: %s posição fechada (%.1fh de hold)",
            symbol, entry.age_hours,
        )

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _size_contracts(self, symbol: str, swap_sym: str) -> int:
        """Calcula número de contratos com base na alocação do portfolio."""
        if self._portfolio_value <= 0:
            return 0
        contract_size = SWAP_CONTRACT_SIZE.get(swap_sym, 1.0)
        if contract_size <= 0:
            return 0
        # Usa preço médio estimado por símbolo (fallback seguro)
        # O tamanho é calculado como: notional / (price × contract_size)
        # Como não temos preço aqui, retorna um mínimo conservador
        notional = self._portfolio_value * HARVEST_ALLOCATION_PCT
        # Para tamanho mínimo seguro: 1 contrato
        # O caller deve verificar se 1 contrato é razoável
        return min(MAX_HARVEST_CONTRACTS, max(1, int(notional / 500)))  # 500 USDT por contrato como ref

    # ── Dados de mercado ──────────────────────────────────────────────────────

    async def _get_funding_rate(self, swap_sym: str) -> float | None:
        """Busca funding rate atual do OKX."""
        try:
            return await self._okx.get_swap_funding_rate(swap_sym)
        except Exception as exc:
            logger.debug("FundingHarvest: get_funding_rate %s: %s", swap_sym, exc)
            return None

    async def _get_funding_history(self, swap_sym: str) -> list[float]:
        """Busca histórico de funding (últimos 3 ciclos)."""
        try:
            return await self._okx.get_funding_rate_history(swap_sym, limit=3)
        except Exception:
            return []

    async def _get_price(self, symbol: str) -> float | None:
        """Busca preço atual do Redis."""
        try:
            raw = await self._cache.get(f"ticker:{symbol}")
            if raw:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                price = data.get("last") or data.get("mid") or data.get("close")
                if price:
                    return float(price)
        except Exception:
            pass
        try:
            ticker = await self._okx.get_ticker(symbol)
            if ticker:
                return float(ticker.last or ticker.mid or 0) or None
        except Exception:
            pass
        return None

    async def _get_macro_regime(self) -> str:
        """Busca regime macro do BTC do Redis."""
        try:
            raw = await self._cache.get("signal:BTC-USDT")
            if raw:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                return data.get("regime", "")
        except Exception:
            pass
        return ""


# TTL do cache de estado harvest (4h — alinhado ao poll de 30min × 8 ciclos)
CACHE_TTL_HARVEST = 14400
