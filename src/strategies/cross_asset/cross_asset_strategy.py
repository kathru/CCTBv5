"""
CrossAssetEngine — Estratégia Market-Neutral Cross-Asset.

Conceito: long(ativo com maior RS momentum) / short(ativo com menor RS momentum)
usando OKX perpetual swaps para o leg curto.

Hedge Alpha garantido: as duas posições se compensam no risco direcional de mercado.
O lucro vem apenas da divergência de performance relativa entre os ativos.

Lógica de entrada:
  1. Rank dos 3 ativos (BTC/ETH/SOL) por score de RS (M7 do collector)
  2. Spread = score_leader - score_laggard
  3. Entrada quando spread > SPREAD_THRESHOLD e regime não é BEAR_TREND / PANIC
  4. Long leg: spot (via pipeline normal)
  5. Short leg: swap perp (via place_swap_order direto no OKX)

Lógica de saída:
  - Spread convergiu abaixo de SPREAD_EXIT_THRESHOLD
  - Max hold period atingido (MAX_HOLD_HOURS)
  - Drawdown da estratégia > MAX_PAIR_DRAWDOWN
  - Mudança de ranking (novo leader/laggard)

Sizing:
  - Dollar-neutral: long notional = short notional
  - Máximo 10% do portfolio por par
  - Short usa contratos inteiros (OKX SWAP sz = inteiros)

Contratos OKX SWAP (face value):
  BTC-USDT-SWAP: 1 contrato = 0.01 BTC
  ETH-USDT-SWAP: 1 contrato = 0.1 ETH
  SOL-USDT-SWAP: 1 contrato = 1 SOL

Ciclo: avalia a cada 15 minutos (acoplado ao RelativeStrengthCollector).
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

SYMBOLS       = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
SWAP_SYMBOLS  = {
    "BTC-USDT": "BTC-USDT-SWAP",
    "ETH-USDT": "ETH-USDT-SWAP",
    "SOL-USDT": "SOL-USDT-SWAP",
}

# OKX SWAP: face value por contrato em unidade base
SWAP_CONTRACT_SIZE = {
    "BTC-USDT-SWAP": 0.01,    # 1 contrato = 0.01 BTC
    "ETH-USDT-SWAP": 0.1,     # 1 contrato = 0.1 ETH
    "SOL-USDT-SWAP": 1.0,     # 1 contrato = 1 SOL
}

POLL_INTERVAL        = 900     # 15 min
SPREAD_THRESHOLD     = 0.12    # spread mínimo para entrar (RS score difference)
SPREAD_EXIT          = 0.05    # fechar quando spread < este threshold
MAX_HOLD_HOURS       = 72      # saída forçada após 72h
MAX_PAIR_DRAWDOWN    = 0.04    # -4% no par → saída (stop loss da estratégia)
MAX_ALLOCATION       = 0.10    # 10% do portfolio por par (cada leg = 5%)
BLOCKED_REGIMES      = {"BEAR_TREND", "PANIC_LIQUIDATION", "HIGH_CORRELATION_RISK"}


# ── Estado de posição aberta ──────────────────────────────────────────────────

@dataclass
class PairPosition:
    """Estado de uma posição market-neutral ativa."""
    long_symbol:      str
    short_symbol:     str
    short_swap_symbol: str
    entry_long_price: float
    entry_short_price:float
    long_qty:         float        # unidades (BTC, ETH, SOL)
    short_contracts:  int          # contratos SWAP (inteiros)
    notional:         float        # tamanho nominal de cada leg (USDT)
    opened_at:        datetime = field(default_factory=lambda: datetime.now(UTC))
    long_exchange_id: str = ""
    short_exchange_id:str = ""
    entry_spread:     float = 0.0

    @property
    def age_hours(self) -> float:
        return (datetime.now(UTC) - self.opened_at).total_seconds() / 3600

    def pnl_pct(self, current_long_price: float, current_short_price: float) -> float:
        """PnL percentual do par (em relação ao notional de cada leg)."""
        long_pnl  = (current_long_price  - self.entry_long_price)  / self.entry_long_price
        short_pnl = (self.entry_short_price - current_short_price) / self.entry_short_price
        return (long_pnl + short_pnl) / 2


# ── Motor principal ───────────────────────────────────────────────────────────

class CrossAssetEngine:
    """
    Motor de estratégia market-neutral.
    Roda como background task independente do pipeline principal.
    """

    def __init__(self, cache, okx_client) -> None:
        self._cache      = cache
        self._okx        = okx_client
        self._position:  PairPosition | None = None
        self._task:      asyncio.Task | None = None
        self._running    = False
        self._portfolio_value = 0.0   # atualizado externamente

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="cross_asset_engine")
        logger.info("CrossAssetEngine started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        await asyncio.sleep(60)   # aguarda sistema estabilizar
        while self._running:
            try:
                await self._evaluate()
            except Exception as exc:
                logger.warning("CrossAssetEngine._evaluate: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    # ── Avaliação principal ───────────────────────────────────────────────────

    async def _evaluate(self) -> None:
        rs_scores = await self._get_rs_scores()
        if not rs_scores:
            return

        prices = await self._get_prices()
        if not prices:
            return

        # Se temos posição aberta, avalia saída
        if self._position:
            await self._check_exit(rs_scores, prices)
            return

        # Sem posição: avalia entrada
        await self._check_entry(rs_scores, prices)

    # ── Entrada ───────────────────────────────────────────────────────────────

    async def _check_entry(self, rs_scores: dict, prices: dict) -> None:
        # Verifica regime macro
        regime = await self._get_macro_regime()
        if regime in BLOCKED_REGIMES:
            logger.debug("CrossAsset: entrada bloqueada regime=%s", regime)
            return

        # Rank por RS score
        ranked = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return

        leader, leader_score   = ranked[0]
        laggard, laggard_score = ranked[-1]
        spread = leader_score - laggard_score

        if spread < SPREAD_THRESHOLD:
            logger.debug(
                "CrossAsset: spread=%.3f < threshold=%.3f — sem entrada",
                spread, SPREAD_THRESHOLD,
            )
            return

        # Sizing: notional = MAX_ALLOCATION × portfolio / 2 (cada leg = metade)
        if self._portfolio_value <= 0:
            return
        notional = self._portfolio_value * MAX_ALLOCATION / 2

        long_price  = prices.get(leader)
        short_price = prices.get(laggard)
        if not long_price or not short_price:
            return

        # Quantidade do long leg (spot)
        long_qty = notional / long_price

        # Quantidade do short leg (swap perp — contratos inteiros)
        swap_sym = SWAP_SYMBOLS.get(laggard)
        contract_size = SWAP_CONTRACT_SIZE.get(swap_sym, 0)
        if not swap_sym or contract_size <= 0:
            return
        short_contracts = max(1, int(notional / (short_price * contract_size)))

        logger.info(
            "CrossAsset ENTRADA: LONG %s @ %.2f (%.4f uni) | SHORT %s @ %.2f (%d contratos) | spread=%.3f",
            leader, long_price, long_qty,
            laggard, short_price, short_contracts,
            spread,
        )

        # Coloca ordens
        try:
            long_id, short_id = await asyncio.gather(
                self._place_spot_long(leader, long_qty, long_price),
                self._place_swap_short(laggard, short_contracts),
            )
        except Exception as exc:
            logger.error("CrossAsset: falha ao abrir par: %s", exc)
            return

        self._position = PairPosition(
            long_symbol=leader,
            short_symbol=laggard,
            short_swap_symbol=swap_sym,
            entry_long_price=long_price,
            entry_short_price=short_price,
            long_qty=long_qty,
            short_contracts=short_contracts,
            notional=notional,
            long_exchange_id=long_id,
            short_exchange_id=short_id,
            entry_spread=spread,
        )

        await self._cache.set(
            "cross_asset:position",
            json.dumps({
                "long": leader, "short": laggard,
                "notional": notional, "spread": spread,
                "opened_at": self._position.opened_at.isoformat(),
            }),
            ttl=86400,
        )

    # ── Saída ─────────────────────────────────────────────────────────────────

    async def _check_exit(self, rs_scores: dict, prices: dict) -> None:
        pos = self._position
        assert pos is not None

        long_price  = prices.get(pos.long_symbol)
        short_price = prices.get(pos.short_symbol)
        if not long_price or not short_price:
            return

        pnl_pct = pos.pnl_pct(long_price, short_price)
        current_spread = rs_scores.get(pos.long_symbol, 0.5) - rs_scores.get(pos.short_symbol, 0.5)
        ranked = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        new_leader  = ranked[0][0] if ranked else pos.long_symbol
        new_laggard = ranked[-1][0] if ranked else pos.short_symbol

        reasons = []
        if current_spread < SPREAD_EXIT:
            reasons.append(f"spread_converged ({current_spread:.3f})")
        if pos.age_hours > MAX_HOLD_HOURS:
            reasons.append(f"max_hold ({pos.age_hours:.1f}h)")
        if pnl_pct < -MAX_PAIR_DRAWDOWN:
            reasons.append(f"stop_loss ({pnl_pct:.1%})")
        if new_leader != pos.long_symbol or new_laggard != pos.short_symbol:
            reasons.append("rank_change")

        if not reasons:
            logger.debug(
                "CrossAsset: posição ativa pnl=%.2f%% spread=%.3f age=%.1fh",
                pnl_pct * 100, current_spread, pos.age_hours,
            )
            return

        logger.info("CrossAsset SAÍDA: %s | pnl=%.2f%%", ", ".join(reasons), pnl_pct * 100)
        await self._close_position(pos)

    async def _close_position(self, pos: PairPosition) -> None:
        try:
            await asyncio.gather(
                self._close_spot_long(pos.long_symbol, pos.long_qty),
                self._close_swap_short(pos.short_swap_symbol, pos.short_contracts),
            )
        except Exception as exc:
            logger.error("CrossAsset: falha ao fechar par: %s", exc)
        finally:
            self._position = None
            await self._cache.delete("cross_asset:position")

    # ── Ordens ───────────────────────────────────────────────────────────────

    async def _place_spot_long(self, symbol: str, qty: float, price: float) -> str:
        coid = f"cx_long_{uuid.uuid4().hex[:12]}"
        return await self._okx.place_order(
            symbol=symbol,
            side="buy",
            order_type="market",
            quantity=qty,
            price=None,
            client_order_id=coid,
        )

    async def _place_swap_short(self, symbol: str, contracts: int) -> str:
        swap_sym = SWAP_SYMBOLS.get(symbol, symbol + "-SWAP")
        coid     = f"cx_short_{uuid.uuid4().hex[:12]}"
        return await self._okx.place_swap_order(
            symbol=swap_sym,
            side="sell",
            pos_side="short",
            quantity=float(contracts),
            order_type="market",
            client_order_id=coid,
        )

    async def _close_spot_long(self, symbol: str, qty: float) -> None:
        coid = f"cx_close_long_{uuid.uuid4().hex[:12]}"
        await self._okx.place_order(
            symbol=symbol,
            side="sell",
            order_type="market",
            quantity=qty,
            price=None,
            client_order_id=coid,
        )

    async def _close_swap_short(self, swap_symbol: str, contracts: int) -> None:
        coid = f"cx_close_short_{uuid.uuid4().hex[:12]}"
        await self._okx.place_swap_order(
            symbol=swap_symbol,
            side="buy",
            pos_side="short",
            quantity=float(contracts),
            order_type="market",
            client_order_id=coid,
        )

    # ── Dados de mercado ──────────────────────────────────────────────────────

    async def _get_rs_scores(self) -> dict[str, float]:
        """Busca scores M7 (relative strength) do Redis para todos os símbolos."""
        scores: dict[str, float] = {}
        for sym in SYMBOLS:
            try:
                raw = await self._cache.get(f"relative_strength:{sym}")
                if raw:
                    data = raw if isinstance(raw, dict) else json.loads(raw)
                    score = data.get("m7_score") or data.get("score")
                    if score is not None:
                        scores[sym] = float(score)
            except Exception:
                pass
        return scores

    async def _get_prices(self) -> dict[str, float]:
        """Busca preços atuais do Redis (último candle)."""
        prices: dict[str, float] = {}
        for sym in SYMBOLS:
            try:
                raw = await self._cache.get(f"ticker:{sym}")
                if raw:
                    data = raw if isinstance(raw, dict) else json.loads(raw)
                    price = data.get("last") or data.get("mid") or data.get("close")
                    if price:
                        prices[sym] = float(price)
            except Exception:
                try:
                    ticker = await self._okx.get_ticker(sym)
                    if ticker:
                        prices[sym] = ticker.last or ticker.mid
                except Exception:
                    pass
        return prices

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

    async def get_status(self) -> dict:
        """Retorna estado atual para o dashboard."""
        pos = self._position
        return {
            "active":       pos is not None,
            "long_symbol":  pos.long_symbol  if pos else None,
            "short_symbol": pos.short_symbol if pos else None,
            "notional":     pos.notional     if pos else None,
            "age_hours":    round(pos.age_hours, 1) if pos else None,
            "entry_spread": pos.entry_spread  if pos else None,
        }
