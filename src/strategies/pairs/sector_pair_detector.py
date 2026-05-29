"""
SectorPairDetector — Módulo v5.20 Phase A  (mean reversion, CHOP regime)

Conceito: quando dois ativos dentro do mesmo setor (BTC/ETH/SOL) divergem
de forma anormal em relação ao histórico recente, a divergência tende a se
reverter. O Z-score da razão de preços detecta esse desvio.

Diferença fundamental em relação ao CrossAssetEngine:
  CrossAssetEngine:    TENDÊNCIA — long líder / short laggard (spread amplia)
  SectorPairDetector:  REVERSÃO  — long laggard / short líder  (spread reverte)

Regime alvo: MEAN_REVERTING_CHOP
  - Em CHOP os ativos oscilem em torno de uma média sem direcional claro.
  - Divergências > 2σ tendem a reverter em 4-8h.
  - Threshold de 6% de divergência como sanidade adicional.

Lógica de entrada:
  Para cada par (A, B):
    ratio = price_A / price_B
    z = (ratio - mean(ratio, N)) / std(ratio, N)
    Se |z| > Z_ENTRY (2.0) E |divergência %| > DIV_THRESHOLD (6%):
      Se ratio está ACIMA da média (z > 0): short A, long B
      Se ratio está ABAIXO da média (z < 0): long A, short B

Histórico de preços:
  Mantém janela deslizante de PRICE_WINDOW (48) ticks por par no Redis.
  Poll a cada POLL_INTERVAL (15min) → ~12h de histórico com 15min de granularidade.

Lógica de saída:
  1. Z-score reverteu para |z| < Z_EXIT (0.5)
  2. Hold máximo: MAX_HOLD_HOURS (8h)
  3. Drawdown > MAX_DRAWDOWN (3%) — stop loss da estratégia
  4. Regime saiu de CHOP

Sizing:
  PAIRS_ALLOCATION_PCT (3%) do portfolio por trade, dividido entre os dois legs.
  Máx 2 pares simultâneos.

Execução:
  Long leg: OKX spot via CrossAssetEngine._place_spot_long()
  Short leg: OKX perpetual swap via CrossAssetEngine._place_swap_short()
  → Reutiliza toda a infraestrutura de ordens já testada.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ...persistence.strategy_trade_log import strategy_trade_log as _stl
from ...risk.global_risk_guard import global_risk_guard as _grg

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

SWAP_SYMBOLS = {
    "BTC-USDT": "BTC-USDT-SWAP",
    "ETH-USDT": "ETH-USDT-SWAP",
    "SOL-USDT": "SOL-USDT-SWAP",
}

SWAP_CONTRACT_SIZE = {
    "BTC-USDT-SWAP": 0.01,
    "ETH-USDT-SWAP": 0.1,
    "SOL-USDT-SWAP": 1.0,
}

# Todos os pares possíveis (sem repetição)
PAIRS: list[tuple[str, str]] = [
    ("BTC-USDT", "ETH-USDT"),
    ("BTC-USDT", "SOL-USDT"),
    ("ETH-USDT", "SOL-USDT"),
]

# Z-score thresholds
Z_ENTRY          = 2.0    # desvios padrão para abrir trade
Z_EXIT           = 0.5    # desvios padrão para fechar (reversão confirmada)

# Divergência mínima em % (sanidade — evita ruído em pares muito correlacionados)
DIV_THRESHOLD    = 0.06   # 6% de divergência mínima

# Histórico de preços para cálculo do Z-score
PRICE_WINDOW     = 48     # ticks (48 × 15min = 12h de lookback)
MIN_WINDOW       = 20     # mínimo para calcular Z-score confiável

# Hold e risco
MAX_HOLD_HOURS   = 8
MAX_DRAWDOWN     = 0.03   # 3% stop loss no par

# Sizing
PAIRS_ALLOC_PCT  = 0.03   # 3% do portfolio por par
MAX_ACTIVE_PAIRS = 2      # máx simultâneos

# Regime único de atuação
TARGET_REGIME    = "MEAN_REVERTING_CHOP"

# Poll e cache
POLL_INTERVAL    = 900    # 15 min (alinha com MarketEngine)
CACHE_KEY_RATIO  = "sector_pair:ratio_history:{a}_{b}"
CACHE_TTL_RATIO  = 86400  # 24h (histórico de 12h + margem)
CACHE_KEY_STATUS = "sector_pair:status"


# ── Posição ativa ─────────────────────────────────────────────────────────────

@dataclass
class PairTrade:
    """Estado de um trade de reversão à média ativo."""
    long_sym:          str
    short_sym:         str
    short_swap_sym:    str
    long_qty:          float
    short_contracts:   int
    entry_long_price:  float
    entry_short_price: float
    entry_ratio:       float      # ratio no momento da entrada
    entry_zscore:      float
    notional:          float
    long_eid:          str = ""
    short_eid:         str = ""
    opened_at:         datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def age_hours(self) -> float:
        return (datetime.now(UTC) - self.opened_at).total_seconds() / 3600

    def pnl_pct(self, long_px: float, short_px: float) -> float:
        long_ret  = (long_px  - self.entry_long_price)  / self.entry_long_price
        short_ret = (self.entry_short_price - short_px) / self.entry_short_price
        return (long_ret + short_ret) / 2

    def pair_key(self) -> str:
        return f"{self.long_sym}_{self.short_sym}"


# ── Motor principal ───────────────────────────────────────────────────────────

class SectorPairDetector:
    """
    Detecta e opera reversões à média entre pares de criptos (BTC/ETH/SOL).
    Roda como background task independente.

    Uso em TradingLoop:
        self._sector_pairs = SectorPairDetector(
            cache=self._cache,
            okx_client=self._okx,
        )
        self._sector_pairs.set_portfolio_value(pv)
        await self._sector_pairs.start()
    """

    def __init__(self, cache, okx_client) -> None:
        self._cache           = cache
        self._okx             = okx_client
        self._portfolio_value = 0.0
        self._running         = False
        self._task: asyncio.Task | None = None
        # Chave: "SYM_A_SYM_B" → PairTrade ativo
        self._active: dict[str, PairTrade] = {}
        # Histórico de ratios em memória (complementa Redis)
        self._ratio_history: dict[str, list[float]] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="sector_pair_detector")
        logger.info("SectorPairDetector: started (Z_ENTRY=%.1f DIV_MIN=%.0f%%)",
                    Z_ENTRY, DIV_THRESHOLD * 100)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_status(self) -> dict:
        active = {k: {
            "long":  v.long_sym, "short": v.short_sym,
            "zscore_entry": v.entry_zscore,
            "age_h": round(v.age_hours, 1),
            "notional": v.notional,
        } for k, v in self._active.items()}
        return {"active_pairs": len(self._active), "positions": active}

    # ── Loop principal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        await asyncio.sleep(120)   # aguarda boot + price feed estabilizar
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.warning("SectorPairDetector._tick error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        prices = await self._get_prices()
        if len(prices) < 2:
            return

        # 1. Atualiza histórico de ratios
        await self._update_ratio_history(prices)

        # 2. Verifica saídas das posições ativas
        for key in list(self._active.keys()):
            await self._check_exit(key, prices)

        # 3. Verifica entradas (se regime correto e slots disponíveis)
        if len(self._active) >= MAX_ACTIVE_PAIRS:
            return

        regime = await self._get_macro_regime()
        if regime != TARGET_REGIME:
            logger.debug("SectorPairDetector: regime=%s ≠ %s — aguarda CHOP", regime, TARGET_REGIME)
            return

        for sym_a, sym_b in PAIRS:
            if len(self._active) >= MAX_ACTIVE_PAIRS:
                break
            # Ignora par se algum símbolo já está em posição ativa
            if any(sym_a in k or sym_b in k for k in self._active):
                continue
            await self._check_entry(sym_a, sym_b, prices)

    # ── Histórico de ratios ───────────────────────────────────────────────────

    async def _update_ratio_history(self, prices: dict[str, float]) -> None:
        """Atualiza janela deslizante de ratios para todos os pares."""
        for sym_a, sym_b in PAIRS:
            px_a = prices.get(sym_a)
            px_b = prices.get(sym_b)
            if not px_a or not px_b or px_b == 0:
                continue

            ratio = px_a / px_b
            key   = f"{sym_a}_{sym_b}"

            # Memória local (rápido)
            hist = self._ratio_history.setdefault(key, [])
            hist.append(ratio)
            if len(hist) > PRICE_WINDOW:
                hist.pop(0)

            # Persistência Redis (sobrevive restart)
            cache_key = CACHE_KEY_RATIO.format(a=sym_a, b=sym_b)
            try:
                raw = await self._cache.get(cache_key)
                persisted: list[float] = []
                if raw:
                    persisted = raw if isinstance(raw, list) else json.loads(raw)
                persisted.append(ratio)
                if len(persisted) > PRICE_WINDOW:
                    persisted = persisted[-PRICE_WINDOW:]
                await self._cache.set(cache_key, json.dumps(persisted), ttl=CACHE_TTL_RATIO)
                # Sincroniza memória local com Redis após restart
                if len(self._ratio_history[key]) < len(persisted):
                    self._ratio_history[key] = persisted
            except Exception as exc:
                logger.debug("SectorPairDetector: ratio history cache error: %s", exc)

    def _zscore(self, sym_a: str, sym_b: str) -> tuple[float, float, float] | None:
        """
        Calcula Z-score do ratio A/B.
        Retorna (zscore, ratio_atual, divergencia_pct) ou None se histórico insuficiente.
        """
        key  = f"{sym_a}_{sym_b}"
        hist = self._ratio_history.get(key, [])
        if len(hist) < MIN_WINDOW:
            return None

        current = hist[-1]
        mean    = sum(hist) / len(hist)
        variance = sum((x - mean) ** 2 for x in hist) / len(hist)
        std     = math.sqrt(variance)

        if std < 1e-10:   # variância quase zero — pares perfeitamente correlacionados
            return None

        z = (current - mean) / std

        # Divergência % em relação à média
        div_pct = abs(current - mean) / mean if mean > 0 else 0.0

        return z, current, div_pct

    # ── Entrada ───────────────────────────────────────────────────────────────

    async def _check_entry(
        self, sym_a: str, sym_b: str, prices: dict[str, float]
    ) -> None:
        result = self._zscore(sym_a, sym_b)
        if result is None:
            return

        z, ratio, div_pct = result

        if abs(z) < Z_ENTRY:
            logger.debug(
                "SectorPairDetector: %s/%s z=%.2f < %.1f — sem entrada",
                sym_a, sym_b, z, Z_ENTRY,
            )
            return

        if div_pct < DIV_THRESHOLD:
            logger.debug(
                "SectorPairDetector: %s/%s div=%.1f%% < %.1f%% — divergência insuficiente",
                sym_a, sym_b, div_pct * 100, DIV_THRESHOLD * 100,
            )
            return

        # Determina direção:
        # z > 0 → ratio ACIMA da média → A sobrevalorizado vs B → short A / long B
        # z < 0 → ratio ABAIXO da média → A subvalorizado vs B → long A / short B
        if z > 0:
            long_sym, short_sym = sym_b, sym_a   # long B, short A
        else:
            long_sym, short_sym = sym_a, sym_b   # long A, short B

        long_px  = prices.get(long_sym)
        short_px = prices.get(short_sym)
        if not long_px or not short_px:
            return

        # ── GlobalRiskGuard: Kill Switch + CB + DrawdownEngine ────────────────
        guard = await _grg.allows_new_entry(strategy_id="sector_pair")
        if not guard.allowed:
            logger.info("SectorPairDetector: entrada bloqueada pelo GlobalRiskGuard — %s", guard.reason)
            return

        # Sizing — kelly_mult do GlobalRiskGuard aplicado ao notional
        notional      = self._portfolio_value * PAIRS_ALLOC_PCT / 2 * guard.kelly_mult
        swap_sym      = SWAP_SYMBOLS.get(short_sym)
        cs            = SWAP_CONTRACT_SIZE.get(swap_sym, 1.0)
        if not swap_sym or notional <= 0:
            return

        long_qty       = notional / long_px
        short_contracts = max(1, int(notional / (short_px * cs)))

        logger.info(
            "SectorPairDetector ENTRADA: LONG %s @ %.4f | SHORT %s @ %.4f "
            "| z=%.2f div=%.1f%% notional=%.0f kelly_mult=%.1f",
            long_sym, long_px, short_sym, short_px,
            z, div_pct * 100, notional, guard.kelly_mult,
        )

        # Coloca ordens
        try:
            coid_long  = f"sp_long_{uuid.uuid4().hex[:12]}"
            coid_short = f"sp_short_{uuid.uuid4().hex[:12]}"

            long_eid, short_eid = await asyncio.gather(
                self._okx.place_order(
                    symbol=long_sym, side="buy", order_type="market",
                    quantity=long_qty, price=None, client_order_id=coid_long,
                ),
                self._okx.place_swap_order(
                    symbol=swap_sym, side="sell", pos_side="short",
                    quantity=float(short_contracts), order_type="market",
                    client_order_id=coid_short,
                ),
            )
        except Exception as exc:
            logger.error("SectorPairDetector: falha ao abrir par %s/%s: %s",
                         long_sym, short_sym, exc)
            return

        trade = PairTrade(
            long_sym=long_sym,
            short_sym=short_sym,
            short_swap_sym=swap_sym,
            long_qty=long_qty,
            short_contracts=short_contracts,
            entry_long_price=long_px,
            entry_short_price=short_px,
            entry_ratio=ratio,
            entry_zscore=z,
            notional=notional,
            long_eid=long_eid or coid_long,
            short_eid=short_eid or coid_short,
        )
        self._active[trade.pair_key()] = trade

        # Persiste estado no Redis
        await self._persist_status()

        logger.info(
            "SectorPairDetector: par %s/%s aberto — z=%.2f age=0h",
            long_sym, short_sym, z,
        )

    # ── Saída ─────────────────────────────────────────────────────────────────

    async def _check_exit(self, key: str, prices: dict[str, float]) -> None:
        trade    = self._active.get(key)
        if not trade:
            return

        long_px  = prices.get(trade.long_sym)
        short_px = prices.get(trade.short_sym)
        if not long_px or not short_px:
            return

        pnl_pct = trade.pnl_pct(long_px, short_px)

        # Z-score atual do par original (sym_a/sym_b da entrada)
        # Reconstrói o par original para calcular o z-score de saída
        sym_a, sym_b = self._resolve_original_pair(trade)
        result = self._zscore(sym_a, sym_b)
        current_z = result[0] if result else trade.entry_zscore

        regime  = await self._get_macro_regime()
        reasons: list[str] = []

        # 1. Z-score reverteu
        if abs(current_z) < Z_EXIT:
            reasons.append(f"zscore_reverteu({current_z:.2f})")

        # 2. Hold máximo
        if trade.age_hours >= MAX_HOLD_HOURS:
            reasons.append(f"max_hold({trade.age_hours:.1f}h)")

        # 3. Stop loss
        if pnl_pct < -MAX_DRAWDOWN:
            reasons.append(f"stop_loss({pnl_pct:.1%})")

        # 4. Regime saiu de CHOP
        if regime and regime != TARGET_REGIME and regime not in ("", "UNKNOWN"):
            reasons.append(f"regime_mudou({regime})")

        if not reasons:
            logger.debug(
                "SectorPairDetector: %s/%s ativo z=%.2f pnl=%.2f%% age=%.1fh",
                trade.long_sym, trade.short_sym, current_z,
                pnl_pct * 100, trade.age_hours,
            )
            return

        logger.info(
            "SectorPairDetector SAÍDA: %s/%s motivo=%s pnl=%.2f%%",
            trade.long_sym, trade.short_sym,
            ", ".join(reasons), pnl_pct * 100,
        )
        await self._close_trade(
            key, trade,
            reason=", ".join(reasons),
            exit_long_px=long_px, exit_short_px=short_px,
        )

    async def _close_trade(
        self, key: str, trade: PairTrade,
        reason: str = "", exit_long_px: float | None = None,
        exit_short_px: float | None = None,
    ) -> None:
        """Fecha ambos os legs do par (spot sell + swap buy)."""
        try:
            coid_close_long  = f"sp_cl_{uuid.uuid4().hex[:12]}"
            coid_close_short = f"sp_cs_{uuid.uuid4().hex[:12]}"
            await asyncio.gather(
                self._okx.place_order(
                    symbol=trade.long_sym, side="sell", order_type="market",
                    quantity=trade.long_qty, price=None,
                    client_order_id=coid_close_long,
                ),
                self._okx.place_swap_order(
                    symbol=trade.short_swap_sym, side="buy", pos_side="short",
                    quantity=float(trade.short_contracts), order_type="market",
                    client_order_id=coid_close_short,
                ),
            )
        except Exception as exc:
            logger.error("SectorPairDetector: falha ao fechar %s: %s", key, exc)
            return

        # Calcula pnl para o registro
        pnl_pct_val: float | None = None
        if exit_long_px and exit_short_px:
            pnl_pct_val = trade.pnl_pct(exit_long_px, exit_short_px)
        pnl_usdt_val = pnl_pct_val * trade.notional if pnl_pct_val is not None else None

        await _stl.log_trade(
            strategy_id="sector_pair",
            symbol=f"{trade.long_sym}/{trade.short_sym}",
            side="pair_long_short",
            entry_price=trade.entry_long_price,
            exit_price=exit_long_px,
            notional=trade.notional,
            pnl_pct=pnl_pct_val,
            pnl_usdt=pnl_usdt_val,
            reason=reason or "manual_close",
            opened_at=trade.opened_at,
            closed_at=datetime.now(UTC),
            extra={
                "entry_zscore": trade.entry_zscore,
                "long_sym":  trade.long_sym,
                "short_sym": trade.short_sym,
            },
        )

        self._active.pop(key, None)
        await self._persist_status()

        logger.info(
            "SectorPairDetector: par %s/%s fechado (%.1fh hold) pnl=%.2f%%",
            trade.long_sym, trade.short_sym, trade.age_hours,
            (pnl_pct_val or 0) * 100,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_original_pair(
        self, trade: PairTrade
    ) -> tuple[str, str]:
        """
        Reconstrói o par (sym_a, sym_b) no formato canônico dos PAIRS,
        independente de quem é long ou short.
        """
        candidates = {trade.long_sym, trade.short_sym}
        for sym_a, sym_b in PAIRS:
            if {sym_a, sym_b} == candidates:
                return sym_a, sym_b
        return trade.long_sym, trade.short_sym

    async def _get_prices(self) -> dict[str, float]:
        """Busca preços atuais do Redis."""
        prices: dict[str, float] = {}
        for sym in SYMBOLS:
            try:
                raw = await self._cache.get(f"ticker:{sym}")
                if raw:
                    data  = raw if isinstance(raw, dict) else json.loads(raw)
                    price = data.get("last") or data.get("mid") or data.get("close")
                    if price:
                        prices[sym] = float(price)
            except Exception:
                pass
            # Fallback OKX direto se Redis falhar
            if sym not in prices:
                try:
                    ticker = await self._okx.get_ticker(sym)
                    if ticker:
                        prices[sym] = float(ticker.last or ticker.mid or 0) or prices.get(sym, 0)
                except Exception:
                    pass
        return {k: v for k, v in prices.items() if v > 0}

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

    async def _persist_status(self) -> None:
        """Persiste estado resumido no Redis para o dashboard."""
        try:
            payload = {
                "active_pairs": len(self._active),
                "positions": [
                    {
                        "long":    t.long_sym,
                        "short":   t.short_sym,
                        "zscore":  round(t.entry_zscore, 2),
                        "age_h":   round(t.age_hours, 1),
                        "notional": round(t.notional, 0),
                    }
                    for t in self._active.values()
                ],
                "updated_at": datetime.now(UTC).isoformat(),
            }
            await self._cache.set(CACHE_KEY_STATUS, json.dumps(payload), ttl=3600)
        except Exception as exc:
            logger.debug("SectorPairDetector: persist_status error: %s", exc)
