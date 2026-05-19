"""
Position Monitor — gerencia saídas automáticas de posições abertas.

Fases implementadas:
  A. ATR-based stop loss + take profit + timeout adaptativo por regime
  B. Trailing stop ativado após +1R de lucro
  C. Saída parcial (50%) em +1.5R, restante corre com trailing
  D. Saída imediata se regime deteriorar para PANIC/VACUUM

Roda a cada 30s verificando todas as posições abertas.
Uma ExitPlan é criada quando uma posição é aberta (via FillEvent)
e destruída quando a posição é fechada.

Integração:
  TradingLoop → PositionMonitor.start()
  FillEvent   → _on_fill() → cria/atualiza ExitPlan
  Loop 30s    → _check_positions() → dispara saídas via OrderManager
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..core.bus import EventBus
from ..core.events import Topic
from ..core.events.order_events import OrderFilledEvent
from ..core.models import Candle
from ..market.engine import MarketEngine
from ..oms.order_manager import OrderManager
from ..persistence.cache import Cache
from ..portfolio.engine import PortfolioEngine

logger = logging.getLogger(__name__)

# ── Constantes de saída ───────────────────────────────────────────────────────

ATR_PERIOD    = 14    # candles para calcular ATR

# Phase A — Multiplicadores de ATR por regime
# SL: distância do stop  |  TP: distância do alvo  |  Ratio TP/SL implícito
REGIME_MULT: dict[str, dict[str, float]] = {
    #                              SL    TP     ratio
    "TREND_EXPANSION":        {"sl": 1.5, "tp": 4.0},  # 1:2.7 — deixa correr
    "VOLATILITY_COMPRESSION": {"sl": 1.5, "tp": 3.0},  # 1:2.0 — padrão
    "TREND_EXHAUSTION":       {"sl": 1.5, "tp": 2.5},  # 1:1.7 — conservador
    "MEAN_REVERTING_CHOP":    {"sl": 1.0, "tp": 1.5},  # 1:1.5 — alvos curtos
    "HIGH_CORRELATION_RISK":  {"sl": 1.2, "tp": 2.0},  # 1:1.7 — risco controlado
}
DEFAULT_SL_MULT = 1.5
DEFAULT_TP_MULT = 3.0

# Phase B — Trailing stop
TRAIL_ACTIVATE_R = 1.0   # ativa trailing após +1R de ganho
TRAIL_ATR_MULT   = 1.0   # distância do trailing = ATR × 1.0

# Phase C — Saída parcial
PARTIAL_EXIT_R   = 1.5   # sai 50% em +1.5R
PARTIAL_EXIT_PCT = 0.50  # fracção da posição a vender

# Phase D — Regimes que forçam saída imediata
EXIT_REGIMES = {"PANIC_LIQUIDATION", "BEAR_TREND"}  # saída imediata nesses regimes

# Timeout adaptativo por regime (horas)
TIMEOUT_HOURS: dict[str, int] = {
    "TREND_EXPANSION":        48,   # BULL  — deixa correr
    "VOLATILITY_COMPRESSION": 24,
    "TREND_EXHAUSTION":       12,
    "MEAN_REVERTING_CHOP":    8,    # CHOP  — sai rápido
    "HIGH_CORRELATION_RISK":  4,    # risco — sai muito rápido
    "BEAR_TREND":             0,    # BEAR  — saída imediata (EXIT_REGIMES)
    "PANIC_LIQUIDATION":      0,    # PANIC — saída imediata
}
DEFAULT_TIMEOUT_HOURS = 12


# ── ExitPlan ──────────────────────────────────────────────────────────────────

@dataclass
class ExitPlan:
    """Plano de saída para uma posição aberta."""
    symbol:        str
    strategy_id:   str
    quantity:      float          # quantidade original da posição
    entry_price:   float
    entry_time:    datetime
    entry_regime:  str
    atr:           float          # ATR no momento da entrada

    # Phase A — níveis fixos
    stop_loss:     float = 0.0
    take_profit:   float = 0.0
    timeout_at:    datetime = field(default_factory=lambda: datetime.now(UTC))

    # Phase B — trailing (None = não ativado ainda)
    trailing_stop:      float | None = None
    trailing_activated: bool = False

    # Phase C — saída parcial
    partial_done:      bool  = False
    qty_remaining:     float = 0.0   # quantidade que ainda está aberta

    def __post_init__(self) -> None:
        mults = REGIME_MULT.get(self.entry_regime, {})
        sl_mult = mults.get("sl", DEFAULT_SL_MULT)
        tp_mult = mults.get("tp", DEFAULT_TP_MULT)
        sl_dist = self.atr * sl_mult
        tp_dist = self.atr * tp_mult
        self.stop_loss   = round(self.entry_price - sl_dist, 4)
        self.take_profit = round(self.entry_price + tp_dist, 4)
        hours = TIMEOUT_HOURS.get(self.entry_regime, DEFAULT_TIMEOUT_HOURS)
        self.timeout_at  = self.entry_time + timedelta(hours=hours)
        self.qty_remaining = self.quantity

    @property
    def r_value(self) -> float:
        """1R = distância entre entrada e stop loss."""
        return self.entry_price - self.stop_loss

    def price_at_r(self, multiples: float) -> float:
        """Preço correspondente a N × R de lucro."""
        return self.entry_price + self.r_value * multiples

    def summary(self) -> str:
        return (
            f"{self.symbol} entry={self.entry_price:.2f} "
            f"sl={self.stop_loss:.2f} tp={self.take_profit:.2f} "
            f"atr={self.atr:.2f} regime={self.entry_regime} "
            f"trail={'ON' if self.trailing_activated else 'off'} "
            f"partial={'done' if self.partial_done else 'pending'}"
        )


# ── PositionMonitor ───────────────────────────────────────────────────────────

class PositionMonitor:
    """
    Monitora posições abertas e dispara saídas automáticas.
    Uma instância por TradingLoop.
    """

    def __init__(
        self,
        bus: EventBus,
        market: MarketEngine,
        portfolio: PortfolioEngine,
        oms: OrderManager,
        cache: Cache,
        interval_seconds: int = 30,
    ) -> None:
        self._bus      = bus
        self._market   = market
        self._portfolio = portfolio
        self._oms      = oms
        self._cache    = cache
        self._interval = interval_seconds

        # ExitPlans ativos: symbol → ExitPlan
        self._plans: dict[str, ExitPlan] = {}

        self._running = False
        self._task: asyncio.Task | None = None
        self._fill_task: asyncio.Task | None = None
        self._fill_queue: asyncio.Queue | None = None
        self._exits_today = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Subscreve fills via queue (API correta do EventBus)
        self._fill_queue = self._bus.subscribe(Topic.FILL)
        self._fill_task  = asyncio.create_task(
            self._consume_fills(), name="position_monitor_fills"
        )

        # Cria planos para posições já abertas (restart recovery)
        await self._recover_existing_positions()

        self._task = asyncio.create_task(self._loop(), name="position_monitor")
        logger.info("PositionMonitor started interval=%ds", self._interval)

    async def stop(self) -> None:
        self._running = False
        for task in (self._task, self._fill_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("PositionMonitor stopped — exits_today=%d", self._exits_today)

    # ── Consumer de fills ─────────────────────────────────────────────────────

    async def _consume_fills(self) -> None:
        """Consome FillEvents do bus e cria ExitPlans para compras."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._fill_queue.get(), timeout=1.0
                )
                await self._on_order_event(event)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("PositionMonitor fill consumer error: %s", exc, exc_info=True)

    # ── Evento de fill → criar ExitPlan ──────────────────────────────────────

    async def _on_order_event(self, event) -> None:
        """Cria ExitPlan quando um fill de compra é confirmado."""
        if not isinstance(event, OrderFilledEvent):
            return
        order = event.order
        if order is None:
            return

        side = str(getattr(order, "side", "")).upper()
        if side not in ("BUY", "LONG"):
            return   # fills de venda não criam planos

        symbol   = order.symbol
        qty      = order.filled_quantity or order.quantity
        price    = order.avg_fill_price or 0.0
        strat_id = order.strategy_id or "unknown"

        if price <= 0 or qty <= 0:
            return

        await self._create_plan(symbol, qty, price, strat_id)

    async def _create_plan(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        strategy_id: str,
    ) -> None:
        # ATR em 30min (14 × 30min = 7h) — responsivo ao timeframe de avaliação
        candles_30m = self._market.get_candles(symbol, "30m", limit=ATR_PERIOD + 5)
        candles_1h  = self._market.get_candles(symbol, "1H",  limit=ATR_PERIOD + 5)
        atr = _calc_atr(candles_30m, ATR_PERIOD) if len(candles_30m) >= ATR_PERIOD else 0.0

        if atr <= 0:
            # Fallback: tenta 1H, depois percentual fixo
            atr = _calc_atr(candles_1h, ATR_PERIOD)
        if atr <= 0:
            atr = entry_price * 0.015
            logger.warning("%s: ATR indisponível — usando fallback %.2f", symbol, atr)

        # Regime usa 1H (decisão macro mais estável)
        regime = _detect_regime(candles_1h)

        plan = ExitPlan(
            symbol=symbol,
            strategy_id=strategy_id,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(UTC),
            entry_regime=regime,
            atr=atr,
        )
        self._plans[symbol] = plan
        logger.info(
            "ExitPlan criado: %s", plan.summary()
        )

    async def _recover_existing_positions(self) -> None:
        """Cria ExitPlans para posições abertas ao reiniciar o sistema."""
        positions = self._portfolio.state.positions
        if not positions:
            return

        logger.info("PositionMonitor: recuperando %d posições abertas…", len(positions))
        for symbol, pos in positions.items():
            if symbol in self._plans:
                continue
            entry_price = float(pos.get("avg_entry", 0))
            quantity    = float(pos.get("quantity", 0))
            strat_id    = pos.get("strategy_id", "recovered")
            if entry_price > 0 and quantity > 0:
                await self._create_plan(symbol, quantity, entry_price, strat_id)

    # ── Loop principal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._check_positions()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("PositionMonitor error: %s", exc, exc_info=True)
            await asyncio.sleep(self._interval)

    async def _check_positions(self) -> None:
        if not self._plans:
            return

        now = datetime.now(UTC)

        for symbol, plan in list(self._plans.items()):
            try:
                await self._evaluate_plan(symbol, plan, now)
            except Exception as exc:
                logger.error("Erro ao avaliar %s: %s", symbol, exc, exc_info=True)

    async def _evaluate_plan(
        self,
        symbol: str,
        plan: ExitPlan,
        now: datetime,
    ) -> None:
        # Preço atual
        price = await self._cache.get_price(symbol)
        if not price or price <= 0:
            return
        price = float(price)

        # Regime check: usa 30m para detectar PANIC/BEAR rapidamente
        candles_30m_chk = self._market.get_candles(symbol, "30m", limit=25)
        candles_1h_chk  = self._market.get_candles(symbol, "1H",  limit=25)
        current_regime = _detect_regime(candles_30m_chk) if len(candles_30m_chk) >= 5 \
            else _detect_regime(candles_1h_chk)

        # ── Phase D — Regime deteriorado ──────────────────────
        if current_regime in EXIT_REGIMES:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason=f"regime_{current_regime.lower()}",
                partial=False,
            )
            return

        # ── Phase A — Stop Loss ───────────────────────────────
        effective_sl = plan.trailing_stop if plan.trailing_stop else plan.stop_loss
        if price <= effective_sl:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason="stop_loss" if not plan.trailing_stop else "trailing_stop",
                partial=False,
            )
            return

        # ── Phase A — Take Profit ────────────────────────────
        if price >= plan.take_profit:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason="take_profit",
                partial=False,
            )
            return

        # ── Phase A — Timeout ────────────────────────────────
        if now >= plan.timeout_at:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason=f"timeout_{plan.entry_regime.lower()}",
                partial=False,
            )
            return

        # ── Phase C — Saída parcial em +1.5R ─────────────────
        if not plan.partial_done and price >= plan.price_at_r(PARTIAL_EXIT_R):
            qty_to_sell = round(plan.quantity * PARTIAL_EXIT_PCT, 8)
            await self._exit(
                plan, price, qty_to_sell,
                reason="partial_take_profit",
                partial=True,
            )
            plan.partial_done  = True
            plan.qty_remaining = round(plan.qty_remaining - qty_to_sell, 8)
            logger.info(
                "%s: saída parcial %.4f unidades @ %.2f (+1.5R) — restante: %.4f",
                symbol, qty_to_sell, price, plan.qty_remaining,
            )

        # ── Phase B — Ativa / atualiza trailing stop ──────────
        if price >= plan.price_at_r(TRAIL_ACTIVATE_R):
            new_trail = round(price - plan.atr * TRAIL_ATR_MULT, 4)
            if not plan.trailing_activated:
                plan.trailing_stop      = new_trail
                plan.trailing_activated = True
                logger.info(
                    "%s: trailing stop ativado @ %.2f (preço=%.2f, +1R atingido)",
                    symbol, new_trail, price,
                )
            elif new_trail > (plan.trailing_stop or 0):
                logger.debug(
                    "%s: trailing stop atualizado %.2f → %.2f",
                    symbol, plan.trailing_stop, new_trail,
                )
                plan.trailing_stop = new_trail

    # ── Execução de saída ─────────────────────────────────────────────────────

    async def _exit(
        self,
        plan: ExitPlan,
        price: float,
        quantity: float,
        reason: str,
        partial: bool,
    ) -> None:
        symbol = plan.symbol
        pnl_r  = (price - plan.entry_price) / plan.r_value if plan.r_value > 0 else 0

        logger.info(
            "SAÍDA %s %s qty=%.4f price=%.2f pnl=%.2fR reason=%s",
            "PARCIAL" if partial else "TOTAL",
            symbol, quantity, price, pnl_r, reason,
        )

        success = await self._oms.exit_position(
            symbol=symbol,
            quantity=quantity,
            reason=reason,
            strategy_id=plan.strategy_id,
        )

        if success:
            self._exits_today += 1
            if not partial:
                # Remove o plano — posição fechada
                self._plans.pop(symbol, None)
                logger.info(
                    "%s: ExitPlan removido após saída total (reason=%s)",
                    symbol, reason,
                )

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running":       self._running,
            "active_plans":  len(self._plans),
            "exits_today":   self._exits_today,
            "plans": {
                sym: {
                    "entry_price":   p.entry_price,
                    "stop_loss":     p.trailing_stop or p.stop_loss,
                    "take_profit":   p.take_profit,
                    "timeout_at":    p.timeout_at.isoformat(),
                    "trailing":      p.trailing_activated,
                    "partial_done":  p.partial_done,
                    "qty_remaining": p.qty_remaining,
                    "regime":        p.entry_regime,
                }
                for sym, p in self._plans.items()
            },
        }


# ── Helpers (espelham v4_strategy, sem import circular) ───────────────────────

def _calc_atr(candles: list[Candle], period: int = 14) -> float:
    """Average True Range dos últimos `period` candles."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(period):
        high       = candles[i].high
        low        = candles[i].low
        prev_close = candles[i + 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)


def _detect_regime(candles: list[Candle]) -> str:
    """Espelho de V4MomentumStrategy._detect_regime (sem importação circular)."""
    if len(candles) < 20:
        return "MEAN_REVERTING_CHOP"

    closes  = [c.close  for c in candles[:20]]
    volumes = [c.volume for c in candles[:20]]

    sma_fast = sum(closes[:5])  / 5
    sma_slow = sum(closes[:20]) / 20
    avg_vol  = sum(volumes) / len(volumes)
    last_vol = volumes[0]

    # Panic: queda > 5% no candle mais recente vs anterior
    if len(closes) >= 2 and closes[1] > 0:
        drop = (closes[0] - closes[1]) / closes[1]
        if drop < -0.05:
            return "PANIC_LIQUIDATION"

    if sma_fast > sma_slow:
        if last_vol > avg_vol * 1.2:
            return "TREND_EXPANSION"
        if last_vol < avg_vol * 0.8:
            return "TREND_EXHAUSTION"
        return "VOLATILITY_COMPRESSION"

    highs = [c.high for c in candles[:10]]
    lows  = [c.low  for c in candles[:10]]
    atr_5 = sum(hi - lo for hi, lo in zip(highs[:5], lows[:5], strict=True)) / 5
    rel_atr = atr_5 / closes[0] if closes[0] > 0 else 0

    # BEAR: preço atual abaixo de 10 candles atrás em > 2%
    if len(closes) >= 11 and closes[10] > 0:
        if (closes[0] - closes[10]) / closes[10] < -0.02:
            return "BEAR_TREND"

    if rel_atr > 0.030:
        return "HIGH_CORRELATION_RISK"

    return "MEAN_REVERTING_CHOP"
