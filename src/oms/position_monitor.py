"""
Position Monitor — gerencia saídas automáticas de posições abertas.

Arquitetura: Buy Engine + Hold Engine + Sell Engine
  • Buy Engine  : StrategyRunner (gera sinais de compra)
  • Hold Engine : HoldEngine (decide se vale continuar no trade)
  • Sell Engine : este módulo (executa a saída via OrderManager)

Fases de saída (em ordem de prioridade):
  D. Regime de emergência (PANIC/BEAR)  → saída imediata
  A. Stop Loss absoluto (ATR-based)     → safety net intocável
  A. Take Profit / TP Conversion        → em TREND_EXPANSION: parcial + running mode
                                          em outros regimes: saída total
  C. Saída parcial (50%)               → garante lucro, deixa metade correr
  B. Trailing stop (conviction-adaptive)→ HOLD forte=2.5× ATR, WATCH=1.0×, ALERT=0.5×
  H. Hold Engine / Conviction Decay    → substitui timeout por deterioração estrutural
     HOLD forte (≥85): trailing 2.5× ATR — winners respiram
     HOLD       (70–84): trailing 2.0× ATR
     WATCH      (50–69): trailing 1.0× ATR
     ALERT      (30–49): trailing 0.5× ATR
     EXIT       (<30):  sair imediatamente

Running Mode (TP convertido em TREND_EXPANSION):
  - Teto de TP removido (posição corre indefinidamente)
  - Trailing ganha +0.5× ATR extra de folga
  - Objetivo: capturar +6R, +8R, +12R que pagam dezenas de losses

Anti-bag-holding (proteções duras dentro do HoldEngine):
  P1: preço < entrada AND conviction < 50%
  P2: conviction < 30% por 3+ ciclos consecutivos
  P3: queda > 8% AND conviction < 60% (colapso crypto)

Roda a cada 30s verificando todas as posições abertas.
Uma ExitPlan é criada quando uma posição é aberta (via FillEvent)
e destruída quando a posição é fechada.
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
from ..oms.hold_engine import HoldEngine, trail_atr_mult
from ..oms.order_manager import OrderManager
from ..persistence.cache import Cache
from ..portfolio.engine import PortfolioEngine
from ..strategies.weight_engine import weight_engine as _weight_engine

logger = logging.getLogger(__name__)

# ── Constantes de saída ───────────────────────────────────────────────────────

ATR_PERIOD = 14

# Phase A — Multiplicadores de ATR por regime (SL + TP)
# v3.0.0 (2026-05-27): SL alargado em CHOP/EXHAUSTION para evitar stop prematuro.
# Análise: CHOP com SL=0.7×ATR era eliminado por ruído normal do mercado (< 0.7% move).
# Novo CHOP: SL=1.0×ATR, TP=2.5×ATR → RR=2.5:1 (antes 2.86:1, mas mais sobrevivível).
# EXPANSION: TP elevado 4.5→5.0×ATR para capturar movimentos maiores.
REGIME_MULT: dict[str, dict[str, float]] = {
    "TREND_EXPANSION":        {"sl": 1.5, "tp": 5.0},  # TP 4.5→5.0 (captura mais)
    "VOLATILITY_COMPRESSION": {"sl": 1.2, "tp": 4.0},  # SL 1.0→1.2, TP 3.5→4.0
    "MEAN_REVERTING_CHOP":    {"sl": 1.0, "tp": 2.5},  # SL 0.7→1.0 (anti-stop prematuro)
    "TREND_EXHAUSTION":       {"sl": 0.8, "tp": 2.5},
    "HIGH_CORRELATION_RISK":  {"sl": 0.8, "tp": 2.0},
    "BEAR_TREND":             {"sl": 0.5, "tp": 1.0},
    "PANIC_LIQUIDATION":      {"sl": 0.5, "tp": 1.0},
}
DEFAULT_SL_MULT = 1.0
DEFAULT_TP_MULT = 3.5

# Phase B — Trailing: R mínimo para ativar por regime
REGIME_TRAIL_ACTIVATE: dict[str, float] = {
    "TREND_EXPANSION":        1.2,
    "VOLATILITY_COMPRESSION": 1.0,
    "TREND_EXHAUSTION":       0.8,
    "MEAN_REVERTING_CHOP":    1.5,
    "HIGH_CORRELATION_RISK":  0.8,
    "BEAR_TREND":             0.3,
    "PANIC_LIQUIDATION":      0.2,
}
TRAIL_ACTIVATE_R = 1.0   # fallback
TRAIL_ATR_MULT   = 1.0   # distância base (ajustada pelo conviction state)

# Phase C — Saída parcial: R para vender 50%
REGIME_PARTIAL_EXIT: dict[str, float] = {
    "TREND_EXPANSION":        3.0,
    "VOLATILITY_COMPRESSION": 2.5,
    "MEAN_REVERTING_CHOP":    1.5,
    "TREND_EXHAUSTION":       1.5,
    "HIGH_CORRELATION_RISK":  1.5,
    "BEAR_TREND":             0.5,
    "PANIC_LIQUIDATION":      0.3,
}
PARTIAL_EXIT_R   = 2.5
PARTIAL_EXIT_PCT = 0.50

# Phase D — Regimes de emergência
EXIT_REGIMES = {"PANIC_LIQUIDATION", "BEAR_TREND"}

# Backstop absoluto (substitui timeout antigo): 30 dias
# Só aciona se TODOS os outros mecanismos falharem silenciosamente.
ABSOLUTE_BACKSTOP_DAYS = 30

# Recovery grace period — ciclos sem conviction exit para posições recuperadas.
# 6 ciclos × 30s = 3 minutos: tempo para o Hold Engine estabelecer baseline.
# Durante a carência: SL, TP, trailing e regime emergency continuam ativos.
RECOVERY_GRACE_CYCLES = 6

# ── Fase A — TP Conversion (convexidade) ─────────────────────────────────────
#
# Em vez de fechar tudo no TP, em regimes de tendência o sistema converte:
#   1. Saída parcial (50%) para garantir lucro
#   2. Remoção do teto de TP (posição corre indefinidamente)
#   3. Trailing largo (2.5× ATR) para capturar +6R, +8R, +12R
#
# Só aplicado em regimes onde trends podem correr muito.
# Outros regimes mantêm saída total no TP (range trades não correm).
REGIMES_CONVERT_TP = {"TREND_EXPANSION"}
TRAIL_ATR_RUNNING  = 2.5   # trailing inicial ao converter TP (running mode)


# ── ShortSwapPlan — plano de saída para SHORTs direcionais via SWAP perp ─────

@dataclass
class ShortSwapPlan:
    """
    Plano de saída para SHORT direcional via OKX perpetual swap (BEAR_TREND).
    Criado pelo TradingLoop após _place_swap_short() ser confirmado.
    Destruído quando o PositionMonitor fecha a posição via _close_swap_short().

    Lógica (invertida em relação ao LONG):
      - Stop Loss  : preço ACIMA da entrada (perda se o ativo subir)
      - Take Profit: preço ABAIXO da entrada (ganho se o ativo cair)
    """
    symbol:       str        # spot symbol, ex: "BTC-USDT"
    swap_sym:     str        # swap symbol, ex: "BTC-USDT-SWAP"
    contracts:    int        # número de contratos OKX
    entry_price:  float      # preço spot no momento da entrada
    stop_loss:    float      # preço ACIMA da entrada (shortSL = entry + atr×sl_mult)
    take_profit:  float      # preço ABAIXO da entrada (shortTP = entry − atr×tp_mult)
    backstop_at:  datetime   # expiração máxima (7 dias)
    strategy_id:  str
    created_at:   datetime = field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> str:
        return (
            f"SHORT {self.symbol} contracts={self.contracts} "
            f"entry={self.entry_price:.2f} sl={self.stop_loss:.2f} tp={self.take_profit:.2f}"
        )


# ── ExitPlan ──────────────────────────────────────────────────────────────────

@dataclass
class ExitPlan:
    """
    Plano de saída para uma posição aberta.
    Criado no fill de compra, destruído no fechamento.
    Carrega todo o contexto do trade do nascimento ao fim.
    """
    symbol:        str
    strategy_id:   str
    quantity:      float
    entry_price:   float
    entry_time:    datetime
    entry_regime:  str
    atr:           float

    signal_factors: dict = field(default_factory=dict)

    # Phase A — níveis fixos
    stop_loss:    float = 0.0
    take_profit:  float = 0.0

    # Backstop absoluto (30 dias) — último recurso
    backstop_at:  datetime = field(default_factory=lambda: datetime.now(UTC))

    # Phase B — trailing
    trailing_stop:      float | None = None
    trailing_activated: bool  = False

    # Phase C — saída parcial
    partial_done:   bool  = False
    qty_remaining:  float = 0.0

    # Phase A — TP Conversion / Running Mode
    tp_converted:   bool  = False   # TP foi convertido em trailing livre
    running_mode:   bool  = False   # posição em modo "let it run" (teto removido)

    # Hold Engine — Conviction Decay
    conviction_score:      float = 100.0
    conviction_state:      str   = "HOLD"      # HOLD/WATCH/ALERT/EXIT
    conviction_history:    list  = field(default_factory=list)   # últimos 10
    conviction_components: dict  = field(default_factory=dict)
    low_conviction_streak: int   = 0

    # Recovery grace period — posições recuperadas após restart precisam de
    # N ciclos antes de conviction exits serem permitidos.
    # Evita que o bot saia imediatamente após restart sem histórico de conviction.
    # SL/TP/Trailing/Regime emergência continuam ativos durante a carência.
    recovered:             bool  = False   # True = posição vinda de _recover_existing_positions
    grace_cycles_remaining: int  = 0       # ciclos restantes de carência

    def __post_init__(self) -> None:
        sl_pct = self.signal_factors.get("sl_pct", 0.0)
        tp_pct = self.signal_factors.get("tp_pct", 0.0)

        if sl_pct > 0 and tp_pct > 0:
            # Estratégias de reversão: SL/TP relativos ao fill price
            self.stop_loss   = round(self.entry_price * (1.0 - sl_pct), 4)
            self.take_profit = round(self.entry_price * (1.0 + tp_pct), 4)
        else:
            # ATR-based (momentum / fallback)
            mults   = REGIME_MULT.get(self.entry_regime, {})
            sl_mult = mults.get("sl", DEFAULT_SL_MULT)
            tp_mult = mults.get("tp", DEFAULT_TP_MULT)
            self.stop_loss   = round(self.entry_price - self.atr * sl_mult, 4)
            self.take_profit = round(self.entry_price + self.atr * tp_mult, 4)

        self.backstop_at   = self.entry_time + timedelta(days=ABSOLUTE_BACKSTOP_DAYS)
        self.qty_remaining = self.quantity

    @property
    def r_value(self) -> float:
        """1R = distância entre entrada e stop loss."""
        return self.entry_price - self.stop_loss

    def price_at_r(self, multiples: float) -> float:
        """Preço correspondente a N × R de lucro."""
        return self.entry_price + self.r_value * multiples

    def update_conviction(self, score: float, state: str, components: dict) -> None:
        """Atualiza conviction e gerencia streak de baixa convicção."""
        self.conviction_score      = score
        self.conviction_state      = state
        self.conviction_components = components
        self.conviction_history    = (self.conviction_history + [score])[-10:]
        if score < 30.0:
            self.low_conviction_streak += 1
        else:
            self.low_conviction_streak = 0

    def summary(self) -> str:
        tp_str = "∞(running)" if self.running_mode else f"{self.take_profit:.2f}"
        return (
            f"{self.symbol} entry={self.entry_price:.2f} "
            f"sl={self.stop_loss:.2f} tp={tp_str} "
            f"atr={self.atr:.2f} regime={self.entry_regime} "
            f"conviction={self.conviction_score:.0f}%({self.conviction_state}) "
            f"trail={'ON' if self.trailing_activated else 'off'} "
            f"partial={'done' if self.partial_done else 'pending'} "
            f"running={'YES' if self.running_mode else 'no'}"
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
        self._bus       = bus
        self._market    = market
        self._portfolio = portfolio
        self._oms       = oms
        self._cache     = cache
        self._interval  = interval_seconds

        # Hold Engine — avalia conviction de cada trade
        self._hold_engine = HoldEngine(market=market, cache=cache)

        # ExitPlans ativos: symbol → ExitPlan
        self._plans: dict[str, ExitPlan] = {}

        self._running    = False
        self._task: asyncio.Task | None       = None
        self._fill_task: asyncio.Task | None  = None
        self._fill_queue: asyncio.Queue | None = None
        self._exits_today = 0

        self._pending_factors: dict[str, dict] = {}

        # ── SHORT direcional via swap perp ────────────────────────────────────
        # ShortSwapPlans ativos: symbol → ShortSwapPlan
        self._short_plans: dict[str, ShortSwapPlan] = {}
        # CrossAssetEngine injetado via set_cross_asset() após startup
        self._cross_asset = None

        # ── Ghost Fill Detection ──────────────────────────────────────────────
        # Detecta ExitPlans sem posição real no OKX (fills fantasma do paper trading).
        # _ghost_streak: quantas verificações consecutivas sem posição no Redis.
        # Após GHOST_STREAK_THRESHOLD checks → ExitPlan removido + alerta.
        self._ghost_streak: dict[str, int] = {}
        self._check_cycle: int = 0   # contador de ciclos para trigger periódico

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running    = True
        self._fill_queue = self._bus.subscribe(Topic.FILL)
        self._fill_task  = asyncio.create_task(
            self._consume_fills(), name="position_monitor_fills"
        )
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
        while self._running:
            try:
                event = await asyncio.wait_for(self._fill_queue.get(), timeout=1.0)
                await self._on_order_event(event)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("PositionMonitor fill consumer: %s", exc, exc_info=True)

    # ── Fill → criar ExitPlan ─────────────────────────────────────────────────

    def set_pending_signal_factors(self, symbol: str, factors: dict) -> None:
        self._pending_factors[symbol] = factors

    async def _on_order_event(self, event) -> None:
        if not isinstance(event, OrderFilledEvent):
            return
        order = event.order
        if order is None:
            return
        side = str(getattr(order, "side", "")).upper()
        if side not in ("BUY", "LONG"):
            return

        symbol   = order.symbol
        qty      = order.filled_quantity or order.quantity
        price    = order.avg_fill_price or 0.0
        strat_id = order.strategy_id or "unknown"

        if price <= 0 or qty <= 0:
            return

        # ── Alerta de fill parcial baixo ──────────────────────────────────────
        # Fill ratio < 70% em paper trading = risco de "ghost fill" no OKX demo:
        # o OKX pode registrar o fill localmente mas não abrir a posição.
        ordered_qty  = order.quantity or qty
        fill_ratio   = qty / ordered_qty if ordered_qty > 0 else 1.0
        if fill_ratio < 0.70:
            logger.warning(
                "FILL PARCIAL BAIXO %s: %.1f%% preenchido (%.4f de %.4f) — "
                "risco de ghost fill no OKX demo. Ghost detector ativo.",
                symbol, fill_ratio * 100, qty, ordered_qty,
            )

        signal_factors = self._pending_factors.pop(symbol, {})
        await self._create_plan(symbol, qty, price, strat_id, signal_factors=signal_factors)

    async def _create_plan(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        strategy_id: str,
        signal_factors: dict | None = None,
        recovered: bool = False,
    ) -> None:
        candles_1h = self._market.get_candles(symbol, "1H", limit=ATR_PERIOD + 5)
        atr    = _calc_atr(candles_1h, ATR_PERIOD) if len(candles_1h) >= ATR_PERIOD else 0.0
        atr    = atr or entry_price * 0.015
        # B4 fix: usa regime do sinal original (ADX-based) se disponível,
        # evitando discrepância com a detecção SMA-based do monitor.
        regime = (signal_factors or {}).get("regime") or _detect_regime(candles_1h)

        plan = ExitPlan(
            symbol=symbol,
            strategy_id=strategy_id,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(UTC),
            entry_regime=regime,
            atr=atr,
            signal_factors=signal_factors or {},
            recovered=recovered,
            grace_cycles_remaining=RECOVERY_GRACE_CYCLES if recovered else 0,
        )
        self._plans[symbol] = plan

        if recovered:
            logger.info(
                "ExitPlan RECUPERADO (carencia=%d ciclos): %s",
                RECOVERY_GRACE_CYCLES, plan.summary(),
            )
        else:
            logger.info("ExitPlan criado: %s", plan.summary())

    async def _recover_existing_positions(self) -> None:
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
                await self._create_plan(
                    symbol, quantity, entry_price, strat_id,
                    recovered=True,
                )

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
        # Verifica SHORTs direcionais via swap (sempre, independente de long plans)
        await self._check_short_plans()

        if not self._plans:
            self._check_cycle = 0
            return

        self._check_cycle += 1

        # Ghost detection a cada 10 ciclos (~5min com interval=30s)
        # Detecta fills fantasma: ExitPlan existe mas OKX não tem a posição.
        if self._check_cycle % 10 == 1:
            await self._detect_ghost_plans()

        # Fix: Refresha TTL do Redis para todas as posições abertas em memória.
        # O TTL padrão é 60s e só é escrito no fill event — posições em hold
        # expiram silenciosamente, causando falso ghost detection.
        for symbol in list(self._plans.keys()):
            try:
                mem_pos = self._portfolio.state.positions.get(symbol)
                if mem_pos:
                    pos_data = mem_pos if isinstance(mem_pos, dict) else vars(mem_pos)
                    await self._cache.set_position(symbol, pos_data)
            except Exception as exc:
                logger.debug("Refresh Redis pos %s: %s", symbol, exc)

        now = datetime.now(UTC)
        for symbol, plan in list(self._plans.items()):
            try:
                await self._evaluate_plan(symbol, plan, now)
            except Exception as exc:
                logger.error("Erro ao avaliar %s: %s", symbol, exc, exc_info=True)

    async def _detect_ghost_plans(self) -> None:
        """
        Detecta ExitPlans sem posição real correspondente no OKX/Redis.

        Contexto: OKX paper trading pode confirmar um fill localmente mas não
        abrir a posição de fato (comum em fills parciais < 70%). O reconciliador
        eventualmente remove a posição do Redis, mas o ExitPlan permanece ativo,
        tentando gerenciar uma posição que não existe.

        Lógica:
          - Verifica cache.get_position(symbol) para cada ExitPlan ativo
          - Se ausente: incrementa ghost_streak
          - ghost_streak >= 2 ciclos consecutivos → ExitPlan fantasma confirmado
          - Remove o plano + loga warning para investigação
          - Reset do streak se posição reaparece (ex: reconciliador a importou)

        Intervalo: a cada 10 ciclos = ~5min (suficiente para reconciliador rodar 2x).
        """
        if not self._plans:
            return

        for symbol in list(self._plans.keys()):
            try:
                pos_data = await self._cache.get_position(symbol)

                if pos_data:
                    # Posição confirmada no Redis → reset do streak
                    if symbol in self._ghost_streak:
                        logger.debug(
                            "Ghost streak reset: %s — posicao reapareceu no Redis",
                            symbol,
                        )
                        self._ghost_streak.pop(symbol, None)
                    continue

                # Posição ausente — incrementa streak
                streak = self._ghost_streak.get(symbol, 0) + 1
                self._ghost_streak[symbol] = streak

                logger.warning(
                    "GHOST CHECK %s: posicao nao encontrada no Redis "
                    "(streak=%d/2) — possivel fill fantasma OKX demo",
                    symbol, streak,
                )

                if streak >= 2:
                    # Fix: Antes de remover, cruza com estado em-memória.
                    # Redis pode ter expirado (TTL 60s) enquanto a posição é real.
                    # Só remove se AMBOS Redis e in-memory concordam que não existe.
                    mem_pos = self._portfolio.state.positions.get(symbol)
                    if mem_pos:
                        logger.warning(
                            "GHOST FALSE POSITIVE %s: Redis expirou mas posicao "
                            "existe em memoria — refresh Redis, ExitPlan mantido.",
                            symbol,
                        )
                        pos_data = mem_pos if isinstance(mem_pos, dict) else vars(mem_pos)
                        await self._cache.set_position(symbol, pos_data)
                        self._ghost_streak.pop(symbol, None)
                        continue

                    # Confirmado: ausente no Redis E em memória → ExitPlan fantasma
                    plan = self._plans.pop(symbol, None)
                    self._ghost_streak.pop(symbol, None)

                    if plan:
                        entry_age = (
                            datetime.now(UTC) - plan.entry_time
                        ).total_seconds() / 60

                        logger.error(
                            "GHOST FILL CONFIRMADO: %s | entry=%.2f | "
                            "entry_age=%.0fmin | regime=%s | "
                            "ExitPlan removido — posicao ausente no Redis E em memoria. "
                            "Verifique fill parcial na OKX demo.",
                            symbol, plan.entry_price, entry_age, plan.entry_regime,
                        )

            except Exception as exc:
                logger.warning("Ghost detection erro %s: %s", symbol, exc)

    # ── Avaliação do plano ────────────────────────────────────────────────────

    async def _evaluate_plan(
        self,
        symbol: str,
        plan: ExitPlan,
        now: datetime,
    ) -> None:
        price = await self._cache.get_price(symbol)
        if not price or price <= 0:
            return
        price = float(price)

        candles_1h     = self._market.get_candles(symbol, "1H", limit=25)
        current_regime = (
            _detect_regime(candles_1h) if len(candles_1h) >= 5
            else "MEAN_REVERTING_CHOP"
        )

        # ── Phase D — Regime de emergência ───────────────────────────────────
        if current_regime in EXIT_REGIMES:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason=f"regime_{current_regime.lower()}", partial=False,
            )
            return

        # ── Phase A — Stop Loss (safety net absoluta) ─────────────────────────
        effective_sl = plan.trailing_stop if plan.trailing_stop else plan.stop_loss
        if price <= effective_sl:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason="stop_loss" if not plan.trailing_stop else "trailing_stop",
                partial=False,
            )
            return

        # ── Phase A — Take Profit / TP Conversion ────────────────────────────
        if price >= plan.take_profit and not plan.tp_converted:
            if plan.entry_regime in REGIMES_CONVERT_TP:
                # ── TREND_EXPANSION: converter TP em Running Mode ─────────────
                # 1. Saída parcial se ainda não foi feita
                if not plan.partial_done:
                    qty_to_sell = round(plan.qty_remaining * PARTIAL_EXIT_PCT, 8)
                    await self._exit(
                        plan, price, qty_to_sell,
                        reason="tp_partial_conversion", partial=True,
                    )
                    plan.qty_remaining = round(plan.qty_remaining - qty_to_sell, 8)
                    plan.partial_done  = True

                # 2. Remover teto de TP e ativar running mode
                plan.tp_converted = True
                plan.running_mode = True
                plan.take_profit  = price * 99   # efetivamente infinito

                # 3. Trailing largo para o winner respirar
                running_trail = round(price - plan.atr * TRAIL_ATR_RUNNING, 4)
                if not plan.trailing_activated or running_trail > (plan.trailing_stop or 0):
                    plan.trailing_stop      = running_trail
                    plan.trailing_activated = True

                logger.info(
                    "%s: TP → RUNNING MODE @ %.2f | trailing=%.2f (%.1f×ATR) "
                    "| qty_remaining=%.4f | regime=%s",
                    symbol, price, plan.trailing_stop, TRAIL_ATR_RUNNING,
                    plan.qty_remaining, plan.entry_regime,
                )
                # Não retorna — continua avaliando neste ciclo
            else:
                # Outros regimes: saída total no TP (range trades não correm)
                await self._exit(
                    plan, price, plan.qty_remaining,
                    reason="take_profit", partial=False,
                )
                return

        # ── Backstop absoluto (30 dias) — último recurso ──────────────────────
        if now >= plan.backstop_at:
            await self._exit(
                plan, price, plan.qty_remaining,
                reason="absolute_backstop_30d", partial=False,
            )
            return

        # ── Hold Engine — Conviction Decay ────────────────────────────────────
        conviction = await self._hold_engine.evaluate(
            symbol=symbol,
            entry_price=plan.entry_price,
            low_conviction_streak=plan.low_conviction_streak,
        )
        plan.update_conviction(
            score=conviction.score,
            state=conviction.state,
            components=conviction.components,
        )

        logger.debug(
            "%s conviction=%.0f state=%s streak=%d grace=%d",
            symbol, conviction.score, conviction.state,
            plan.low_conviction_streak, plan.grace_cycles_remaining,
        )

        # Recovery grace period — posições recuperadas após restart aguardam
        # RECOVERY_GRACE_CYCLES ciclos antes de conviction exits serem permitidos.
        # Previne saída imediata após restart (sem histórico de conviction).
        # SL, TP, trailing e regime emergency continuam ativos durante a carência.
        if plan.grace_cycles_remaining > 0:
            plan.grace_cycles_remaining -= 1
            if conviction.state == "EXIT":
                logger.info(
                    "%s conviction EXIT ignorado — carência ativa (%d ciclos restantes)",
                    symbol, plan.grace_cycles_remaining,
                )
        elif conviction.state == "EXIT":
            await self._exit(
                plan, price, plan.qty_remaining,
                reason=conviction.exit_reason or "conviction_exit",
                partial=False,
            )
            return

        # ── Phase C — Saída parcial dinâmica ──────────────────────────────────
        partial_r = REGIME_PARTIAL_EXIT.get(plan.entry_regime, PARTIAL_EXIT_R)
        if not plan.partial_done and price >= plan.price_at_r(partial_r):
            qty_to_sell = round(plan.quantity * PARTIAL_EXIT_PCT, 8)
            await self._exit(
                plan, price, qty_to_sell,
                reason="partial_take_profit", partial=True,
            )
            plan.partial_done  = True
            plan.qty_remaining = round(plan.qty_remaining - qty_to_sell, 8)
            logger.info(
                "%s: saída parcial %.4f @ %.2f (+%.1fR, regime=%s) — restante: %.4f",
                symbol, qty_to_sell, price, partial_r,
                plan.entry_regime, plan.qty_remaining,
            )

        # ── Phase B — Trailing stop (conviction-adaptive + running mode) ────────
        #
        # ATR multiplier = trail_atr_mult(conviction_score, running_mode)
        #   HOLD forte (≥85): 2.5× ATR   — winners respiram
        #   HOLD       (70–84): 2.0× ATR
        #   WATCH      (50–69): 1.0× ATR
        #   ALERT      (30–49): 0.5× ATR
        #   Running mode: +0.5× em todos os níveis
        #
        # Regra: trailing só sobe (ratchet), nunca desce para proteger de whipsaw.
        # Exceção: quando conviction degrada E trailing está frouxo demais,
        #          o trailing APERTA para o nível da nova conviction.
        trail_r    = REGIME_TRAIL_ACTIVATE.get(plan.entry_regime, TRAIL_ACTIVATE_R)
        atr_mult   = trail_atr_mult(conviction.score, plan.running_mode)

        if price >= plan.price_at_r(trail_r) or plan.running_mode:
            new_trail = round(price - plan.atr * atr_mult, 4)
            current   = plan.trailing_stop or 0.0

            if not plan.trailing_activated:
                plan.trailing_stop      = new_trail
                plan.trailing_activated = True
                logger.info(
                    "%s: trailing ativado @ %.2f (%.1f×ATR conviction=%.0f%% %s%s)",
                    symbol, new_trail, atr_mult, conviction.score,
                    conviction.state, " RUNNING" if plan.running_mode else "",
                )
            elif new_trail > current:
                # Ratchet up — trailing segue o preço para cima
                plan.trailing_stop = new_trail
                logger.debug(
                    "%s: trailing ↑ %.2f→%.2f (%.1f×ATR conv=%.0f%%)",
                    symbol, current, new_trail, atr_mult, conviction.score,
                )
            elif new_trail < current:
                # Conviction degradou → tighten se o novo mult é mais apertado
                # (ex: era HOLD 2.0×, virou WATCH 1.0× — aperta o trailing)
                tighter = round(price - plan.atr * atr_mult, 4)
                if tighter > current:
                    plan.trailing_stop = tighter
                    logger.info(
                        "%s: trailing apertado por conviction %s→%.0f%% → %.2f",
                        symbol, conviction.state, conviction.score, tighter,
                    )

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
            "SAÍDA %s %s qty=%.4f price=%.2f pnl=%.2fR reason=%s conviction=%.0f(%s)",
            "PARCIAL" if partial else "TOTAL",
            symbol, quantity, price, pnl_r, reason,
            plan.conviction_score, plan.conviction_state,
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
                self._plans.pop(symbol, None)

                # ── Online Learning: registra trade real no WeightEngine ───────
                # Permite que o Portfolio Allocator converja da simulação para
                # a realidade à medida que trades reais se acumulam.
                try:
                    # Usa `quantity` (quantidade efetivamente vendida neste exit),
                    # não plan.quantity (total original) — evita superestimar PnL
                    # em casos de saída parcial prévia.
                    pnl_usd = (price - plan.entry_price) * quantity
                    _weight_engine.record_trade(
                        strategy_id=plan.strategy_id,
                        regime=plan.entry_regime,
                        pnl=pnl_usd,
                        entry_price=plan.entry_price,
                    )
                except Exception as we_exc:
                    logger.debug("WeightEngine.record_trade falhou: %s", we_exc)

    # ── SHORT direcional via swap perp ───────────────────────────────────────

    def set_cross_asset(self, engine) -> None:
        """Injeta o CrossAssetEngine para poder fechar SHORTs via swap."""
        self._cross_asset = engine

    def register_short_plan(
        self,
        symbol: str,
        swap_sym: str,
        contracts: int,
        entry_price: float,
        atr: float,
        strategy_id: str,
    ) -> None:
        """
        Registra um ShortSwapPlan após abertura confirmada de SHORT direcional.
        Chamado pelo TradingLoop após _place_swap_short() retornar.

        SL/TP invertidos (SHORT):
          - SL = entry + atr × 0.5   (fecha com perda se mercado SUBIR)
          - TP = entry − atr × 1.0   (fecha com lucro se mercado CAIR)
          - Backstop: 7 dias (shorts direcionais não devem durar mais)
        """
        sl_mult = REGIME_MULT.get("BEAR_TREND", {}).get("sl", 0.5)
        tp_mult = REGIME_MULT.get("BEAR_TREND", {}).get("tp", 1.0)
        plan = ShortSwapPlan(
            symbol=symbol,
            swap_sym=swap_sym,
            contracts=contracts,
            entry_price=entry_price,
            stop_loss=round(entry_price + atr * sl_mult, 2),
            take_profit=round(entry_price - atr * tp_mult, 2),
            backstop_at=datetime.now(UTC) + timedelta(days=7),
            strategy_id=strategy_id,
        )
        self._short_plans[symbol] = plan
        logger.info("ShortSwapPlan criado: %s", plan.summary())

    async def _check_short_plans(self) -> None:
        """Verifica SL/TP/backstop para todos os SHORTs direcionais ativos."""
        if not self._short_plans:
            return
        now = datetime.now(UTC)
        for symbol, plan in list(self._short_plans.items()):
            try:
                price_raw = await self._cache.get_price(symbol)
                if not price_raw or float(price_raw) <= 0:
                    continue
                price = float(price_raw)

                reason: str | None = None
                if price >= plan.stop_loss:
                    reason = "short_stop_loss"
                elif price <= plan.take_profit:
                    reason = "short_take_profit"
                elif now >= plan.backstop_at:
                    reason = "short_backstop_7d"

                if reason:
                    await self._close_short_plan(plan, price, reason)
            except Exception as exc:
                logger.error("ShortSwapPlan check %s: %s", symbol, exc, exc_info=True)

    async def _close_short_plan(
        self, plan: ShortSwapPlan, price: float, reason: str
    ) -> None:
        """Fecha SHORT via BUY_COVER no swap perp e remove o plano."""
        symbol = plan.symbol
        # PnL SHORT = (entrada − saída) × qty_base (positivo quando mercado cai)
        cs_map = {"BTC-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.1, "SOL-USDT-SWAP": 1.0}
        cs     = cs_map.get(plan.swap_sym, 1.0)
        pnl    = (plan.entry_price - price) * plan.contracts * cs

        logger.info(
            "SHORT EXIT %s contracts=%d entry=%.2f exit=%.2f pnl≈%.2f USDT reason=%s",
            symbol, plan.contracts, plan.entry_price, price, pnl, reason,
        )

        if self._cross_asset is None:
            logger.error(
                "ShortSwapPlan: CrossAssetEngine não injetado — não é possível "
                "fechar %s automaticamente. Intervenção manual necessária.", symbol,
            )
            return

        try:
            await self._cross_asset._close_swap_short(plan.swap_sym, plan.contracts)
            self._short_plans.pop(symbol, None)
            self._exits_today += 1
            logger.info("SHORT EXIT %s executado com sucesso (reason=%s)", symbol, reason)
        except Exception as exc:
            logger.error("SHORT EXIT falha %s: %s", symbol, exc, exc_info=True)

    # ── Status (para dashboard e API) ────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running":        self._running,
            "active_plans":   len(self._plans),
            "active_shorts":  len(self._short_plans),
            "exits_today":    self._exits_today,
            "short_plans": {
                sym: {
                    "swap_sym":    p.swap_sym,
                    "contracts":   p.contracts,
                    "entry_price": p.entry_price,
                    "stop_loss":   p.stop_loss,
                    "take_profit": p.take_profit,
                    "backstop_at": p.backstop_at.isoformat(),
                    "age_hours":   round(
                        (datetime.now(UTC) - p.created_at).total_seconds() / 3600, 2
                    ),
                }
                for sym, p in self._short_plans.items()
            },
            "ghost_suspects": dict(self._ghost_streak),   # symbols sob suspeita
            "plans": {
                sym: {
                    "entry_price":         p.entry_price,
                    "stop_loss":           p.trailing_stop or p.stop_loss,
                    "take_profit":         p.take_profit,
                    "trailing":            p.trailing_activated,
                    "partial_done":        p.partial_done,
                    "qty_remaining":       p.qty_remaining,
                    "regime":              p.entry_regime,
                    # Running Mode (convexidade)
                    "tp_converted":        p.tp_converted,
                    "running_mode":        p.running_mode,
                    # Hold Engine
                    "conviction_score":    round(p.conviction_score, 1),
                    "conviction_state":    p.conviction_state,
                    "conviction_history":  p.conviction_history,
                    "conviction_components": {
                        k: round(v, 3)
                        for k, v in p.conviction_components.items()
                    },
                    "low_conviction_streak": p.low_conviction_streak,
                    "backstop_at":         p.backstop_at.isoformat(),
                }
                for sym, p in self._plans.items()
            },
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _calc_atr(candles: list[Candle], period: int = 14) -> float:
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
    if len(candles) < 20:
        return "MEAN_REVERTING_CHOP"

    closes  = [c.close  for c in candles[:20]]
    volumes = [c.volume for c in candles[:20]]

    sma_fast = sum(closes[:5])  / 5
    sma_slow = sum(closes[:20]) / 20
    avg_vol  = sum(volumes) / len(volumes)
    last_vol = volumes[0]

    if len(closes) >= 2 and closes[1] > 0:
        if (closes[0] - closes[1]) / closes[1] < -0.05:
            return "PANIC_LIQUIDATION"

    if sma_fast > sma_slow:
        if last_vol > avg_vol * 1.2:
            return "TREND_EXPANSION"
        if last_vol < avg_vol * 0.8:
            return "TREND_EXHAUSTION"
        return "VOLATILITY_COMPRESSION"

    highs  = [c.high for c in candles[:10]]
    lows   = [c.low  for c in candles[:10]]
    atr_5  = sum(hi - lo for hi, lo in zip(highs[:5], lows[:5], strict=True)) / 5
    rel_atr = atr_5 / closes[0] if closes[0] > 0 else 0

    if len(closes) >= 11 and closes[10] > 0:
        if (closes[0] - closes[10]) / closes[10] < -0.02:
            return "BEAR_TREND"

    if rel_atr > 0.030:
        return "HIGH_CORRELATION_RISK"

    return "MEAN_REVERTING_CHOP"
