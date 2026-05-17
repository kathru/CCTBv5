"""
Trading Loop — wires all engines together and runs the main cycle.

Pipeline completo (corrigido):
  MarketEngine → (CandleEvent) → StrategyRunner
  StrategyRunner → (SignalEvent) → [signal_consumer task]
  signal_consumer → RiskEngine.evaluate() → sizing → OrderManager
  OrderManager → ExecutionRouter → OKX
  OKX fill → (FillEvent) → PositionMonitor → exit orders

Parallel services:
  - PeriodicReconciler (every 2 min)
  - HeartbeatWatchdog (every 15s)
  - WebSocketWatchdog (every 5s)
  - AlertListener (continuous)
  - PositionMonitor (every 30s)

Boot sequence (BootSequence handles this):
  Connect → Schema → RECONCILING → Load state → Reconcile → RUNNING
"""

import asyncio
import logging
from pathlib import Path

from ..alerts.discord import create_alert_channel
from ..alerts.listener import AlertListener
from ..exchange.okx.client import OKXClient
from ..market.engine import MarketEngine
from ..metrics.infra_metrics import InfraMetrics
from ..oms.execution_router import ExecutionRouter
from ..oms.order_manager import OrderManager
from ..oms.position_monitor import PositionMonitor
from ..persistence import Cache, Database
from ..portfolio.engine import PortfolioEngine
from ..recovery.boot import BootSequence
from ..recovery.periodic_reconciler import PeriodicReconciler
from ..risk.engine import RiskContext, RiskEngine
from ..risk.kill_switch import KillSwitch
from ..strategies.meta_layer import MetaStrategyLayer
from ..strategies.ml.inference import MLInferenceEngine
from ..strategies.momentum.v4_strategy import V4MomentumStrategy
from ..strategies.runner import StrategyRunner
from ..watchdog.heartbeat import HeartbeatWatchdog
from ..watchdog.resource_watchdog import ResourceWatchdog
from ..watchdog.websocket_watchdog import WebSocketWatchdog
from .bus import EventBus
from .config import settings
from .events import SignalEvent, Topic
from .events.risk_events import RiskAction

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
MODELS_DIR = Path("data") / "models"

# Quantidade mínima por símbolo (OKX spot minimums)
MIN_QTY = {
    "BTC-USDT": 0.00001,
    "ETH-USDT": 0.0001,
    "SOL-USDT": 0.01,
}
# Precisão de casas decimais por símbolo
QTY_PRECISION = {
    "BTC-USDT": 5,
    "ETH-USDT": 4,
    "SOL-USDT": 2,
}


class TradingLoop:
    """
    Main trading loop — composes all engines and runs them.
    Call start() to begin, stop() to shut down gracefully.
    """

    def __init__(
        self,
        db: Database,
        cache: Cache,
        app_state=None,
    ) -> None:
        self._db = db
        self._cache = cache
        self._app_state = app_state
        self._running = False

        # ── Core infrastructure ───────────────────────────────
        self._bus = EventBus()
        self._kill_switch = KillSwitch()
        self._infra_metrics = InfraMetrics()

        # ── OKX client ────────────────────────────────────────
        self._okx = OKXClient(
            api_key=settings.okx_api_key,
            secret_key=settings.okx_secret_key,
            passphrase=settings.okx_passphrase,
            paper_trading=settings.okx_paper_trading,
        )

        # ── Market Engine ─────────────────────────────────────
        self._market = MarketEngine(
            bus=self._bus,
            okx=self._okx,
            cache=self._cache,
            symbols=SYMBOLS,
            granularities=["1H", "6H"],
            poll_interval=15,
        )

        # ── OMS ───────────────────────────────────────────────
        router = ExecutionRouter(exchange=self._okx)
        self._oms = OrderManager(
            bus=self._bus,
            router=router,
        )

        # ── Risk Engine ───────────────────────────────────────
        self._risk = RiskEngine(
            bus=self._bus,
            kill_switch=self._kill_switch,
        )

        # ── Portfolio Engine ──────────────────────────────────
        self._portfolio = PortfolioEngine(
            bus=self._bus,
            initial_capital=10000.0,
        )

        # ── ML Inference ──────────────────────────────────────
        self._ml = MLInferenceEngine(models_dir=MODELS_DIR)

        # ── Meta Layer ────────────────────────────────────────
        self._meta = MetaStrategyLayer()

        # ── Strategy Runner ───────────────────────────────────
        self._runner = StrategyRunner(
            bus=self._bus,
            market=self._market,
            cache=self._cache,
        )
        v4 = V4MomentumStrategy(symbols=SYMBOLS)
        self._runner.register(v4)
        self._meta.register(v4.strategy_id)

        # ── Alerts ────────────────────────────────────────────
        self._alert_channel = create_alert_channel(settings.discord_webhook_url)
        self._alert_listener = AlertListener(
            bus=self._bus,
            channel=self._alert_channel,
        )

        # ── Watchdogs ─────────────────────────────────────────
        self._heartbeat = HeartbeatWatchdog(
            bus=self._bus,
            kill_switch=self._kill_switch,
            timeout_seconds=60,
            interval_seconds=15,
        )
        self._ws_watchdog = WebSocketWatchdog(
            on_dead=self._on_ws_dead,
            dead_threshold=120,
            name="okx_market",
        )
        self._resource_watchdog = ResourceWatchdog(
            kill_switch=self._kill_switch,
            interval_seconds=30,
        )

        # ── Position Monitor ──────────────────────────────────
        self._position_monitor = PositionMonitor(
            bus=self._bus,
            market=self._market,
            portfolio=self._portfolio,
            oms=self._oms,
            cache=self._cache,
            interval_seconds=30,
        )

        # ── Periodic Reconciler ───────────────────────────────
        self._reconciler = PeriodicReconciler(
            bus=self._bus,
            db=self._db,
            exchange=self._okx,
            order_manager=self._oms,
            kill_switch=self._kill_switch,
            interval_seconds=120,
        )

        # ── Signal consumer (pipeline SIGNAL → RISK → OMS) ───
        self._signal_queue: asyncio.Queue | None = None
        self._signal_task: asyncio.Task | None = None

        self._market.set_on_poll_callback(self._ws_watchdog.record_message)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        logger.info("TradingLoop: starting boot sequence...")

        boot = BootSequence(
            bus=self._bus,
            db=self._db,
            cache=self._cache,
            exchange=self._okx,
            order_manager=self._oms,
        )
        ready = await boot.run()
        if not ready:
            logger.error("TradingLoop: boot failed — not starting")
            return False

        if self._app_state:
            self._app_state.ws_watchdog       = self._ws_watchdog
            self._app_state.heartbeat_watchdog = self._heartbeat
            self._app_state.resource_watchdog  = self._resource_watchdog
            self._app_state.portfolio          = self._portfolio
            self._app_state.position_monitor   = self._position_monitor

        # ── Subscreve ao pipeline de sinais ──────────────────
        self._signal_queue = self._bus.subscribe(Topic.SIGNAL)
        self._signal_task  = asyncio.create_task(
            self._consume_signals(), name="signal_consumer"
        )

        await self._alert_listener.start()
        await self._heartbeat.start()
        await self._ws_watchdog.start()
        await self._resource_watchdog.start()
        await self._market.start()
        await self._runner.start()
        await self._position_monitor.start()
        await self._reconciler.start()

        self._running = True
        logger.info("TradingLoop: all services started — RUNNING")

        await self._alert_channel.info(
            title="CCTBv5 Started",
            message="Sistema iniciado com sucesso.",
            mode="paper" if settings.okx_paper_trading else "live",
            symbols=", ".join(SYMBOLS),
        )

        await self._run_loop()
        return True

    async def stop(self) -> None:
        self._running = False
        logger.info("TradingLoop: shutting down...")

        if self._signal_task and not self._signal_task.done():
            self._signal_task.cancel()
            try:
                await self._signal_task
            except asyncio.CancelledError:
                pass

        await self._market.stop()
        await self._runner.stop()
        await self._position_monitor.stop()
        await self._reconciler.stop()
        await self._heartbeat.stop()
        await self._ws_watchdog.stop()
        await self._resource_watchdog.stop()
        await self._alert_listener.stop()

        await self._alert_channel.info(
            title="CCTBv5 Stopped",
            message="Sistema encerrado graciosamente.",
        )
        logger.info("TradingLoop: shutdown complete")

    # ── Pipeline: Signal → Risk → Sizing → OMS ───────────────────────────────

    async def _consume_signals(self) -> None:
        """
        Consumer do pipeline de sinais.
        Recebe cada SignalEvent, avalia risco, dimensiona e envia ao OMS.
        """
        logger.info("Signal consumer iniciado — aguardando sinais...")
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._signal_queue.get(), timeout=1.0
                )
                if isinstance(event, SignalEvent) and event.signal:
                    await self._process_signal(event)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Signal consumer error: %s", exc, exc_info=True)

    async def _process_signal(self, event: SignalEvent) -> None:
        """
        Processa um sinal aprovado pela estratégia:
          1. RiskEngine — verifica limites de drawdown, exposição, cooldown
          2. Sizing     — calcula quantidade usando kelly_fraction e preço atual
          3. OMS        — cria e submete ordem de mercado
        """
        signal = event.signal

        # 1. Avaliação de risco
        portfolio_value = self._portfolio.state.total_value or 10000.0
        risk_ctx = RiskContext(
            portfolio_value=portfolio_value,
            open_positions=[],
            strategy_id=signal.strategy_id,
        )
        action = await self._risk.evaluate(risk_ctx)

        if action != RiskAction.NORMAL:
            logger.info(
                "Sinal rejeitado pelo RiskEngine action=%s symbol=%s strategy=%s",
                action, signal.symbol, signal.strategy_id,
            )
            return

        # 2. Sizing usando kelly_fraction e preço atual
        price_raw = await self._cache.get_price(signal.symbol)
        if not price_raw:
            logger.warning(
                "Sem preço em cache para %s — sinal descartado", signal.symbol
            )
            return

        price = float(price_raw)
        if price <= 0:
            return

        # Kelly já vem ajustado pelo regime_mult da estratégia
        kelly    = signal.kelly_fraction or 0.05
        regime   = getattr(signal, "regime", "MEAN_REVERTING_CHOP")

        # Cap máximo de Kelly por família de regime (segurança extra)
        KELLY_CAP = {
            "TREND_EXPANSION":        0.15,
            "VOLATILITY_COMPRESSION": 0.12,
            "TREND_EXHAUSTION":       0.10,
            "MEAN_REVERTING_CHOP":    0.08,
            "HIGH_CORRELATION_RISK":  0.05,
        }
        kelly    = min(kelly, KELLY_CAP.get(regime, 0.08))
        notional = portfolio_value * kelly
        precision = QTY_PRECISION.get(signal.symbol, 4)
        quantity  = round(notional / price, precision)
        min_qty   = MIN_QTY.get(signal.symbol, 0.0001)

        if quantity < min_qty:
            logger.info(
                "Quantidade %.6f < mínimo %.6f para %s — sinal descartado",
                quantity, min_qty, signal.symbol,
            )
            return

        logger.info(
            "Sinal aprovado: %s %s regime=%s qty=%.6f price=%.2f "
            "notional=%.2f kelly=%.1f%% (cap=%.0f%%)",
            signal.direction, signal.symbol, regime,
            quantity, price, notional, kelly * 100,
            KELLY_CAP.get(regime, 0.08) * 100,
        )

        # 3. Criar e submeter ordem via OMS
        await self._oms.create_order_from_signal(event, quantity)

        # Alerta Discord para sinais reais
        if not settings.okx_paper_trading:
            await self._alert_channel.info(
                title=f"Sinal: {signal.symbol}",
                message=(
                    f"Direção: {signal.direction} | "
                    f"Score: {signal.calibrated_score:.3f} | "
                    f"Qty: {quantity} | Notional: ${notional:.0f}"
                ),
            )

    # ── Paper fill simulator ──────────────────────────────────────────────────

    async def _simulate_paper_fills(self) -> None:
        """
        Simula fills para ordens PAPER-xxx (paper trading local).
        Preenche pelo preço atual do Redis cache — fill imediato.
        Publica OrderFilledEvent para o PositionMonitor processar.
        """
        if not settings.okx_paper_trading:
            return
        open_orders = self._oms.get_open_orders()
        for order in open_orders:
            eid = order.exchange_order_id or ""
            if not eid.startswith("PAPER-"):
                continue
            price = await self._cache.get_price(order.symbol)
            if not price:
                continue
            from ..core.events import OrderFilledEvent
            from .models import OrderStatus
            order.status          = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price  = price
            order.filled_at       = __import__('datetime').datetime.now(
                __import__('datetime').timezone.utc
            )
            order.fees_paid       = round(price * order.quantity * 0.001, 6)
            logger.info(
                "[PAPER-FILL] order=%s symbol=%s qty=%s price=%s fee=%s",
                order.client_order_id, order.symbol,
                order.quantity, price, order.fees_paid,
            )
            await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order))

    # ── Main heartbeat loop ────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while self._running:
            try:
                self._heartbeat.beat()
                self._infra_metrics.record_ws_message()

                # Simula fills para ordens paper
                await self._simulate_paper_fills()

                # Atualiza portfolio value no runner
                self._runner.update_portfolio_value(
                    self._portfolio.state.total_value
                )

                if not self._kill_switch.allows_any_operation:
                    logger.critical("HARD kill switch active — stopping loop")
                    break

                await asyncio.sleep(15)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("TradingLoop error: %s", exc, exc_info=True)
                self._infra_metrics.record_error()

    async def _on_ws_dead(self) -> None:
        logger.warning("WS dead — triggering soft kill switch")
        self._kill_switch.trigger_soft(reason="websocket_dead")
        self._infra_metrics.record_error()
        await self._alert_channel.warning(
            title="WebSocket Morto",
            message="Conexao com OKX perdida. Kill switch SOFT ativado.",
        )
