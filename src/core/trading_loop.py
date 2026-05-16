"""
Trading Loop — wires all engines together and runs the main cycle.

Component wiring:
  MarketEngine → (CandleEvent) → StrategyRunner
  StrategyRunner → (SignalEvent) → RiskEngine
  RiskEngine → (RiskEvaluatedEvent) → OrderManager
  OrderManager → (OrderEvent) → ExecutionRouter → OKX

Parallel services:
  - PeriodicReconciler (every 2 min)
  - HeartbeatWatchdog (every 15s)
  - WebSocketWatchdog (every 5s)
  - AlertListener (continuous)
  - PeriodicReconciler (continuous)

Boot sequence (BootSequence handles this):
  Connect → Schema → RECONCILING → Load state → Reconcile → RUNNING
"""

import asyncio
import logging
from pathlib import Path

from .bus import EventBus
from .config import settings
from ..exchange.okx.client import OKXClient
from ..market.engine import MarketEngine
from ..oms.order_manager import OrderManager
from ..oms.execution_router import ExecutionRouter
from ..risk.engine import RiskEngine, RiskContext
from ..risk.kill_switch import KillSwitch
from ..portfolio.engine import PortfolioEngine
from ..strategies.runner import StrategyRunner
from ..strategies.meta_layer import MetaStrategyLayer
from ..strategies.momentum.v4_strategy import V4MomentumStrategy
from ..strategies.ml.inference import MLInferenceEngine
from ..recovery.boot import BootSequence
from ..recovery.periodic_reconciler import PeriodicReconciler
from ..oms.position_monitor import PositionMonitor
from ..watchdog.heartbeat import HeartbeatWatchdog
from ..watchdog.websocket_watchdog import WebSocketWatchdog
from ..watchdog.resource_watchdog import ResourceWatchdog
from ..alerts.discord import create_alert_channel
from ..alerts.listener import AlertListener
from ..persistence import Database, Cache
from ..metrics.infra_metrics import InfraMetrics

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
MODELS_DIR = Path("data") / "models"


class TradingLoop:
    """
    Main trading loop — composes all engines and runs them.
    Call start() to begin, stop() to shut down gracefully.
    """

    def __init__(
        self,
        db: Database,
        cache: Cache,
        app_state=None,    # FastAPI app.state for watchdog registration
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
        # Register strategies
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
            dead_threshold=120,   # REST polling — threshold maior até WS real
            name="okx_market",
        )
        self._resource_watchdog = ResourceWatchdog(
            kill_switch=self._kill_switch,
            interval_seconds=30,
        )

        # ── Position Monitor (saídas automáticas) ────────────
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

        # Wire MarketEngine poll → WS watchdog (REST polling acts as WS heartbeat)
        self._market.set_on_poll_callback(self._ws_watchdog.record_message)

    async def start(self) -> bool:
        """
        Run boot sequence then start all services.
        Returns True if ready to trade.
        """
        logger.info("TradingLoop: starting boot sequence...")

        # Boot sequence
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

        # Register watchdogs in app.state for dashboard
        if self._app_state:
            self._app_state.ws_watchdog = self._ws_watchdog
            self._app_state.heartbeat_watchdog = self._heartbeat
            self._app_state.resource_watchdog = self._resource_watchdog
            self._app_state.portfolio        = self._portfolio
            self._app_state.position_monitor = self._position_monitor

        # Start all services
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

        # Send startup alert
        await self._alert_channel.info(
            title="✅ CCTBv5 Started",
            message="Sistema iniciado com sucesso.",
            mode="paper" if settings.okx_paper_trading else "live",
            symbols=", ".join(SYMBOLS),
        )

        # Main heartbeat loop
        await self._run_loop()
        return True

    async def _run_loop(self) -> None:
        """Main loop — beats heartbeat and checks kill switch."""
        while self._running:
            try:
                self._heartbeat.beat()
                self._infra_metrics.record_ws_message()

                # Check kill switch
                if not self._kill_switch.allows_any_operation:
                    logger.critical("HARD kill switch active — stopping loop")
                    break

                await asyncio.sleep(15)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("TradingLoop error: %s", exc, exc_info=True)
                self._infra_metrics.record_error()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        logger.info("TradingLoop: shutting down...")

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

    async def _on_ws_dead(self) -> None:
        """Called when WebSocket is declared dead."""
        logger.warning("WS dead — triggering soft kill switch")
        self._kill_switch.trigger_soft(reason="websocket_dead")
        self._infra_metrics.record_error()
        await self._alert_channel.warning(
            title="⚠️ WebSocket Morto",
            message="Conexão com OKX perdida. Kill switch SOFT ativado.",
        )
