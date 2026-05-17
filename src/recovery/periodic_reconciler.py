"""
Periodic Reconciler — runs every 1-5 minutes in background.

Complements the BootReconciler (which runs once at startup).
This one runs continuously to catch mid-session divergences:
  - Fill received by exchange but not by our WebSocket
  - Order cancelled by exchange due to timeout/insufficient balance
  - Position closed by exchange stop-loss but not reflected locally

Flow on divergence found:
  1. Close OMS gate (freeze new entries)
  2. Emit alert event (Discord via alerts module)
  3. Fix local state
  4. Reopen OMS gate if all resolved

If divergences cannot be resolved → trigger SOFT kill switch.
"""

import asyncio
import logging
from datetime import UTC, datetime

from ..core.bus import EventBus
from ..core.events import ReconciliationEvent, SystemStatusEvent, Topic
from ..core.events.system_events import SystemStatus
from ..persistence.postgres import Database
from .reconciler import BootReconciler, ExchangeStateProtocol

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class PeriodicReconciler:
    """
    Background task that reconciles state every N seconds.
    Inject into the application and call start() after boot.
    """

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        exchange: ExchangeStateProtocol,
        order_manager=None,
        kill_switch=None,
        interval_seconds: int = 120,   # 2 minutes default
    ) -> None:
        self._bus = bus
        self._db = db
        self._exchange = exchange
        self._order_manager = order_manager
        self._kill_switch = kill_switch
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_run: datetime | None = None
        self._run_count = 0
        self._divergence_count = 0

    async def start(self) -> None:
        """Start the background reconciliation loop."""
        if self._running:
            logger.warning("PeriodicReconciler already running")
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="periodic_reconciler"
        )
        logger.info(
            "PeriodicReconciler started interval=%ds", self._interval
        )

    async def stop(self) -> None:
        """Stop the background loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PeriodicReconciler stopped")

    async def _loop(self) -> None:
        """Main background loop."""
        # Wait one full interval before first run
        # (BootReconciler already ran at startup)
        await asyncio.sleep(self._interval)

        while self._running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "PeriodicReconciler cycle error: %s", exc, exc_info=True
                )
            await asyncio.sleep(self._interval)

    async def _run_cycle(self) -> None:
        """Execute one reconciliation cycle."""
        self._run_count += 1
        self._last_run = _now()
        logger.debug("PeriodicReconciler cycle #%d", self._run_count)

        # Get current open orders from OMS memory
        open_orders = []
        if self._order_manager:
            open_orders = self._order_manager.get_open_orders()

        if not open_orders:
            logger.debug("No open orders — skipping reconciliation")
            return

        # Run reconciliation
        reconciler = BootReconciler(
            bus=self._bus,
            db=self._db,
            exchange=self._exchange,
        )

        # Build a minimal LoadedState-like object
        from .state_loader import LoadedState
        state = LoadedState(
            open_orders=open_orders,
            open_positions=[],
            loaded_at=self._last_run.isoformat(),
        )

        report = await reconciler.run(
            loaded_state=state,
            order_manager=self._order_manager,
        )

        if report.has_unresolved:
            self._divergence_count += 1
            logger.warning(
                "Periodic reconciliation: unresolved divergences=%d "
                "total_divergence_cycles=%d",
                report.orders_unresolved,
                self._divergence_count,
            )

            # Close OMS gate during divergence
            if self._order_manager:
                self._order_manager.close_gate(
                    reason=f"reconciliation_divergence: {report.summary}"
                )

            # Trigger soft kill switch if divergences persist
            if self._divergence_count >= 3 and self._kill_switch:
                self._kill_switch.trigger_soft(
                    reason="repeated_reconciliation_divergences"
                )
                await self._bus.publish(
                    Topic.SYSTEM,
                    SystemStatusEvent(
                        status=SystemStatus.SUSPENDED,
                        reason="repeated_reconciliation_divergences",
                    ),
                )
        else:
            # All clean — ensure gate is open
            if self._order_manager and not self._order_manager._accepting_orders:
                self._order_manager.open_gate()
                logger.info(
                    "PeriodicReconciler: divergences resolved — gate reopened"
                )
            # Reset divergence counter on clean cycle
            self._divergence_count = 0
            # Reset soft kill switch if it was triggered by reconciliation
            if self._kill_switch and self._kill_switch.is_armed:
                ks_reason = getattr(
                    self._kill_switch._current_event, "reason", ""
                )
                if "reconciliation" in ks_reason:
                    self._kill_switch.reset_soft(
                        reason="reconciliation_clean_cycle"
                    )
                    logger.info(
                        "PeriodicReconciler: kill switch reset — "
                        "reconciliation clean"
                    )
                    await self._bus.publish(
                        Topic.SYSTEM,
                        SystemStatusEvent(
                            status=SystemStatus.RUNNING,
                            reason="reconciliation_clean_cycle",
                        ),
                    )

        # Emit reconciliation event for dashboard/alerts
        await self._bus.publish(
            Topic.SYSTEM,
            ReconciliationEvent(
                divergences_found=report.orders_unresolved,
                resolved=not report.has_unresolved,
                detail=f"cycle #{self._run_count}: {report.summary}",
            ),
        )

    def status(self) -> dict:
        return {
            "running": self._running,
            "interval_seconds": self._interval,
            "run_count": self._run_count,
            "divergence_count": self._divergence_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
        }
