"""
Boot Sequence Orchestrator — runs the full startup protocol.

Sequence:
  1. Connect to PostgreSQL and Redis
  2. Apply schema (idempotent)
  3. Set system status = RECONCILING (OMS gate stays CLOSED)
  4. Load state from DB (StateLoader)
  5. Restore OMS in-memory state
  6. Reconcile with exchange (BootReconciler)
  7. If reconciliation OK → open OMS gate, set status = RUNNING
  8. If unresolved divergences → stay SUSPENDED, alert

The OMS gate NEVER opens before step 7 completes successfully.
"""

import logging
from pathlib import Path

from ..core.bus import EventBus
from ..core.events import SystemStatusEvent, Topic
from ..core.events.system_events import SystemStatus
from ..persistence.cache import Cache
from ..persistence.postgres import Database
from .reconciler import BootReconciler, ExchangeStateProtocol
from .state_loader import StateLoader

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "infra" / "schema.sql"


class BootSequence:
    """
    Runs the full startup protocol.
    Returns True if system is ready to trade, False if manual intervention needed.
    """

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        cache: Cache,
        exchange: ExchangeStateProtocol,
        order_manager=None,   # injected after construction
    ) -> None:
        self._bus = bus
        self._db = db
        self._cache = cache
        self._exchange = exchange
        self._order_manager = order_manager

    async def run(self) -> bool:
        """
        Execute full boot sequence.
        Returns True if ready to trade.
        """
        logger.info("=" * 60)
        logger.info("CCTBv5 BOOT SEQUENCE STARTING")
        logger.info("=" * 60)

        try:
            # Step 1: Connect to infrastructure
            logger.info("[BOOT 1/5] Connecting to PostgreSQL and Redis...")
            await self._db.connect()
            await self._cache.connect()
            await self._cache.set_system_status(SystemStatus.STARTING)

            # Step 2: Apply schema (idempotent — safe to run every boot)
            logger.info("[BOOT 2/5] Applying database schema...")
            if SCHEMA_PATH.exists():
                await self._db.apply_schema(str(SCHEMA_PATH))
            else:
                logger.warning("Schema file not found at %s", SCHEMA_PATH)

            # Step 3: Set RECONCILING — OMS gate stays CLOSED
            logger.info("[BOOT 3/5] Entering RECONCILING mode...")
            await self._cache.set_system_status(SystemStatus.RECONCILING)
            await self._bus.publish(
                Topic.SYSTEM,
                SystemStatusEvent(
                    status=SystemStatus.RECONCILING,
                    reason="boot_sequence",
                ),
            )

            # Step 4: Load state from DB
            logger.info("[BOOT 4/5] Loading persisted state...")
            loader = StateLoader(db=self._db, cache=self._cache)
            loaded_state = await loader.load()

            # Restore OMS in-memory state
            if self._order_manager:
                await loader.restore_oms_state(
                    self._order_manager, loaded_state
                )

            # Step 5: Reconcile with exchange
            logger.info("[BOOT 5/5] Reconciling with exchange...")
            reconciler = BootReconciler(
                bus=self._bus,
                db=self._db,
                exchange=self._exchange,
            )
            report = await reconciler.run(
                loaded_state=loaded_state,
                order_manager=self._order_manager,
            )

            # Evaluate result
            if report.has_unresolved:
                logger.error(
                    "Boot reconciliation has unresolved divergences: %s",
                    report.summary,
                )
                await self._cache.set_system_status(SystemStatus.SUSPENDED)
                await self._bus.publish(
                    Topic.SYSTEM,
                    SystemStatusEvent(
                        status=SystemStatus.SUSPENDED,
                        reason=f"unresolved_divergences: {report.summary}",
                    ),
                )
                return False

            # All good — open gate and start trading
            if self._order_manager:
                self._order_manager.open_gate()

            await self._cache.set_system_status(SystemStatus.RUNNING)
            await self._bus.publish(
                Topic.SYSTEM,
                SystemStatusEvent(
                    status=SystemStatus.RUNNING,
                    reason="boot_complete",
                ),
            )

            logger.info("=" * 60)
            logger.info("BOOT COMPLETE — system RUNNING")
            logger.info(loaded_state.summary)
            logger.info("=" * 60)
            return True

        except Exception as exc:
            logger.critical("BOOT SEQUENCE FAILED: %s", exc, exc_info=True)
            await self._cache.set_system_status(SystemStatus.SUSPENDED)
            return False
