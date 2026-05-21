"""
State Loader — restores system state from PostgreSQL on boot.

Boot sequence (must complete BEFORE OMS gate opens):
  1. Load open positions from DB
  2. Load open orders from DB
  3. Rebuild in-memory OMS state from loaded orders
  4. Update Redis cache with loaded positions
  5. Hand off to Reconciler for exchange comparison

This ensures that even after a crash, the system knows exactly
what was open before it went down.
"""

import logging
from dataclasses import dataclass
from datetime import UTC

from ..core.models import Order
from ..persistence.cache import Cache
from ..persistence.postgres import Database
from ..persistence.repositories.orders import OrderRepository
from ..persistence.repositories.positions import PositionRepository

logger = logging.getLogger(__name__)


@dataclass
class LoadedState:
    """Result of a state load operation."""
    open_orders: list[Order]
    open_positions: list[dict]
    loaded_at: str  # ISO timestamp

    @property
    def summary(self) -> str:
        return (
            f"Loaded {len(self.open_orders)} open orders, "
            f"{len(self.open_positions)} open positions"
        )


class StateLoader:
    """
    Loads persisted state from PostgreSQL on system boot.
    Populates Redis cache with hot state.
    """

    def __init__(
        self,
        db: Database,
        cache: Cache,
    ) -> None:
        self._db = db
        self._cache = cache
        self._order_repo = OrderRepository(db)
        self._position_repo = PositionRepository(db)

    async def load(self) -> LoadedState:
        """
        Main boot loader. Call once before starting the trading loop.
        Returns the loaded state for handoff to Reconciler.
        """
        from datetime import datetime
        logger.info("StateLoader: starting boot sequence")

        # Step 1: Load open orders from DB
        logger.info("Step 1/3: Loading open orders from PostgreSQL...")
        open_orders = await self._order_repo.get_open()
        logger.info("Found %d open orders", len(open_orders))

        # Step 2: Load open positions from DB
        logger.info("Step 2/3: Loading open positions from PostgreSQL...")
        open_positions = await self._position_repo.get_open()
        logger.info("Found %d open positions", len(open_positions))

        # Step 3: Populate Redis cache with hot state
        logger.info("Step 3/3: Warming Redis cache...")
        await self._warm_cache(open_positions)

        loaded_at = datetime.now(UTC).isoformat()
        state = LoadedState(
            open_orders=open_orders,
            open_positions=open_positions,
            loaded_at=loaded_at,
        )

        logger.info("StateLoader complete: %s", state.summary)
        return state

    async def _warm_cache(self, positions: list[dict]) -> None:
        """Push open positions into Redis for fast access."""
        import decimal
        import uuid as _uuid
        from datetime import datetime

        def _safe(v):
            """Converte tipos não-JSON para tipos primitivos."""
            if isinstance(v, (_uuid.UUID, decimal.Decimal, datetime)):
                return str(v)
            if isinstance(v, dict):
                return {kk: _safe(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [_safe(i) for i in v]
            return v

        for pos in positions:
            symbol = pos.get("symbol", "")
            if not symbol:
                continue
            safe_pos = {k: _safe(v) for k, v in pos.items()}
            await self._cache.set_position(symbol, safe_pos)
            logger.debug("Cache warmed for symbol=%s", symbol)

    async def restore_oms_state(
        self,
        order_manager,  # OrderManager — avoid circular import
        loaded_state: LoadedState,
    ) -> None:
        """
        Inject loaded orders back into the OMS in-memory state.
        Called after StateLoader.load() and before Reconciler runs.
        Orders that are terminal (filled/cancelled) are skipped.
        """
        restored = 0
        skipped = 0

        for order in loaded_state.open_orders:
            if not order.is_open:
                skipped += 1
                continue
            # Re-register in OMS memory without re-submitting to exchange
            order_manager._orders[order.client_order_id] = order
            restored += 1

        logger.info(
            "OMS state restored: %d orders active, %d skipped",
            restored, skipped,
        )
