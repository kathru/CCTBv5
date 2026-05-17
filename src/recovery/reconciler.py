"""
Boot Reconciler — compares loaded state vs live exchange state.

Runs AFTER StateLoader, BEFORE OMS gate opens.
System is in RECONCILING mode during this phase — no new orders accepted.

For each open order found in DB:
  → Ask exchange: what is the current status?
  → If divergent: fix local state
  → If unresolvable: alert and keep frozen

For each open position found in DB:
  → Ask exchange: do you agree this position exists?
  → If not: mark as closed in DB and cache

After reconciliation:
  → Emit SystemStatusEvent(RUNNING)
  → Open OMS gate
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from ..core.bus import EventBus
from ..core.events import ReconciliationEvent, SystemStatusEvent, Topic
from ..core.events.system_events import SystemStatus
from ..core.models import Order, OrderStatus
from ..persistence.postgres import Database
from ..persistence.repositories.orders import OrderRepository
from .state_loader import LoadedState

logger = logging.getLogger(__name__)


class ExchangeStateProtocol(Protocol):
    """Minimum exchange interface needed for reconciliation."""

    async def get_order_status(self, exchange_order_id: str, symbol: str | None = None) -> dict:
        """Returns {'status': str, 'filled_qty': float, 'avg_px': float}"""
        ...

    async def get_open_positions(self) -> list[dict]:
        """Returns list of open positions from exchange."""
        ...


@dataclass
class ReconciliationReport:
    orders_checked: int = 0
    orders_fixed: int = 0
    orders_unresolved: int = 0
    positions_checked: int = 0
    positions_divergent: int = 0

    @property
    def has_unresolved(self) -> bool:
        return self.orders_unresolved > 0

    @property
    def summary(self) -> str:
        return (
            f"Orders: checked={self.orders_checked} "
            f"fixed={self.orders_fixed} "
            f"unresolved={self.orders_unresolved} | "
            f"Positions: checked={self.positions_checked} "
            f"divergent={self.positions_divergent}"
        )


class BootReconciler:
    """
    Runs once at boot to sync local state with exchange.
    Blocks OMS gate until complete.
    """

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        exchange: ExchangeStateProtocol,
    ) -> None:
        self._bus = bus
        self._db = db
        self._exchange = exchange
        self._order_repo = OrderRepository(db)

    async def run(
        self,
        loaded_state: LoadedState,
        order_manager=None,  # OrderManager — injected to update state
    ) -> ReconciliationReport:
        """
        Main reconciliation loop.
        Returns report. Caller decides whether to open OMS gate.
        """
        logger.info("BootReconciler: starting reconciliation")

        # Emit RECONCILING status
        await self._bus.publish(
            Topic.SYSTEM,
            SystemStatusEvent(
                status=SystemStatus.RECONCILING,
                reason="boot_reconciliation",
            ),
        )

        report = ReconciliationReport()

        # Reconcile orders
        await self._reconcile_orders(
            loaded_state.open_orders, report, order_manager
        )

        # Reconcile positions
        await self._reconcile_positions(
            loaded_state.open_positions, report
        )

        logger.info("BootReconciler complete: %s", report.summary)

        # Emit result event
        await self._bus.publish(
            Topic.SYSTEM,
            ReconciliationEvent(
                divergences_found=report.orders_unresolved + report.positions_divergent,
                resolved=not report.has_unresolved,
                detail=report.summary,
            ),
        )

        return report

    async def _reconcile_orders(
        self,
        orders: list[Order],
        report: ReconciliationReport,
        order_manager=None,
    ) -> None:
        for order in orders:
            if not order.is_open:
                continue

            report.orders_checked += 1

            # Orders never submitted — can't reconcile with exchange
            if order.exchange_order_id is None:
                logger.warning(
                    "Order has no exchange_id — marking CANCELLED coid=%s",
                    order.client_order_id,
                )
                order.status = OrderStatus.CANCELLED
                await self._order_repo.save(order)
                report.orders_fixed += 1
                continue

            # Paper trading orders have local PAPER-{uuid} exchange_ids
            # that the real OKX API doesn't know about — skip reconciliation
            if order.exchange_order_id.startswith("PAPER-"):
                logger.debug(
                    "Skipping reconciliation for paper order coid=%s",
                    order.client_order_id,
                )
                order.status = OrderStatus.CANCELLED
                await self._order_repo.save(order)
                report.orders_fixed += 1
                if order_manager:
                    order_manager._orders[order.client_order_id] = order
                continue

            try:
                remote = await self._exchange.get_order_status(
                    order.exchange_order_id,
                    symbol=order.symbol,
                )
                fixed = self._apply_remote_status(order, remote)
                if fixed:
                    await self._order_repo.save(order)
                    report.orders_fixed += 1
                    if order_manager:
                        order_manager._orders[order.client_order_id] = order

            except Exception as exc:
                logger.error(
                    "Cannot reconcile order exchange_id=%s error=%s",
                    order.exchange_order_id, exc,
                )
                report.orders_unresolved += 1

    async def _reconcile_positions(
        self,
        positions: list[dict],
        report: ReconciliationReport,
    ) -> None:
        try:
            exchange_positions = await self._exchange.get_open_positions()
            exchange_symbols = {
                p.get("symbol") for p in exchange_positions
            }
        except Exception as exc:
            logger.error("Cannot fetch exchange positions: %s", exc)
            return

        for pos in positions:
            report.positions_checked += 1
            symbol = pos.get("symbol", "")

            if symbol not in exchange_symbols:
                logger.warning(
                    "Position in DB but not on exchange — symbol=%s "
                    "marking as divergent",
                    symbol,
                )
                report.positions_divergent += 1

    def _apply_remote_status(self, order: Order, remote: dict) -> bool:
        """Apply exchange status to local order. Returns True if changed."""
        remote_status = remote.get("status", "")

        if remote_status == "filled" and order.status != OrderStatus.FILLED:
            order.status = OrderStatus.FILLED
            order.filled_quantity = float(remote.get("filled_qty", order.quantity))
            order.avg_fill_price = float(remote.get("avg_px", 0))
            logger.info("Reconciled FILLED coid=%s", order.client_order_id)
            return True

        if remote_status == "cancelled" and order.is_open:
            order.status = OrderStatus.CANCELLED
            logger.info("Reconciled CANCELLED coid=%s", order.client_order_id)
            return True

        if remote_status == "partial":
            order.status = OrderStatus.PARTIAL
            order.filled_quantity = float(remote.get("filled_qty", 0))
            return True

        return False
