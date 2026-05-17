"""
OMS Reconciliation — compares local state vs exchange state.

When to run:
  1. On boot (before OMS gate opens)
  2. Every 1-5 minutes in background

If divergence found:
  → freeze trading (close gate)
  → alert Discord
  → fix local state
  → reopen gate
"""

import logging
from typing import Protocol

from ..core.models import Order, OrderStatus

logger = logging.getLogger(__name__)


class ExchangeOrderFetcher(Protocol):
    """Minimum needed from exchange to reconcile."""

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Returns list of open orders from exchange."""
        ...

    async def get_order_status(self, exchange_order_id: str) -> dict:
        """Returns current status of a specific order."""
        ...


class ReconciliationResult:
    def __init__(self) -> None:
        self.divergences: list[str] = []
        self.fixed: list[str] = []
        self.unresolved: list[str] = []

    @property
    def has_divergences(self) -> bool:
        return len(self.divergences) > 0

    @property
    def all_resolved(self) -> bool:
        return len(self.unresolved) == 0


class Reconciler:
    """
    Compares OMS in-memory state with exchange state.
    Fixes divergences where possible, reports what it cannot fix.
    """

    def __init__(self, fetcher: ExchangeOrderFetcher) -> None:
        self._fetcher = fetcher

    async def reconcile(
        self,
        local_orders: list[Order],
    ) -> ReconciliationResult:
        """
        Main reconciliation loop.
        Checks every local open order against exchange state.
        """
        result = ReconciliationResult()

        for order in local_orders:
            if not order.is_open:
                continue
            if order.exchange_order_id is None:
                # Never made it to exchange — can be retried or cancelled
                logger.warning(
                    "Order has no exchange_id coid=%s — marking for retry",
                    order.client_order_id,
                )
                result.divergences.append(order.client_order_id)
                continue

            # Paper trading orders have a local PAPER-{uuid} exchange_id
            # that the real OKX API doesn't know about — skip reconciliation
            if order.exchange_order_id.startswith("PAPER-"):
                logger.debug(
                    "Skipping reconciliation for paper order coid=%s",
                    order.client_order_id,
                )
                continue

            try:
                remote = await self._fetcher.get_order_status(
                    order.exchange_order_id
                )
                divergence = self._check_divergence(order, remote)
                if divergence:
                    result.divergences.append(order.client_order_id)
                    fixed = self._fix(order, remote)
                    if fixed:
                        result.fixed.append(order.client_order_id)
                    else:
                        result.unresolved.append(order.client_order_id)

            except Exception as exc:
                logger.error(
                    "Failed to fetch order status exchange_id=%s error=%s",
                    order.exchange_order_id, exc,
                )
                result.unresolved.append(order.client_order_id)

        if result.has_divergences:
            logger.warning(
                "Reconciliation found %d divergences fixed=%d unresolved=%d",
                len(result.divergences),
                len(result.fixed),
                len(result.unresolved),
            )
        else:
            logger.info("Reconciliation clean — no divergences")

        return result

    def _check_divergence(self, order: Order, remote: dict) -> bool:
        """Return True if local state differs from exchange state."""
        remote_status = remote.get("status", "")

        # Exchange says filled but we think it's still open
        if remote_status == "filled" and order.status != OrderStatus.FILLED:
            logger.warning(
                "DIVERGENCE: exchange=filled local=%s coid=%s",
                order.status, order.client_order_id,
            )
            return True

        # Exchange says cancelled but we think it's open
        if remote_status == "cancelled" and order.is_open:
            logger.warning(
                "DIVERGENCE: exchange=cancelled local=%s coid=%s",
                order.status, order.client_order_id,
            )
            return True

        return False

    def _fix(self, order: Order, remote: dict) -> bool:
        """
        Apply remote state to local order.
        Returns True if fixed, False if manual intervention needed.
        """
        remote_status = remote.get("status", "")

        if remote_status == "filled":
            order.status = OrderStatus.FILLED
            filled_qty = float(remote.get("filled_qty", order.quantity))
            avg_px = float(remote.get("avg_px", 0))
            order.filled_quantity = filled_qty
            order.avg_fill_price = avg_px
            logger.info(
                "Fixed: marked FILLED from exchange coid=%s",
                order.client_order_id,
            )
            return True

        if remote_status == "cancelled":
            order.status = OrderStatus.CANCELLED
            logger.info(
                "Fixed: marked CANCELLED from exchange coid=%s",
                order.client_order_id,
            )
            return True

        return False
