"""
Execution Router — decides WHERE and HOW to send each order.

Today: OKX only.
Tomorrow: add more exchanges without changing the OMS.

The router is the only place that knows about exchanges.
The OMS only calls router.submit() and router.cancel().
"""

import logging
from typing import Protocol

from ..core.models import Order, OrderMode, OrderType

logger = logging.getLogger(__name__)


class ExchangeClientProtocol(Protocol):
    """Minimum interface any exchange adapter must implement."""

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        client_order_id: str,
    ) -> str:
        """Place order. Returns exchange_order_id."""
        ...

    async def cancel_order(
        self,
        symbol: str,
        exchange_order_id: str,
    ) -> bool:
        """Cancel order. Returns True if successful."""
        ...


class ExecutionRouter:
    """
    Routes orders to the correct exchange client.
    Applies mode-specific logic (passive limit, staggered, market).
    """

    def __init__(self, exchange: ExchangeClientProtocol) -> None:
        self._exchange = exchange

    async def submit(self, order: Order) -> str:
        """
        Submit order to exchange.
        Returns exchange_order_id.
        """
        order_type = self._resolve_order_type(order)
        price = self._resolve_price(order)

        logger.info(
            "Routing order coid=%s symbol=%s side=%s type=%s price=%s qty=%s",
            order.client_order_id,
            order.symbol,
            order.side,
            order_type,
            price,
            order.quantity,
        )

        exchange_id = await self._exchange.place_order(
            symbol=order.symbol,
            side=order.side.value,
            order_type=order_type,
            quantity=order.quantity,
            price=price,
            client_order_id=order.client_order_id,
        )
        return exchange_id

    async def cancel(self, order: Order) -> bool:
        if order.exchange_order_id is None:
            logger.warning(
                "Cannot cancel — no exchange_order_id coid=%s",
                order.client_order_id,
            )
            return False

        return await self._exchange.cancel_order(
            symbol=order.symbol,
            exchange_order_id=order.exchange_order_id,
        )

    def _resolve_order_type(self, order: Order) -> str:
        if order.mode == OrderMode.MARKET:
            return "market"
        if order.order_type == OrderType.LIMIT_MAKER:
            return "post_only"
        return "limit"

    def _resolve_price(self, order: Order) -> float | None:
        if order.mode == OrderMode.MARKET:
            return None
        return order.limit_price
