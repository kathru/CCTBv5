"""
Order Manager — the OMS state machine.

Responsibilities:
  - Own the lifecycle of every order (NEW → terminal state)
  - Receive signals and create orders
  - Track all open orders in memory (source of truth)
  - Emit order events to the bus on every state transition
  - Never touch the exchange directly (delegates to ExecutionRouter)
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Protocol

from ..core.bus import EventBus
from ..core.events import (
    OrderCancelledEvent,
    OrderCreatedEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartialEvent,
    OrderRejectedEvent,
    OrderSubmittedEvent,
    Topic,
)
from ..core.events.signal_events import SignalEvent
from ..core.models import (
    Fill,
    Order,
    OrderMode,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
)

logger = logging.getLogger(__name__)

# Orders older than this with no fill are expired
ORDER_TIMEOUT = timedelta(minutes=15)


def _now() -> datetime:
    return datetime.now(UTC)


class ExecutionRouterProtocol(Protocol):
    """What the OMS needs from the execution layer."""

    async def submit(self, order: Order) -> str:
        """Submit order to exchange. Returns exchange_order_id."""
        ...

    async def cancel(self, order: Order) -> bool:
        """Cancel order on exchange. Returns True if successful."""
        ...


class OrderManager:
    """
    Central OMS. One instance per application.
    Inject the EventBus and ExecutionRouter.
    """

    def __init__(
        self,
        bus: EventBus,
        router: ExecutionRouterProtocol,
        order_timeout: timedelta = ORDER_TIMEOUT,
        db=None,   # Database — injected after async init; enables persistence
    ) -> None:
        self._bus = bus
        self._router = router
        self._order_timeout = order_timeout
        self._db = db   # opcional — sem DB, ordens ficam só em memória

        # In-memory state — persisted to DB when db is available
        self._orders: dict[str, Order] = {}   # client_order_id → Order

        # System gate — set to False during RECONCILING or HARD kill switch
        self._accepting_orders: bool = False

    def set_db(self, db) -> None:
        """Injeta o Database após inicialização assíncrona."""
        self._db = db

    async def _persist(self, order: Order) -> None:
        """Salva/atualiza ordem no PostgreSQL (fire-and-forget)."""
        if self._db is None:
            return
        try:
            from ..persistence.repositories.orders import OrderRepository
            repo = OrderRepository(self._db)
            await repo.save(order)
        except Exception as exc:
            logger.warning("OMS persist error coid=%s: %s", order.client_order_id, exc)

    # ── Gate control ─────────────────────────────────────────

    def open_gate(self) -> None:
        """Allow new orders (called after reconciliation completes)."""
        self._accepting_orders = True
        logger.info("OMS gate OPEN — accepting orders")

    def close_gate(self, reason: str = "") -> None:
        """Block new orders (kill switch / reconciling)."""
        self._accepting_orders = False
        logger.warning("OMS gate CLOSED reason=%s", reason)

    # ── Signal → Order ────────────────────────────────────────

    async def on_signal(self, event: SignalEvent) -> None:
        """
        Called by the signal consumer coroutine.
        Converts a signal into an order and starts its lifecycle.
        """
        if not self._accepting_orders:
            logger.info(
                "OMS gate closed — ignoring signal strategy=%s symbol=%s",
                event.signal.strategy_id if event.signal else "?",
                event.signal.symbol if event.signal else "?",
            )
            return

        signal = event.signal
        if signal is None or signal.direction == SignalDirection.FLAT:
            return

        order = self._build_order(event)
        await self._register(order)
        await self._submit(order)

    def _build_order(self, event: SignalEvent) -> Order:
        signal = event.signal
        assert signal is not None
        side = (
            OrderSide.BUY
            if signal.direction == SignalDirection.LONG
            else OrderSide.SELL
        )
        return Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=0.0,          # NOTE: sized externally — on_signal() path is unused in prod; use create_order_from_signal()
            strategy_id=signal.strategy_id,
            signal_id=event.event_id,   # ID do SignalEvent, não do Signal
            mode=OrderMode.PASSIVE_LIMIT,
        )

    # ── Lifecycle transitions ─────────────────────────────────

    async def _register(self, order: Order) -> None:
        """Record order and emit CREATED event."""
        self._orders[order.client_order_id] = order
        import asyncio
        asyncio.create_task(self._persist(order), name=f"persist_order_{order.client_order_id[:8]}")
        await self._bus.publish(
            Topic.ORDER,
            OrderCreatedEvent(order=order),
        )
        logger.info(
            "Order CREATED coid=%s symbol=%s side=%s",
            order.client_order_id, order.symbol, order.side,
        )

    async def _submit(self, order: Order) -> None:
        """Send to exchange via router with retry."""
        from .retry_policy import RetryPolicy
        policy = RetryPolicy()
        order.status = OrderStatus.PENDING

        for attempt in range(policy.max_attempts):
            try:
                exchange_id = await self._router.submit(order)
                order.exchange_order_id = exchange_id
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = _now()
                await self._bus.publish(
                    Topic.ORDER,
                    OrderSubmittedEvent(order=order),
                )
                import asyncio
                asyncio.create_task(
                    self._persist(order),
                    name=f"persist_submitted_{order.client_order_id[:8]}",
                )
                logger.info(
                    "Order SUBMITTED coid=%s exchange_id=%s",
                    order.client_order_id, exchange_id,
                )
                return

            except Exception as exc:
                order.retry_count += 1
                order.last_error = str(exc)
                logger.warning(
                    "Submit failed coid=%s attempt=%d error=%s",
                    order.client_order_id, attempt, exc,
                )
                if policy.should_retry(attempt + 1):
                    await policy.wait(attempt)
                else:
                    await self._reject(order, reason=str(exc))

    async def on_fill(self, fill: Fill) -> None:
        """Called when exchange confirms a fill."""
        order = self._orders.get(fill.client_order_id)
        if order is None:
            logger.warning("Fill for unknown order coid=%s", fill.client_order_id)
            return

        order.filled_quantity += fill.quantity
        order.fees_paid += fill.fee
        order.fill_ids = getattr(order, "fill_ids", [])

        # Recalculate average fill price
        prev_notional = (
            (order.filled_quantity - fill.quantity) * order.avg_fill_price
        )
        order.avg_fill_price = (
            (prev_notional + fill.notional) / order.filled_quantity
        )

        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            order.filled_at = _now()
            await self._bus.publish(
                Topic.ORDER,
                OrderFilledEvent(order=order, fill=fill),
            )
            await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order, fill=fill))
            import asyncio
            asyncio.create_task(
                self._persist(order),
                name=f"persist_filled_{order.client_order_id[:8]}",
            )
            logger.info(
                "Order FILLED coid=%s avg_px=%.4f", order.client_order_id, order.avg_fill_price,
            )
        else:
            order.status = OrderStatus.PARTIAL
            await self._bus.publish(
                Topic.ORDER,
                OrderPartialEvent(order=order, fill=fill),
            )

    async def cancel_order(self, client_order_id: str, reason: str = "") -> None:
        """Cancel an open order."""
        order = self._orders.get(client_order_id)
        if order is None or not order.is_open:
            return

        success = await self._router.cancel(order)
        if success:
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = _now()
            await self._bus.publish(
                Topic.ORDER,
                OrderCancelledEvent(order=order, reason=reason),
            )
            logger.info("Order CANCELLED coid=%s reason=%s", client_order_id, reason)

    async def cancel_all(self, reason: str = "kill_switch") -> None:
        """Cancel every open order — used by kill switch."""
        open_orders = [o for o in self._orders.values() if o.is_open]
        logger.warning("Cancelling %d open orders reason=%s", len(open_orders), reason)
        for order in open_orders:
            await self.cancel_order(order.client_order_id, reason=reason)

    async def expire_stale_orders(self) -> None:
        """
        Called periodically (e.g. every minute).
        Expires orders that have been open too long without a fill.
        """
        now = _now()
        for order in list(self._orders.values()):
            if not order.is_open:
                continue
            age = now - order.created_at
            if age > self._order_timeout:
                order.status = OrderStatus.EXPIRED
                await self._bus.publish(
                    Topic.ORDER,
                    OrderExpiredEvent(order=order),
                )
                logger.warning(
                    "Order EXPIRED coid=%s age=%s", order.client_order_id, age
                )

    async def _reject(self, order: Order, reason: str) -> None:
        order.status = OrderStatus.REJECTED
        await self._bus.publish(
            Topic.ORDER,
            OrderRejectedEvent(order=order, reason=reason),
        )
        logger.error("Order REJECTED coid=%s reason=%s", order.client_order_id, reason)

    # ── Entrada de sinal (chamado pelo TradingLoop) ───────────

    async def create_order_from_signal(
        self,
        event: SignalEvent,
        quantity: float,
    ) -> None:
        """
        Cria e submete ordem de mercado a partir de um sinal aprovado.
        Quantidade já calculada pelo TradingLoop (kelly sizing).
        Ordem MARKET — execução imediata ao preço atual.
        """
        if not self._accepting_orders:
            logger.info(
                "OMS gate fechado — sinal descartado symbol=%s",
                event.signal.symbol if event.signal else "?",
            )
            return

        signal = event.signal
        if signal is None or quantity <= 0:
            return

        from ..core.models import SignalDirection
        side = (
            OrderSide.BUY
            if signal.direction == SignalDirection.LONG
            else OrderSide.SELL
        )

        order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy_id=signal.strategy_id,
            signal_id=event.event_id,
            mode=OrderMode.MARKET,
        )

        await self._register(order)
        await self._submit(order)

    # ── Exit position (chamado pelo PositionMonitor) ───────────

    async def exit_position(
        self,
        symbol: str,
        quantity: float,
        reason: str,
        strategy_id: str = "position_monitor",
    ) -> bool:
        """
        Cria e submete uma ordem de venda a mercado para fechar (ou reduzir)
        uma posição. Bypassa o gate de entradas — saídas são sempre permitidas.
        Returns True se a ordem foi submetida com sucesso.
        """
        import uuid
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=round(quantity, 8),
            strategy_id=strategy_id,
            signal_id=f"exit_{reason}_{uuid.uuid4().hex[:8]}",
            mode=OrderMode.MARKET,
        )
        await self._register(order)
        try:
            await self._submit(order)
            logger.info(
                "Exit order SUBMITTED symbol=%s qty=%.6f reason=%s",
                symbol, quantity, reason,
            )
            return True
        except Exception as exc:
            logger.error(
                "Exit order FAILED symbol=%s qty=%.6f reason=%s error=%s",
                symbol, quantity, reason, exc,
            )
            return False

    # ── Queries ───────────────────────────────────────────────

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def get_order(self, client_order_id: str) -> Order | None:
        return self._orders.get(client_order_id)

    @property
    def open_order_count(self) -> int:
        return len(self.get_open_orders())
