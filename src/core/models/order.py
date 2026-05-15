from datetime import datetime
from dataclasses import dataclass, field
from enum import StrEnum
import uuid


class OrderSide(StrEnum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET       = "market"
    LIMIT        = "limit"
    LIMIT_MAKER  = "limit_maker"   # post-only (maker fee)


class OrderStatus(StrEnum):
    NEW        = "new"          # created locally, not sent yet
    PENDING    = "pending"      # being submitted to exchange
    SUBMITTED  = "submitted"    # accepted by exchange, awaiting fill
    PARTIAL    = "partial"      # partially filled
    FILLED     = "filled"       # fully filled
    CANCELLED  = "cancelled"    # cancelled by us or exchange
    REJECTED   = "rejected"     # refused by exchange
    EXPIRED    = "expired"      # GTT/GTC order that timed out


class OrderMode(StrEnum):
    PASSIVE_LIMIT = "passive_limit"
    STAGGERED     = "staggered"
    MARKET        = "market"


@dataclass
class Order:
    """
    Core order model — lives in the OMS state machine.
    All exchange-specific fields go in the exchange adapter, not here.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    strategy_id: str
    signal_id: str                          # links back to Signal

    # Price
    limit_price: float | None = None        # None for market orders
    stop_loss: float | None = None
    take_profit: float | None = None

    # Identity
    client_order_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    exchange_order_id: str | None = None

    # State machine
    status: OrderStatus = OrderStatus.NEW
    mode: OrderMode = OrderMode.PASSIVE_LIMIT

    # Fill tracking
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees_paid: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Retry tracking
    retry_count: int = 0
    last_error: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status in {
            OrderStatus.NEW,
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        }

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
