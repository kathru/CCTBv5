from dataclasses import dataclass

from ..models import Fill, Order
from .base import BaseEvent


@dataclass(frozen=True)
class OrderCreatedEvent(BaseEvent):
    """OMS created a new order (not yet submitted)."""
    order: Order | None = None


@dataclass(frozen=True)
class OrderSubmittedEvent(BaseEvent):
    """Order was accepted by the exchange."""
    order: Order | None = None


@dataclass(frozen=True)
class OrderFilledEvent(BaseEvent):
    """Order fully filled."""
    order: Order | None = None
    fill: Fill | None = None


@dataclass(frozen=True)
class OrderPartialEvent(BaseEvent):
    """Order partially filled."""
    order: Order | None = None
    fill: Fill | None = None


@dataclass(frozen=True)
class OrderCancelledEvent(BaseEvent):
    """Order cancelled (by us or by exchange)."""
    order: Order | None = None
    reason: str = ""


@dataclass(frozen=True)
class OrderRejectedEvent(BaseEvent):
    """Order rejected by exchange."""
    order: Order | None = None
    reason: str = ""


@dataclass(frozen=True)
class OrderExpiredEvent(BaseEvent):
    """GTT/GTC order expired without fill."""
    order: Order | None = None
