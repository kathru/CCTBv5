from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import StrEnum


class PositionSide(StrEnum):
    LONG  = "long"
    SHORT = "short"


class PositionStatus(StrEnum):
    OPEN   = "open"
    CLOSED = "closed"


@dataclass
class Position:
    """
    Tracks an open or closed position across its full lifecycle.
    Updated by the OMS as fills arrive.
    """

    symbol: str
    side: PositionSide
    strategy_id: str

    # Size and cost basis
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    total_fees: float = 0.0

    # Risk levels
    stop_loss: float | None = None
    take_profit: float | None = None

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Lifecycle
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None

    # Audit
    signal_ids: list[str] = field(default_factory=list)
    fill_ids: list[str] = field(default_factory=list)

    @property
    def notional(self) -> float:
        return self.quantity * self.avg_entry_price

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def update_unrealized(self, current_price: float) -> None:
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (
                (current_price - self.avg_entry_price) * self.quantity
            )
        else:
            self.unrealized_pnl = (
                (self.avg_entry_price - current_price) * self.quantity
            )
