"""
PortfolioState — snapshot of the portfolio at a point in time.

Lives in /portfolio, NOT in /core (avoids bidirectional coupling).
Updated by the Portfolio Engine every cycle.
Read by strategies via StrategyContext injection.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class PortfolioState:
    """
    Complete portfolio snapshot.
    Passed to strategies as read-only context — they never modify it.
    """

    # Capital
    total_value: float = 0.0           # cash + open positions value
    cash_available: float = 0.0        # deployable cash
    initial_capital: float = 0.0       # starting capital (for drawdown calc)

    # Exposure
    total_exposure_pct: float = 0.0    # fraction of capital deployed
    open_position_count: int = 0

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_return_pct: float = 0.0

    # Risk
    portfolio_beta: float = 0.0        # weighted beta vs BTC
    avg_correlation: float = 0.0       # avg correlation between open positions
    concentration_risk: float = 0.0   # Herfindahl index (0=diversified, 1=concentrated)

    # Per-symbol breakdown
    positions: dict[str, dict] = field(default_factory=dict)

    # Metadata
    updated_at: datetime = field(default_factory=_now)

    @property
    def is_overexposed(self) -> bool:
        return self.total_exposure_pct > 0.5

    @property
    def is_correlated(self) -> bool:
        return self.avg_correlation > 0.85

    @property
    def drawdown_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return max(0.0, (self.initial_capital - self.total_value) / self.initial_capital)
