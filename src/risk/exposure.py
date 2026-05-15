"""
Exposure control — limits total capital at risk across all open positions.

MAX_TOTAL_EXPOSURE = 0.5 means at most 50% of portfolio can be deployed
at any given time. Prevents over-allocation in correlated markets.
"""

from dataclasses import dataclass

from ..core.models import Position, PositionStatus


@dataclass
class ExposureState:
    max_exposure: float = 0.5       # 50% of portfolio max
    portfolio_value: float = 0.0

    @property
    def max_notional(self) -> float:
        return self.portfolio_value * self.max_exposure


class ExposureEngine:
    """
    Tracks current total exposure and gates new entries.
    Exposure = sum of all open position notionals / portfolio value.
    """

    def __init__(self, max_exposure: float = 0.5) -> None:
        self._max_exposure = max_exposure

    def current_exposure(
        self,
        positions: list[Position],
        portfolio_value: float,
    ) -> float:
        """Returns current exposure as a fraction of portfolio (0.0–1.0)."""
        if portfolio_value <= 0:
            return 0.0
        open_notional = sum(
            p.notional
            for p in positions
            if p.status == PositionStatus.OPEN
        )
        return open_notional / portfolio_value

    def can_open(
        self,
        positions: list[Position],
        portfolio_value: float,
        new_notional: float,
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        Checks if adding new_notional would breach max exposure.
        """
        current = self.current_exposure(positions, portfolio_value)
        new_exposure = current + (new_notional / portfolio_value if portfolio_value > 0 else 0)

        if new_exposure > self._max_exposure:
            return False, (
                f"Exposure limit breached: current={current:.1%} "
                f"new={new_exposure:.1%} max={self._max_exposure:.1%}"
            )
        return True, ""

    def remaining_capacity(
        self,
        positions: list[Position],
        portfolio_value: float,
    ) -> float:
        """Returns remaining notional that can be deployed."""
        current = self.current_exposure(positions, portfolio_value)
        remaining_pct = max(0.0, self._max_exposure - current)
        return remaining_pct * portfolio_value
