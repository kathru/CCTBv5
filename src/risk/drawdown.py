"""
Drawdown control — daily and peak-based drawdown tracking.

Two layers:
  1. Daily drawdown  — resets every day (MAX_DAILY_DD = 3%)
  2. Peak drawdown   — from all-time high (for longer-term health)
  3. Acceleration    — velocity of drawdown (detects collapses early)
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime


def _today() -> date:
    return datetime.now(UTC).date()


@dataclass
class DrawdownState:
    # Daily tracking
    day_start_value: float = 0.0
    day_date: date = field(default_factory=_today)

    # Peak tracking
    peak_value: float = 0.0

    # Recent portfolio values for acceleration detection
    recent_values: list[float] = field(default_factory=list)
    max_recent_samples: int = 10


class DrawdownEngine:
    """
    Tracks daily and peak drawdown.
    Emits block signals when limits are breached.
    """

    def __init__(
        self,
        max_daily_dd: float = 0.03,     # 3% daily limit
        max_peak_dd: float = 0.10,      # 10% from peak
        acceleration_threshold: float = 0.015,  # 1.5%/cycle acceleration
    ) -> None:
        self._max_daily_dd = max_daily_dd
        self._max_peak_dd = max_peak_dd
        self._accel_threshold = acceleration_threshold
        self._state = DrawdownState()

    def update(self, current_value: float) -> None:
        """Call every cycle with current portfolio value."""
        today = _today()

        # Reset daily tracking on new day
        if self._state.day_date != today:
            self._state.day_start_value = current_value
            self._state.day_date = today

        # Initialize on first call
        if self._state.day_start_value == 0.0:
            self._state.day_start_value = current_value

        if self._state.peak_value == 0.0:
            self._state.peak_value = current_value

        # Update peak
        if current_value > self._state.peak_value:
            self._state.peak_value = current_value

        # Track recent values for acceleration
        self._state.recent_values.append(current_value)
        if len(self._state.recent_values) > self._state.max_recent_samples:
            self._state.recent_values.pop(0)

    @property
    def daily_drawdown(self) -> float:
        """Current daily drawdown as positive fraction (0.03 = 3% down)."""
        if self._state.day_start_value <= 0:
            return 0.0
        # Get latest value
        if not self._state.recent_values:
            return 0.0
        current = self._state.recent_values[-1]
        dd = (self._state.day_start_value - current) / self._state.day_start_value
        return max(0.0, dd)

    @property
    def peak_drawdown(self) -> float:
        """Drawdown from all-time peak as positive fraction."""
        if self._state.peak_value <= 0:
            return 0.0
        if not self._state.recent_values:
            return 0.0
        current = self._state.recent_values[-1]
        dd = (self._state.peak_value - current) / self._state.peak_value
        return max(0.0, dd)

    @property
    def acceleration(self) -> float:
        """
        Rate of change of drawdown over recent samples.
        Positive = drawdown accelerating (getting worse faster).
        """
        values = self._state.recent_values
        if len(values) < 3:
            return 0.0
        # Simple: compare last third vs first third
        n = len(values) // 3
        early_avg = sum(values[:n]) / n
        late_avg = sum(values[-n:]) / n
        if early_avg <= 0:
            return 0.0
        return (early_avg - late_avg) / early_avg  # positive = declining

    def is_daily_limit_breached(self) -> tuple[bool, str]:
        dd = self.daily_drawdown
        if dd >= self._max_daily_dd:
            return True, (
                f"Daily drawdown limit breached: {dd:.2%} >= {self._max_daily_dd:.2%}"
            )
        return False, ""

    def is_peak_limit_breached(self) -> tuple[bool, str]:
        dd = self.peak_drawdown
        if dd >= self._max_peak_dd:
            return True, (
                f"Peak drawdown limit breached: {dd:.2%} >= {self._max_peak_dd:.2%}"
            )
        return False, ""

    def is_accelerating(self) -> tuple[bool, str]:
        accel = self.acceleration
        if accel >= self._accel_threshold:
            return True, f"Drawdown accelerating: {accel:.2%}/cycle"
        return False, ""
