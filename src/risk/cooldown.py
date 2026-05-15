"""
Cooldown — pauses a strategy after consecutive losses.

Why: consecutive losses often indicate regime change or edge death.
A short pause prevents the system from digging deeper into a hole
while the meta-layer adapts weights.

Cooldown is per-strategy, not global — one bad strategy
shouldn't freeze the entire system.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class StrategyCooldownState:
    strategy_id: str
    consecutive_losses: int = 0
    in_cooldown: bool = False
    cooldown_until: datetime | None = None
    total_cooldowns: int = 0


class CooldownEngine:
    """
    Tracks consecutive losses per strategy.
    Activates cooldown when threshold is breached.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        cooldown_duration: timedelta = timedelta(hours=2),
    ) -> None:
        self._max_losses = max_consecutive_losses
        self._cooldown_duration = cooldown_duration
        self._states: dict[str, StrategyCooldownState] = {}

    def _get_state(self, strategy_id: str) -> StrategyCooldownState:
        if strategy_id not in self._states:
            self._states[strategy_id] = StrategyCooldownState(strategy_id)
        return self._states[strategy_id]

    def record_loss(self, strategy_id: str) -> bool:
        """
        Record a loss for a strategy.
        Returns True if cooldown was just activated.
        """
        state = self._get_state(strategy_id)
        state.consecutive_losses += 1

        if state.consecutive_losses >= self._max_losses and not state.in_cooldown:
            state.in_cooldown = True
            state.cooldown_until = _now() + self._cooldown_duration
            state.total_cooldowns += 1
            return True  # cooldown activated

        return False

    def record_win(self, strategy_id: str) -> None:
        """A win resets the consecutive loss counter."""
        state = self._get_state(strategy_id)
        state.consecutive_losses = 0

    def is_allowed(self, strategy_id: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        Checks if strategy is currently in cooldown.
        """
        state = self._get_state(strategy_id)

        if not state.in_cooldown:
            return True, ""

        # Check if cooldown has expired
        if state.cooldown_until and _now() >= state.cooldown_until:
            state.in_cooldown = False
            state.cooldown_until = None
            state.consecutive_losses = 0
            return True, ""

        remaining = state.cooldown_until - _now() if state.cooldown_until else timedelta()
        return False, (
            f"Strategy {strategy_id} in cooldown for "
            f"{remaining.seconds // 60}min more "
            f"(after {state.consecutive_losses} consecutive losses)"
        )

    def force_reset(self, strategy_id: str) -> None:
        """Manual reset — used by kill switch recovery."""
        state = self._get_state(strategy_id)
        state.in_cooldown = False
        state.cooldown_until = None
        state.consecutive_losses = 0

    def status(self) -> dict[str, dict]:
        return {
            sid: {
                "consecutive_losses": s.consecutive_losses,
                "in_cooldown": s.in_cooldown,
                "cooldown_until": s.cooldown_until.isoformat() if s.cooldown_until else None,
                "total_cooldowns": s.total_cooldowns,
            }
            for sid, s in self._states.items()
        }
