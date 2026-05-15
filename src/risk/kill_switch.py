"""
Kill Switch — emergency stop with two modes.

SOFT: closes all open positions, blocks new entries.
      Used when: daily DD breached, websocket unstable, high latency.
      Recovery: automatic after conditions normalize.

HARD: stops everything immediately, no new orders, no auto-recovery.
      Used when: exchange error, critical system failure, manual trigger.
      Recovery: requires explicit manual reset.

The kill switch state is the final gate — it overrides everything else.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class KillSwitchState(StrEnum):
    ARMED    = "armed"     # normal operation
    SOFT     = "soft"      # positions closing, no new entries
    HARD     = "hard"      # full stop


@dataclass
class KillSwitchEvent:
    mode: KillSwitchState
    reason: str
    triggered_at: datetime = field(default_factory=_now)
    resolved_at: datetime | None = None


class KillSwitch:
    """
    Global safety gate. Injected into OMS and Risk Engine.
    Any component can trigger it. Only manual action resets HARD.
    """

    def __init__(self) -> None:
        self._state: KillSwitchState = KillSwitchState.ARMED
        self._history: list[KillSwitchEvent] = []
        self._current_event: KillSwitchEvent | None = None

    # ── Triggers ──────────────────────────────────────────────

    def trigger_soft(self, reason: str) -> None:
        """
        SOFT kill: block new entries, close existing positions.
        Can recover automatically.
        """
        if self._state == KillSwitchState.HARD:
            logger.warning("HARD kill active — ignoring SOFT trigger")
            return
        if self._state == KillSwitchState.SOFT:
            return  # already soft

        self._state = KillSwitchState.SOFT
        event = KillSwitchEvent(mode=KillSwitchState.SOFT, reason=reason)
        self._current_event = event
        self._history.append(event)
        logger.warning("KILL SWITCH SOFT triggered reason=%s", reason)

    def trigger_hard(self, reason: str) -> None:
        """
        HARD kill: stop everything, requires manual reset.
        """
        self._state = KillSwitchState.HARD
        event = KillSwitchEvent(mode=KillSwitchState.HARD, reason=reason)
        self._current_event = event
        self._history.append(event)
        logger.critical("KILL SWITCH HARD triggered reason=%s", reason)

    # ── Recovery ──────────────────────────────────────────────

    def reset_soft(self, reason: str = "conditions_normalized") -> bool:
        """
        Recover from SOFT kill. Returns True if reset was applied.
        HARD kill cannot be reset this way.
        """
        if self._state == KillSwitchState.HARD:
            logger.error("Cannot reset SOFT — HARD kill is active")
            return False
        if self._state == KillSwitchState.SOFT:
            if self._current_event:
                self._current_event.resolved_at = _now()
            self._state = KillSwitchState.ARMED
            self._current_event = None
            logger.info("Kill switch RESET to ARMED reason=%s", reason)
            return True
        return False

    def reset_hard(self, reason: str = "manual_reset") -> None:
        """
        Manual reset of HARD kill. Only humans should call this.
        """
        if self._current_event:
            self._current_event.resolved_at = _now()
        self._state = KillSwitchState.ARMED
        self._current_event = None
        logger.warning("HARD kill switch MANUALLY RESET reason=%s", reason)

    # ── Gates ─────────────────────────────────────────────────

    @property
    def is_armed(self) -> bool:
        """True = normal operation."""
        return self._state == KillSwitchState.ARMED

    @property
    def allows_new_entries(self) -> bool:
        """False during SOFT or HARD."""
        return self._state == KillSwitchState.ARMED

    @property
    def allows_any_operation(self) -> bool:
        """False only during HARD."""
        return self._state != KillSwitchState.HARD

    @property
    def state(self) -> KillSwitchState:
        return self._state

    def status(self) -> dict:
        return {
            "state": self._state.value,
            "reason": self._current_event.reason if self._current_event else None,
            "triggered_at": (
                self._current_event.triggered_at.isoformat()
                if self._current_event else None
            ),
            "total_events": len(self._history),
        }
