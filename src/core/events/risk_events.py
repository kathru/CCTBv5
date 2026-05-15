from dataclasses import dataclass
from enum import StrEnum
from .base import BaseEvent


class RiskAction(StrEnum):
    NORMAL  = "normal"    # proceed as planned
    REDUCE  = "reduce"    # halve position size
    CLOSE   = "close"     # exit all open positions
    SUSPEND = "suspend"   # no new entries


class KillSwitchMode(StrEnum):
    SOFT = "soft"   # close positions, block new entries
    HARD = "hard"   # stop everything immediately


@dataclass(frozen=True)
class RiskEvaluatedEvent(BaseEvent):
    """Risk engine completed evaluation cycle."""
    action: RiskAction = RiskAction.NORMAL
    reason: str = ""
    var_pct: float = 0.0
    drawdown_pct: float = 0.0


@dataclass(frozen=True)
class KillSwitchEvent(BaseEvent):
    """Kill switch triggered."""
    mode: KillSwitchMode = KillSwitchMode.SOFT
    reason: str = ""
