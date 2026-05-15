from dataclasses import dataclass
from enum import StrEnum

from .base import BaseEvent


class SystemStatus(StrEnum):
    STARTING      = "starting"
    RECONCILING   = "reconciling"   # boot: comparing local vs exchange
    RUNNING       = "running"
    SUSPENDED     = "suspended"     # risk block
    SHUTTING_DOWN = "shutting_down"


@dataclass(frozen=True)
class HeartbeatEvent(BaseEvent):
    """Periodic liveness signal from the main loop."""
    status: SystemStatus = SystemStatus.RUNNING
    latency_ms: float = 0.0


@dataclass(frozen=True)
class SystemStatusEvent(BaseEvent):
    """System changed operational state."""
    status: SystemStatus = SystemStatus.STARTING
    reason: str = ""


@dataclass(frozen=True)
class ReconciliationEvent(BaseEvent):
    """Reconciliation cycle completed."""
    divergences_found: int = 0
    resolved: bool = True
    detail: str = ""
