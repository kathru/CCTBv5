from dataclasses import dataclass
from .base import BaseEvent
from ..models import Signal


@dataclass(frozen=True)
class SignalEvent(BaseEvent):
    """A strategy produced a signal."""
    signal: Signal | None = None
