from dataclasses import dataclass

from ..models import Signal
from .base import BaseEvent


@dataclass(frozen=True)
class SignalEvent(BaseEvent):
    """A strategy produced a signal."""
    signal: Signal | None = None
