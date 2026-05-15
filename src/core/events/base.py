"""
Base event — all events in the system inherit from this.
Frozen dataclass = immutable + hashable + serializable.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
import uuid


@dataclass(frozen=True)
class BaseEvent:
    """Every event is immutable and carries a unique ID and timestamp."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict — required for replay and persistence."""
        return asdict(self)
