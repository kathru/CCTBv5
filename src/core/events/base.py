"""
Base event — all events in the system inherit from this.
Frozen dataclass = immutable + hashable + serializable.
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class BaseEvent:
    """Every event is immutable and carries a unique ID and timestamp."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict — required for replay and persistence."""
        return asdict(self)
