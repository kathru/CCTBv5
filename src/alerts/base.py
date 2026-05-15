"""
AlertChannel — abstract interface for all alert providers.

Design: the system never depends on Discord directly.
It depends on AlertChannel. This means:
  - Switching provider = swap the implementation
  - Testing = use MockAlertChannel
  - Multiple channels = CompositeAlertChannel

Alert levels:
  INFO    — routine notifications (trade opened, closed)
  WARNING — degraded performance (high DD, reconnect)
  CRITICAL— system needs human attention (kill switch, divergence)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from datetime import datetime, timezone


class AlertLevel(StrEnum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a single alert to be sent."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime | None = None
    fields: dict[str, str] | None = None   # key-value pairs shown in embed

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class AlertChannel(ABC):
    """Abstract base — all alert providers must implement send()."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """
        Send an alert. Returns True if sent successfully.
        Must never raise — catch and log internally.
        """
        ...

    async def info(self, title: str, message: str, **fields) -> bool:
        return await self.send(Alert(
            level=AlertLevel.INFO,
            title=title,
            message=message,
            fields={k: str(v) for k, v in fields.items()} if fields else None,
        ))

    async def warning(self, title: str, message: str, **fields) -> bool:
        return await self.send(Alert(
            level=AlertLevel.WARNING,
            title=title,
            message=message,
            fields={k: str(v) for k, v in fields.items()} if fields else None,
        ))

    async def critical(self, title: str, message: str, **fields) -> bool:
        return await self.send(Alert(
            level=AlertLevel.CRITICAL,
            title=title,
            message=message,
            fields={k: str(v) for k, v in fields.items()} if fields else None,
        ))


class NullAlertChannel(AlertChannel):
    """No-op channel — used when DISCORD_WEBHOOK_URL is not set."""

    async def send(self, alert: Alert) -> bool:
        return True   # silently discard


class CompositeAlertChannel(AlertChannel):
    """Fan-out to multiple channels simultaneously."""

    def __init__(self, channels: list[AlertChannel]) -> None:
        self._channels = channels

    async def send(self, alert: Alert) -> bool:
        results = []
        for ch in self._channels:
            results.append(await ch.send(alert))
        return all(results)
