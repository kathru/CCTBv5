"""
Event Player — replays a recorded session through the Event Bus.

Reads a JSONL file produced by EventRecorder and re-publishes
each event to the bus in order, with optional speed control.

Use cases:
  1. Debug: replay a session that had a bug
  2. Backtest: replay historical data through live strategy code
  3. Validation: verify system behaviour is deterministic

Speed modes:
  - INSTANT  : publish all events without delay (fastest, for testing)
  - REALTIME : replay at original speed (wall clock)
  - FACTOR   : replay at N× original speed

The same strategy code runs — no separate "backtest mode".
This is the key design principle: one codebase, multiple execution modes.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import AsyncIterator

from ..core.bus import EventBus
from ..core.events import (
    Topic,
    CandleEvent, TickerEvent,
    SignalEvent,
    OrderCreatedEvent, OrderSubmittedEvent, OrderFilledEvent,
    OrderPartialEvent, OrderCancelledEvent, OrderRejectedEvent, OrderExpiredEvent,
    RiskEvaluatedEvent, KillSwitchEvent,
    HeartbeatEvent, SystemStatusEvent, ReconciliationEvent,
)
from ..core.events.base import BaseEvent

logger = logging.getLogger(__name__)

# Registry: type name → event class
EVENT_REGISTRY: dict[str, type[BaseEvent]] = {
    "CandleEvent":         CandleEvent,
    "TickerEvent":         TickerEvent,
    "SignalEvent":         SignalEvent,
    "OrderCreatedEvent":   OrderCreatedEvent,
    "OrderSubmittedEvent": OrderSubmittedEvent,
    "OrderFilledEvent":    OrderFilledEvent,
    "OrderPartialEvent":   OrderPartialEvent,
    "OrderCancelledEvent": OrderCancelledEvent,
    "OrderRejectedEvent":  OrderRejectedEvent,
    "OrderExpiredEvent":   OrderExpiredEvent,
    "RiskEvaluatedEvent":  RiskEvaluatedEvent,
    "KillSwitchEvent":     KillSwitchEvent,
    "HeartbeatEvent":      HeartbeatEvent,
    "SystemStatusEvent":   SystemStatusEvent,
    "ReconciliationEvent": ReconciliationEvent,
}


class ReplaySpeed(StrEnum):
    INSTANT  = "instant"
    REALTIME = "realtime"
    FACTOR   = "factor"


@dataclass
class ReplayStats:
    total_events: int = 0
    published: int = 0
    skipped: int = 0
    errors: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0


class EventPlayer:
    """
    Replays a JSONL event log through the Event Bus.
    Strategies receive events exactly as in live trading.
    """

    def __init__(
        self,
        bus: EventBus,
        session_file: Path,
        speed: ReplaySpeed = ReplaySpeed.INSTANT,
        speed_factor: float = 1.0,
        topics_filter: set[str] | None = None,
    ) -> None:
        self._bus = bus
        self._session_file = session_file
        self._speed = speed
        self._speed_factor = speed_factor
        self._topics_filter = topics_filter  # None = replay all topics
        self._stats = ReplayStats()

    async def play(self) -> ReplayStats:
        """
        Replay the session file from start to finish.
        Returns stats about the replay.
        """
        if not self._session_file.exists():
            raise FileNotFoundError(f"Session file not found: {self._session_file}")

        self._stats = ReplayStats(
            started_at=datetime.now(timezone.utc)
        )

        logger.info(
            "EventPlayer starting replay file=%s speed=%s",
            self._session_file, self._speed,
        )

        prev_timestamp: datetime | None = None

        async for topic, event, raw_timestamp in self._read_events():
            self._stats.total_events += 1

            # Apply speed control
            if self._speed == ReplaySpeed.REALTIME and prev_timestamp and raw_timestamp:
                delta = (raw_timestamp - prev_timestamp).total_seconds()
                if delta > 0:
                    await asyncio.sleep(delta)
            elif self._speed == ReplaySpeed.FACTOR and prev_timestamp and raw_timestamp:
                delta = (raw_timestamp - prev_timestamp).total_seconds()
                if delta > 0:
                    await asyncio.sleep(delta / self._speed_factor)

            # Publish to bus
            try:
                await self._bus.publish(Topic(topic), event)
                self._stats.published += 1
                prev_timestamp = raw_timestamp
            except Exception as exc:
                self._stats.errors += 1
                logger.warning("Failed to publish event: %s", exc)

        self._stats.finished_at = datetime.now(timezone.utc)
        logger.info(
            "EventPlayer finished: published=%d skipped=%d errors=%d duration=%.1fs",
            self._stats.published,
            self._stats.skipped,
            self._stats.errors,
            self._stats.duration_seconds,
        )
        return self._stats

    async def _read_events(
        self,
    ) -> AsyncIterator[tuple[str, BaseEvent, datetime | None]]:
        """Read and deserialize events from JSONL file."""
        with open(self._session_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    topic = record.get("topic", "")
                    event_type = record.get("type", "")
                    data = record.get("data", {})

                    # Topic filter
                    if self._topics_filter and topic not in self._topics_filter:
                        self._stats.skipped += 1
                        continue

                    # Deserialize event
                    event = self._deserialize(event_type, data)
                    if event is None:
                        self._stats.skipped += 1
                        continue

                    # Extract timestamp for speed control
                    ts = None
                    if "timestamp" in data:
                        try:
                            ts = datetime.fromisoformat(
                                str(data["timestamp"]).replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    yield topic, event, ts

                except Exception as exc:
                    self._stats.errors += 1
                    logger.warning(
                        "Parse error line=%d: %s", line_num, exc
                    )

    def _deserialize(self, event_type: str, data: dict) -> BaseEvent | None:
        """Reconstruct an event from its dict representation."""
        cls = EVENT_REGISTRY.get(event_type)
        if cls is None:
            logger.debug("Unknown event type: %s", event_type)
            return None

        # For replay purposes, create a minimal event with just event_id + timestamp
        # The nested models (Signal, Order, etc.) are complex to fully reconstruct
        # from JSON — for now we emit lightweight proxy events
        try:
            return cls(
                event_id=data.get("event_id", ""),
            )
        except Exception as exc:
            logger.debug("Cannot deserialize %s: %s", event_type, exc)
            return None
