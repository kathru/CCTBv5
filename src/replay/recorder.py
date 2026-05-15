"""
Event Recorder — subscribes to all topics and saves events to JSONL.

Each line in the output file is one JSON event.
Format: {"topic": "market", "type": "CandleEvent", "data": {...}}

This enables:
  1. Replay of any session for debugging
  2. Backtesting with real event sequences
  3. Audit trail of every decision

Usage:
    recorder = EventRecorder(bus, output_path="logs/session_2026-05-15.jsonl")
    await recorder.start()
    ...
    await recorder.stop()
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..core.bus import EventBus
from ..core.events import Topic
from ..core.events.base import BaseEvent

logger = logging.getLogger(__name__)


def _default_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"session_{ts}.jsonl"


class EventRecorder:
    """
    Subscribes to all event topics and writes each event to a JSONL file.
    One event per line — easy to stream, parse, and replay.
    """

    def __init__(
        self,
        bus: EventBus,
        output_path: Path | None = None,
        topics: list[Topic] | None = None,
    ) -> None:
        self._bus = bus
        self._output_path = output_path or _default_path()
        self._topics = topics or list(Topic)
        self._queues: list[asyncio.Queue] = []
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._file = None
        self._event_count = 0

    async def start(self) -> None:
        if self._running:
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_path, "a", encoding="utf-8")
        self._running = True

        # Subscribe to all topics
        for topic in self._topics:
            queue = self._bus.subscribe(topic)
            self._queues.append(queue)
            task = asyncio.create_task(
                self._consume(topic, queue),
                name=f"recorder_{topic}",
            )
            self._tasks.append(task)

        logger.info(
            "EventRecorder started topics=%d output=%s",
            len(self._topics), self._output_path,
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        logger.info(
            "EventRecorder stopped events_recorded=%d", self._event_count
        )

    async def _consume(self, topic: Topic, queue: asyncio.Queue) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                self._write(topic, event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Recorder error topic=%s: %s", topic, exc)

    def _write(self, topic: Topic, event: BaseEvent) -> None:
        if self._file is None:
            return
        try:
            record = {
                "topic": topic.value,
                "type": type(event).__name__,
                "data": event.to_dict(),
            }
            self._file.write(json.dumps(record, default=str) + "\n")
            self._event_count += 1
            # Flush every 10 events to avoid data loss on crash
            if self._event_count % 10 == 0:
                self._file.flush()
        except Exception as exc:
            logger.error("Failed to write event: %s", exc)

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def output_path(self) -> Path:
        return self._output_path
