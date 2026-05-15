"""
Event Bus — asyncio-based dispatcher with fan-out per topic.

Design:
  - Each topic has N subscriber queues (one per consumer).
  - Publishing to a topic delivers the event to ALL subscribers.
  - No Kafka, no Redis Streams — pure asyncio.Queue (Oracle Free friendly).
  - All events are immutable (frozen dataclasses) → safe to share across queues.

Usage:
    bus = EventBus()

    # Subscribe
    queue = bus.subscribe(Topic.SIGNAL)

    # Publish (from producer)
    await bus.publish(Topic.SIGNAL, SignalEvent(signal=my_signal))

    # Consume (in consumer coroutine)
    event = await queue.get()
"""

import asyncio
import logging
from collections import defaultdict
from typing import TypeVar

from ..events.base import BaseEvent
from ..events.topics import Topic

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseEvent)


class EventBus:
    """
    Central dispatcher. One instance per application.
    Inject via dependency injection — do not use as global singleton.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        # topic → list of subscriber queues
        self._subscribers: dict[Topic, list[asyncio.Queue[BaseEvent]]] = (
            defaultdict(list)
        )
        self._maxsize = maxsize
        self._publish_count: dict[Topic, int] = defaultdict(int)
        self._drop_count: dict[Topic, int] = defaultdict(int)

    def subscribe(self, topic: Topic) -> asyncio.Queue[BaseEvent]:
        """
        Register a new subscriber for a topic.
        Returns a dedicated queue — the consumer owns it.
        Call before the event loop starts consuming.
        """
        queue: asyncio.Queue[BaseEvent] = asyncio.Queue(
            maxsize=self._maxsize
        )
        self._subscribers[topic].append(queue)
        logger.debug(
            "New subscriber on topic=%s total_subscribers=%d",
            topic,
            len(self._subscribers[topic]),
        )
        return queue

    async def publish(self, topic: Topic, event: BaseEvent) -> None:
        """
        Deliver event to all subscribers of the topic (fan-out).
        If a subscriber queue is full, the event is dropped for that
        subscriber and counted — never blocks the producer.
        """
        subscribers = self._subscribers.get(topic, [])
        self._publish_count[topic] += 1

        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._drop_count[topic] += 1
                logger.warning(
                    "Queue full — dropping event topic=%s event_id=%s",
                    topic,
                    event.event_id,
                )

    def subscriber_count(self, topic: Topic) -> int:
        return len(self._subscribers.get(topic, []))

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            "published": dict(self._publish_count),
            "dropped": dict(self._drop_count),
            "subscribers": {
                t: len(qs) for t, qs in self._subscribers.items()
            },
        }
