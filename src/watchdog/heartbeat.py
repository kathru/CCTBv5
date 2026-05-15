"""
Heartbeat Watchdog — verifies the main trading loop is alive.

The main loop must call beat() regularly.
If beat() is not called within the timeout, the system is considered hung.

On hang detected:
  - Log critical alert
  - Trigger soft kill switch
  - Emit HeartbeatEvent with degraded status
"""

import asyncio
import logging
from datetime import datetime, timezone

from ..core.bus import EventBus
from ..core.events import Topic
from ..core.events.system_events import SystemStatus
from ..core.events import HeartbeatEvent

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HeartbeatWatchdog:
    """
    Monitors trading loop liveness via periodic beat() calls.
    Runs as a background task and publishes HeartbeatEvent each cycle.
    """

    def __init__(
        self,
        bus: EventBus,
        timeout_seconds: int = 60,
        interval_seconds: int = 15,
        kill_switch=None,
    ) -> None:
        self._bus = bus
        self._timeout = timeout_seconds
        self._interval = interval_seconds
        self._kill_switch = kill_switch

        self._last_beat: datetime | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._beat_count = 0
        self._hang_count = 0

    def beat(self) -> None:
        """Call this from the main trading loop every cycle."""
        self._last_beat = _now()
        self._beat_count += 1

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_beat = _now()
        self._task = asyncio.create_task(
            self._loop(), name="heartbeat_watchdog"
        )
        logger.info(
            "HeartbeatWatchdog started timeout=%ds interval=%ds",
            self._timeout, self._interval,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                await self._check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("HeartbeatWatchdog error: %s", exc)

    async def _check(self) -> None:
        if self._last_beat is None:
            return

        age = (_now() - self._last_beat).total_seconds()
        latency_ms = age * 1000

        if age > self._timeout:
            self._hang_count += 1
            logger.critical(
                "HEARTBEAT TIMEOUT — main loop hung! last_beat=%.0fs ago",
                age,
            )
            # Trigger soft kill switch
            if self._kill_switch:
                self._kill_switch.trigger_soft(
                    reason=f"heartbeat_timeout_{age:.0f}s"
                )
            # Publish degraded heartbeat
            await self._bus.publish(
                Topic.SYSTEM,
                HeartbeatEvent(
                    status=SystemStatus.SUSPENDED,
                    latency_ms=latency_ms,
                ),
            )
        else:
            # Normal heartbeat
            await self._bus.publish(
                Topic.SYSTEM,
                HeartbeatEvent(
                    status=SystemStatus.RUNNING,
                    latency_ms=latency_ms,
                ),
            )
            logger.debug("Heartbeat OK latency=%.0fms", latency_ms)

    @property
    def is_alive(self) -> bool:
        if self._last_beat is None:
            return False
        return (_now() - self._last_beat).total_seconds() <= self._timeout

    def status(self) -> dict:
        age = (_now() - self._last_beat).total_seconds() if self._last_beat else None
        return {
            "running": self._running,
            "is_alive": self.is_alive,
            "beat_count": self._beat_count,
            "hang_count": self._hang_count,
            "seconds_since_last_beat": round(age, 1) if age else None,
        }
