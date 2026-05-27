"""
WebSocket Watchdog — detects dead connections and triggers reconnect.

Rule: if no WS message received in > 30 seconds → connection is dead.
Action: emit KillSwitchEvent(SOFT) + trigger reconnect callback.

The watchdog itself never reconnects — it delegates to the caller
via a callback. This keeps it decoupled from the exchange adapter.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

DEAD_THRESHOLD_SECONDS = 30
CHECK_INTERVAL_SECONDS = 5


def _now() -> datetime:
    return datetime.now(UTC)


class WebSocketWatchdog:
    """
    Monitors WebSocket liveness by tracking last message timestamp.
    Call record_message() on every received WS message.
    Start the watchdog as a background task.
    """

    def __init__(
        self,
        on_dead: Callable[[], Awaitable[None]],
        dead_threshold: int = DEAD_THRESHOLD_SECONDS,
        check_interval: int = CHECK_INTERVAL_SECONDS,
        name: str = "ws",
    ) -> None:
        """
        Args:
            on_dead: Async callback called when connection is declared dead.
            dead_threshold: Seconds without a message before declaring dead.
            check_interval: How often to check liveness (seconds).
            name: Label for logging (e.g. "okx_public", "okx_private").
        """
        self._on_dead = on_dead
        self._threshold = dead_threshold
        self._interval = check_interval
        self._name = name

        self._last_message: datetime | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._dead_count = 0
        self._reconnect_count = 0

    def record_message(self) -> None:
        """Call this on every received WebSocket message."""
        self._last_message = _now()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_message = _now()   # assume alive at start
        self._task = asyncio.create_task(
            self._loop(), name=f"ws_watchdog_{self._name}"
        )
        logger.info(
            "WebSocketWatchdog started name=%s threshold=%ds",
            self._name, self._threshold,
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
            except Exception:
                logger.exception("WebSocketWatchdog error")

    async def _check(self) -> None:
        if self._last_message is None:
            return

        age = (_now() - self._last_message).total_seconds()

        if age > self._threshold:
            self._dead_count += 1
            logger.warning(
                "WebSocket DEAD name=%s last_message=%.0fs ago count=%d",
                self._name, age, self._dead_count,
            )
            await self._on_dead()
            self._reconnect_count += 1
            # Reset last_message to avoid firing repeatedly
            self._last_message = _now()
        else:
            logger.debug(
                "WebSocket alive name=%s last_message=%.1fs ago",
                self._name, age,
            )

    @property
    def is_alive(self) -> bool:
        if self._last_message is None:
            return False
        age = (_now() - self._last_message).total_seconds()
        return age <= self._threshold

    @property
    def seconds_since_last_message(self) -> float:
        if self._last_message is None:
            return float("inf")
        return (_now() - self._last_message).total_seconds()

    def status(self) -> dict:
        return {
            "name": self._name,
            "running": self._running,
            "is_alive": self.is_alive,
            "seconds_since_last_message": round(self.seconds_since_last_message, 1),
            "dead_count": self._dead_count,
            "reconnect_count": self._reconnect_count,
        }
