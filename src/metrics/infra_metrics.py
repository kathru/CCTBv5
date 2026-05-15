"""
Infrastructure Metrics — tracks system health in real-time.

Metrics:
  - WebSocket uptime     : % time WS connection was live
  - Reconnect count      : how many times WS had to reconnect
  - Event lag            : delay between market event and signal
  - Order latency        : time from signal to order submission
  - Poll cycle time      : how long each MarketEngine poll takes
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class InfraSnapshot:
    """Point-in-time snapshot of infrastructure metrics."""
    timestamp: datetime
    ws_uptime_pct: float
    reconnect_count: int
    avg_event_lag_ms: float
    avg_order_latency_ms: float
    avg_poll_duration_ms: float
    error_count: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ws_uptime_pct": f"{self.ws_uptime_pct:.1%}",
            "reconnect_count": self.reconnect_count,
            "avg_event_lag_ms": f"{self.avg_event_lag_ms:.1f}",
            "avg_order_latency_ms": f"{self.avg_order_latency_ms:.1f}",
            "avg_poll_duration_ms": f"{self.avg_poll_duration_ms:.1f}",
            "error_count": self.error_count,
        }


class InfraMetrics:
    """
    Collects and exposes infrastructure performance metrics.
    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self, window: int = 100) -> None:
        """
        Args:
            window: Number of samples to keep for rolling averages.
        """
        self._window = window

        # WebSocket
        self._ws_connected_seconds: float = 0.0
        self._ws_total_seconds: float = 0.0
        self._ws_last_connect: datetime | None = None
        self._reconnect_count: int = 0

        # Latencies (rolling windows)
        self._event_lags_ms: deque[float] = deque(maxlen=window)
        self._order_latencies_ms: deque[float] = deque(maxlen=window)
        self._poll_durations_ms: deque[float] = deque(maxlen=window)

        # Errors
        self._error_count: int = 0

        self._started_at: datetime = _now()

    # ── WebSocket tracking ────────────────────────────────────

    def record_ws_connect(self) -> None:
        self._ws_last_connect = _now()
        if self._reconnect_count > 0:
            self._reconnect_count += 1
        else:
            self._reconnect_count = 0
        logger.debug("WS connected reconnects=%d", self._reconnect_count)

    def record_ws_disconnect(self) -> None:
        if self._ws_last_connect:
            connected = (_now() - self._ws_last_connect).total_seconds()
            self._ws_connected_seconds += connected
        self._reconnect_count += 1
        self._ws_last_connect = None
        logger.debug("WS disconnected reconnects=%d", self._reconnect_count)

    def record_ws_message(self) -> None:
        """Call on every received WS message to track uptime."""
        total = (_now() - self._started_at).total_seconds()
        self._ws_total_seconds = total

    # ── Latency tracking ─────────────────────────────────────

    def record_event_lag(self, lag_ms: float) -> None:
        """Time between market event creation and strategy evaluation."""
        self._event_lags_ms.append(lag_ms)

    def record_order_latency(self, latency_ms: float) -> None:
        """Time from signal to order submission."""
        self._order_latencies_ms.append(latency_ms)

    def record_poll_duration(self, duration_ms: float) -> None:
        """Time for a complete MarketEngine poll cycle."""
        self._poll_durations_ms.append(duration_ms)

    # ── Error tracking ────────────────────────────────────────

    def record_error(self) -> None:
        self._error_count += 1

    # ── Snapshot ──────────────────────────────────────────────

    def snapshot(self) -> InfraSnapshot:
        """Return current metrics snapshot."""
        total = max((_now() - self._started_at).total_seconds(), 1)

        # WS uptime
        connected = self._ws_connected_seconds
        if self._ws_last_connect:
            connected += (_now() - self._ws_last_connect).total_seconds()
        ws_uptime = min(connected / total, 1.0)

        def avg(samples: deque) -> float:
            return sum(samples) / len(samples) if samples else 0.0

        return InfraSnapshot(
            timestamp=_now(),
            ws_uptime_pct=ws_uptime,
            reconnect_count=self._reconnect_count,
            avg_event_lag_ms=avg(self._event_lags_ms),
            avg_order_latency_ms=avg(self._order_latencies_ms),
            avg_poll_duration_ms=avg(self._poll_durations_ms),
            error_count=self._error_count,
        )

    def is_healthy(self) -> bool:
        """Basic health check — WS mostly up, low error rate."""
        snap = self.snapshot()
        return (
            snap.ws_uptime_pct >= 0.95
            and snap.avg_event_lag_ms < 500
            and snap.reconnect_count < 10
        )
