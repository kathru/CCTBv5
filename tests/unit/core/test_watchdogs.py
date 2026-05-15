import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.watchdog.websocket_watchdog import WebSocketWatchdog
from src.watchdog.heartbeat import HeartbeatWatchdog
from src.core.bus import EventBus


# ── WebSocket Watchdog ────────────────────────────────────────

@pytest.mark.asyncio
async def test_ws_watchdog_alive_after_message():
    callback = AsyncMock()
    wd = WebSocketWatchdog(on_dead=callback, dead_threshold=30)
    wd.record_message()
    assert wd.is_alive is True
    assert wd.seconds_since_last_message < 1.0


@pytest.mark.asyncio
async def test_ws_watchdog_calls_callback_when_dead():
    callback = AsyncMock()
    wd = WebSocketWatchdog(on_dead=callback, dead_threshold=30)
    # Manually set last_message to old time
    from datetime import timedelta
    wd._last_message = datetime.now(timezone.utc) - timedelta(seconds=60)
    await wd._check()
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_ws_watchdog_no_callback_when_alive():
    callback = AsyncMock()
    wd = WebSocketWatchdog(on_dead=callback, dead_threshold=30)
    wd.record_message()
    await wd._check()
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_ws_watchdog_start_stop():
    callback = AsyncMock()
    wd = WebSocketWatchdog(on_dead=callback, dead_threshold=30, check_interval=60)
    await wd.start()
    assert wd._running is True
    await wd.stop()
    assert wd._running is False


def test_ws_watchdog_status():
    callback = AsyncMock()
    wd = WebSocketWatchdog(on_dead=callback, name="test_ws")
    wd.record_message()
    s = wd.status()
    assert s["name"] == "test_ws"
    assert "is_alive" in s
    assert "dead_count" in s


# ── Heartbeat Watchdog ────────────────────────────────────────

@pytest.mark.asyncio
async def test_heartbeat_alive_after_beat():
    bus = EventBus()
    wd = HeartbeatWatchdog(bus=bus, timeout_seconds=60)
    wd.beat()
    assert wd.is_alive is True


@pytest.mark.asyncio
async def test_heartbeat_triggers_kill_switch_on_hang():
    bus = EventBus()
    kill_switch = MagicMock()
    wd = HeartbeatWatchdog(bus=bus, timeout_seconds=30, kill_switch=kill_switch)

    from datetime import timedelta
    wd._last_beat = datetime.now(timezone.utc) - timedelta(seconds=60)
    await wd._check()

    kill_switch.trigger_soft.assert_called_once()


@pytest.mark.asyncio
async def test_heartbeat_publishes_event_on_check():
    bus = EventBus()
    queue = bus.subscribe(topic=__import__('src.core.events', fromlist=['Topic']).Topic.SYSTEM)
    wd = HeartbeatWatchdog(bus=bus, timeout_seconds=60, interval_seconds=60)
    wd.beat()
    await wd._check()
    # Should have published HeartbeatEvent
    assert not queue.empty()


def test_heartbeat_status():
    bus = EventBus()
    wd = HeartbeatWatchdog(bus=bus)
    wd.beat()
    s = wd.status()
    assert "beat_count" in s
    assert "hang_count" in s
    assert s["beat_count"] == 1
