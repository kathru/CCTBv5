import asyncio
import json
from datetime import UTC, datetime

import pytest

from src.core.bus import EventBus
from src.core.events import CandleEvent, Topic
from src.core.models import Candle
from src.replay.event_player import EventPlayer, ReplaySpeed
from src.replay.recorder import EventRecorder


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def tmp_session(tmp_path):
    return tmp_path / "test_session.jsonl"


@pytest.mark.asyncio
async def test_recorder_creates_file(bus, tmp_session):
    recorder = EventRecorder(bus, output_path=tmp_session, topics=[Topic.MARKET])
    await recorder.start()

    # Publish a candle event
    candle = Candle(
        symbol="BTC-USDT",
        granularity="1H",
        timestamp=datetime.now(UTC),
        open=42000, high=43000, low=41000, close=42500, volume=100,
    )
    await bus.publish(Topic.MARKET, CandleEvent(candle=candle))
    await asyncio.sleep(0.05)  # let consumer write

    await recorder.stop()
    assert tmp_session.exists()
    assert recorder.event_count >= 1


@pytest.mark.asyncio
async def test_recorder_writes_valid_json(bus, tmp_session):
    recorder = EventRecorder(bus, output_path=tmp_session, topics=[Topic.SYSTEM])
    await recorder.start()

    from src.core.events import HeartbeatEvent
    from src.core.events.system_events import SystemStatus
    await bus.publish(Topic.SYSTEM, HeartbeatEvent(status=SystemStatus.RUNNING))
    await asyncio.sleep(0.05)

    await recorder.stop()

    lines = tmp_session.read_text().strip().split("\n")
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert "topic" in record
    assert "type" in record
    assert "data" in record


@pytest.mark.asyncio
async def test_player_instant_replay(bus, tmp_session):
    """Write events to file, replay them, count published."""
    # Write test JSONL manually
    records = [
        {"topic": "system", "type": "HeartbeatEvent",
         "data": {"event_id": "a1", "timestamp": "2026-05-15T00:00:00Z",
                  "status": "running", "latency_ms": 0.0}},
        {"topic": "system", "type": "HeartbeatEvent",
         "data": {"event_id": "a2", "timestamp": "2026-05-15T00:01:00Z",
                  "status": "running", "latency_ms": 0.0}},
    ]
    tmp_session.write_text(
        "\n".join(json.dumps(r) for r in records)
    )

    player = EventPlayer(bus, tmp_session, speed=ReplaySpeed.INSTANT)
    stats = await player.play()

    assert stats.total_events == 2
    assert stats.published == 2
    assert stats.errors == 0


@pytest.mark.asyncio
async def test_player_skips_unknown_event_types(bus, tmp_session):
    records = [
        {"topic": "market", "type": "UnknownEventXYZ",
         "data": {"event_id": "b1", "timestamp": "2026-05-15T00:00:00Z"}},
    ]
    tmp_session.write_text(json.dumps(records[0]))

    player = EventPlayer(bus, tmp_session, speed=ReplaySpeed.INSTANT)
    stats = await player.play()

    assert stats.skipped == 1
    assert stats.published == 0


@pytest.mark.asyncio
async def test_player_raises_if_file_not_found(bus, tmp_path):
    player = EventPlayer(
        bus,
        tmp_path / "nonexistent.jsonl",
        speed=ReplaySpeed.INSTANT,
    )
    with pytest.raises(FileNotFoundError):
        await player.play()
