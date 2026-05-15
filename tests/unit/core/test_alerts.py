from unittest.mock import AsyncMock, patch

import pytest

from src.alerts.base import Alert, AlertLevel, CompositeAlertChannel, NullAlertChannel
from src.alerts.discord import DiscordAlertChannel, create_alert_channel

# ── Base / Null ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_null_channel_always_returns_true():
    ch = NullAlertChannel()
    result = await ch.send(Alert(AlertLevel.CRITICAL, "test", "msg"))
    assert result is True


@pytest.mark.asyncio
async def test_alert_channel_convenience_methods():
    ch = NullAlertChannel()
    assert await ch.info("title", "msg") is True
    assert await ch.warning("title", "msg") is True
    assert await ch.critical("title", "msg") is True


@pytest.mark.asyncio
async def test_composite_sends_to_all_channels():
    results = []

    class RecordingChannel(NullAlertChannel):
        async def send(self, alert):
            results.append(alert.level)
            return True

    ch = CompositeAlertChannel([RecordingChannel(), RecordingChannel()])
    await ch.send(Alert(AlertLevel.WARNING, "t", "m"))
    assert len(results) == 2


# ── Discord ───────────────────────────────────────────────────

def test_create_alert_channel_returns_null_when_no_url():
    ch = create_alert_channel("")
    assert isinstance(ch, NullAlertChannel)


def test_create_alert_channel_returns_discord_when_url():
    ch = create_alert_channel("https://discord.com/api/webhooks/test")
    assert isinstance(ch, DiscordAlertChannel)


@pytest.mark.asyncio
async def test_discord_channel_builds_correct_payload():
    ch = DiscordAlertChannel(webhook_url="https://fake.webhook")
    alert = Alert(
        level=AlertLevel.CRITICAL,
        title="Kill Switch",
        message="Sistema parado",
        fields={"modo": "HARD", "motivo": "teste"},
    )
    payload = ch._build_payload(alert)
    assert payload["embeds"][0]["title"].startswith("🚨")
    assert payload["embeds"][0]["color"] == 0xE74C3C
    assert len(payload["embeds"][0]["fields"]) == 2


@pytest.mark.asyncio
async def test_discord_channel_level_filter():
    ch = DiscordAlertChannel(
        webhook_url="https://fake.webhook",
        min_level=AlertLevel.WARNING,
    )
    # INFO should be filtered
    assert ch._should_send(Alert(AlertLevel.INFO, "t", "m")) is False
    # WARNING should pass
    assert ch._should_send(Alert(AlertLevel.WARNING, "t", "m")) is True
    # CRITICAL should pass
    assert ch._should_send(Alert(AlertLevel.CRITICAL, "t", "m")) is True


@pytest.mark.asyncio
async def test_discord_send_success():
    alert = Alert(AlertLevel.INFO, "Test", "Hello")
    # Use NullAlertChannel to avoid real HTTP calls in tests
    null_ch = NullAlertChannel()
    result = await null_ch.send(alert)
    assert result is True
