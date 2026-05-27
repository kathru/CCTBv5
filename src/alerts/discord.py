"""
Discord Alert Channel — sends alerts via Discord webhook.

Uses rich embeds with color coding by level:
  INFO     → blue   (#3498db)
  WARNING  → yellow (#f39c12)
  CRITICAL → red    (#e74c3c)

Webhook URL is set via DISCORD_WEBHOOK_URL env variable.
If not set, NullAlertChannel is used automatically.

Rate limiting: Discord allows ~30 messages/minute per webhook.
We add a small delay between sends to stay safe.
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime

import httpx

from .base import Alert, AlertChannel, AlertLevel, NullAlertChannel

# Discord allows ~30 messages/minute per webhook (≈ 2s per message).
# We enforce a minimum gap and a per-minute bucket to stay under the limit.
_MIN_SEND_INTERVAL = 2.0   # seconds between any two sends
_MAX_PER_MINUTE    = 25    # conservative limit (Discord hard limit is 30)

logger = logging.getLogger(__name__)

# Color codes for Discord embeds
LEVEL_COLORS = {
    AlertLevel.INFO:     0x3498DB,   # blue
    AlertLevel.WARNING:  0xF39C12,   # yellow
    AlertLevel.CRITICAL: 0xE74C3C,   # red
}

LEVEL_EMOJI = {
    AlertLevel.INFO:     "ℹ️",
    AlertLevel.WARNING:  "⚠️",
    AlertLevel.CRITICAL: "🚨",
}


class DiscordAlertChannel(AlertChannel):
    """
    Sends alerts to a Discord channel via webhook.
    Uses rich embeds for formatted output.
    """

    def __init__(
        self,
        webhook_url: str,
        bot_name: str = "CCTBv5",
        min_level: AlertLevel = AlertLevel.INFO,
    ) -> None:
        self._webhook_url = webhook_url
        self._bot_name = bot_name
        self._min_level = min_level
        self._sent_count = 0
        self._error_count = 0
        self._dropped_count = 0
        # Rate limiting state
        self._last_sent_at: float = 0.0
        self._window_start: float = time.monotonic()
        self._window_count: int = 0
        self._send_lock = asyncio.Lock()

    async def send(self, alert: Alert) -> bool:
        """Send alert as a Discord embed. Returns True if successful."""
        if not self._should_send(alert):
            return True

        async with self._send_lock:
            # Per-minute bucket reset
            now = time.monotonic()
            if now - self._window_start >= 60.0:
                self._window_start = now
                self._window_count = 0

            # Hard drop if over per-minute limit (avoids Discord 429)
            if self._window_count >= _MAX_PER_MINUTE:
                self._dropped_count += 1
                logger.warning(
                    "Discord rate limit reached — alert dropped level=%s title=%s",
                    alert.level, alert.title,
                )
                return False

            # Enforce minimum inter-message gap
            elapsed = now - self._last_sent_at
            if elapsed < _MIN_SEND_INTERVAL:
                await asyncio.sleep(_MIN_SEND_INTERVAL - elapsed)

            payload = self._build_payload(alert)

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        self._webhook_url,
                        content=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    )
                    self._last_sent_at = time.monotonic()
                    if resp.status_code in {200, 204}:
                        self._sent_count += 1
                        self._window_count += 1
                        logger.debug(
                            "Discord alert sent level=%s title=%s",
                            alert.level, alert.title,
                        )
                        return True
                    elif resp.status_code == 429:
                        # Discord is telling us to back off
                        self._error_count += 1
                        retry_after = float(resp.headers.get("Retry-After", 5))
                        logger.warning(
                            "Discord 429 rate-limited — backing off %.1fs", retry_after
                        )
                        await asyncio.sleep(retry_after)
                        return False
                    else:
                        self._error_count += 1
                        logger.warning(
                            "Discord alert failed status=%d body=%s",
                            resp.status_code, resp.text[:200],
                        )
                        return False

            except Exception as exc:
                self._error_count += 1
                logger.error("Discord alert error: %s", exc)
                return False

    def _should_send(self, alert: Alert) -> bool:
        """Filter by minimum level."""
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL]
        return levels.index(alert.level) >= levels.index(self._min_level)

    def _build_payload(self, alert: Alert) -> dict:
        """Build Discord webhook payload with embed."""
        emoji = LEVEL_EMOJI.get(alert.level, "")
        color = LEVEL_COLORS.get(alert.level, 0x95A5A6)
        ts = (alert.timestamp or datetime.now(UTC)).isoformat()

        embed: dict = {
            "title": f"{emoji} {alert.title}",
            "description": alert.message,
            "color": color,
            "timestamp": ts,
            "footer": {"text": self._bot_name},
        }

        if alert.fields:
            embed["fields"] = [
                {"name": k, "value": v, "inline": True}
                for k, v in alert.fields.items()
            ]

        return {
            "username": self._bot_name,
            "embeds": [embed],
        }

    def status(self) -> dict:
        return {
            "provider": "discord",
            "sent_count": self._sent_count,
            "error_count": self._error_count,
            "dropped_count": self._dropped_count,
            "window_count": self._window_count,
        }


def create_alert_channel(webhook_url: str = "", bot_name: str = "CCTBv5") -> AlertChannel:
    """
    Factory — returns DiscordAlertChannel if URL is set,
    otherwise NullAlertChannel.
    """
    if webhook_url:
        logger.info("AlertChannel: Discord webhook configured — bot_name=%s", bot_name)
        return DiscordAlertChannel(webhook_url=webhook_url, bot_name=bot_name)
    else:
        logger.warning(
            "AlertChannel: DISCORD_WEBHOOK_URL not set — alerts silently discarded"
        )
        return NullAlertChannel()
