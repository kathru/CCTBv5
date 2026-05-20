"""
Redis cache — hot state for low-latency reads.

What lives here (NOT the source of truth — PostgreSQL is):
  - Open positions (avoid DB round-trip on every cycle)
  - Last known price per symbol (for unrealized P&L)
  - Last fill per symbol
  - System status (RECONCILING, RUNNING, etc.)

TTLs:
  - Positions: 60s (refreshed by OMS on every state change)
  - Prices:    10s (refreshed by Market Engine)
  - Status:    no TTL (explicit set/delete)
"""

import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Key prefixes
_PFX_POSITION = "pos:"
_PFX_PRICE    = "price:"
_PFX_STATUS   = "system:status"
_PFX_FILL     = "lastfill:"

_TTL_POSITION = 60   # seconds
_TTL_PRICE    = 60   # 60s — cobre gap entre polls de 15s com folga
_TTL_FILL     = 300


class Cache:
    """Redis cache wrapper — async."""

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("Redis connected url=%s", self._url)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Redis disconnected")

    def _r(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Cache not connected — call connect() first")
        return self._client

    # ── Positions ─────────────────────────────────────────────

    async def set_position(self, symbol: str, data: dict) -> None:
        await self._r().setex(
            f"{_PFX_POSITION}{symbol}",
            _TTL_POSITION,
            json.dumps(data),
        )

    async def get_position(self, symbol: str) -> dict | None:
        raw = await self._r().get(f"{_PFX_POSITION}{symbol}")
        return json.loads(raw) if raw else None

    async def delete_position(self, symbol: str) -> None:
        await self._r().delete(f"{_PFX_POSITION}{symbol}")

    # ── Prices ────────────────────────────────────────────────

    async def set_price(self, symbol: str, price: float) -> None:
        await self._r().setex(
            f"{_PFX_PRICE}{symbol}",
            _TTL_PRICE,
            str(price),
        )

    async def get_price(self, symbol: str) -> float | None:
        raw = await self._r().get(f"{_PFX_PRICE}{symbol}")
        return float(raw) if raw else None

    async def set_ticker(self, symbol: str, data: dict) -> None:
        import json
        await self._r().setex(f"ticker:{symbol}", _TTL_PRICE, json.dumps(data))

    async def get_ticker(self, symbol: str) -> dict | None:
        import json
        raw = await self._r().get(f"ticker:{symbol}")
        return json.loads(raw) if raw else None

    # ── Last fill ─────────────────────────────────────────────

    async def set_last_fill(self, symbol: str, data: dict) -> None:
        await self._r().setex(
            f"{_PFX_FILL}{symbol}",
            _TTL_FILL,
            json.dumps(data),
        )

    async def get_last_fill(self, symbol: str) -> dict | None:
        raw = await self._r().get(f"{_PFX_FILL}{symbol}")
        return json.loads(raw) if raw else None

    # ── System status ─────────────────────────────────────────

    async def set_system_status(self, status: str) -> None:
        await self._r().set(_PFX_STATUS, status)

    async def get_system_status(self) -> str | None:
        return await self._r().get(_PFX_STATUS)

    # ── Generic ───────────────────────────────────────────────

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        v = json.dumps(value) if not isinstance(value, str) else value
        if ttl:
            await self._r().setex(key, ttl, v)
        else:
            await self._r().set(key, v)

    async def get(self, key: str) -> Any | None:
        raw = await self._r().get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def delete(self, key: str) -> None:
        await self._r().delete(key)

    async def ping(self) -> bool:
        try:
            await self._r().ping()
            return True
        except Exception:
            return False
