"""
Market Engine — the single source of market data for all strategies.

Responsibilities:
  - Poll OKX for candles and tickers on a schedule
  - Publish CandleEvent and TickerEvent to the bus
  - Maintain a local cache of recent candles per symbol/granularity
  - Strategies NEVER access OKX directly — they consume events from the bus

Design:
  - Runs as a background asyncio task
  - Polls REST API (WebSocket upgrade in Passo 9 extension)
  - Normalizes all data via OKX normalizer before publishing
"""

import asyncio
import logging
from datetime import UTC, datetime

from ..core.bus import EventBus
from ..core.events import CandleEvent, TickerEvent, Topic
from ..core.models import Candle, Ticker
from ..exchange.okx.client import OKXClient
from ..persistence.cache import Cache

logger = logging.getLogger(__name__)


class MarketEngine:
    """
    Polls market data and publishes to Event Bus.
    One instance per application.
    """

    def __init__(
        self,
        bus: EventBus,
        okx: OKXClient,
        cache: Cache,
        symbols: list[str],
        granularities: list[str] | None = None,
        poll_interval: int = 15,       # seconds between polls
    ) -> None:
        self._bus = bus
        self._okx = okx
        self._cache = cache
        self._symbols = symbols
        self._granularities = granularities or ["1H", "6H"]
        self._poll_interval = poll_interval
        self._on_poll_callback = None   # called after each successful poll
        self._running = False
        self._task: asyncio.Task | None = None

        # Local candle cache: (symbol, granularity) → list[Candle]
        self._candles: dict[tuple[str, str], list[Candle]] = {}
        # Último timestamp publicado por (symbol, granularity) — evita republicar histórico
        self._last_published: dict[tuple[str, str], datetime] = {}
        self._last_poll: datetime | None = None
        self._poll_count = 0
        self._error_count = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="market_engine"
        )
        logger.info(
            "MarketEngine started symbols=%s granularities=%s interval=%ds",
            self._symbols, self._granularities, self._poll_interval,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MarketEngine stopped")

    def set_on_poll_callback(self, callback) -> None:
        """Register callback called after each successful poll (e.g. ws_watchdog.record_message)."""
        self._on_poll_callback = callback

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(self._poll_cycle(), timeout=30.0)
                if self._on_poll_callback:
                    self._on_poll_callback()
            except asyncio.CancelledError:
                break
            except TimeoutError:
                self._error_count += 1
                logger.warning("MarketEngine poll timeout (>30s) — skipping cycle")
            except Exception as exc:
                self._error_count += 1
                logger.error("MarketEngine poll error: %s", exc, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    async def _poll_cycle(self) -> None:
        self._poll_count += 1
        self._last_poll = datetime.now(UTC)

        client = self._okx
        for symbol in self._symbols:
            # Fetch and publish ticker
            try:
                ticker = await client.get_ticker(symbol)
                await self._publish_ticker(ticker)
            except Exception as exc:
                logger.warning(
                    "Ticker fetch failed symbol=%s error=%s", symbol, exc
                )

            # Fetch and publish candles per granularity
            for gran in self._granularities:
                try:
                    candles = await client.get_candles(
                        symbol, granularity=gran, limit=100
                    )
                    await self._publish_candles(symbol, gran, candles)
                except Exception as exc:
                    logger.warning(
                        "Candles fetch failed symbol=%s gran=%s error=%s",
                        symbol, gran, exc,
                    )

    async def _publish_ticker(self, ticker: Ticker) -> None:
        # Update Redis price cache
        await self._cache.set_price(ticker.symbol, ticker.last)
        # Publish to bus
        await self._bus.publish(Topic.MARKET, TickerEvent(ticker=ticker))

    async def _publish_candles(
        self,
        symbol: str,
        granularity: str,
        candles: list[Candle],
    ) -> None:
        key = (symbol, granularity)
        self._candles[key] = candles

        # Publica apenas candles confirmados MAIS NOVOS que o último publicado.
        # Sem isso, todos os 100 candles históricos são republicados a cada poll.
        last_ts = self._last_published.get(key)
        new_candles = [
            c for c in candles
            if c.confirmed and (last_ts is None or c.timestamp > last_ts)
        ]

        for candle in new_candles:
            await self._bus.publish(Topic.MARKET, CandleEvent(candle=candle))

        if new_candles:
            self._last_published[key] = max(c.timestamp for c in new_candles)
            logger.debug(
                "Published %d new candles symbol=%s gran=%s",
                len(new_candles), symbol, granularity,
            )

    # ── Query interface for strategies ────────────────────────

    def get_candles(
        self,
        symbol: str,
        granularity: str,
        limit: int = 100,
    ) -> list[Candle]:
        """Return cached candles. Strategies call this, not OKX directly."""
        key = (symbol, granularity)
        candles = self._candles.get(key, [])
        return candles[:limit]

    def get_latest_candle(
        self, symbol: str, granularity: str
    ) -> Candle | None:
        candles = self.get_candles(symbol, granularity, limit=1)
        return candles[0] if candles else None

    def status(self) -> dict:
        return {
            "running": self._running,
            "symbols": self._symbols,
            "granularities": self._granularities,
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "last_poll": (
                self._last_poll.isoformat() if self._last_poll else None
            ),
            "cached_feeds": len(self._candles),
        }
