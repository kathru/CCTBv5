"""
Strategy Runner — evaluates all registered strategies each cycle.

Responsibilities:
  - Subscribe to MARKET events from the bus
  - Build StrategyContext from market data
  - Call each enabled strategy's evaluate()
  - Publish SignalEvent for every non-None result
  - Never touches the OMS or exchange directly

This is the bridge between Market Engine and OMS.
"""

import asyncio
import logging

from ..core.bus import EventBus
from ..core.events import CandleEvent, SignalEvent, Topic
from ..core.models import Signal
from ..market.engine import MarketEngine
from ..persistence.cache import Cache
from .base import BaseStrategy, StrategyContext

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Evaluates all strategies on each market data update.
    Runs as a background consumer of MARKET events.
    """

    def __init__(
        self,
        bus: EventBus,
        market: MarketEngine,
        cache: Cache,
        portfolio_value: float = 0.0,
    ) -> None:
        self._bus = bus
        self._market = market
        self._cache = cache
        self._portfolio_value = portfolio_value
        self._strategies: dict[str, BaseStrategy] = {}
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._signal_count = 0

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy plugin."""
        self._strategies[strategy.strategy_id] = strategy
        logger.info("Strategy registered: %s symbols=%s",
                    strategy.strategy_id, strategy.symbols)

    def unregister(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Subscribe to MARKET events
        self._queue = self._bus.subscribe(Topic.MARKET)
        self._task = asyncio.create_task(
            self._consume(), name="strategy_runner"
        )
        logger.info(
            "StrategyRunner started with %d strategies",
            len(self._strategies),
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _consume(self) -> None:
        """Consume MARKET events and trigger strategy evaluation."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                if isinstance(event, CandleEvent) and event.candle:
                    candle = event.candle
                    # Only evaluate on confirmed candles
                    if candle.confirmed:
                        await self._evaluate_all(candle.symbol)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("StrategyRunner error: %s", exc, exc_info=True)

    async def _evaluate_all(self, symbol: str) -> None:
        """Evaluate all strategies that trade the given symbol."""
        for strategy in self._strategies.values():
            if not strategy.is_enabled:
                continue
            if symbol not in strategy.symbols:
                continue
            try:
                ctx = await self._build_context(symbol)
                signal = await strategy.evaluate(ctx)
                if signal is not None:
                    await self._publish_signal(signal)
            except Exception as exc:
                logger.error(
                    "Strategy %s raised: %s",
                    strategy.strategy_id, exc, exc_info=True,
                )

    async def _build_context(self, symbol: str) -> StrategyContext:
        """Build the injection context for a strategy."""
        candles_1h = self._market.get_candles(symbol, "1H")
        candles_6h = self._market.get_candles(symbol, "6H")
        ticker = self._market.get_latest_candle(symbol, "1H")

        # Get open positions from Redis cache
        pos_data = await self._cache.get_position(symbol)
        open_positions = [pos_data] if pos_data else []

        return StrategyContext(
            symbol=symbol,
            candles_1h=candles_1h,
            candles_6h=candles_6h,
            ticker=None,          # Ticker injected via TickerEvent separately
            portfolio_value=self._portfolio_value,
            open_positions=open_positions,
        )

    async def _publish_signal(self, signal: Signal) -> None:
        self._signal_count += 1
        await self._bus.publish(Topic.SIGNAL, SignalEvent(signal=signal))
        logger.info(
            "Signal published strategy=%s symbol=%s direction=%s score=%.3f",
            signal.strategy_id, signal.symbol,
            signal.direction, signal.calibrated_score,
        )

    def update_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    def status(self) -> dict:
        return {
            "running": self._running,
            "strategies": list(self._strategies.keys()),
            "signal_count": self._signal_count,
        }
