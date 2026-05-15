from datetime import UTC, datetime

import pytest

from src.core.models import Candle, SignalDirection
from src.strategies.base import BaseStrategy, StrategyContext
from src.strategies.momentum.v4_strategy import V4MomentumStrategy


def make_candles(n: int = 30, trend: str = "up") -> list[Candle]:
    """Generate synthetic candles for testing."""
    candles = []
    base = 42000.0
    for i in range(n):
        if trend == "up":
            close = base + (n - i) * 100
        else:
            close = base - (n - i) * 100
        candles.append(Candle(
            symbol="BTC-USDT",
            granularity="1H",
            timestamp=datetime.now(UTC),
            open=close - 50,
            high=close + 100,
            low=close - 100,
            close=close,
            volume=500 + i * 10,
            confirmed=True,
        ))
    return candles


def make_context(symbol: str = "BTC-USDT", trend: str = "up") -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        candles_1h=make_candles(30, trend),
        candles_6h=make_candles(10, trend),
        ticker=None,
        portfolio_value=10000.0,
    )


def test_strategy_cannot_be_instantiated_directly():
    """BaseStrategy is abstract — cannot instantiate directly."""
    with pytest.raises(TypeError):
        BaseStrategy("test", ["BTC-USDT"])


def test_strategy_has_correct_id():
    s = V4MomentumStrategy(symbols=["BTC-USDT"], strategy_id="test_v4")
    assert s.strategy_id == "test_v4"


def test_strategy_starts_enabled():
    s = V4MomentumStrategy(symbols=["BTC-USDT"])
    assert s.is_enabled is True


def test_strategy_can_be_disabled():
    s = V4MomentumStrategy(symbols=["BTC-USDT"])
    s.disable()
    assert s.is_enabled is False
    s.enable()
    assert s.is_enabled is True


@pytest.mark.asyncio
async def test_strategy_returns_none_with_insufficient_candles():
    s = V4MomentumStrategy(symbols=["BTC-USDT"])
    ctx = StrategyContext(
        symbol="BTC-USDT",
        candles_1h=[],    # empty — should return None
        candles_6h=[],
        ticker=None,
        portfolio_value=10000.0,
    )
    result = await s.evaluate(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_strategy_returns_signal_or_none_on_uptrend():
    """Strategy must return Signal or None — never raises."""
    s = V4MomentumStrategy(symbols=["BTC-USDT"])
    ctx = make_context("BTC-USDT", trend="up")
    result = await s.evaluate(ctx)
    # Must be Signal or None — never an exception
    assert result is None or hasattr(result, "calibrated_score")


@pytest.mark.asyncio
async def test_signal_direction_is_valid():
    """Any signal returned must have a valid direction."""
    s = V4MomentumStrategy(symbols=["BTC-USDT"])
    ctx = make_context("BTC-USDT", trend="up")
    result = await s.evaluate(ctx)
    if result is not None:
        assert result.direction in {
            SignalDirection.LONG,
            SignalDirection.SHORT,
            SignalDirection.FLAT,
        }
        assert result.strategy_id == "v4_momentum"
        assert result.symbol == "BTC-USDT"
        assert 0.0 <= result.calibrated_score <= 1.0
        assert result.kelly_fraction <= 0.15   # cap enforced
