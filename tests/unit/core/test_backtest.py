from datetime import UTC, datetime

import pytest

from src.core.models import Candle
from src.replay.backtest_engine import BacktestEngine, BacktestResult, SimulatedFillEngine
from src.strategies.momentum.v4_strategy import V4MomentumStrategy


def make_candles(n: int = 50, trend: str = "up") -> list[Candle]:
    """Generate synthetic candles oldest first."""
    candles = []
    base = 42000.0
    for i in range(n):
        price = base + (i * 200 if trend == "up" else -i * 200)
        candles.append(Candle(
            symbol="BTC-USDT",
            granularity="1H",
            timestamp=datetime(2026, 1, 1, i % 24, tzinfo=UTC),
            open=price - 100,
            high=price + 200,
            low=price - 200,
            close=price,
            volume=500 + i * 10,
            confirmed=True,
        ))
    return candles


@pytest.mark.asyncio
async def test_backtest_runs_without_error():
    strategy = V4MomentumStrategy(symbols=["BTC-USDT"], strategy_id="test")
    engine = BacktestEngine(
        strategy=strategy,
        symbol="BTC-USDT",
        initial_capital=10000.0,
        seed=42,
    )
    candles = make_candles(50, "up")
    result = await engine.run(candles, warmup=20)
    assert isinstance(result, BacktestResult)
    assert result.symbol == "BTC-USDT"
    assert result.initial_capital == 10000.0


@pytest.mark.asyncio
async def test_backtest_result_has_valid_metrics():
    strategy = V4MomentumStrategy(symbols=["BTC-USDT"])
    engine = BacktestEngine(strategy=strategy, symbol="BTC-USDT", seed=42)
    candles = make_candles(60, "up")
    result = await engine.run(candles, warmup=20)

    assert 0.0 <= result.win_rate <= 1.0
    assert result.total_trades >= 0
    assert isinstance(result.final_capital, float)
    assert isinstance(result.summary(), dict)


@pytest.mark.asyncio
async def test_backtest_requires_minimum_candles():
    strategy = V4MomentumStrategy(symbols=["BTC-USDT"])
    engine = BacktestEngine(strategy=strategy, symbol="BTC-USDT")
    with pytest.raises(ValueError):
        await engine.run(make_candles(5), warmup=20)


def test_simulated_fill_engine_partial_fills():
    engine = SimulatedFillEngine(passive_fill_prob=1.0, seed=42)
    candle = make_candles(1)[0]

    from src.core.models import Signal, SignalDirection
    signal = Signal(
        strategy_id="test",
        symbol="BTC-USDT",
        direction=SignalDirection.LONG,
        timestamp=datetime.now(UTC),
        score=0.75,
        calibrated_score=0.72,
        confidence=0.8,
        expected_value=3.5,
        kelly_fraction=0.1,
        regime="TREND_EXPANSION",
        timeframe="1H",
    )

    trade = engine.simulate_entry(signal, candle, capital=10000.0, position_size_pct=0.10)
    assert trade is not None
    assert 0.0 < trade.fill_ratio <= 1.0
    assert trade.entry_price > 0
    assert trade.entry_fee > 0


def test_backtest_result_summary_keys():
    from src.replay.backtest_engine import BacktestResult
    result = BacktestResult(
        symbol="BTC-USDT",
        strategy_id="test",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
        initial_capital=10000.0,
    )
    summary = result.summary()
    required_keys = {"symbol", "strategy_id", "total_trades", "win_rate",
                     "profit_factor", "total_pnl", "total_return", "expectancy"}
    assert required_keys.issubset(summary.keys())
