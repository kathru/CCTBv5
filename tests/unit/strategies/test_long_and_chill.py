"""Tests for LongAndChill — SMA 6H trailing stop module."""

import pytest

from src.strategies.longchill.long_and_chill import (
    ATR_CHILL_MULT,
    CHILL_SCORE_THRESHOLD,
    MIN_CANDLES,
    SMA_PERIOD,
    LongAndChill,
)

# ── _calc_atr ─────────────────────────────────────────────────────────────────

def make_price_series(n=20, base=100.0, step=0.5):
    closes = [base + i * step for i in range(n)]
    highs  = [c + 1.0 for c in closes]
    lows   = [c - 1.0 for c in closes]
    return closes, highs, lows


def test_calc_atr_basic():
    closes, highs, lows = make_price_series(20)
    atr = LongAndChill._calc_atr(closes, highs, lows, period=14)
    # high-low = 2.0 every candle → ATR should be ~2.0
    assert 1.5 < atr < 2.5


def test_calc_atr_returns_zero_for_single_candle():
    atr = LongAndChill._calc_atr([100.0], [101.0], [99.0], period=14)
    assert atr == 0.0


def test_calc_atr_returns_zero_for_empty():
    atr = LongAndChill._calc_atr([], [], [], period=14)
    assert atr == 0.0


def test_calc_atr_uses_last_n_periods():
    """ATR uses only the last `period` true ranges."""
    closes = [100.0] * 30
    highs  = [101.0] * 30
    lows   = [99.0]  * 30
    # Last 5 candles have wider range
    for i in range(25, 30):
        highs[i]  = 106.0
        lows[i]   = 94.0
    atr14 = LongAndChill._calc_atr(closes, highs, lows, period=14)
    atr5  = LongAndChill._calc_atr(closes, highs, lows, period=5)
    # period=5 uses only the wide-range candles
    assert atr5 > atr14


# ── _compute_sma_trail (mocked market) ───────────────────────────────────────

class FakeCandle:
    def __init__(self, close, high=None, low=None):
        self.close = close
        self.high  = high if high is not None else close + 1.0
        self.low   = low  if low  is not None else close - 1.0


class FakeMarket:
    def __init__(self, candles):
        self._candles = candles

    def get_candles(self, symbol, gran, limit):
        return self._candles[-limit:] if limit else self._candles


def make_module_with_candles(n_candles=34, base=50000.0):
    candles = [FakeCandle(base + i * 10) for i in range(n_candles)]
    market  = FakeMarket(candles)
    lac = LongAndChill.__new__(LongAndChill)
    lac._market  = market
    lac._pm      = None
    lac._running = False
    lac._task    = None
    return lac


@pytest.mark.asyncio
async def test_compute_sma_trail_returns_float():
    lac = make_module_with_candles(34)
    trail = await lac._compute_sma_trail("BTC-USDT")
    assert trail is not None
    assert isinstance(trail, float)
    assert trail > 0


@pytest.mark.asyncio
async def test_compute_sma_trail_returns_none_for_insufficient_candles():
    lac = make_module_with_candles(n_candles=5)   # below MIN_CANDLES=10
    trail = await lac._compute_sma_trail("BTC-USDT")
    assert trail is None


@pytest.mark.asyncio
async def test_compute_sma_trail_is_below_sma():
    """Trail = SMA - ATR*mult should be strictly below SMA."""
    base = 50000.0
    lac  = make_module_with_candles(34, base=base)
    trail = await lac._compute_sma_trail("BTC-USDT")
    # SMA of last 20 candles around base+140..base+330
    sma_approx = base + (14 + 33) / 2 * 10  # midpoint ≈ base+235
    assert trail < sma_approx


# ── Constants sanity ──────────────────────────────────────────────────────────

def test_constants():
    assert CHILL_SCORE_THRESHOLD == 0.72
    assert SMA_PERIOD == 20
    assert MIN_CANDLES == 10
    assert ATR_CHILL_MULT == 1.0
