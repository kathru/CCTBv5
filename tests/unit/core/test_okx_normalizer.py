import pytest
from datetime import timezone
from src.exchange.okx.normalizer import candle_from_okx, ticker_from_okx


def test_candle_from_okx_confirmed():
    row = ["1700000000000", "42000", "43000", "41000", "42500", "100.5",
           "0", "0", "1"]
    candle = candle_from_okx("BTC-USDT", "1H", row)
    assert candle.symbol == "BTC-USDT"
    assert candle.granularity == "1H"
    assert candle.open == 42000.0
    assert candle.high == 43000.0
    assert candle.low == 41000.0
    assert candle.close == 42500.0
    assert candle.volume == 100.5
    assert candle.confirmed is True
    assert candle.timestamp.tzinfo == timezone.utc


def test_candle_from_okx_forming():
    row = ["1700000000000", "42000", "43000", "41000", "42500", "100.5",
           "0", "0", "0"]
    candle = candle_from_okx("BTC-USDT", "1H", row)
    assert candle.confirmed is False


def test_ticker_from_okx():
    data = {
        "instId": "BTC-USDT",
        "ts": "1700000000000",
        "bidPx": "41990",
        "askPx": "42010",
        "last": "42000",
        "vol24h": "5000",
        "open24h": "41000",
    }
    ticker = ticker_from_okx(data)
    assert ticker.symbol == "BTC-USDT"
    assert ticker.bid == 41990.0
    assert ticker.ask == 42010.0
    assert ticker.last == 42000.0
    assert abs(ticker.mid - 42000.0) < 1.0
    assert ticker.spread == 20.0


def test_ticker_spread_pct():
    data = {
        "instId": "ETH-USDT",
        "ts": "1700000000000",
        "bidPx": "2000",
        "askPx": "2002",
        "last": "2001",
        "vol24h": "10000",
        "open24h": "1990",
    }
    ticker = ticker_from_okx(data)
    assert ticker.spread == 2.0
    assert abs(ticker.spread_pct - 0.001) < 0.0001
