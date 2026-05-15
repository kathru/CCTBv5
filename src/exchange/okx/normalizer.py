"""
OKX Normalizer — converts OKX raw API responses into core models.

This is the ONLY place that knows OKX field names.
Everything above this layer uses core models (Candle, Ticker, etc.)

OKX candle format:
  [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
  confirm: "0" = forming, "1" = confirmed

OKX ticker format: standard REST/WS response dict
"""

from datetime import datetime, timezone
from ...core.models import Candle, Ticker

# OKX granularity string → our internal label
GRANULARITY_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1H":  "1H",
    "4H":  "4H",
    "6H":  "6H",
    "1D":  "1D",
}


def _ts(ms: str | int) -> datetime:
    """Convert OKX millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def candle_from_okx(
    symbol: str,
    granularity: str,
    row: list,
) -> Candle:
    """
    Parse one OKX candle row into a Candle model.
    row = [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    """
    confirmed = len(row) < 9 or row[8] == "1"
    return Candle(
        symbol=symbol,
        granularity=GRANULARITY_MAP.get(granularity, granularity),
        timestamp=_ts(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        confirmed=confirmed,
    )


def ticker_from_okx(data: dict) -> Ticker:
    """
    Parse OKX ticker response into a Ticker model.
    Works for both REST (/api/v5/market/ticker) and WebSocket push.
    """
    return Ticker(
        symbol=data["instId"],
        timestamp=_ts(data.get("ts", 0)),
        bid=float(data.get("bidPx", data.get("bid", 0))),
        ask=float(data.get("askPx", data.get("ask", 0))),
        last=float(data.get("last", 0)),
        volume_24h=float(data.get("vol24h", 0)),
        open_24h=float(data.get("open24h", 0)),
    )
