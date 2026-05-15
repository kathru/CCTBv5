from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Candle:
    """OHLCV candle — immutable, exchange-agnostic."""

    symbol: str
    granularity: str          # "1H", "6H", "1D", etc.
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool = True    # False = candle still forming
