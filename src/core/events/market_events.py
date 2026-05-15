from dataclasses import dataclass
from .base import BaseEvent
from ..models import Candle, Ticker


@dataclass(frozen=True)
class CandleEvent(BaseEvent):
    """New candle received from exchange (confirmed or forming)."""
    candle: Candle | None = None


@dataclass(frozen=True)
class TickerEvent(BaseEvent):
    """Real-time price update."""
    ticker: Ticker | None = None
