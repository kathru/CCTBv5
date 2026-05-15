from datetime import datetime
from dataclasses import dataclass, field
from enum import StrEnum


class SignalDirection(StrEnum):
    LONG  = "long"
    SHORT = "short"
    FLAT  = "flat"      # exit / no position


class SignalStrength(StrEnum):
    WEAK   = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


@dataclass(frozen=True)
class Signal:
    """
    Trading signal produced by a strategy.
    Strategies return Signal | None — nothing else.
    """

    strategy_id: str
    symbol: str
    direction: SignalDirection
    timestamp: datetime

    # Probabilistic scoring (from signal engine)
    score: float                        # raw 0.0–1.0
    calibrated_score: float             # Platt-scaled probability
    confidence: float                   # 0.0–1.0
    expected_value: float               # EV in R-multiples

    # Sizing hint (Kelly fraction — sizing engine decides final size)
    kelly_fraction: float

    # Context that generated the signal
    regime: str                         # e.g. "TREND_EXPANSION"
    timeframe: str                      # e.g. "1H"

    # Sub-model breakdown (for audit/replay)
    factors: dict[str, float] = field(default_factory=dict)

    @property
    def strength(self) -> SignalStrength:
        if self.calibrated_score >= 0.72:
            return SignalStrength.STRONG
        if self.calibrated_score >= 0.60:
            return SignalStrength.MEDIUM
        return SignalStrength.WEAK
