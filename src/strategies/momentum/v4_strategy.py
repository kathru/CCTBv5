"""
V4 Momentum Strategy — plugin wrapper around the V4 signal logic.

This is a THIN wrapper. The actual signal logic lives in the
V4 engines (regime, signal, sizing) from the original codebase.
This plugin:
  1. Receives StrategyContext by injection
  2. Delegates to V4 engines
  3. Returns Signal | None

Migration note:
  The V4 engines will be migrated module by module in subsequent steps.
  For now this stub demonstrates the plugin contract is correct.
"""

import logging
from datetime import datetime, timezone

from ..base import BaseStrategy, StrategyContext
from ...core.models import Signal, SignalDirection

logger = logging.getLogger(__name__)


class V4MomentumStrategy(BaseStrategy):
    """
    Probabilistic momentum strategy based on the V4 architecture:
    - 7-regime detection
    - 4 probabilistic sub-models
    - Platt-scaled calibration
    - Kelly-based sizing hint

    Thresholds by regime (from V4):
      TREND_EXPANSION     : 0.56
      VOLATILITY_COMPRESS : 0.60
      TREND_EXHAUSTION    : 0.68
      MEAN_REVERTING_CHOP : 0.72
      HIGH_CORRELATION    : 0.75
      PANIC_LIQUIDATION   : 0.99
      LIQUIDITY_VACUUM    : 0.99
    """

    REGIME_THRESHOLDS: dict[str, float] = {
        "TREND_EXPANSION":        0.56,
        "VOLATILITY_COMPRESSION": 0.60,
        "TREND_EXHAUSTION":       0.68,
        "MEAN_REVERTING_CHOP":    0.72,
        "HIGH_CORRELATION_RISK":  0.75,
        "PANIC_LIQUIDATION":      0.99,
        "LIQUIDITY_VACUUM":       0.99,
    }

    MIN_EV_MULTIPLIER = 3.0       # signal EV must be > 3x round-trip fee
    ROUND_TRIP_FEE = 0.005        # 0.5% (maker + taker + slippage)

    def __init__(
        self,
        symbols: list[str],
        strategy_id: str = "v4_momentum",
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        """
        Evaluate market conditions and return a signal if conditions are met.
        Returns None if no trade opportunity exists.
        """
        if len(ctx.candles_1h) < 20:
            logger.debug(
                "Not enough candles symbol=%s have=%d need=20",
                ctx.symbol, len(ctx.candles_1h),
            )
            return None

        # ── Step 1: Detect regime ─────────────────────────────
        regime = self._detect_regime(ctx)
        threshold = self.REGIME_THRESHOLDS.get(regime, 0.65)

        # Hard stop regimes
        if regime in {"PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"}:
            return None

        # ── Step 2: Score signal ──────────────────────────────
        score, factors = self._score_signal(ctx, regime)
        calibrated = self._calibrate(score)

        if calibrated < threshold:
            return None

        # ── Step 3: Check EV ─────────────────────────────────
        ev = self._expected_value(calibrated)
        if ev < self.MIN_EV_MULTIPLIER * self.ROUND_TRIP_FEE:
            return None

        # ── Step 4: Direction ────────────────────────────────
        direction = self._direction(ctx)
        if direction == SignalDirection.FLAT:
            return None

        # ── Step 5: Kelly fraction ───────────────────────────
        kelly = min(calibrated * 0.25, 0.15)   # cap at 15%

        return Signal(
            strategy_id=self._strategy_id,
            symbol=ctx.symbol,
            direction=direction,
            timestamp=datetime.now(timezone.utc),
            score=score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ev,
            kelly_fraction=kelly,
            regime=regime,
            timeframe="1H",
            factors=factors,
        )

    # ── Internal helpers ──────────────────────────────────────
    # These will be replaced by full V4 engine calls in migration

    def _detect_regime(self, ctx: StrategyContext) -> str:
        """Simplified regime detection — full V4 migration in next step."""
        if not ctx.candles_1h:
            return "MEAN_REVERTING_CHOP"

        closes = [c.close for c in ctx.candles_1h[:20]]
        volumes = [c.volume for c in ctx.candles_1h[:20]]

        sma_fast = sum(closes[:5]) / 5
        sma_slow = sum(closes[:20]) / 20
        avg_vol = sum(volumes) / len(volumes)
        last_vol = volumes[0]

        # Panic: large sudden drop
        if len(closes) >= 2:
            drop = (closes[1] - closes[0]) / closes[1]
            if drop < -0.05:
                return "PANIC_LIQUIDATION"

        # Trend expansion: price above both MAs + high volume
        if sma_fast > sma_slow and last_vol > avg_vol * 1.2:
            return "TREND_EXPANSION"

        # Trend exhaustion: fast MA converging with slow
        if sma_fast > sma_slow and last_vol < avg_vol * 0.8:
            return "TREND_EXHAUSTION"

        return "MEAN_REVERTING_CHOP"

    def _score_signal(
        self, ctx: StrategyContext, regime: str
    ) -> tuple[float, dict[str, float]]:
        """Simplified 4-model scoring — full V4 migration in next step."""
        closes = [c.close for c in ctx.candles_1h[:20]]

        # Sub-model 1: momentum
        momentum = (closes[0] - closes[-1]) / closes[-1] if closes[-1] > 0 else 0
        m1 = min(max((momentum + 0.05) / 0.10, 0), 1)

        # Sub-model 2: structure (higher highs)
        highs = [c.high for c in ctx.candles_1h[:5]]
        m2 = 1.0 if highs[0] > highs[1] > highs[2] else 0.4

        # Sub-model 3: volume
        vols = [c.volume for c in ctx.candles_1h[:5]]
        avg_v = sum(vols) / len(vols)
        m3 = min(vols[0] / avg_v, 2.0) / 2.0

        # Sub-model 4: regime alignment
        m4 = 0.8 if regime == "TREND_EXPANSION" else 0.5

        score = (m1 * 0.3 + m2 * 0.3 + m3 * 0.2 + m4 * 0.2)
        factors = {"momentum": m1, "structure": m2, "volume": m3, "regime": m4}
        return score, factors

    def _calibrate(self, score: float) -> float:
        """
        Platt scaling placeholder.
        Full calibration uses coefficients from calibration_coef.json.
        Migration: load actual A, B coefficients from V4.
        """
        import math
        A, B = -2.5, 1.2   # placeholder — will be replaced with real coefs
        return 1 / (1 + math.exp(-(A * score + B)))

    def _expected_value(self, calibrated: float) -> float:
        """EV in R-multiples: win_prob * reward - loss_prob * risk."""
        win_prob = calibrated
        loss_prob = 1 - calibrated
        reward_r = 2.5    # average R target
        risk_r = 1.0
        return win_prob * reward_r - loss_prob * risk_r

    def _direction(self, ctx: StrategyContext) -> SignalDirection:
        """Determine long/short based on price structure."""
        if len(ctx.candles_1h) < 2:
            return SignalDirection.FLAT
        closes = [c.close for c in ctx.candles_1h[:5]]
        if closes[0] > closes[1]:
            return SignalDirection.LONG
        return SignalDirection.FLAT   # spot only — no shorts for now
