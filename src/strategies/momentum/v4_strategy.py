"""
V4 Momentum Strategy — plugin wrapper around the V4 signal logic.

This is a THIN wrapper. The actual signal logic lives in the
V4 engines (regime, signal, sizing) from the original codebase.
This plugin:
  1. Receives StrategyContext by injection
  2. Delegates to V4 engines
  3. Returns Signal | None
  4. Logs every evaluation to SignalAuditLog (visível no dashboard)
"""

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from ...core.models import Signal, SignalDirection
from ...monitoring.signal_log import SignalAuditEntry, signal_audit_log
from ..base import BaseStrategy, StrategyContext
from ..ml.inference import PlattCalibrator

MODELS_DIR = Path("data") / "models"

logger = logging.getLogger(__name__)


class V4MomentumStrategy(BaseStrategy):
    """
    Probabilistic momentum strategy based on the V4 architecture:
    - 7-regime detection
    - 4 probabilistic sub-models
    - Platt-scaled calibration (lê coeficientes reais do JSON)
    - Kelly-based sizing hint

    Thresholds by regime:
      TREND_EXPANSION     : 0.56
      VOLATILITY_COMPRESS : 0.60
      TREND_EXHAUSTION    : 0.68
      MEAN_REVERTING_CHOP : 0.72
      HIGH_CORRELATION    : 0.75
      PANIC_LIQUIDATION   : bloqueado
      LIQUIDITY_VACUUM    : bloqueado
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

    MIN_EV_MULTIPLIER = 3.0
    ROUND_TRIP_FEE    = 0.005   # 0.5% round-trip

    def __init__(
        self,
        symbols: list[str],
        strategy_id: str = "v4_momentum",
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._platt = PlattCalibrator(
            coef_path=MODELS_DIR / "calibration_coef.json"
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        """
        Avalia condições de mercado e retorna sinal se aprovado em todos os filtros.
        Registra o resultado (com motivo detalhado) no SignalAuditLog.
        """
        ts     = datetime.now(UTC)
        symbol = ctx.symbol

        def _log(result: str, detail: str,
                 regime: str = "–", score: float = 0.0,
                 calibrated: float = 0.0, threshold: float = 0.0,
                 ev: float = 0.0, direction: str = "N/A",
                 factors: dict | None = None) -> None:
            signal_audit_log.record(SignalAuditEntry(
                timestamp=ts, symbol=symbol, regime=regime,
                score=score, calibrated=calibrated,
                threshold=threshold, ev=ev,
                direction=direction, result=result, detail=detail,
                factors=factors or {},
            ))

        # ── Filtro 0: candles suficientes ────────────────────
        if len(ctx.candles_1h) < 20:
            _log("NO_CANDLES",
                 f"Candles insuficientes: {len(ctx.candles_1h)}/20")
            return None

        # ── Filtro 1: regime ──────────────────────────────────
        regime    = self._detect_regime(ctx)
        hardcoded = self.REGIME_THRESHOLDS.get(regime, 0.65)
        threshold = self._platt.get_regime_threshold(regime, hardcoded)

        if regime in {"PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"}:
            _log("REGIME_BLOCKED",
                 f"Regime bloqueado: {regime}",
                 regime=regime, threshold=threshold)
            return None

        # ── Filtro 2: score ───────────────────────────────────
        score, factors = self._score_signal(ctx, regime)
        calibrated = self._calibrate(score)

        if calibrated < threshold:
            _log("SCORE_LOW",
                 f"Score {calibrated:.3f} < threshold {threshold:.3f} ({regime})",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, factors=factors)
            return None

        # ── Filtro 3: expected value ──────────────────────────
        ev      = self._expected_value(calibrated)
        min_ev  = self.MIN_EV_MULTIPLIER * self.ROUND_TRIP_FEE

        if ev < min_ev:
            _log("EV_LOW",
                 f"EV {ev:.3f} < mínimo {min_ev:.3f}",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, ev=ev, factors=factors)
            return None

        # ── Filtro 4: direção ────────────────────────────────
        direction = self._direction(ctx)
        if direction == SignalDirection.FLAT:
            _log("DIRECTION_FLAT",
                 "Sem direção clara (preço lateralizado)",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, ev=ev, direction="FLAT",
                 factors=factors)
            return None

        # ── ✅ Sinal aprovado ─────────────────────────────────
        kelly = min(calibrated * 0.25, 0.15)
        dir_str = "LONG" if direction == SignalDirection.LONG else "SHORT"

        _log("SIGNAL",
             f"Sinal LONG gerado — score={score:.3f} "
             f"prob={calibrated:.3f} EV={ev:.3f} kelly={kelly:.1%}",
             regime=regime, score=score, calibrated=calibrated,
             threshold=threshold, ev=ev, direction=dir_str,
             factors=factors)

        logger.info(
            "SIGNAL %s %s regime=%s score=%.3f prob=%.3f EV=%.3f kelly=%.1f%%",
            dir_str, symbol, regime, score, calibrated, ev, kelly * 100,
        )

        return Signal(
            strategy_id=self._strategy_id,
            symbol=symbol,
            direction=direction,
            timestamp=ts,
            score=score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ev,
            kelly_fraction=kelly,
            regime=regime,
            timeframe="1H",
            factors=factors,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_regime(self, ctx: StrategyContext) -> str:
        if not ctx.candles_1h:
            return "MEAN_REVERTING_CHOP"

        closes  = [c.close  for c in ctx.candles_1h[:20]]
        volumes = [c.volume for c in ctx.candles_1h[:20]]

        sma_fast = sum(closes[:5])  / 5
        sma_slow = sum(closes[:20]) / 20
        avg_vol  = sum(volumes) / len(volumes)
        last_vol = volumes[0]

        if len(closes) >= 2:
            drop = (closes[1] - closes[0]) / closes[1]
            if drop < -0.05:
                return "PANIC_LIQUIDATION"

        if sma_fast > sma_slow:
            if last_vol > avg_vol * 1.2:
                return "TREND_EXPANSION"
            if last_vol < avg_vol * 0.8:
                return "TREND_EXHAUSTION"
            return "VOLATILITY_COMPRESSION"

        highs = [c.high for c in ctx.candles_1h[:10]]
        lows  = [c.low  for c in ctx.candles_1h[:10]]
        atr_5 = sum(h - l for h, l in zip(highs[:5], lows[:5])) / 5
        rel_atr = atr_5 / closes[0] if closes[0] > 0 else 0

        if rel_atr < 0.005:
            return "LIQUIDITY_VACUUM"
        if rel_atr > 0.025:
            return "HIGH_CORRELATION_RISK"

        return "MEAN_REVERTING_CHOP"

    def _score_signal(
        self, ctx: StrategyContext, regime: str
    ) -> tuple[float, dict[str, float]]:
        closes = [c.close  for c in ctx.candles_1h[:20]]
        highs  = [c.high   for c in ctx.candles_1h[:5]]
        vols   = [c.volume for c in ctx.candles_1h[:5]]

        # M1 — Momentum (30%)
        momentum = (closes[0] - closes[-1]) / closes[-1] if closes[-1] > 0 else 0
        m1 = min(max((momentum + 0.05) / 0.10, 0.0), 1.0)

        # M2 — Estrutura: higher highs (30%)
        m2 = 1.0 if (len(highs) >= 3 and highs[0] > highs[1] > highs[2]) else 0.4

        # M3 — Volume relativo (20%)
        avg_v = sum(vols) / len(vols) if vols else 1.0
        m3 = min(vols[0] / avg_v, 2.0) / 2.0 if avg_v > 0 else 0.5

        # M4 — Alinhamento de regime (20%)
        m4 = 0.8 if regime == "TREND_EXPANSION" else 0.5

        score   = m1 * 0.3 + m2 * 0.3 + m3 * 0.2 + m4 * 0.2
        factors = {"m1_momentum": m1, "m2_structure": m2,
                   "m3_volume": m3, "m4_regime": m4}
        return score, factors

    def _calibrate(self, score: float) -> float:
        """
        Platt scaling usando coeficientes reais do calibration_coef.json.
        Fallback para defaults (A=2.5, B=-1.2) se arquivo não encontrado.
        """
        return self._platt.calibrate(score)

    def _expected_value(self, calibrated: float) -> float:
        win_prob  = calibrated
        loss_prob = 1 - calibrated
        reward_r  = 2.5
        risk_r    = 1.0
        return win_prob * reward_r - loss_prob * risk_r

    def _direction(self, ctx: StrategyContext) -> SignalDirection:
        if len(ctx.candles_1h) < 2:
            return SignalDirection.FLAT
        closes = [c.close for c in ctx.candles_1h[:5]]
        return SignalDirection.LONG if closes[0] > closes[1] else SignalDirection.FLAT
