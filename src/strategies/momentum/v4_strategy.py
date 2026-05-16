"""
V4 Momentum Strategy — BULL / CHOP / BEAR aware.

Camadas de proteção:
  1. Detecção explícita de BEAR (downtrend gradual)
  2. Sizing adaptativo por regime (Kelly multiplier)
  3. Confirmação multi-timeframe 1H + 6H

Regimes e comportamento:
  TREND_EXPANSION      → BULL  : threshold 0.50, Kelly 100%, timeout 48h
  VOLATILITY_COMPRESSION       : threshold 0.52, Kelly 80%,  timeout 24h
  TREND_EXHAUSTION             : threshold 0.54, Kelly 60%,  timeout 12h
  MEAN_REVERTING_CHOP  → CHOP  : threshold 0.56, Kelly 50%,  timeout 8h
  HIGH_CORRELATION_RISK        : threshold 0.60, Kelly 30%,  timeout 6h
  BEAR_TREND           → BEAR  : BLOQUEADO para novas entradas
  PANIC_LIQUIDATION            : BLOQUEADO + saída imediata
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from ...core.models import Signal, SignalDirection
from ...monitoring.signal_log import SignalAuditEntry, signal_audit_log
from ..base import BaseStrategy, StrategyContext
from ..ml.inference import PlattCalibrator

MODELS_DIR = Path("data") / "models"
logger     = logging.getLogger(__name__)


# ── Configuração por regime ───────────────────────────────────────────────────

# Thresholds em RAW score space (0-1)
REGIME_THRESHOLDS: dict[str, float] = {
    "TREND_EXPANSION":        0.50,
    "VOLATILITY_COMPRESSION": 0.52,
    "TREND_EXHAUSTION":       0.54,
    "MEAN_REVERTING_CHOP":    0.56,
    "HIGH_CORRELATION_RISK":  0.60,
    "BEAR_TREND":             0.99,   # bloqueado
    "PANIC_LIQUIDATION":      0.99,   # bloqueado
}

# Multiplicador Kelly por regime (aplicado sobre o kelly base)
REGIME_KELLY_MULT: dict[str, float] = {
    "TREND_EXPANSION":        1.00,   # 100% — mercado favorável
    "VOLATILITY_COMPRESSION": 0.80,   # 80%
    "TREND_EXHAUSTION":       0.60,   # 60%
    "MEAN_REVERTING_CHOP":    0.50,   # 50% — mercado lateral
    "HIGH_CORRELATION_RISK":  0.30,   # 30% — alto risco
    "BEAR_TREND":             0.00,   # bloqueado
    "PANIC_LIQUIDATION":      0.00,   # bloqueado
}

# M4 regime alignment score
REGIME_M4: dict[str, float] = {
    "TREND_EXPANSION":        0.85,
    "VOLATILITY_COMPRESSION": 0.65,
    "TREND_EXHAUSTION":       0.50,
    "MEAN_REVERTING_CHOP":    0.45,
    "HIGH_CORRELATION_RISK":  0.30,
    "BEAR_TREND":             0.10,
    "PANIC_LIQUIDATION":      0.00,
}

# Regimes que bloqueiam novas entradas
BLOCKED_REGIMES = {"BEAR_TREND", "PANIC_LIQUIDATION"}


class V4MomentumStrategy(BaseStrategy):

    REGIME_THRESHOLDS = REGIME_THRESHOLDS
    MIN_EV_MULTIPLIER = 3.0
    ROUND_TRIP_FEE    = 0.005

    def __init__(self, symbols: list[str], strategy_id: str = "v4_momentum") -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._platt = PlattCalibrator(coef_path=MODELS_DIR / "calibration_coef.json")

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        ts     = datetime.now(UTC)
        symbol = ctx.symbol

        def _log(result, detail, regime="–", score=0.0, calibrated=0.0,
                 threshold=0.0, ev=0.0, direction="N/A", factors=None):
            signal_audit_log.record(SignalAuditEntry(
                timestamp=ts, symbol=symbol, regime=regime,
                score=score, calibrated=calibrated,
                threshold=threshold, ev=ev,
                direction=direction, result=result, detail=detail,
                factors=factors or {},
            ))

        # ── Filtro 0: candles ────────────────────────────────
        if len(ctx.candles_1h) < 20:
            _log("NO_CANDLES", f"Candles 1H insuficientes: {len(ctx.candles_1h)}/20")
            return None

        # ── Camada 1: Detecção de regime 1H ─────────────────
        regime_1h = self._detect_regime_1h(ctx)

        # ── Camada 3: Confirmação multi-timeframe ────────────
        regime = self._confirm_regime_mtf(ctx, regime_1h)

        threshold  = REGIME_THRESHOLDS.get(regime, 0.56)
        kelly_mult = REGIME_KELLY_MULT.get(regime, 0.5)

        if regime in BLOCKED_REGIMES:
            _log("REGIME_BLOCKED",
                 f"Regime bloqueado: {regime} (sem entradas LONG em bear/panic)",
                 regime=regime, threshold=threshold)
            return None

        # ── Filtro 2: score bruto ────────────────────────────
        score, factors = self._score_signal(ctx, regime)
        calibrated     = self._calibrate(score)

        if score < threshold:
            _log("SCORE_LOW",
                 f"Score {score:.3f} < thr {threshold:.3f} [{regime}]",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, factors=factors)
            return None

        # ── Filtro 3: EV ─────────────────────────────────────
        ev     = self._expected_value(calibrated)
        min_ev = self.MIN_EV_MULTIPLIER * self.ROUND_TRIP_FEE
        if ev < min_ev:
            _log("EV_LOW",
                 f"EV {ev:.3f} < min {min_ev:.3f}",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, ev=ev, factors=factors)
            return None

        # ── Filtro 4: Direção ────────────────────────────────
        direction = self._direction(ctx)
        if direction == SignalDirection.FLAT:
            _log("DIRECTION_FLAT", "Preço lateralizado",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, ev=ev, direction="FLAT", factors=factors)
            return None

        # ── Camada 2: Kelly adaptativo por regime ────────────
        base_kelly = min(calibrated * 0.25, 0.15)
        kelly      = round(base_kelly * kelly_mult, 4)
        dir_str    = "LONG"

        _log("SIGNAL",
             f"BUY {regime} score={score:.3f} prob={calibrated:.3f} "
             f"kelly={kelly:.1%} (mult={kelly_mult:.0%})",
             regime=regime, score=score, calibrated=calibrated,
             threshold=threshold, ev=ev, direction=dir_str, factors=factors)

        logger.info(
            "SIGNAL %s %s regime=%s score=%.3f prob=%.3f "
            "EV=%.3f kelly=%.1f%% (regime_mult=%.0f%%)",
            dir_str, symbol, regime, score, calibrated, ev,
            kelly * 100, kelly_mult * 100,
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

    # ── Camada 1: Detecção de regime 1H ──────────────────────────────────────

    def _detect_regime_1h(self, ctx: StrategyContext) -> str:
        closes  = [c.close  for c in ctx.candles_1h[:20]]
        volumes = [c.volume for c in ctx.candles_1h[:20]]

        sma_fast = sum(closes[:5])  / 5
        sma_slow = sum(closes[:20]) / 20
        avg_vol  = sum(volumes)     / len(volumes)
        last_vol = volumes[0]

        # Panic: queda brusca > 5%
        if len(closes) >= 2 and closes[1] > 0:
            if (closes[0] - closes[1]) / closes[1] < -0.05:
                return "PANIC_LIQUIDATION"

        # BULL: preço acima da SMA lenta
        if sma_fast > sma_slow:
            if last_vol > avg_vol * 1.2:
                return "TREND_EXPANSION"
            if last_vol < avg_vol * 0.8:
                return "TREND_EXHAUSTION"
            return "VOLATILITY_COMPRESSION"

        # Abaixo da SMA lenta — diferencia BEAR de CHOP
        # BEAR: preço atual abaixo de 10 candles atrás em > 2%
        if len(closes) >= 11 and closes[10] > 0:
            decline = (closes[0] - closes[10]) / closes[10]
            if decline < -0.02:
                return "BEAR_TREND"

        # ATR alto → correlação / volatilidade extrema
        highs  = [c.high for c in ctx.candles_1h[:5]]
        lows   = [c.low  for c in ctx.candles_1h[:5]]
        atr_5  = sum(hi - lo for hi, lo in zip(highs, lows, strict=True)) / 5
        rel_atr = atr_5 / closes[0] if closes[0] > 0 else 0
        if rel_atr > 0.030:
            return "HIGH_CORRELATION_RISK"

        return "MEAN_REVERTING_CHOP"

    # ── Camada 3: Confirmação multi-timeframe ─────────────────────────────────

    def _confirm_regime_mtf(self, ctx: StrategyContext, regime_1h: str) -> str:
        """
        Confirma regime 1H com o contexto de 6H.
        Regras:
          - Se 6H concorda (mesma família) → regime_1h confirmado
          - Se 6H diz BEAR mas 1H diz BULL → downgrade para CHOP
          - Se 6H diz BULL mas 1H diz CHOP → pequeno upgrade (VOLATILITY_COMPRESSION)
          - Se 6H diz BEAR e 1H diz CHOP   → upgrade para BEAR_TREND
          - Sem candles 6H → usa só 1H (sem penalidade)
        """
        if not ctx.candles_6h or len(ctx.candles_6h) < 5:
            return regime_1h   # sem 6H, confia no 1H

        closes_6h = [c.close for c in ctx.candles_6h[:10]]
        sma_fast_6h = sum(closes_6h[:3]) / 3
        sma_slow_6h = sum(closes_6h[:10]) / 10

        # Tendência de 6H
        if sma_fast_6h > sma_slow_6h:
            trend_6h = "BULL"
        elif len(closes_6h) >= 5 and closes_6h[4] > 0:
            decline_6h = (closes_6h[0] - closes_6h[4]) / closes_6h[4]
            trend_6h = "BEAR" if decline_6h < -0.03 else "CHOP"
        else:
            trend_6h = "CHOP"

        # Família do regime 1H
        family_1h = ("BULL" if regime_1h in {"TREND_EXPANSION", "VOLATILITY_COMPRESSION", "TREND_EXHAUSTION"}
                     else "BEAR" if regime_1h in {"BEAR_TREND", "PANIC_LIQUIDATION"}
                     else "CHOP")

        # Regras de confirmação
        if family_1h == "BULL" and trend_6h == "BEAR":
            # 1H acha BULL mas 6H está em BEAR → sinal fraco, downgrade
            logger.debug("MTF: 1H=%s (BULL) conflita com 6H BEAR → CHOP", regime_1h)
            return "MEAN_REVERTING_CHOP"

        if family_1h == "CHOP" and trend_6h == "BEAR":
            # 1H lateral mas 6H em queda → confirma downtrend
            logger.debug("MTF: 1H=CHOP + 6H BEAR → BEAR_TREND")
            return "BEAR_TREND"

        if family_1h == "CHOP" and trend_6h == "BULL":
            # 1H lateral mas 6H em alta → pode ser consolidação antes de subida
            logger.debug("MTF: 1H=CHOP + 6H BULL → VOLATILITY_COMPRESSION")
            return "VOLATILITY_COMPRESSION"

        # Regimes concordantes ou PANIC/BEAR confirmado
        return regime_1h

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_signal(self, ctx: StrategyContext, regime: str) -> tuple[float, dict]:
        closes = [c.close  for c in ctx.candles_1h[:20]]
        highs  = [c.high   for c in ctx.candles_1h[:5]]
        vols   = [c.volume for c in ctx.candles_1h[:5]]

        # M1 — Momentum 20 candles (30%)
        momentum = (closes[0] - closes[-1]) / closes[-1] if closes[-1] > 0 else 0
        m1 = min(max((momentum + 0.05) / 0.10, 0.0), 1.0)

        # M2 — Estrutura: higher highs (30%)
        m2 = 1.0 if (len(highs) >= 3 and highs[0] > highs[1] > highs[2]) else 0.4

        # M3 — Volume relativo (20%)
        avg_v = sum(vols) / len(vols) if vols else 1.0
        m3 = min(vols[0] / avg_v, 2.0) / 2.0 if avg_v > 0 else 0.5

        # M4 — Alinhamento de regime (20%) — penaliza BEAR/CHOP
        m4 = REGIME_M4.get(regime, 0.45)

        score   = m1 * 0.3 + m2 * 0.3 + m3 * 0.2 + m4 * 0.2
        factors = {"m1_momentum": round(m1,3), "m2_structure": round(m2,3),
                   "m3_volume":   round(m3,3), "m4_regime":    round(m4,3)}
        return score, factors

    def _calibrate(self, score: float) -> float:
        return self._platt.calibrate(score)

    def _expected_value(self, calibrated: float) -> float:
        return calibrated * 2.5 - (1 - calibrated) * 1.0

    def _direction(self, ctx: StrategyContext) -> SignalDirection:
        if len(ctx.candles_1h) < 2:
            return SignalDirection.FLAT
        closes = [c.close for c in ctx.candles_1h[:5]]
        return SignalDirection.LONG if closes[0] > closes[1] else SignalDirection.FLAT
