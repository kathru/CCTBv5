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
from ...monitoring.feature_governance import governance
from ...monitoring.signal_log import SignalAuditEntry, signal_audit_log
from ..base import BaseStrategy, StrategyContext
from ..ml.inference import PlattCalibrator

MODELS_DIR = Path("data") / "models"
logger     = logging.getLogger(__name__)


# ── Configuração por regime ───────────────────────────────────────────────────

# Thresholds em RAW score space — ajustados para 30min
# 30min tem mais ruído → thresholds ~0.06 abaixo dos valores 1H
# Objetivo: gerar 2-4 trades/dia para validação estatística do paper trading
REGIME_THRESHOLDS: dict[str, float] = {
    "TREND_EXPANSION":        0.44,
    "VOLATILITY_COMPRESSION": 0.46,
    "TREND_EXHAUSTION":       0.48,
    "MEAN_REVERTING_CHOP":    0.50,
    "HIGH_CORRELATION_RISK":  0.54,
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


class MomentumStrategy(BaseStrategy):

    REGIME_THRESHOLDS = REGIME_THRESHOLDS
    # EV mínimo reduzido para 30min — permite validação estatística do paper trading
    # Com WR=25-30% em 30min, exigir EV alto bloqueia tudo (matematicamente impossível)
    # EV calculado como: calibrated × TP_mult - (1-calibrated) × 1.0
    MIN_EV_MULTIPLIER = 0.5   # era 3.0 — min_ev agora = 0.5×0.005 = 0.0025
    ROUND_TRIP_FEE    = 0.005

    def __init__(self, symbols: list[str], strategy_id: str = "momentum_v2") -> None:
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
        if not ctx.candles_30m or len(ctx.candles_30m) < 4:
            _log("NO_CANDLES", f"Candles 30m insuficientes: {len(ctx.candles_30m) if ctx.candles_30m else 0}/4")
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
        ev     = self._expected_value(calibrated, regime)
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
            timeframe="30m",
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

    # ── Scoring v2 (5 fatores, scoring contínuo) ─────────────────────────────

    def _score_signal(self, ctx: StrategyContext, regime: str) -> tuple[float, dict]:
        """
        Modelo de scoring com 5 fatores contínuos.
        M1 usa retornos 30min (curto prazo) + 1H (médio prazo) para capturar moves intra-hora.

        Fatores:
          M1 Adaptive Momentum  (25%): blend 30min + 1H (captura moves rápidos)
          M2 Trend Consistency   (25%): % candles bullish + higher-highs E higher-lows
          M3 Volume Confirmation (20%): volume crescente + confirmação direcional
          M4 Regime Strength     (20%): distância SMA5-SMA20 normalizada
          M5 Candle Structure    (10%): close no terço superior do range
        """
        # Candles 1H (regime/tendência macro — SMA, ATR, M4)
        closes = [c.close  for c in ctx.candles_1h[:21]]
        highs  = [c.high   for c in ctx.candles_1h[:10]]
        lows   = [c.low    for c in ctx.candles_1h[:10]]
        opens  = [c.open   for c in ctx.candles_1h[:10]]
        vols_1h = [c.volume for c in ctx.candles_1h[:20]]

        # Candles 30min (estrutura recente — M2, M3, M5)
        c30 = ctx.candles_30m or []
        highs_30m = [c.high   for c in c30[:10]]
        lows_30m  = [c.low    for c in c30[:10]]
        opens_30m = [c.open   for c in c30[:10]]
        closes_30m = [c.close for c in c30[:10]]
        vols_30m  = [c.volume for c in c30[:20]]

        # ── M1: Adaptive Momentum (25%) — blend 30min + 1H ───
        atr_20 = sum(highs[i] - lows[i] for i in range(min(10, len(highs)))) / min(10, len(highs)) if highs else closes[0] * 0.01
        norm   = max(atr_20 * 2, closes[0] * 0.005)

        # Horizonte 1H (médio prazo: 5h, 10h, 20h)
        r5  = (closes[0] - closes[5])  / closes[5]  if len(closes) > 5  and closes[5]  > 0 else 0
        r10 = (closes[0] - closes[10]) / closes[10] if len(closes) > 10 and closes[10] > 0 else 0
        r20 = (closes[0] - closes[20]) / closes[20] if len(closes) > 20 and closes[20] > 0 else 0
        m1_1h = r5 * 0.5 + r10 * 0.3 + r20 * 0.2

        # Horizonte 30min (curto prazo: 30min, 2h em candles 30m)
        if len(closes_30m) >= 4:
            r1_30 = (closes_30m[0] - closes_30m[1]) / closes_30m[1] if closes_30m[1] > 0 else 0
            r4_30 = (closes_30m[0] - closes_30m[3]) / closes_30m[3] if closes_30m[3] > 0 else 0
            m1_30m = r1_30 * 0.6 + r4_30 * 0.4
        else:
            m1_30m = m1_1h

        momentum_weighted = m1_1h * 0.60 + m1_30m * 0.40
        m1 = min(max((momentum_weighted / (norm / closes[0])) * 0.5 + 0.5, 0.0), 1.0)

        # ── M2: Trend Consistency (25%) — usa candles 30min ──
        # Bullish count nos últimos 6 candles 30m (= 3h)
        src_opens  = opens_30m  if len(opens_30m)  >= 5 else opens
        src_closes = closes_30m if len(closes_30m) >= 5 else closes
        src_highs  = highs_30m  if len(highs_30m)  >= 5 else highs
        src_lows   = lows_30m   if len(lows_30m)   >= 5 else lows

        n = min(6, len(src_closes) - 1)
        bullish_count = sum(1 for i in range(n) if src_closes[i] > src_opens[i])
        pct_bullish   = bullish_count / n if n > 0 else 0.5

        hh_count  = sum(1 for i in range(min(4, len(src_highs)-1)) if src_highs[i] > src_highs[i+1])
        hl_count  = sum(1 for i in range(min(4, len(src_lows)-1))  if src_lows[i]  > src_lows[i+1])
        structure = (hh_count + hl_count) / 8

        m2 = pct_bullish * 0.5 + structure * 0.5

        # ── M3: Volume Confirmation (20%) — usa volumes 30min ─
        vols = vols_30m if len(vols_30m) >= 6 else vols_1h
        avg_vol_5  = sum(vols[:5])  / 5  if len(vols) >= 5  else vols[0] if vols else 1
        avg_vol_20 = sum(vols[:20]) / 20 if len(vols) >= 20 else avg_vol_5

        vol_ratio = min(vols[0] / avg_vol_5, 3.0) / 3.0 if avg_vol_5 > 0 else 0.5
        vol_trend = (sum(vols[:3]) / sum(vols[3:6])) if len(vols) >= 6 and sum(vols[3:6]) > 0 else 1.0
        vol_trend = min(max(vol_trend, 0.3), 2.0)
        vol_trend_score = (vol_trend - 0.3) / 1.7

        # Confirmação direcional com candle 30min mais recente
        cur_close = src_closes[0] if src_closes else closes[0]
        cur_open  = src_opens[0]  if src_opens  else opens[0]
        candle_confirm = 1.0 if (cur_close > cur_open and vols[0] > avg_vol_20) else 0.4

        m3 = vol_ratio * 0.4 + vol_trend_score * 0.3 + candle_confirm * 0.3

        # ── M4: Regime Strength (20%) — mantém 1H (SMA macro) ─
        sma5  = sum(closes[:5])  / 5
        sma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else sma5
        sma_distance = (sma5 - sma20) / sma20 if sma20 > 0 else 0
        m4_raw    = min(max((sma_distance + 0.02) / 0.04, 0.0), 1.0)
        m4_regime = REGIME_M4.get(regime, 0.45)
        m4 = m4_raw * 0.6 + m4_regime * 0.4

        # ── M5: Candle Structure (10%) — usa candles 30min ────
        # Close no terço superior do range dos 3 candles 30m mais recentes
        m5_highs  = src_highs[:3]  if src_highs  else highs[:3]
        m5_lows   = src_lows[:3]   if src_lows   else lows[:3]
        m5_closes = src_closes[:3] if src_closes else closes[:3]
        candle_scores = []
        for i in range(min(3, len(m5_closes))):
            rng = m5_highs[i] - m5_lows[i] if i < len(m5_highs) else 0
            if rng > 0:
                pos = (m5_closes[i] - m5_lows[i]) / rng
                candle_scores.append(pos)
        m5 = sum(candle_scores) / len(candle_scores) if candle_scores else 0.5

        # ── Score final ────────────────────────────────────────
        score = m1 * 0.25 + m2 * 0.25 + m3 * 0.20 + m4 * 0.20 + m5 * 0.10
        score = round(min(max(score, 0.0), 1.0), 4)

        factors = {
            "m1_momentum":   round(m1, 3),
            "m2_consistency":round(m2, 3),
            "m3_volume":     round(m3, 3),
            "m4_regime_str": round(m4, 3),
            "m5_candle":     round(m5, 3),
        }
        # Registra features no drift monitor (nunca bloqueia o trading)
        try:
            governance.record_live(ctx.symbol, factors)
        except Exception:
            pass
        return score, factors

    def _calibrate(self, score: float) -> float:
        return self._platt.calibrate(score)

    def _expected_value(self, calibrated: float, regime: str = "") -> float:
        """
        EV dinâmico alinhado com o TP real por regime (position_monitor.py).
        Reward = TP_mult, Risk = 1.0 (SL relativo ao ATR normalizado).
        """
        tp_by_regime = {
            "TREND_EXPANSION":        4.0,
            "VOLATILITY_COMPRESSION": 3.0,
            "TREND_EXHAUSTION":       2.5,
            "MEAN_REVERTING_CHOP":    1.5,
            "HIGH_CORRELATION_RISK":  2.0,
        }
        tp = tp_by_regime.get(regime, 3.0)
        return calibrated * tp - (1 - calibrated) * 1.0

    def _direction(self, ctx: StrategyContext) -> SignalDirection:
        """
        Direção baseada na maioria dos últimos 3 candles 1H (menos binário).
        Também considera momentum 30min recente para não bloquear moves rápidos.
        """
        if len(ctx.candles_1h) < 3:
            return SignalDirection.FLAT

        closes_1h = [c.close for c in ctx.candles_1h[:4]]
        # Maioria dos últimos 3 candles 1H são bullish (close > close anterior)?
        bullish_count = sum(1 for i in range(3) if closes_1h[i] > closes_1h[i + 1])

        # Desempate via 30min recente
        if bullish_count >= 2:
            return SignalDirection.LONG

        # Se 30min recente é fortemente bullish (último close > 2 closes atrás), aceita
        if ctx.candles_30m and len(ctx.candles_30m) >= 3:
            c30 = [c.close for c in ctx.candles_30m[:3]]
            if c30[0] > c30[2] * 1.002:   # +0.2% nos últimos 2 candles 30min
                return SignalDirection.LONG

        return SignalDirection.FLAT
