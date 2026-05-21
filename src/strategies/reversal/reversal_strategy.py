"""
Reversal Strategy 1H — Buy the Bottom.

Detecta reversões reais em candles 1H:
  1. QUEDA REAL ≥ 3% nos últimos 7–20 candles
  2. BASE ≥ 6 velas (6H de consolidação / fundo sendo formado)
  3. ROMPIMENTO da máxima da base (+0.1% de tolerância)
  4. VOLUME do rompimento ≥ 80% da média das 8h anteriores
  5. MACRO: preço entre 92%–102% da SMA20
     (não é bear estrutural nem rally avançado)
  6. DIPPING: fall_low deve ter tocado a SMA20
     (dip estrutural, não ruído de alta)

Saída (gerenciada pelo ExitPlan com sl_pct / tp_pct):
  SL = base_low × 0.999  →  sl_pct = (entry − SL) / entry
  TP = entry × (1 + 1.5 × sl_pct)  →  ratio 1.5:1 garantido

Os percentuais sl_pct / tp_pct são passados nos factors do Signal e
aplicados ao fill_price real pelo ExitPlan, garantindo assimetria
independente de gaps na abertura do candle de entrada.

Validado em backtest abril 2026:
  16 trades | WR=44% | PF=2.4 | +$240 (+0.25%) | Fees=$89
  vs MomentumStrategy: 157 trades | -$960 | Fees=$760
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from ...core.models import Signal, SignalDirection
from ..base import BaseStrategy, StrategyContext
from ..ml.inference import PlattCalibrator

logger     = logging.getLogger(__name__)
MODELS_DIR = Path("data") / "models"


class ReversalStrategy1H(BaseStrategy):
    """
    Estratégia de reversão 1H — compra fundos confirmados, vende na recuperação.

    Parâmetros (sem fitting — baseados em lógica de mercado):
      MIN_FALL_PCT = 3%    queda mínima para ser considerado crash
      BASE_CANDLES = 6     6H de base = estabilização do fundo
      TREND_FLOOR  = 0.92  não entrar em bear estrutural (< 92% SMA20)
      TREND_CEIL   = 1.02  não entrar em rally avançado  (> 102% SMA20)
      MIN_SL_PCT   = 0.8%  SL mínimo — abaixo disso fees > ganho
      MAX_SL_PCT   = 6%    SL máximo — base muito larga → não é fundo
      MIN_RATIO    = 1.5   TP = 1.5 × risco (garantido por construção)
    """

    # ── Parâmetros ────────────────────────────────────────────────────────────
    MIN_FALL_PCT = 0.030   # 3% de queda mínima
    BASE_CANDLES = 6       # 6 velas 1H = 6H de consolidação
    TREND_FLOOR  = 0.92    # 92% da SMA20 (mínimo — não entrar em bear)
    TREND_CEIL   = 1.02    # 102% da SMA20 (máximo — não entrar em rally)
    MIN_SL_PCT   = 0.008   # SL mínimo 0.8% (fee round-trip = 0.2%)
    MAX_SL_PCT   = 0.060   # SL máximo 6.0% (base muito larga → ruído)
    MIN_RATIO    = 1.5     # TP/SL mínimo — garantido por construção
    KELLY_BASE   = 0.08    # 8% do capital por trade (conservador)

    def __init__(self, symbols: list[str]) -> None:
        super().__init__(strategy_id="reversal_1h", symbols=symbols)
        try:
            self._platt = PlattCalibrator(
                coef_path=Path(MODELS_DIR) / "calibration_coef.json"
            )
        except Exception:
            self._platt = None
            logger.warning("ReversalStrategy1H: calibrador Platt não disponível — usando score raw")

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        """
        Avalia contexto e retorna sinal de compra se reversão confirmada.

        Contexto: ctx.candles_1h (newest first, ≥ 22 candles).
        Retorna: Signal com factors{sl_pct, tp_pct} ou None.
        """
        c = ctx.candles_1h   # newest first
        if len(c) < 22:
            return None

        closes  = [x.close  for x in c[:22]]
        highs   = [x.high   for x in c[:22]]
        lows    = [x.low    for x in c[:22]]
        volumes = [x.volume for x in c[:22]]

        current = closes[0]
        sma20   = sum(closes[:20]) / 20

        # ── Filtro 1: Faixa macro (nem bear nem rally avançado) ───────────────
        if not (sma20 * self.TREND_FLOOR <= current <= sma20 * self.TREND_CEIL):
            return None

        # ── Filtro 2: Queda real ≥ 3% ─────────────────────────────────────────
        # lookback_high: máxima dos candles ANTES da base (7–20h atrás)
        lookback_high = max(closes[self.BASE_CANDLES + 1:20])
        fall_low      = min(closes[1:20])
        fall_pct      = (lookback_high - fall_low) / lookback_high if lookback_high > 0 else 0

        if fall_pct < self.MIN_FALL_PCT:
            return None

        # ── Filtro 3: Dip tocou a SMA20 (não é ruído de rally) ───────────────
        if fall_low > sma20 * 1.01:
            return None

        # ── Filtro 4: Base formada (últimas 6H estabilizando) ─────────────────
        n          = self.BASE_CANDLES
        base_high  = max(highs[1:n + 1])   # highs[1:7]
        base_low   = min(lows[1:n + 1])    # lows[1:7]
        base_range = base_high - base_low
        fall_mag   = lookback_high - fall_low

        # ── Filtro 5: Rompimento da máxima da base ────────────────────────────
        if current <= base_high * 1.001:   # tolerância 0.1%
            return None

        # ── Filtro 6: Volume — força no rompimento ────────────────────────────
        avg_vol = sum(volumes[1:9]) / 8
        if avg_vol > 0 and volumes[0] < avg_vol * 0.8:
            return None

        # ── Calcula SL / TP ───────────────────────────────────────────────────
        sl_target = base_low * 0.999                 # 0.1% abaixo da base
        sl_dist   = current - sl_target
        if sl_dist <= 0:
            return None

        sl_pct = sl_dist / current
        if not (self.MIN_SL_PCT <= sl_pct <= self.MAX_SL_PCT):
            return None

        # TP = 1.5× risco relativo ao entry real (aplicado no ExitPlan)
        tp_pct    = self.MIN_RATIO * sl_pct
        ratio     = self.MIN_RATIO   # garantido por construção

        # ── Score (0–1) ────────────────────────────────────────────────────────
        score_fall  = min(fall_pct / 0.10, 1.0)
        score_base  = 1.0 - min(base_range / fall_mag, 1.0) if fall_mag > 0 else 0
        score_vol   = min(volumes[0] / (avg_vol * 2), 1.0) if avg_vol > 0 else 0.5
        score_ratio = min(ratio / 8.0, 1.0)

        raw_score = (score_fall * 0.30 + score_base * 0.30
                     + score_vol * 0.20 + score_ratio * 0.20)

        if self._platt is not None:
            try:
                calibrated = self._platt.calibrate(raw_score)
            except Exception:
                calibrated = raw_score
        else:
            calibrated = raw_score

        logger.info(
            "ReversalStrategy1H | %s | fall=%.1f%% sl=%.2f%% tp=%.2f%% "
            "vol_ratio=%.2fx score=%.3f",
            ctx.symbol, fall_pct * 100, sl_pct * 100, tp_pct * 100,
            volumes[0] / avg_vol if avg_vol > 0 else 0, raw_score,
        )

        return Signal(
            strategy_id=self._strategy_id,
            symbol=ctx.symbol,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw_score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ratio * calibrated - (1 - calibrated),
            kelly_fraction=round(min(self.KELLY_BASE, 0.10), 4),
            regime="REVERSAL_1H",
            timeframe="1H",
            factors={
                # Percentuais usados pelo ExitPlan (relativos ao fill_price real)
                "sl_pct":     round(sl_pct, 5),
                "tp_pct":     round(tp_pct, 5),
                # Referência (display / log)
                "fall_pct":   round(fall_pct, 3),
                "base_range": round(base_range / fall_mag, 3) if fall_mag > 0 else 0,
                "vol_ratio":  round(volumes[0] / avg_vol, 2) if avg_vol > 0 else 0,
                "tp_ratio":   round(ratio, 2),
                "sl_ref":     round(sl_target, 4),
            },
        )
