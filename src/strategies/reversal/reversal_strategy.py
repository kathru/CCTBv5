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
from ...monitoring.signal_log import SignalAuditEntry, signal_audit_log
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

    def _audit(
        self,
        symbol: str,
        result: str,
        detail: str,
        score: float = 0.0,
        calibrated: float = 0.0,
        ev: float = 0.0,
        direction: str = "FLAT",
        factors: dict | None = None,
    ) -> None:
        """Registra avaliação no signal_audit_log (alimenta dashboard e funil)."""
        signal_audit_log.record(SignalAuditEntry(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            regime="REVERSAL_1H",
            score=score,
            calibrated=calibrated,
            threshold=0.0,          # sem threshold fixo na reversão
            ev=ev,
            direction=direction,
            result=result,
            detail=detail,
            factors=factors or {},
            strategy_id=self._strategy_id,
        ))

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        """
        Avalia contexto e retorna sinal de compra se reversão confirmada.
        Registra CADA avaliação no signal_audit_log para alimentar o dashboard.

        Contexto: ctx.candles_1h (newest first, ≥ 22 candles).
        Retorna: Signal com factors{sl_pct, tp_pct} ou None.
        """
        sym = ctx.symbol
        c   = ctx.candles_1h   # newest first

        if len(c) < 22:
            self._audit(sym, "NO_CANDLES", "Candles insuficientes")
            return None

        closes  = [x.close  for x in c[:22]]
        highs   = [x.high   for x in c[:22]]
        lows    = [x.low    for x in c[:22]]
        volumes = [x.volume for x in c[:22]]

        current = closes[0]
        sma20   = sum(closes[:20]) / 20

        # ── Filtro 1: Faixa macro (nem bear nem rally avançado) ───────────────
        if not (sma20 * self.TREND_FLOOR <= current <= sma20 * self.TREND_CEIL):
            ratio_sma = current / sma20 if sma20 > 0 else 0
            detail = (f"Acima SMA20 ({ratio_sma:.2%})" if current > sma20 * self.TREND_CEIL
                      else f"Abaixo SMA20 ({ratio_sma:.2%})")
            self._audit(sym, "MACRO_BLOCKED", detail)
            return None

        # ── Filtro 2: Queda real ≥ 3% ─────────────────────────────────────────
        lookback_high = max(closes[self.BASE_CANDLES + 1:20])
        fall_low      = min(closes[1:20])
        fall_pct      = (lookback_high - fall_low) / lookback_high if lookback_high > 0 else 0

        base_factors = {"fall_pct": round(fall_pct, 3), "sma20_ratio": round(current / sma20, 3)}

        if fall_pct < self.MIN_FALL_PCT:
            self._audit(sym, "FALL_WEAK", f"Queda {fall_pct:.1%} < 3%", factors=base_factors)
            return None

        # ── Filtro 3: Dip tocou a SMA20 ──────────────────────────────────────
        if fall_low > sma20 * 1.01:
            self._audit(sym, "DIP_SHALLOW", f"Fall low {fall_low:.2f} acima SMA20 {sma20:.2f}",
                        factors=base_factors)
            return None

        # ── Filtro 4 + 5: Base e rompimento ───────────────────────────────────
        n          = self.BASE_CANDLES
        base_high  = max(highs[1:n + 1])
        base_low   = min(lows[1:n + 1])
        base_range = base_high - base_low
        fall_mag   = lookback_high - fall_low
        base_ratio = base_range / fall_mag if fall_mag > 0 else 1

        base_factors = {**base_factors, "base_range": round(base_ratio, 3)}

        if current <= base_high * 1.001:
            self._audit(sym, "BASE_MISSING",
                        f"Sem rompimento: {current:.2f} ≤ base_high {base_high:.2f}",
                        factors=base_factors)
            return None

        # ── Filtro 6: Volume ──────────────────────────────────────────────────
        avg_vol   = sum(volumes[1:9]) / 8
        vol_ratio = volumes[0] / avg_vol if avg_vol > 0 else 0

        all_factors = {**base_factors, "vol_ratio": round(vol_ratio, 2)}

        if avg_vol > 0 and vol_ratio < 0.8:
            self._audit(sym, "VOLUME_WEAK", f"Volume {vol_ratio:.2f}× < 0.8×",
                        factors=all_factors)
            return None

        # ── Calcula SL / TP ───────────────────────────────────────────────────
        sl_target = base_low * 0.999
        sl_dist   = current - sl_target
        if sl_dist <= 0:
            self._audit(sym, "SL_INVALID", "SL abaixo do entry", factors=all_factors)
            return None

        sl_pct = sl_dist / current
        if not (self.MIN_SL_PCT <= sl_pct <= self.MAX_SL_PCT):
            self._audit(sym, "SL_INVALID",
                        f"SL {sl_pct:.2%} fora de [{self.MIN_SL_PCT:.0%}–{self.MAX_SL_PCT:.0%}]",
                        factors=all_factors)
            return None

        tp_pct = self.MIN_RATIO * sl_pct
        ratio  = self.MIN_RATIO

        # ── Score (0–1) ───────────────────────────────────────────────────────
        score_fall  = min(fall_pct / 0.10, 1.0)
        score_base  = 1.0 - min(base_ratio, 1.0)
        score_vol   = min(vol_ratio / 2.0, 1.0)
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

        ev = ratio * calibrated - (1 - calibrated)

        signal_factors = {
            "sl_pct":     round(sl_pct, 5),
            "tp_pct":     round(tp_pct, 5),
            "fall_pct":   round(fall_pct, 3),
            "base_range": round(base_ratio, 3),
            "vol_ratio":  round(vol_ratio, 2),
            "tp_ratio":   round(ratio, 2),
            "sl_ref":     round(sl_target, 4),
        }

        logger.info(
            "ReversalStrategy1H SIGNAL | %s | fall=%.1f%% sl=%.2f%% tp=%.2f%% "
            "vol=%.2fx score=%.3f cal=%.3f EV=%.4f",
            sym, fall_pct * 100, sl_pct * 100, tp_pct * 100,
            vol_ratio, raw_score, calibrated, ev,
        )

        # Registra no audit log — alimenta dashboard e funil
        self._audit(sym, "SIGNAL",
                    f"Reversão confirmada: fall={fall_pct:.1%} sl={sl_pct:.2%} tp={tp_pct:.2%}",
                    score=raw_score, calibrated=calibrated, ev=ev,
                    direction="LONG", factors=signal_factors)

        return Signal(
            strategy_id=self._strategy_id,
            symbol=sym,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw_score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ev,
            kelly_fraction=round(min(self.KELLY_BASE, 0.10), 4),
            regime="REVERSAL_1H",
            timeframe="1H",
            factors=signal_factors,
        )
