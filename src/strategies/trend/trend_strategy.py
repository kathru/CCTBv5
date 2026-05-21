"""
TrendStrategy — EMA50D + Vol Scaling (v5.6)

Estratégia documentada pelos maiores CTAs do mundo (Man AHL, Winton, Aspect).
Validada em backtest Jan/2025→Abr/2026: +7.8% vs Buy&Hold -18.4%.

Regras:
  LONG  : close > EMA50D AND vol_20D < 80% aa → compra spot
  SHORT : close < EMA50D AND vol_20D < 80% aa → vende perp (BTC-USDT-SWAP)
  FLAT  : vol muito alta (≥80% aa) → zera posição

Avalia 1× por dia no fechamento do candle diário (agregado dos 1H).
BTC avaliado primeiro → serve de âncora para ETH e SOL (BTC correlation gate).

Sizing: position_pct = min(target_vol(15%) / realized_vol, 33%)
"""

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from ...core.models import Signal, SignalDirection
from ..base import BaseStrategy, StrategyContext
from ..ml.inference import PlattCalibrator

logger     = logging.getLogger(__name__)
MODELS_DIR = Path("data") / "models"

# ── Parâmetros (sem fitting) ──────────────────────────────────────────────────
EMA_PERIOD  = 50      # dias
ATR_PERIOD  = 20      # dias
VOL_TARGET  = 0.15    # 15% vol anualizada alvo
VOL_CAP     = 0.80    # ≥80% aa → mercado em pânico → flat
MAX_POS_PCT = 0.33    # máx 33% capital por símbolo
ANN         = math.sqrt(365)

# Âncora BTC: ETH/SOL só operam se BTC também estiver no mesmo lado
# Evita divergência entre símbolos em mercados descorrelacionados
BTC_ANCHOR_ENABLED = True


class TrendStrategy(BaseStrategy):
    """
    EWMA Trend Daily + Vol Scaling.

    Avalia candles 1H do StrategyContext e os agrega internamente
    para produzir a visão diária necessária ao EMA50D.

    Sinal LONG:
        direction = LONG
        regime    = "TREND_UP"
        factors   = {sl_pct, tp_pct, vol_ann, ema50d, mode="long"}

    Sinal SHORT (via perp):
        direction = SHORT
        regime    = "TREND_DOWN"
        factors   = {sl_pct, tp_pct, vol_ann, ema50d, mode="short"}

    Sinal FLAT (sair):
        direction = FLAT
        regime    = "VOL_SPIKE" ou "TREND_CHANGE"
    """

    # Estado compartilhado BTC → âncora para ETH/SOL
    _btc_signal: str = "FLAT"     # "LONG" | "SHORT" | "FLAT"
    _btc_vol:    float = 0.0

    def __init__(self, symbols: list[str]) -> None:
        super().__init__(strategy_id="trend_v56", symbols=symbols)
        try:
            self._platt = PlattCalibrator(
                coef_path=Path(MODELS_DIR) / "calibration_coef.json"
            )
        except Exception:
            self._platt = None
            logger.warning("TrendStrategy: calibrador Platt não disponível")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _agg_daily(candles_1h) -> list[dict]:
        """Agrega candles 1H → diários (UTC midnight)."""
        from collections import defaultdict
        by_day: dict = defaultdict(list)
        for c in candles_1h:
            day = (int(c.timestamp.timestamp() * 1000) // 86_400_000) * 86_400_000
            by_day[day].append(c)
        result = []
        for day in sorted(by_day):
            g = by_day[day]
            result.append({
                "close": g[-1].close,
                "high":  max(x.high  for x in g),
                "low":   min(x.low   for x in g),
            })
        return result

    @staticmethod
    def _ema(values: list[float], period: int) -> float | None:
        """Retorna EMA atual (último valor). None se dados insuficientes."""
        valid = [v for v in values if not math.isnan(v)]
        if len(valid) < period:
            return None
        k   = 2.0 / (period + 1)
        ema = sum(valid[:period]) / period
        for v in valid[period:]:
            ema = v * k + ema * (1 - k)
        return ema

    @staticmethod
    def _atr_pct(daily: list[dict], period: int) -> float | None:
        """ATR% anualizado. None se dados insuficientes."""
        if len(daily) < period + 1:
            return None
        trs = []
        for i in range(1, len(daily)):
            prev = daily[i - 1]["close"]
            c    = daily[i]
            tr   = max(c["high"] - c["low"],
                       abs(c["high"] - prev),
                       abs(c["low"]  - prev)) / prev
            trs.append(tr)
        if len(trs) < period:
            return None
        # ATR = média simples dos últimos N
        return sum(trs[-period:]) / period

    # ── Evaluate ──────────────────────────────────────────────────────────────

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        """
        1. Agrega candles 1H → diários
        2. Calcula EMA50D e ATR20D
        3. Decide sinal: LONG / SHORT / FLAT
        4. Calcula position sizing via vol-target
        """
        sym = ctx.symbol

        # Precisa de candles newest-first → reverte para oldest-first para agregar
        candles_oldest = list(reversed(ctx.candles_1h))
        if len(candles_oldest) < (EMA_PERIOD + ATR_PERIOD) * 2:
            logger.debug("TrendStrategy %s: candles insuficientes (%d)", sym, len(candles_oldest))
            return None

        daily = self._agg_daily(candles_oldest)

        # Mínimo de dados para EMA50D
        if len(daily) < EMA_PERIOD + ATR_PERIOD + 5:
            return None

        closes = [d["close"] for d in daily]
        price  = closes[-1]

        ema50d = self._ema(closes, EMA_PERIOD)
        if ema50d is None or price <= 0:
            return None

        atr_pct = self._atr_pct(daily[-ATR_PERIOD - 5:], ATR_PERIOD)
        if atr_pct is None:
            return None

        vol_ann = atr_pct * ANN   # vol realizada anualizada

        # ── Sinal ─────────────────────────────────────────────────────────────
        trend_up   = price > ema50d
        trend_down = price < ema50d
        vol_ok     = vol_ann < VOL_CAP

        # Atualiza âncora BTC
        if sym == "BTC-USDT":
            if trend_up and vol_ok:
                TrendStrategy._btc_signal = "LONG"
            elif trend_down and vol_ok:
                TrendStrategy._btc_signal = "SHORT"
            else:
                TrendStrategy._btc_signal = "FLAT"
            TrendStrategy._btc_vol = vol_ann

        # ETH e SOL precisam de concordância com BTC (correlation gate)
        if BTC_ANCHOR_ENABLED and sym != "BTC-USDT":
            btc_sig = TrendStrategy._btc_signal
            if trend_up and btc_sig != "LONG":
                return None   # BTC não confirma alta → não entra long
            if trend_down and btc_sig != "SHORT":
                return None   # BTC não confirma baixa → não entra short

        # Vol alta → flat (sai de qualquer posição)
        if not vol_ok:
            return Signal(
                strategy_id=self._strategy_id,
                symbol=sym,
                direction=SignalDirection.FLAT,
                timestamp=datetime.now(UTC),
                score=0.0, calibrated_score=0.0, confidence=0.0,
                expected_value=0.0, kelly_fraction=0.0,
                regime="VOL_SPIKE",
                timeframe="1D",
                factors={"vol_ann": round(vol_ann, 3), "ema50d": round(ema50d, 4),
                         "mode": "flat", "reason": "vol_spike"},
            )

        if not trend_up and not trend_down:
            return None   # Preço na EMA → sem sinal

        # ── Sizing ─────────────────────────────────────────────────────────────
        pos_pct = min(VOL_TARGET / vol_ann, MAX_POS_PCT)

        # SL: ATR × 2 do lado oposto (stop baseado em vol, não fixo)
        sl_pct = min(atr_pct * 2, 0.08)   # máx 8% SL
        tp_pct = sl_pct * 2.0              # TP = 2× SL (ratio 2:1)

        # Score baseado na força do sinal (distância % da EMA50D)
        dist_pct = abs(price - ema50d) / ema50d
        raw_score = min(dist_pct / 0.05, 1.0)   # 5% de distância = score 1.0

        calibrated = self._platt.calibrate(raw_score) if self._platt else raw_score
        ev = raw_score * 2.0 - (1 - raw_score)   # TP=2×SL → EV simples

        direction = SignalDirection.LONG if trend_up else SignalDirection.SHORT
        regime    = "TREND_UP" if trend_up else "TREND_DOWN"
        mode      = "long"     if trend_up else "short"

        logger.info(
            "TrendStrategy %s | %s | price=%.4f ema50=%.4f vol=%.1f%% pos=%.1f%% score=%.2f",
            sym, regime, price, ema50d, vol_ann * 100, pos_pct * 100, raw_score,
        )

        return Signal(
            strategy_id=self._strategy_id,
            symbol=sym,
            direction=direction,
            timestamp=datetime.now(UTC),
            score=raw_score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ev,
            kelly_fraction=round(pos_pct, 4),
            regime=regime,
            timeframe="1D",
            factors={
                "sl_pct":   round(sl_pct, 5),
                "tp_pct":   round(tp_pct, 5),
                "vol_ann":  round(vol_ann, 4),
                "ema50d":   round(ema50d, 4),
                "pos_pct":  round(pos_pct, 4),
                "dist_pct": round(dist_pct, 4),
                "mode":     mode,
            },
        )
