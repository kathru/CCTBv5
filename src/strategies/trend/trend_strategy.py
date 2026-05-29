"""
TrendStrategy — EMA50D + Vol Scaling (v5.6 Brasil)

Estratégia documentada pelos maiores CTAs do mundo (Man AHL, Winton, Aspect).
Validada em backtest Jan/2025→Abr/2026: -1.4% vs Buy&Hold -18.4% (+17pp alpha).

Regras (Long + Flat — sem derivativos, compatível com regulação BR):
  LONG  : close > EMA50D AND vol_20D < 80% aa → compra spot
  FLAT  : close < EMA50D OR  vol_20D ≥ 80% aa → vende spot, aguarda em USDT

Avalia 1× por dia no fechamento do candle diário (agregado dos 1H).
BTC avaliado primeiro → serve de âncora para ETH e SOL (BTC correlation gate).

Sizing: position_pct = min(target_vol(15%) / realized_vol, 33%)
"""

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from ...core.models import Signal, SignalDirection
from ...monitoring.signal_log import SignalAuditEntry, signal_audit_log
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
    _btc_signal:  str  = "FLAT"
    _btc_vol:     float = 0.0
    # Cache histórico de candles diários por símbolo
    _daily_cache: dict = {}
    # Controle de deduplicação diária: {symbol: date} — avalia 1× por dia
    _last_eval_day: dict = {}

    def __init__(self, symbols: list[str]) -> None:
        super().__init__(strategy_id="trend_v56", symbols=symbols)
        try:
            self._platt = PlattCalibrator(
                coef_path=Path(MODELS_DIR) / "calibration_coef.json"
            )
        except Exception:
            self._platt = None
            logger.warning("TrendStrategy: calibrador Platt não disponível")
        self._preload_history(symbols)

    def _preload_history(self, symbols: list[str]) -> None:
        """
        Pré-carrega candles diários históricos do cache JSON.
        Garante EMA50D disponível imediatamente, sem esperar 50 dias de feed ao vivo.
        """
        import json
        from collections import defaultdict
        cache_dir = Path("data") / "cache"
        if not cache_dir.exists():
            logger.warning("TrendStrategy: cache dir não encontrado: %s", cache_dir)
            return
        for sym in symbols:
            key  = sym.replace("-", "_")
            path = cache_dir / f"{key}_1H.json"
            if not path.exists():
                continue
            try:
                raw  = json.loads(path.read_text())
                # Últimos 120 dias (suficiente para EMA50D + ATR20D com margem)
                sorted_raw = sorted(raw, key=lambda c: c["ts"])
                cutoff = sorted_raw[-1]["ts"] - 120 * 86_400_000
                recent = [c for c in sorted_raw if c["ts"] >= cutoff]
                by_day: dict = defaultdict(list)
                for c in recent:
                    day = (c["ts"] // 86_400_000) * 86_400_000
                    by_day[day].append(c)
                daily = [
                    {"ts": d,
                     "close": g[-1]["close"],
                     "high":  max(x["high"] for x in g),
                     "low":   min(x["low"]  for x in g)}
                    for d, g in sorted(by_day.items())
                    for g in [by_day[d]]
                ]
                TrendStrategy._daily_cache[sym] = daily
                logger.info("TrendStrategy: %d dias históricos carregados para %s", len(daily), sym)
            except Exception as exc:
                logger.warning("TrendStrategy: falha ao carregar histórico %s: %s", sym, exc)

    # ── Audit log ─────────────────────────────────────────────────────────────

    def _audit(self, symbol: str, result: str, detail: str,
               regime: str = "TREND_UP", score: float = 0.0,
               calibrated: float = 0.0, ev: float = 0.0,
               direction: str = "FLAT", factors: dict | None = None) -> None:
        signal_audit_log.record(SignalAuditEntry(
            timestamp=datetime.now(UTC), symbol=symbol,
            regime=regime, score=score, calibrated=calibrated,
            threshold=0.0, ev=ev, direction=direction,
            result=result, detail=detail, factors=factors or {},
            strategy_id=self._strategy_id,
        ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _agg_daily(candles_1h) -> list[dict]:
        """Agrega candles 1H → diários (UTC midnight). Inclui 'ts' para merge com histórico."""
        from collections import defaultdict
        by_day: dict = defaultdict(list)
        for c in candles_1h:
            day = (int(c.timestamp.timestamp() * 1000) // 86_400_000) * 86_400_000
            by_day[day].append(c)
        result = []
        for day in sorted(by_day):
            g = by_day[day]
            result.append({
                "ts":    day,
                "close": g[-1].close,
                "high":  max(x.high for x in g),
                "low":   min(x.low  for x in g),
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
        sym      = ctx.symbol
        today    = datetime.now(UTC).date()

        # ── Deduplicação diária ───────────────────────────────────────────────
        # A TrendStrategy usa EMA50D (diário) — só faz sentido avaliar 1× por dia.
        # O warm-start no boot é permitido (today != last_eval_day na primeira vez).
        # Avaliações subsequentes no mesmo dia retornam None silenciosamente.
        last_day = TrendStrategy._last_eval_day.get(sym)
        if last_day == today:
            return None   # já avaliou hoje — aguarda próximo dia UTC

        # Agrega candles 1H ao vivo → diários
        candles_oldest = list(reversed(ctx.candles_1h))
        live_daily     = self._agg_daily(candles_oldest) if candles_oldest else []

        # Mescla com histórico pré-carregado (garante EMA50D sem esperar 50 dias ao vivo)
        hist = TrendStrategy._daily_cache.get(sym, [])
        if hist and live_daily:
            live_ts = {d["ts"] for d in live_daily}
            merged  = [d for d in hist if d["ts"] not in live_ts] + live_daily
            daily   = sorted(merged, key=lambda d: d["ts"])[-120:]
        elif hist:
            daily = hist[-120:]
        else:
            daily = live_daily

        if len(daily) < EMA_PERIOD + ATR_PERIOD + 5:
            self._audit(sym, "NO_CANDLES",
                        f"Apenas {len(daily)} dias (precisa {EMA_PERIOD+ATR_PERIOD+5})")
            return None

        closes = [d["close"] for d in daily]
        price  = closes[-1]

        ema50d = self._ema(closes, EMA_PERIOD)
        if ema50d is None or price <= 0:
            self._audit(sym, "NO_CANDLES", "EMA50D indisponível")
            return None

        atr_pct = self._atr_pct(daily[-ATR_PERIOD - 5:], ATR_PERIOD)
        if atr_pct is None:
            self._audit(sym, "NO_CANDLES", "ATR20D indisponível")
            return None

        vol_ann = atr_pct * ANN   # vol realizada anualizada

        # ── Sinal ─────────────────────────────────────────────────────────────
        trend_up = price > ema50d
        vol_ok   = vol_ann < VOL_CAP

        # Marca que já avaliamos hoje (impede avaliações duplicadas na mesma hora)
        TrendStrategy._last_eval_day[sym] = today

        # Atualiza âncora BTC para ETH/SOL
        if sym == "BTC-USDT":
            TrendStrategy._btc_signal = "LONG" if (trend_up and vol_ok) else "FLAT"
            TrendStrategy._btc_vol    = vol_ann

        # ETH e SOL só entram se BTC também estiver em tendência de alta
        if BTC_ANCHOR_ENABLED and sym != "BTC-USDT":
            if TrendStrategy._btc_signal != "LONG":
                self._audit(sym, "BTC_NOT_TRENDING",
                            f"BTC sinal={TrendStrategy._btc_signal} — ETH/SOL bloqueados",
                            regime="BTC_FLAT",
                            factors={"ema50d": round(ema50d, 4), "vol_ann": round(vol_ann, 4)})
                return Signal(
                    strategy_id=self._strategy_id, symbol=sym,
                    direction=SignalDirection.FLAT,
                    timestamp=datetime.now(UTC),
                    score=0.0, calibrated_score=0.0, confidence=0.0,
                    expected_value=0.0, kelly_fraction=0.0,
                    regime="BTC_FLAT", timeframe="1D",
                    factors={"vol_ann": round(vol_ann, 4), "ema50d": round(ema50d, 4),
                             "mode": "flat", "reason": "btc_not_trending"},
                )

        # Vol spike ou tendência de baixa → FLAT
        if not vol_ok or not trend_up:
            reason = "vol_spike" if not vol_ok else "below_ema50d"
            regime = "VOL_SPIKE" if not vol_ok else "TREND_DOWN"
            dist   = (price - ema50d) / ema50d
            detail = (f"Vol {vol_ann*100:.0f}% ≥ 80% aa" if not vol_ok
                      else f"Preço {dist*100:.1f}% abaixo da EMA50D")
            self._audit(sym, "TREND_FLAT", detail, regime=regime,
                        factors={"ema50d": round(ema50d,4), "vol_ann": round(vol_ann,4),
                                 "dist_pct": round(dist,4), "mode": "flat"})
            logger.info(
                "TrendStrategy %s | FLAT | price=%.2f ema50=%.2f vol=%.1f%% reason=%s",
                sym, price, ema50d, vol_ann * 100, reason,
            )
            return Signal(
                strategy_id=self._strategy_id, symbol=sym,
                direction=SignalDirection.FLAT,
                timestamp=datetime.now(UTC),
                score=0.0, calibrated_score=0.0, confidence=0.0,
                expected_value=0.0, kelly_fraction=0.0,
                regime=regime, timeframe="1D",
                factors={"vol_ann": round(vol_ann, 4), "ema50d": round(ema50d, 4),
                         "mode": "flat", "reason": reason},
            )

        # ── LONG: acima da EMA50D com vol controlada ───────────────────────────
        pos_pct  = min(VOL_TARGET / vol_ann, MAX_POS_PCT)
        sl_pct   = min(atr_pct * 2, 0.10)   # SL emergência: 2× ATR, máx 10%
        tp_pct   = sl_pct * 2.0              # TP = 2× SL

        dist_pct  = (price - ema50d) / ema50d
        raw_score = min(dist_pct / 0.05, 1.0)   # 5% acima da EMA = score máximo
        calibrated = self._platt.calibrate(raw_score) if self._platt else raw_score

        logger.info(
            "TrendStrategy %s | LONG | price=%.2f ema50=%.2f dist=+%.1f%% vol=%.1f%% pos=%.1f%%",
            sym, price, ema50d, dist_pct * 100, vol_ann * 100, pos_pct * 100,
        )

        sig_factors = {
            "sl_pct": round(sl_pct, 5), "tp_pct": round(tp_pct, 5),
            "vol_ann": round(vol_ann, 4), "ema50d": round(ema50d, 4),
            "pos_pct": round(pos_pct, 4), "dist_pct": round(dist_pct, 4),
            "mode": "long",
        }
        self._audit(
            sym, "SIGNAL",
            f"LONG: preço +{dist_pct*100:.1f}% acima EMA50D"
            f" | vol {vol_ann*100:.0f}% aa | pos {pos_pct*100:.0f}%",
            regime="TREND_UP", score=raw_score, calibrated=calibrated,
            ev=raw_score*2.0-(1-raw_score), direction="LONG", factors=sig_factors,
        )

        return Signal(
            strategy_id=self._strategy_id, symbol=sym,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw_score, calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=raw_score * 2.0 - (1 - raw_score),
            kelly_fraction=round(pos_pct, 4),
            regime="TREND_UP", timeframe="1D",
            factors={
                "sl_pct":   round(sl_pct, 5),
                "tp_pct":   round(tp_pct, 5),
                "vol_ann":  round(vol_ann, 4),
                "ema50d":   round(ema50d, 4),
                "pos_pct":  round(pos_pct, 4),
                "dist_pct": round(dist_pct, 4),
                "mode":     "long",
            },
        )
