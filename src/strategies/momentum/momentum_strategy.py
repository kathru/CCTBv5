"""
V4 Momentum Strategy — BULL / CHOP / BEAR aware.

Camadas de proteção:
  1. Detecção explícita de BEAR (downtrend gradual)
  2. Sizing adaptativo por regime (Kelly multiplier)
  3. Confirmação multi-timeframe 1H + 6H

Regimes e comportamento (v2.5.0 — 2026-05-23):
  TREND_EXPANSION      → BULL  : threshold 0.56, Kelly 100%, SL 1.5×ATR, TP 4.5×ATR, timeout 48h
  VOLATILITY_COMPRESSION       : threshold 0.58, Kelly 80%,  SL 1.0×ATR, TP 3.5×ATR, timeout 24h
  MEAN_REVERTING_CHOP  → CHOP  : threshold 0.68, Kelly 50%,  SL 0.7×ATR, TP 2.0×ATR, timeout 8h
  TREND_EXHAUSTION             : BLOQUEADO (threshold 0.99) — comprar topo é errado
  HIGH_CORRELATION_RISK        : BLOQUEADO (threshold 0.99)
  BEAR_TREND           → BEAR  : BLOQUEADO para novas entradas
  PANIC_LIQUIDATION            : BLOQUEADO + saída imediata
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from ...core.models import Signal, SignalDirection
from ...market.alpha_orthogonality import alpha_orthogonality
from ...monitoring.feature_governance import governance
from ...monitoring.signal_log import SignalAuditEntry, signal_audit_log
from ...oms.sizing_engine import SizingEngine
from ..base import BaseStrategy, StrategyContext
from ..edge_conditioner import EdgeConditioner
from ..ml.inference import PlattCalibrator

MODELS_DIR = Path("data") / "models"
logger     = logging.getLogger(__name__)


# ── Configuração por regime ───────────────────────────────────────────────────

# Thresholds em RAW score space — calibrados para 1H
# 1H tem menos ruído que 30m → thresholds mais conservadores (+0.04 vs 30m)
# Objetivo: 1-3 trades/dia de alta qualidade com menor ruído
REGIME_THRESHOLDS: dict[str, float] = {
    # v2.4.0 — thresholds mais seletivos baseados em análise live (WR 10.7% em 28 trades)
    # Eleva barra de entrada para reduzir trades em regimes menos confiáveis
    "TREND_EXPANSION":        0.56,   # era 0.50 → +0.06 (evita entradas em fakeouts)
    "VOLATILITY_COMPRESSION": 0.58,   # era 0.52 → +0.06 (exige confirmação de breakout)
    "MEAN_REVERTING_CHOP":    0.68,   # era 0.60 → +0.08 (só sinais muito fortes em lateral)
    "TREND_EXHAUSTION":       0.99,   # BLOQUEADO — comprar topo é errado
    "HIGH_CORRELATION_RISK":  0.99,   # BLOQUEADO — risco não justifica
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
# Princípio: "comprar fraqueza dentro de força — nunca comprar força exaurida"
BLOCKED_REGIMES = {
    "BEAR_TREND",
    "PANIC_LIQUIDATION",
    "TREND_EXHAUSTION",      # comprar topo de movimento fraco = errado
    "HIGH_CORRELATION_RISK", # risco sistêmico não justifica entrada
}

# EV mínimo por regime — quanto mais favorável o regime, mais exigente
# EXPANSION: mercado claro → exige EV positivo real
# CHOP: mercado lateral → aceita EV quase zero (entry oportunista)
REGIME_MIN_EV_MULT: dict[str, float] = {
    "TREND_EXPANSION":        1.5,   # mercado favorável → mais seletivo
    "VOLATILITY_COMPRESSION": 1.0,
    "TREND_EXHAUSTION":       0.5,
    "MEAN_REVERTING_CHOP":    0.2,   # mercado lateral → mais permissivo
    "HIGH_CORRELATION_RISK":  0.3,
    "BEAR_TREND":             0.0,
    "PANIC_LIQUIDATION":      0.0,
}

# Threshold de momentum 1H para _direction (quão forte deve ser o move)
# 1H move esperado é maior — thresholds dobrados vs 30m
# EXPANSION: 0.2% confirma tendência | CHOP: exige move de 0.5%+
REGIME_DIRECTION_THRESH: dict[str, float] = {
    "TREND_EXPANSION":        0.002,   # 0.2% — sensível em tendência clara
    "VOLATILITY_COMPRESSION": 0.003,
    "TREND_EXHAUSTION":       0.004,   # 0.4% — padrão 1H
    "MEAN_REVERTING_CHOP":    0.005,   # 0.5% — exige move mais forte em lateral
    "HIGH_CORRELATION_RISK":  0.007,   # 0.7% — muito seletivo em risco alto
    "BEAR_TREND":             0.02,
    "PANIC_LIQUIDATION":      0.02,
}


class MomentumStrategy(BaseStrategy):

    REGIME_THRESHOLDS = REGIME_THRESHOLDS
    ROUND_TRIP_FEE    = 0.005

    def __init__(self, symbols: list[str], strategy_id: str = "momentum_v2") -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._platt     = PlattCalibrator(coef_path=MODELS_DIR / "calibration_coef.json")
        self._sizing    = SizingEngine()
        self._edge_cond = EdgeConditioner()

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
        # Ciclo 1H: candles_30m não coletados — usamos somente 1H e 6H

        # ── Camada 1: Detecção de regime 1H ─────────────────
        regime_1h = self._detect_regime_1h(ctx)

        # ── Camada 3: Confirmação multi-timeframe ────────────
        regime = self._confirm_regime_mtf(ctx, regime_1h)

        threshold  = REGIME_THRESHOLDS.get(regime, 0.56)
        kelly_mult = REGIME_KELLY_MULT.get(regime, 0.5)

        # ── Phase 13: Meta Regime — modula threshold macro ───────────────────
        # Multiplica o threshold pelo modificador do regime macro (cross-asset).
        # RISK_ON ×0.92 → mais fácil entrar | RISK_OFF ×1.20 → muito difícil
        # Nunca bloqueia sozinho — apenas facilita/dificulta a passagem.
        meta_regime_data = (ctx.extra or {}).get("meta_regime") or {}
        meta_regime_name = meta_regime_data.get("regime", "UNKNOWN")
        # Phase 16: usa threshold_mult BLENDADO (softmax EMA) se disponível.
        # Fallback para threshold_mult_hard (legado) se distribuição não presente.
        meta_thr_mult = float(
            meta_regime_data.get("threshold_mult")          # blendado (novo)
            or meta_regime_data.get("threshold_mult_hard")  # hard label (legado)
            or 1.0
        )
        # Expõe distribuição de probabilidade para diagnóstico
        regime_distribution = meta_regime_data.get("regime_distribution", {})
        threshold = round(min(threshold * meta_thr_mult, 0.99), 4)

        # ── Phase 5: Adaptive Threshold — percentil de ATR 1H ────────────────
        # ATR alto (mercado agitado) → exige score maior → mult > 1.0
        # ATR baixo (mercado calmo)  → threshold levemente reduzido → mult < 1.0
        # Cap: [0.92, 1.12] — nunca bloqueia, nunca facilita demais.
        atr_mult, atr_pct_now = self._atr_threshold_mult(ctx)
        threshold = round(min(threshold * atr_mult, 0.99), 4)

        # ── Fase C: Edge Conditioning — gates de qualidade de edge ───────────
        # Avalia 4 condições: PSI drift, model health, liquidez, WR calibration.
        # Resultado: eleva threshold quando condições degradam.
        #            bloqueia APENAS em situações catastróficas (PSI>0.35, vol<25%, WR diff<-35%).
        # WR drift graduado: -0.15 a -0.35 → micro-trades (sizing reduzido) ao invés de bloqueio.
        # Isso evita deadlock onde ausência de trades impede recuperação do WR live.
        # Dados vêm de ctx.extra["model_health"] + candles_1h (já disponíveis).
        edge = self._edge_cond.evaluate(
            candles_1h=ctx.candles_1h,
            model_health_data=(ctx.extra or {}).get("model_health"),
        )

        # ── Score e fatores — calculados SEMPRE (mesmo em regime bloqueado) ──
        # Garantia: o dashboard sempre exibe M1-M9 com valores reais,
        # independente do resultado final. Permite diagnóstico contínuo.
        score, factors = self._score_signal(ctx, regime)
        # Adiciona moduladores ao factors para diagnóstico no dashboard
        factors["meta_thr_mult"]      = round(meta_thr_mult, 3)
        factors["atr_thr_mult"]       = round(atr_mult, 3)
        factors["atr_pct_now"]        = round(atr_pct_now, 3)
        factors["meta_regime_name"]   = meta_regime_name
        # Phase 16: distribuição probabilística de regime (para dashboard)
        for rname, rprob in regime_distribution.items():
            factors[f"mr_{rname.lower()}"] = round(rprob, 4)
        factors.update(edge.to_factors())
        calibrated = self._calibrate(score)

        # ── Phase 18: Alpha Orthogonality ────────────────────────────────────
        # 4 sinais ortogonais ao M1-M9 (FD, LV, OD, MRM).
        # Calculados SEMPRE — mesmo em regime bloqueado (para diagnóstico).
        # Somente aplicados ao kelly/threshold se o sinal passar todos os filtros.
        spread_pct_now = float((ctx.extra or {}).get("spread_pct", 0.0))
        alpha_result = alpha_orthogonality.evaluate(
            candles_1h=ctx.candles_1h,
            spread_pct=spread_pct_now,
            futures_flow=(ctx.extra or {}).get("futures_flow") or {},
            regime=regime,
        )
        factors.update(alpha_result.to_factors())

        # Aplica edge threshold mult (eleva a barra quando condições degradam)
        threshold = round(min(threshold * edge.threshold_mult, 0.99), 4)

        # Aplica alpha threshold adjustment (aditivo, clampado em [0.44, 0.99])
        threshold = round(min(max(threshold + alpha_result.threshold_adj, 0.44), 0.99), 4)

        if regime in BLOCKED_REGIMES:
            _log("REGIME_BLOCKED",
                 f"Regime bloqueado: {regime} | meta={meta_regime_name}",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, factors=factors)
            return None

        # Edge gate hard block (PSI extremo, liquidez seca, WR divergência extrema)
        if edge.should_block:
            _log("GATE_CLOSED",
                 f"Edge gate: {edge.block_reason}",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, factors=factors)
            return None

        if score < threshold:
            # Calcula direção tentativa mesmo em SCORE_LOW (só para info no log)
            try:
                _dir_hint = self._direction(ctx, regime)
            except Exception:
                _dir_hint = "N/A"
            _log("SCORE_LOW",
                 f"Score {score:.3f} < thr {threshold:.3f} [{regime}]",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, direction=_dir_hint, factors=factors)
            return None

        # ── Filtro 3: EV dinâmico por regime ─────────────────
        ev     = self._expected_value(calibrated, regime)
        ev_mult = REGIME_MIN_EV_MULT.get(regime, 0.5)
        min_ev = ev_mult * self.ROUND_TRIP_FEE
        if ev < min_ev:
            _log("EV_LOW",
                 f"EV {ev:.3f} < min {min_ev:.3f}",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, ev=ev, factors=factors)
            return None

        # ── Filtro 4: Direção dinâmica por regime ────────────
        direction = self._direction(ctx, regime)
        if direction == SignalDirection.FLAT:
            _log("DIRECTION_FLAT", "Preço lateralizado",
                 regime=regime, score=score, calibrated=calibrated,
                 threshold=threshold, ev=ev, direction="FLAT", factors=factors)
            return None

        # ── Camada 2: Kelly composto (Position Sizing Dinâmico) ─────────────
        # base_kelly × regime_mult × drift_mult × vol_state_mult
        #            × calibration_mult × score_mult × ec_sizing_mult
        # Cada dimensão modula o sizing de forma independente.
        # ec_sizing_mult: reduz kelly quando WR drift está em zona de micro-trade.
        # Size sobe quando edge sobe, cai agressivamente quando degrada.
        base_kelly = min(calibrated * 0.25, 0.15)
        sizing     = self._sizing.compute(
            base_kelly=base_kelly,
            regime_mult=kelly_mult,
            calibrated_score=calibrated,
            vol_state_data=(ctx.extra or {}).get("vol_state"),
            model_health_data=(ctx.extra or {}).get("model_health"),
        )
        # Aplica edge conditioning sizing multiplier (anti-deadlock WR gate)
        kelly = round(max(sizing.final_kelly * edge.sizing_mult, 0.01), 4) \
            if edge.sizing_mult < 0.99 else sizing.final_kelly
        dir_str = "LONG"
        # Adiciona breakdown do sizing aos factors para rastreabilidade no dashboard
        factors.update(sizing.to_factors())

        # Phase 18 — Alpha Orthogonality kelly boost (após todos os outros mults)
        if abs(alpha_result.kelly_boost - 1.0) >= 0.01:
            kelly_pre = kelly
            kelly = round(min(max(kelly * alpha_result.kelly_boost, 0.01), 0.20), 4)
            logger.info(
                "AlphaOrtho %s: kelly %.1f%% → %.1f%% (boost×%.2f active=%s)",
                symbol, kelly_pre * 100, kelly * 100,
                alpha_result.kelly_boost, alpha_result.active,
            )

        ao_tag = f" [AO boost×{alpha_result.kelly_boost:.2f} active={alpha_result.active}]" \
            if alpha_result.active else ""
        micro_tag = f" [MICRO-TRADE ec_sizing={edge.sizing_mult:.0%}]" \
            if edge.sizing_mult < 0.99 else ""
        _log("SIGNAL",
             f"BUY {regime} score={score:.3f} prob={calibrated:.3f} "
             f"kelly={kelly:.1%} [{sizing.summary()}]{micro_tag}{ao_tag}",
             regime=regime, score=score, calibrated=calibrated,
             threshold=threshold, ev=ev, direction=dir_str, factors=factors)

        logger.info(
            "SIGNAL %s %s regime=%s score=%.3f prob=%.3f "
            "EV=%.3f kelly=%.1f%%%s | drift=%.2f vol=%s calib=%.2f score_m=%.2f",
            dir_str, symbol, regime, score, calibrated, ev,
            kelly * 100, micro_tag,
            sizing.drift_mult, sizing.vol_state, sizing.calibration_mult, sizing.score_mult,
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
        Confirma regime 1H com contexto 6H — ciclo 1H.

        Em ciclo 1H o 6H representa 6 candles de avaliação (mais rigoroso).
        Regras para 1H (menos ruído, maior confiança na confirmação):
          - 6H BULL + 1H BULL  → confirma (upgrade possível)
          - 6H BULL + 1H CHOP  → VOLATILITY_COMPRESSION (upgrade leve)
          - 6H BEAR + 1H BULL  → downgrade para CHOP (conflito → cautela)
          - 6H BEAR + 1H CHOP  → HIGH_CORRELATION_RISK (penaliza sizing)
          - 6H BEAR + 1H BEAR  → BEAR_TREND (bloqueio confirmado por ambos)
          - Sem 6H             → usa só 1H
        """
        if not ctx.candles_6h or len(ctx.candles_6h) < 5:
            return regime_1h

        closes_6h = [c.close for c in ctx.candles_6h[:10]]
        sma_fast_6h = sum(closes_6h[:3]) / 3
        sma_slow_6h = sum(closes_6h[:10]) / 10

        # Tendência 6H — exige queda de 3% para confirmar BEAR (mais rigoroso)
        if sma_fast_6h > sma_slow_6h:
            trend_6h = "BULL"
        elif len(closes_6h) >= 5 and closes_6h[4] > 0:
            decline_6h = (closes_6h[0] - closes_6h[4]) / closes_6h[4]
            trend_6h = "BEAR" if decline_6h < -0.03 else "CHOP"
        else:
            trend_6h = "CHOP"

        family_1h = (
            "BULL" if regime_1h in {"TREND_EXPANSION", "VOLATILITY_COMPRESSION", "TREND_EXHAUSTION"}
            else "BEAR" if regime_1h in {"BEAR_TREND", "PANIC_LIQUIDATION"}
            else "CHOP"
        )

        # 6H BEAR + 1H BEAR → bloqueio total (ambos confirmam downtrend)
        if family_1h == "BEAR" and trend_6h == "BEAR":
            logger.debug("MTF: 1H=BEAR + 6H=BEAR → BEAR_TREND confirmado")
            return "BEAR_TREND"

        # 6H BEAR + 1H BULL → downgrade para CHOP (conflito entre timeframes)
        if family_1h == "BULL" and trend_6h == "BEAR":
            logger.debug("MTF: 1H=%s (BULL) conflita 6H BEAR → CHOP (downgrade)", regime_1h)
            return "MEAN_REVERTING_CHOP"

        # 6H BEAR + 1H CHOP → HIGH_CORRELATION_RISK (penaliza sizing)
        if family_1h == "CHOP" and trend_6h == "BEAR":
            logger.debug("MTF: 1H=CHOP + 6H=BEAR → HIGH_CORRELATION_RISK (não bloqueia)")
            return "HIGH_CORRELATION_RISK"

        # 6H BULL + 1H CHOP → consolidação antes de subida
        if family_1h == "CHOP" and trend_6h == "BULL":
            logger.debug("MTF: 1H=CHOP + 6H=BULL → VOLATILITY_COMPRESSION")
            return "VOLATILITY_COMPRESSION"

        return regime_1h

    # ── Scoring v2 (5 fatores, scoring contínuo) ─────────────────────────────

    def _score_signal(self, ctx: StrategyContext, regime: str) -> tuple[float, dict]:
        """
        Modelo de scoring com 6 fatores contínuos — ciclo 1H.
        Todos os fatores usam candles 1H (sem granularidade 30m).

        Fatores (pesos v2.5.0 — M9 adicionado 2026-05-23):
          M1 Adaptive Momentum  ( 2%): correlação Spearman=-0.048 → peso residual
          M2 Trend Consistency  (11%): % candles bullish + higher-highs/lows (1H)
          M3 Volume Confirmation(33%): único fator com edge claro (Spearman=+0.229)
          M4 Regime Strength    ( 0%): correlação Spearman=-0.057 → removido
          M5 Candle Structure   ( 6%): close no terço superior do range (1H)
          M6 Futures Flow       (12%): funding rate + OI change (perp market signal)
          M7 Relative Strength  (11%): RS vs BTC multi-horizonte + BTC leadership
          M8 Volatility State   (19%): state machine 5-estados (EXPANDING/TREND/...)
          M9 News Sentiment     ( 6%): Fear&Greed Index + CoinGecko social sentiment
        """
        # Candles 1H — única granularidade em ciclo 1H
        closes  = [c.close  for c in ctx.candles_1h[:21]]
        highs   = [c.high   for c in ctx.candles_1h[:10]]
        lows    = [c.low    for c in ctx.candles_1h[:10]]
        opens   = [c.open   for c in ctx.candles_1h[:10]]
        vols_1h = [c.volume for c in ctx.candles_1h[:20]]

        # ── M1: Adaptive Momentum (25%) — retornos 1H multi-horizonte ─
        atr_20 = (
            sum(highs[i] - lows[i] for i in range(min(10, len(highs)))) / min(10, len(highs))
            if highs else closes[0] * 0.01
        )
        norm   = max(atr_20 * 2, closes[0] * 0.005)

        # Horizonte curto (1h, 5h) — captura momentum recente
        r1  = (closes[0] - closes[1])  / closes[1]  if len(closes) > 1  and closes[1]  > 0 else 0
        r5  = (closes[0] - closes[5])  / closes[5]  if len(closes) > 5  and closes[5]  > 0 else 0
        # Horizonte médio (10h, 20h) — tendência estabelecida
        r10 = (closes[0] - closes[10]) / closes[10] if len(closes) > 10 and closes[10] > 0 else 0
        r20 = (closes[0] - closes[20]) / closes[20] if len(closes) > 20 and closes[20] > 0 else 0

        # Blend: peso maior no curto prazo (mais reativo) sem ignorar médio prazo
        momentum_weighted = r1 * 0.30 + r5 * 0.30 + r10 * 0.25 + r20 * 0.15
        m1 = min(max((momentum_weighted / (norm / closes[0])) * 0.5 + 0.5, 0.0), 1.0)

        # ── M2: Trend Consistency (25%) — candles 1H ──────────
        # Bullish count nos últimos 6 candles 1H (= 6h de histórico)
        n = min(6, len(closes) - 1)
        bullish_count = sum(1 for i in range(n) if closes[i] > opens[i])
        pct_bullish   = bullish_count / n if n > 0 else 0.5

        hh_count  = sum(1 for i in range(min(4, len(highs)-1)) if highs[i] > highs[i+1])
        hl_count  = sum(1 for i in range(min(4, len(lows)-1))  if lows[i]  > lows[i+1])
        structure = (hh_count + hl_count) / 8

        m2 = pct_bullish * 0.5 + structure * 0.5

        # ── M3: Volume Confirmation (20%) — volumes 1H ────────
        vols = vols_1h
        avg_vol_5  = sum(vols[:5])  / 5  if len(vols) >= 5  else vols[0] if vols else 1
        avg_vol_20 = sum(vols[:20]) / 20 if len(vols) >= 20 else avg_vol_5

        vol_ratio = min(vols[0] / avg_vol_5, 3.0) / 3.0 if avg_vol_5 > 0 else 0.5
        vol_trend = (
            (sum(vols[:3]) / sum(vols[3:6])) if len(vols) >= 6 and sum(vols[3:6]) > 0 else 1.0
        )
        vol_trend = min(max(vol_trend, 0.3), 2.0)
        vol_trend_score = (vol_trend - 0.3) / 1.7

        # Confirmação direcional com candle 1H mais recente
        candle_confirm = 1.0 if (closes[0] > opens[0] and vols[0] > avg_vol_20) else 0.4

        m3 = vol_ratio * 0.4 + vol_trend_score * 0.3 + candle_confirm * 0.3

        # ── M4: Regime Strength (20%) — SMA macro 1H ──────────
        sma5  = sum(closes[:5])  / 5
        sma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else sma5
        sma_distance = (sma5 - sma20) / sma20 if sma20 > 0 else 0
        m4_raw    = min(max((sma_distance + 0.02) / 0.04, 0.0), 1.0)
        m4_regime = REGIME_M4.get(regime, 0.45)
        m4 = m4_raw * 0.6 + m4_regime * 0.4

        # ── M5: Candle Structure (10%) — 3 candles 1H recentes ─
        # Close no terço superior do range dos 3 candles 1H mais recentes
        m5_highs  = highs[:3]
        m5_lows   = lows[:3]
        m5_closes = closes[:3]
        candle_scores = []
        for i in range(min(3, len(m5_closes))):
            rng = m5_highs[i] - m5_lows[i] if i < len(m5_highs) else 0
            if rng > 0:
                pos = (m5_closes[i] - m5_lows[i]) / rng
                candle_scores.append(pos)
        m5 = sum(candle_scores) / len(candle_scores) if candle_scores else 0.5

        # ── M6: Futures Flow (10%) — funding rate + OI (Phase 10) ────────────
        ff_data   = (ctx.extra or {}).get("futures_flow") or {}
        ff_scores = ff_data.get("scores", {})
        m6 = float(ff_scores.get("m6", 0.5))

        # ── M7: Relative Strength (9%) — RS vs BTC + leadership (Phase 11) ──
        rs_data   = (ctx.extra or {}).get("relative_strength") or {}
        rs_scores = rs_data.get("scores", {})
        m7 = float(rs_scores.get("m7", 0.5))

        # ── M8: Volatility State (13%) — state machine (Phase 12) ────────────
        # Lê do ctx.extra injetado pelo StrategyRunner (VolatilityStateCollector).
        # Fallback neutro (0.5) se dados indisponíveis — não bloqueia o trading.
        vol_data = (ctx.extra or {}).get("vol_state") or {}
        m8 = float(vol_data.get("m8_score", 0.5))

        # ── M9: News Sentiment (Phase 4) — Fear&Greed + CoinGecko ────────────
        # Combina Fear & Greed Index global + sentimento CoinGecko por moeda.
        # Fallback neutro (0.5) se API indisponível — não bloqueia o trading.
        news_data = (ctx.extra or {}).get("news_sentiment") or {}
        m9 = float(news_data.get("m9_score", 0.5))

        # ── Score final — pesos v2.5.0 (M9 adicionado 2026-05-23) ─────────────
        # M1  2% (Spearman=-0.048, residual)
        # M2 11% (era 12% → -1%: espaço para M9)
        # M3 33% (era 35% → -2%: continua dominante, cede espaço para M9)
        # M4  0% (removido)
        # M5  6% (mantido)
        # M6 12% (era 13% → -1%: pequeno ajuste)
        # M7 11% (era 12% → -1%: pequeno ajuste)
        # M8 19% (era 20% → -1%: pequeno ajuste)
        # M9  6% (NOVO: Fear&Greed + CoinGecko sentiment)
        # Soma: 2+11+33+0+6+12+11+19+6 = 100% ✓
        score = (m1 * 0.02 + m2 * 0.11 + m3 * 0.33 +
                 m5 * 0.06 + m6 * 0.12 +
                 m7 * 0.11 + m8 * 0.19 + m9 * 0.06)
        score = round(min(max(score, 0.0), 1.0), 4)

        factors = {
            "m1_momentum":    round(m1, 3),
            "m2_consistency": round(m2, 3),
            "m3_volume":      round(m3, 3),
            "m4_regime_str":  round(m4, 3),
            "m5_candle":      round(m5, 3),
            "m6_futures":     round(m6, 3),
            "m7_rel_strength":round(m7, 3),
            "m8_vol_state":   round(m8, 3),
            "m9_sentiment":   round(m9, 3),
            # Sub-scores M9
            "m9_fng":         round(float(news_data.get("fng",           50.0)) / 100, 3),
            "m9_coin_sent":   round(float(news_data.get("coin_sentiment", 0.5)), 3),
            "m9_fng_class":   str(news_data.get("fng_class", "Neutral")),
            # Sub-scores M6
            "m6_funding":     round(float(ff_scores.get("funding",       0.5)), 3),
            "m6_oi_change":   round(float(ff_scores.get("oi_change",     0.5)), 3),
            "m6_fr_trend":    round(float(ff_scores.get("funding_trend", 0.5)), 3),
            # Sub-scores M7
            "m7_rs_1h":       round(float(rs_scores.get("rs_1h",      0.5)), 3),
            "m7_leadership":  round(float(rs_scores.get("leadership",  0.5)), 3),
            "m7_rs_trend":    round(float(rs_scores.get("rs_trend",    0.5)), 3),
            # Sub-scores M8 — apenas numéricos (strings não são aceitas em factors)
            "m8_atr_pct":     round(float(vol_data.get("metrics", {}).get("atr_pct",        0)), 3),
            "m8_dir_consist": round(
                float(vol_data.get("metrics", {}).get("dir_consistency", 0.5)), 3
            ),
        }
        # Registra features no drift monitor (nunca bloqueia o trading)
        try:
            governance.record_live(ctx.symbol, factors)
        except Exception as _gov_exc:
            logger.warning("governance.record_live falhou: %s", _gov_exc)
        return score, factors

    # ── Phase 5: Adaptive Threshold por ATR percentil ────────────────────────

    def _atr_threshold_mult(
        self, ctx: StrategyContext, lookback: int = 50
    ) -> tuple[float, float]:
        """
        Calcula multiplicador de threshold baseado no percentil do ATR atual
        em relação ao histórico recente (últimos `lookback` candles 1H).

        Lógica:
          - Computa ATR de cada candle: high - low (True Range simplificado)
          - Percentil do ATR atual dentro dos últimos `lookback` ATRs
          - Mapeia percentil → multiplicador [0.92, 1.12]:
              p0-p20  (ATR baixo / mercado calmo)     → ×0.92 a ×0.97  (facilita levemente)
              p20-p50 (ATR normal)                     → ×0.97 a ×1.00  (neutro)
              p50-p80 (ATR moderado)                   → ×1.00 a ×1.06  (eleva levemente)
              p80-p100 (ATR alto / mercado agitado)    → ×1.06 a ×1.12  (eleva mais)

        Retorna:
            (mult, atr_percentile_0_to_1)
        """
        candles = ctx.candles_1h
        n = min(lookback, len(candles))
        if n < 5:
            return 1.0, 0.5   # sem dados suficientes — neutro

        # ATR = high - low (simplificado, suficiente para percentil relativo)
        atrs = [c.high - c.low for c in candles[:n] if c.high > c.low]
        if not atrs:
            return 1.0, 0.5

        atr_now = atrs[0]   # ATR do candle mais recente
        sorted_atrs = sorted(atrs)
        rank = sum(1 for a in sorted_atrs if a <= atr_now)
        pct  = rank / len(sorted_atrs)   # percentil [0, 1]

        # Interpolação linear por faixa
        if pct <= 0.20:
            # Baixa volatilidade: [0, 0.20] → [0.92, 0.97]
            mult = 0.92 + (pct / 0.20) * 0.05
        elif pct <= 0.50:
            # Normal: [0.20, 0.50] → [0.97, 1.00]
            mult = 0.97 + ((pct - 0.20) / 0.30) * 0.03
        elif pct <= 0.80:
            # Moderado: [0.50, 0.80] → [1.00, 1.06]
            mult = 1.00 + ((pct - 0.50) / 0.30) * 0.06
        else:
            # Alta volatilidade: [0.80, 1.00] → [1.06, 1.12]
            mult = 1.06 + ((pct - 0.80) / 0.20) * 0.06

        return round(mult, 4), round(pct, 4)

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

    def _direction(self, ctx: StrategyContext, regime: str = "") -> SignalDirection:
        """
        Detecção de DIP + RECUPERAÇÃO dentro de uptrend.

        Princípio: "comprar fraqueza dentro de força — nunca comprar força".

        3 condições obrigatórias (todas devem ser verdade):
          a) Dip real: preço caiu ≥ 0.3% do máximo das últimas 4 horas
          b) Uptrend intacto: preço atual acima da SMA20 1H
          c) Recuperação iniciada: último candle 1H fechou acima da abertura

        Se não há dip detectado → FLAT (aguarda oportunidade).
        """
        if len(ctx.candles_1h) < 6:
            return SignalDirection.FLAT

        closes = [c.close for c in ctx.candles_1h[:21]]
        opens  = [c.open  for c in ctx.candles_1h[:6]]

        current = closes[0]

        # ── Condição B: uptrend intacto (SMA20 1H) ──────────
        sma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else closes[-1]
        if current < sma20 * 0.998:   # tolerância 0.2% abaixo da SMA20
            return SignalDirection.FLAT

        # ── Condição A: dip real nas últimas 4 horas ────────
        recent_high = max(closes[1:5])   # máximo dos últimos 4 candles
        recent_low  = min(closes[1:5])   # mínimo dos últimos 4 candles
        dip_pct = (recent_high - recent_low) / recent_high if recent_high > 0 else 0

        if dip_pct < 0.003:   # dip mínimo de 0.3%
            return SignalDirection.FLAT

        # ── Condição C: recuperação iniciada ────────────────
        # Último candle 1H fechou acima da abertura (vela verde)
        last_candle_bullish = closes[0] > opens[0]
        if not last_candle_bullish:
            return SignalDirection.FLAT

        # ── Todas as condições OK → DIP + RECUPERAÇÃO ───────
        return SignalDirection.LONG
