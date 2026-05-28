"""
V5 Momentum Strategy — BULL / CHOP / BEAR aware.

Changelog:
  v3.0.0 (2026-05-27) — Reforma Quant: Edge First
    • Desativados M1, M2, M5 (permutation importance negativa no IS)
    • M1 reformulado para RSI-based (diagnóstico, peso=0%)
    • M3 reformulado para volume direcional (buyer vs seller pressure)
    • Regime detection: SMA substituído por ADX(10) + DI± (menos lag)
    • Novos pesos: M3=40%, M6=22%, M7=18%, M9=14%, M8=6% = 100%

Camadas de proteção:
  1. ADX(10) para detecção de tendência (ADX≥25=trend, <25=range)
  2. DI+/DI- para direcionalidade dentro de tendência
  3. Sizing adaptativo por regime (Kelly multiplier)
  4. Confirmação multi-timeframe 1H + 6H

Regimes e comportamento (v3.0.0 — 2026-05-27):
  TREND_EXPANSION      → BULL  : threshold 0.62, Kelly 100%, SL 1.5×ATR, TP 4.5×ATR
  VOLATILITY_COMPRESSION       : threshold 0.64, Kelly 80%,  SL 1.0×ATR, TP 3.5×ATR
  MEAN_REVERTING_CHOP  → CHOP  : threshold 0.72, Kelly 50%,  SL 1.0×ATR, TP 2.5×ATR
  TREND_EXHAUSTION             : BLOQUEADO (threshold 0.99)
  HIGH_CORRELATION_RISK        : BLOQUEADO (threshold 0.99)
  BEAR_TREND           → BEAR  : BLOQUEADO para novas entradas
  PANIC_LIQUIDATION            : BLOQUEADO + saída imediata
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from ...core.config import OKX_ROUND_TRIP_FEE
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
    "TREND_EXPANSION":        0.62,   # era 0.56 → +0.06 (WR=33% exige seletividade maior)
    "VOLATILITY_COMPRESSION": 0.64,   # era 0.58 → +0.06 (confirmação mais exigente)
    "MEAN_REVERTING_CHOP":    0.72,   # era 0.68 → +0.04 (apenas sinais de alta confiança em lateral)
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
    ROUND_TRIP_FEE    = OKX_ROUND_TRIP_FEE   # 0.25% — entrada taker + saída maker

    def __init__(self, symbols: list[str], strategy_id: str = "momentum_v2") -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._platt     = PlattCalibrator(coef_path=MODELS_DIR / "calibration_coef.json")
        self._sizing    = SizingEngine()
        self._edge_cond = EdgeConditioner()

    @property
    def is_platt_using_defaults(self) -> bool:
        return self._platt.is_using_defaults

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
            # BEAR_TREND: avalia oportunidade de short via swap perp
            if regime == "BEAR_TREND":
                short_signal = self._evaluate_short_opportunity(
                    ctx=ctx, regime=regime, score=score,
                    calibrated=calibrated, factors=factors, ts=ts,
                )
                if short_signal is not None:
                    return short_signal
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
        # NOTA(🟡 Kelly): base_kelly = calibrated * 0.25 é um Fractional Kelly heurístico,
        # não o Full Kelly. Full Kelly = (p*b - q) / b onde p=win_rate, b=avg_win/avg_loss.
        # Aqui `calibrated` é a probabilidade de ganho (p) estimada pelo Platt calibrator,
        # e o fator 0.25 é a fração de segurança (Quarter-Kelly), aceita pela indústria.
        # O cap de 0.15 (15%) é o teto institucional padrão para crypto.
        # Não é um bug — é design deliberado mais conservador que o Full Kelly.
        # O SizingEngine depois modula este base_kelly com 5 dimensões adicionais.
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

    # ── Camada 1: Detecção de regime 1H — ADX-based (v3.0.0) ────────────────

    @staticmethod
    def _calc_adx(
        highs: list[float], lows: list[float], closes: list[float], period: int = 10
    ) -> tuple[float, float, float]:
        """
        Wilder's ADX simplificado.

        Inputs em ordem reversa (candles[0]=mais recente) — inverte internamente.
        Retorna (adx, di_plus, di_minus).
        ADX ≥ 25 → mercado em tendência.
        DI+ > DI- → tendência de alta; DI- > DI+ → tendência de baixa.

        Com dados insuficientes retorna (20.0, 0.5, 0.5):
          ADX=20 = zona cinza (neutro / range).
        """
        n = min(len(highs), len(lows), len(closes))
        if n < period + 2:
            return 20.0, 0.5, 0.5

        # Cronológico (mais antigo primeiro)
        h = list(reversed(highs[:n]))
        lo = list(reversed(lows[:n]))
        c = list(reversed(closes[:n]))

        trs, dmp, dmm = [], [], []
        for i in range(1, n):
            tr   = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            up   = h[i] - h[i - 1]
            down = lo[i - 1] - lo[i]
            dp   = up   if (up > down   and up   > 0) else 0.0
            dm   = down if (down > up   and down > 0) else 0.0
            trs.append(tr)
            dmp.append(dp)
            dmm.append(dm)

        if len(trs) < period:
            return 20.0, 0.5, 0.5

        # Wilder smoothing: seed = média dos primeiros `period` valores
        def _ws(data: list[float]) -> float:
            s = sum(data[:period]) / period
            for v in data[period:]:
                s = (s * (period - 1) + v) / period
            return s

        atr_w = _ws(trs)
        dp_w  = _ws(dmp)
        dm_w  = _ws(dmm)

        if atr_w <= 0:
            return 20.0, 0.5, 0.5

        di_p = 100.0 * dp_w / atr_w
        di_m = 100.0 * dm_w / atr_w
        dx   = 100.0 * abs(di_p - di_m) / (di_p + di_m) if (di_p + di_m) > 0 else 0.0

        return round(dx, 2), round(di_p, 2), round(di_m, 2)

    def _detect_regime_1h(self, ctx: StrategyContext) -> str:
        """
        Detecção de regime baseada em ADX(10) em vez de SMA cruzada.

        ADX mede FORÇA da tendência sem lag de média móvel:
          ADX ≥ 25 + DI+ > DI- → tendência de alta  → EXPANSION / COMPRESSION
          ADX ≥ 25 + DI- > DI+ → tendência de baixa → BEAR_TREND
          ADX < 25              → mercado lateral    → CHOP / HIGH_CORR

        Panic override: queda > 5% em 1 candle ignora ADX.
        """
        closes  = [c.close  for c in ctx.candles_1h[:21]]
        volumes = [c.volume for c in ctx.candles_1h[:21]]
        highs   = [c.high   for c in ctx.candles_1h[:21]]
        lows    = [c.low    for c in ctx.candles_1h[:21]]

        if len(closes) < 12:
            return "MEAN_REVERTING_CHOP"

        # ── Panic: queda brusca > 5% num único candle ────────────────────────
        if closes[1] > 0 and (closes[0] - closes[1]) / closes[1] < -0.05:
            return "PANIC_LIQUIDATION"

        # ── ADX(10) ──────────────────────────────────────────────────────────
        adx, di_plus, di_minus = self._calc_adx(highs, lows, closes, period=10)

        # ── BEAR: declínio confirmado por ADX + DI- dominante ────────────────
        if len(closes) >= 11 and closes[10] > 0:
            decline = (closes[0] - closes[10]) / closes[10]
            if decline < -0.02 and di_minus > di_plus:
                return "BEAR_TREND"

        # ── Tendência forte (ADX ≥ 25) ───────────────────────────────────────
        if adx >= 25:
            if di_plus >= di_minus:
                # Tendência de alta — volume confirma intensidade
                avg_vol  = sum(volumes) / len(volumes)
                last_vol = volumes[0]
                if last_vol > avg_vol * 1.15:
                    return "TREND_EXPANSION"
                if last_vol < avg_vol * 0.65:
                    return "TREND_EXHAUSTION"
                return "VOLATILITY_COMPRESSION"
            else:
                # DI- dominante em tendência forte → BEAR
                return "BEAR_TREND"

        # ── Mercado lateral (ADX < 25) ───────────────────────────────────────
        # Volatilidade extrema → risco alto
        atr_5   = sum(hi - lo for hi, lo in zip(highs[:5], lows[:5], strict=True)) / 5
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

    # ── Scoring v3 (reforma quant 2026-05-27) ────────────────────────────────

    def _score_signal(self, ctx: StrategyContext, regime: str) -> tuple[float, dict]:
        """
        Modelo de scoring com M1-M9 — ciclo 1H.

        Reforma v3.0.0 (quant specialist 2026-05-27):
          DESATIVADOS (peso=0%): M1 perm_imp=-0.0013 | M2 perm_imp=-0.0033
                                  M4 perm_imp=0.0 | M5 perm_imp=-0.001
          MANTIDOS:  M8=6% (estrutural, vol regime)
          ELEVADOS:  M3=40% (directional vol, único edge real)
                     M6=22% (Spearman=0.52 live) | M7=18% | M9=14%

        M1 reformulado para RSI(14) — mais robusto que retornos brutos.
        M3 reformulado para pressão de compra direcional (buy/sell vol ratio).
        """
        # Candles 1H — única granularidade em ciclo 1H
        closes  = [c.close  for c in ctx.candles_1h[:21]]
        highs   = [c.high   for c in ctx.candles_1h[:10]]
        lows    = [c.low    for c in ctx.candles_1h[:10]]
        opens   = [c.open   for c in ctx.candles_1h[:21]]
        vols_1h = [c.volume for c in ctx.candles_1h[:20]]

        # ── M1: RSI Momentum — diagnóstico (peso=0%) ──────────────────────────
        # RSI(14) 1H — substitui retornos brutos (menos ruído, escala consistente).
        # Zona ótima de compra: RSI 40-65 (momentum sem excesso).
        # NOTA: peso=0% → não entra no score mas aparece no dashboard para diagnóstico.
        n_rsi = min(16, len(closes))
        if n_rsi >= 15:
            gains  = [max(closes[i] - closes[i + 1], 0.0) for i in range(n_rsi - 1)]
            losses = [max(closes[i + 1] - closes[i], 0.0) for i in range(n_rsi - 1)]
            avg_g  = sum(gains[:14])  / 14
            avg_l  = sum(losses[:14]) / 14
            if avg_l > 0:
                rsi = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
            else:
                rsi = 100.0 if avg_g > 0 else 50.0
            # Mapeia RSI → [0,1]: zona ótima 40-65, penaliza >70
            if rsi < 20:
                m1 = 0.50   # oversold profundo — risco de queda livre
            elif rsi < 40:
                m1 = 0.50 + (rsi - 20) / 20 * 0.30   # 0.50-0.80
            elif rsi < 65:
                m1 = 0.80 + (rsi - 40) / 25 * 0.20   # 0.80-1.00 (zona ótima)
            elif rsi < 70:
                m1 = 0.80 - (rsi - 65) / 5  * 0.30   # 0.80-0.50 (aquecendo)
            else:
                m1 = max(0.10, 0.50 - (rsi - 70) / 30 * 0.40)  # overbought
        else:
            rsi = 50.0
            m1  = 0.50

        # ── M2: Trend Consistency — diagnóstico (peso=0%) ─────────────────────
        # Mantido para monitoramento, removido do score por perm_imp=-0.0033.
        n_m2 = min(6, len(closes) - 1)
        bullish_count = sum(1 for i in range(n_m2) if i < len(opens) and closes[i] > opens[i])
        pct_bullish   = bullish_count / n_m2 if n_m2 > 0 else 0.5
        hh_count      = sum(1 for i in range(min(4, len(highs) - 1)) if highs[i] > highs[i + 1])
        hl_count      = sum(1 for i in range(min(4, len(lows)  - 1)) if lows[i]  > lows[i + 1])
        m2            = pct_bullish * 0.5 + (hh_count + hl_count) / 8 * 0.5

        # ── M3: Directional Volume — PRINCIPAL EDGE (40%) ─────────────────────
        # Reformulado v3.0: volume direcional (buy vs sell pressure)
        # buyer_vol  = soma de vol em candles de alta (close > open)
        # seller_vol = soma de vol em candles de baixa (close < open)
        # directional_ratio: 0.0=tudo venda → 1.0=tudo compra
        n_dir = min(12, len(vols_1h), len(closes), len(opens))
        buy_vol  = sum(vols_1h[i] for i in range(n_dir) if closes[i] > opens[i])
        sell_vol = sum(vols_1h[i] for i in range(n_dir) if closes[i] < opens[i])
        total_dir = buy_vol + sell_vol
        directional_ratio = buy_vol / total_dir if total_dir > 0 else 0.5

        # Relative volume (vs média 20h) — confirma participação
        avg_vol_20 = sum(vols_1h[:20]) / 20 if len(vols_1h) >= 20 else (vols_1h[0] if vols_1h else 1)
        vol_ratio  = min(vols_1h[0] / avg_vol_20, 3.0) / 3.0 if avg_vol_20 > 0 else 0.5

        # Volume momentum (recente vs passado) — confirma aceleração
        vol_trend = (
            sum(vols_1h[:3]) / sum(vols_1h[3:6])
            if len(vols_1h) >= 6 and sum(vols_1h[3:6]) > 0 else 1.0
        )
        vol_trend = min(max(vol_trend, 0.3), 2.0)
        vol_trend_score = (vol_trend - 0.3) / 1.7

        # M3: 50% direcional + 30% volume relativo + 20% momentum de volume
        m3 = directional_ratio * 0.50 + vol_ratio * 0.30 + vol_trend_score * 0.20

        # ── M4: Regime Strength — diagnóstico (peso=0%) ───────────────────────
        sma5  = sum(closes[:5])  / 5
        sma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else sma5
        sma_distance = (sma5 - sma20) / sma20 if sma20 > 0 else 0
        m4_raw    = min(max((sma_distance + 0.02) / 0.04, 0.0), 1.0)
        m4_regime = REGIME_M4.get(regime, 0.45)
        m4        = m4_raw * 0.6 + m4_regime * 0.4

        # ── M5: Candle Structure — diagnóstico (peso=0%) ──────────────────────
        # Mantido para monitoramento, removido do score por perm_imp=-0.001.
        candle_scores = []
        for i in range(min(3, len(closes))):
            rng = highs[i] - lows[i] if i < len(highs) else 0
            if rng > 0:
                candle_scores.append((closes[i] - lows[i]) / rng)
        m5 = sum(candle_scores) / len(candle_scores) if candle_scores else 0.5

        # ── M6: Futures Flow (22%) — funding rate + OI (Phase 10) ────────────
        ff_data   = (ctx.extra or {}).get("futures_flow") or {}
        ff_scores = ff_data.get("scores", {})
        m6 = float(ff_scores.get("m6", 0.5))

        # ── M7: Relative Strength (18%) — RS vs BTC + leadership (Phase 11) ─
        rs_data   = (ctx.extra or {}).get("relative_strength") or {}
        rs_scores = rs_data.get("scores", {})
        m7 = float(rs_scores.get("m7", 0.5))

        # ── M8: Volatility State (6%) — state machine (Phase 12) ─────────────
        vol_data = (ctx.extra or {}).get("vol_state") or {}
        m8 = float(vol_data.get("m8_score", 0.5))

        # ── M9: News Sentiment (14%) — Fear&Greed + CoinGecko (Phase 4) ──────
        news_data = (ctx.extra or {}).get("news_sentiment") or {}
        m9 = float(news_data.get("m9_score", 0.5))

        # ── Features compostas (interações ortogonais) ───────────────────────
        # Permutation importance individual negativa não exclui interações.
        # mc1 = M1 × M2: RSI × consistência — captura "RSI alto em tendência firme"
        # mc2 = M4 × M3: regime_strength × volume — "volume confirma força do regime"
        # Normaliza ao range [0,1]: produto de dois [0,1] já está em [0,1]
        mc1 = round(m1 * m2, 4)           # RSI × consistency
        mc2 = round(m4 * m3, 4)           # regime_strength × volume

        # ── Score final — pesos v3.1.0 (composites adicionados 2026-05-28) ────
        # M1  0% individual (RSI momentum — perm_imp=-0.0013)
        # M2  0% individual (Trend Consistency — perm_imp=-0.0033)
        # M3 37% (Directional Volume — principal fonte de edge)
        # M4  0% individual (Regime Strength — perm_imp=0.0)
        # M5  0% (Candle Structure — perm_imp=-0.001)
        # M6 20% (Futures Flow — Spearman=0.52 live)
        # M7 17% (Relative Strength — Spearman=0.52 live)
        # M8  5% (Vol State — estrutural)
        # M9 13% (News Sentiment — Spearman=0.52 live)
        # mc1 4% (M1×M2 — interação RSI×consistency)
        # mc2 4% (M4×M3 — interação regime_strength×volume)
        # Soma: 37+20+17+5+13+4+4 = 100% ✓
        score = (
            m3 * 0.37 + m6 * 0.20 + m7 * 0.17
            + m8 * 0.05 + m9 * 0.13
            + mc1 * 0.04 + mc2 * 0.04
        )
        score = round(min(max(score, 0.0), 1.0), 4)

        factors = {
            "m1_momentum":    round(m1, 3),
            "m2_consistency": round(m2, 3),
            "m3_volume":      round(m3, 3),
            "m4_regime_str":  round(m4, 3),
            "m5_candle":      round(m5, 3),
            "mc1_rsi_cons":   round(mc1, 3),   # M1×M2 composta
            "mc2_reg_vol":    round(mc2, 3),   # M4×M3 composta
            "m6_futures":     round(m6, 3),
            "m7_rel_strength":round(m7, 3),
            "m8_vol_state":   round(m8, 3),
            "m9_sentiment":   round(m9, 3),
            # Sub-scores M1 — RSI diagnóstico
            "m1_rsi":         round(rsi, 1),
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
            # Sub-scores M8 — apenas numéricos
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

    # ── Short opportunity em BEAR_TREND ──────────────────────────────────────

    def _evaluate_short_opportunity(
        self,
        ctx:        "StrategyContext",
        regime:     str,
        score:      float,
        calibrated: float,
        factors:    dict,
        ts:         "datetime",
    ) -> "Signal | None":
        """
        Avalia se há oportunidade de short via swap perp em BEAR_TREND.

        Condições (mais exigentes que LONG — sem recuperação, sem dip):
          a) Downtrend forte: 3 candles 1H consecutivos com close < open
          b) 6H também bearish: close_6h[0] < SMA5_6H
          c) Score ≥ 0.45 (sinal presente mesmo em bear)
          d) DI- > DI+ confirmado (ADX bear direcional)

        Retorna Signal com direction=SHORT e strategy_id="momentum_v2" para
        roteamento pelo TradingLoop ao CrossAssetEngine (via swap perp).
        """
        if len(ctx.candles_1h) < 10 or len(ctx.candles_6h) < 6:
            return None

        closes_1h = [c.close for c in ctx.candles_1h[:6]]
        opens_1h  = [c.open  for c in ctx.candles_1h[:6]]
        closes_6h = [c.close for c in ctx.candles_6h[:6]]

        # Condição A: 3 velas 1H consecutivas bearish
        bearish_candles = sum(1 for c, o in zip(closes_1h[:3], opens_1h[:3]) if c < o)
        if bearish_candles < 3:
            return None

        # Condição B: 6H bearish (close abaixo da SMA5 dos últimos 5 candles 6H)
        sma5_6h = sum(closes_6h[1:6]) / 5 if len(closes_6h) >= 6 else closes_6h[-1]
        if closes_6h[0] >= sma5_6h * 0.999:
            return None

        # Condição C: score mínimo presente (indica tendência detectada)
        if score < 0.45:
            return None

        # Condição D: ADX bear direcional
        highs  = [c.high for c in ctx.candles_1h[:21]]
        lows   = [c.low  for c in ctx.candles_1h[:21]]
        _, di_plus, di_minus = self._calc_adx(highs, lows, closes_1h + [c.close for c in ctx.candles_1h[6:21]], period=10)
        if di_minus <= di_plus:
            return None

        bear_score = round(min(di_minus / (di_plus + di_minus + 1e-6), 1.0), 3)
        logger.info(
            "SHORT opportunity %s: BEAR_TREND confirmado 3×1H + 6H bearish | "
            "DI-=%.1f DI+=%.1f bear_strength=%.3f",
            ctx.symbol, di_minus, di_plus, bear_score,
        )

        from dataclasses import replace
        from ..base import StrategyContext as _SC  # noqa
        from ...core.models import Signal as _Signal

        kelly_short = 0.03   # tamanho conservador para shorts (3% do portfolio)
        short_factors = dict(factors)
        short_factors["bear_strength"]  = bear_score
        short_factors["di_minus"]       = round(di_minus, 2)
        short_factors["di_plus"]        = round(di_plus, 2)
        short_factors["short_via_swap"] = 1.0   # flag para roteamento no TradingLoop

        return _Signal(
            symbol=ctx.symbol,
            direction=SignalDirection.SHORT,
            strategy_id=self._strategy_id,
            score=score,
            calibrated_score=calibrated,
            kelly_fraction=kelly_short,
            regime=regime,
            factors=short_factors,
        )

    def _detect_bear_strength(self, ctx: "StrategyContext") -> float:
        """Força do sinal bear: proporção DI- vs DI+ em ADX(10)."""
        if len(ctx.candles_1h) < 21:
            return 0.0
        highs  = [c.high  for c in ctx.candles_1h[:21]]
        lows   = [c.low   for c in ctx.candles_1h[:21]]
        closes = [c.close for c in ctx.candles_1h[:21]]
        _, di_plus, di_minus = self._calc_adx(highs, lows, closes, period=10)
        denom = di_plus + di_minus
        return round(di_minus / denom, 3) if denom > 0 else 0.0
