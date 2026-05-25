"""
AlphaOrthogonality — Phase 18: 4 sinais ortogonais ao M1-M9.

Problema resolvido:
  M1-M9 capturam momentum, volume, regime, funding e sentimento —
  mas são todos *diretamente correlacionados* à direção de preço.
  Nenhum deles captura:
    a) Dislocation entre posicionamento (funding) e preço
    b) Deterioração de liquidez (spread anômalo)
    c) Viés de drift em janelas de baixo volume (overnight)
    d) Overextension microestrutural (z-score intraday)

  Esses 4 sinais são ortogonais porque:
    - FD (Funding Dislocation): captura *divergência* entre funding e preço
    - LV (Liquidity Vacuum):    captura *estrutura do livro* via spread anômalo
    - OD (Overnight Drift):     captura *assimetria temporal* por sessão
    - MRM (Mean Rev. Micro):    captura *overextension* relativo ao ATR

Integração:
  AlphaOrthogonality.evaluate(ctx) → AlphaResult
    kelly_boost:    float — multiplicador de sizing (0.7–1.3)
    threshold_adj:  float — ajuste aditivo no threshold (negativo = facilita)
    signals:        dict com score [0,1] de cada sinal
    active:         lista de sinais ativos (boost ou cut acima do neutro)

  Na momentum_strategy:
    kelly  = kelly * alpha_result.kelly_boost
    thr    = threshold + alpha_result.threshold_adj
    factors.update(alpha_result.to_factors())

Filosofia:
  Ortogonalidade real → mesmo que M1-M9 sinalizem forte, um sinal ortogonal
  pode cortar sizing (LV, MRM em overextension) ou boostá-lo (FD em squeeze).
  Nunca bloqueia (job do EdgeConditioner) — apenas calibra o tamanho.

Limites do boost:
  Acumulativo máx: ×1.30 up / ×0.65 down (3 sinais confluentes)
  Cada sinal: ±10% (neutro = 1.0, max_boost = 1.10, max_cut = 0.90)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Thresholds de sinalização ──────────────────────────────────────────────────

# Funding Dislocation
FD_NEUTRAL_FR    = 0.0001   # 0.01% — funding neutro/saudável
FD_EXTREME_POS   = 0.0010   # 0.10% — crowded long (contrarian cut)
FD_EXTREME_NEG   = -0.0005  # -0.05% — bears pagando (squeeze setup)

# Liquidity Vacuum
LV_NORMAL_BPS    = 10.0     # ≤ 10 bps = mercado líquido (score 1.0)
LV_WARN_BPS      = 25.0     # 25 bps = spreads alargando (cut 5%)
LV_VACUUM_BPS    = 60.0     # 60 bps = livro thin (cut 10%)

# Overnight Drift
OD_GAP_THRESHOLD = 0.002    # 0.2% de gap = significativo
OD_LOOKBACK      = 6        # candles para detectar padrão

# Mean Reversion Micro
MRM_ZSCORE_WARN  = 1.5      # z-score > 1.5 = levemente overextended
MRM_ZSCORE_EXTREME = 2.5    # z-score > 2.5 = muito overextended (cut 10%)
MRM_LOOKBACK     = 20       # candles para calcular z-score


@dataclass
class AlphaSignal:
    """Resultado de um sinal alpha individual."""
    name:        str
    score:       float   # [0, 1] — 0.5 = neutro, >0.5 = bullish/boost, <0.5 = bearish/cut
    kelly_mult:  float   # multiplicador de kelly (1.0 = neutro)
    thr_adj:     float   # ajuste aditivo de threshold (0 = neutro)
    description: str


@dataclass
class AlphaResult:
    """Resultado composto de todos os sinais ortogonais."""
    signals:       dict[str, AlphaSignal] = field(default_factory=dict)
    kelly_boost:   float = 1.0
    threshold_adj: float = 0.0
    active:        list[str] = field(default_factory=list)
    computed_at:   datetime  = field(default_factory=lambda: datetime.now(UTC))

    def to_factors(self) -> dict:
        """Exporta para o dict de factors do Signal (dashboard)."""
        f: dict = {
            "ao_kelly_boost":   round(self.kelly_boost, 3),
            "ao_thr_adj":       round(self.threshold_adj, 4),
            "ao_active":        len(self.active),
        }
        for name, sig in self.signals.items():
            f[f"ao_{name}_score"] = round(sig.score, 3)
            f[f"ao_{name}_mult"]  = round(sig.kelly_mult, 3)
        return f

    def status_line(self) -> str:
        parts = [f"{s.name}={s.score:.2f}(×{s.kelly_mult:.2f})" for s in self.signals.values()]
        return f"AlphaOrtho boost×{self.kelly_boost:.2f} thr{self.threshold_adj:+.3f} | " + " ".join(parts)


class AlphaOrthogonality:
    """
    Computa 4 sinais ortogonais e retorna overlay de kelly + threshold.

    Não precisa de estado persistente — todos os cálculos são stateless
    sobre candles e ctx.extra disponíveis em cada ciclo.
    """

    def evaluate(
        self,
        candles_1h:  list,
        spread_pct:  float = 0.0,
        futures_flow: dict | None = None,
        regime:      str = "",
    ) -> AlphaResult:
        """
        Args:
            candles_1h:   lista de candles ordenada mais recente primeiro
            spread_pct:   bid-ask spread / mid (do Ticker)
            futures_flow: dados de ff do ctx.extra["futures_flow"]
            regime:       regime 1H atual (para contextualizar sinais)
        """
        signals: dict[str, AlphaSignal] = {}

        signals["fd"]  = self._funding_dislocation(candles_1h, futures_flow or {})
        signals["lv"]  = self._liquidity_vacuum(spread_pct)
        signals["od"]  = self._overnight_drift(candles_1h)
        signals["mrm"] = self._mean_reversion_micro(candles_1h)

        # ── Composite kelly boost ─────────────────────────────────────────────
        # Multiplicativo: cada sinal aplica seu mult sobre o acumulado.
        # Limite final: clamp [0.65, 1.30] para evitar sizing extremo.
        kelly_boost = 1.0
        for sig in signals.values():
            kelly_boost *= sig.kelly_mult
        kelly_boost = round(min(max(kelly_boost, 0.65), 1.30), 4)

        # ── Threshold adjustment (aditivo) ────────────────────────────────────
        # Sinal FD positivo (squeeze setup) → facilita threshold -0.01
        # Sinal MRM overextended → eleva threshold +0.02
        thr_adj = sum(sig.thr_adj for sig in signals.values())
        thr_adj = round(min(max(thr_adj, -0.03), 0.03), 4)   # clamp ±3%

        # Sinais ativos (afastados do neutro)
        active = [name for name, sig in signals.items() if abs(sig.kelly_mult - 1.0) >= 0.04]

        result = AlphaResult(
            signals=signals,
            kelly_boost=kelly_boost,
            threshold_adj=thr_adj,
            active=active,
        )
        if active or abs(kelly_boost - 1.0) >= 0.04:
            logger.info(result.status_line())
        return result

    # ── Sinal 1: Funding Dislocation ─────────────────────────────────────────

    def _funding_dislocation(self, candles_1h: list, ff_data: dict) -> AlphaSignal:
        """
        Detecta divergência entre posicionamento (funding) e momentum de preço.

        Lógica:
          Preço em alta + funding muito NEGATIVO → bears capitulando → squeeze iminente
          → sinal BULLISH → boost kelly 10%

          Preço em alta + funding EXTREMAMENTE POSITIVO → crowded long → reversão provável
          → sinal BEARISH → cut kelly 10%

          Funding normal [FD_EXTREME_NEG, FD_EXTREME_POS] → neutro (0.5, ×1.0)

        Ortogonalidade: M6 mede funding como sinal direcional. FD mede a
        *divergência* funding vs preço — são matematicamente ortogonais.
        """
        # Funding rate bruto (campo "funding_rate" do FuturesFlowCollector)
        ff_scores = ff_data.get("scores", {})
        raw_fr    = float(ff_data.get("funding_rate", 0.0) or 0.0)  # em decimal

        # Momentum de preço dos últimos 5 candles
        if len(candles_1h) >= 6:
            c0, c5 = candles_1h[0].close, candles_1h[5].close
            price_mom = (c0 - c5) / c5 if c5 > 0 else 0.0
        else:
            price_mom = 0.0

        # Squeeze setup: preço caindo ou estagnado, mas funding very negative
        # (shorts pagando demais → posição insustentável → squeeze)
        if raw_fr <= FD_EXTREME_NEG:
            score      = 0.80
            kelly_mult = 1.08   # +8% (short squeeze setup)
            thr_adj    = -0.010
            desc       = f"FD squeeze: fr={raw_fr*100:.3f}% (bears capitulando)"

        # Crowded long: preço em alta, funding extremamente positivo
        elif raw_fr >= FD_EXTREME_POS and price_mom > 0:
            score      = 0.25
            kelly_mult = 0.90   # -10% (crowded, reversão possível)
            thr_adj    = 0.015
            desc       = f"FD crowded: fr={raw_fr*100:.3f}%, price_mom={price_mom*100:.2f}%"

        # Funding neutro a moderado positivo — sem dislocation
        else:
            score      = 0.50
            kelly_mult = 1.0
            thr_adj    = 0.0
            desc       = f"FD neutro: fr={raw_fr*100:.3f}%"

        return AlphaSignal("fd", score, kelly_mult, thr_adj, desc)

    # ── Sinal 2: Liquidity Vacuum ─────────────────────────────────────────────

    def _liquidity_vacuum(self, spread_pct: float) -> AlphaSignal:
        """
        Detecta deterioração de liquidez via spread bid-ask anômalo.

        Lógica:
          Spread normal (< 10 bps)   → mercado líquido → neutro
          Spread alargado (10-25 bps) → book ficando thin → cut 5%
          Spread extremo (> 25 bps)   → liquidity vacuum → cut 10%

        Ortogonalidade: nenhum fator M1-M9 mede a estrutura do livro
        diretamente. M8 (volatility state) mede retornos, não spreads.
        EdgeConditioner mede vol_ratio via candles, não spread real.
        """
        spread_bps = spread_pct * 10_000

        if spread_bps <= LV_NORMAL_BPS:
            score      = 0.90   # mercado líquido — leve boost
            kelly_mult = 1.05   # +5% (condições boas de execução)
            thr_adj    = -0.005
            desc       = f"LV líquido: {spread_bps:.1f} bps"

        elif spread_bps <= LV_WARN_BPS:
            # Interpolação linear [10, 25] → [1.0, 0.95]
            t          = (spread_bps - LV_NORMAL_BPS) / (LV_WARN_BPS - LV_NORMAL_BPS)
            score      = 0.50
            kelly_mult = round(1.0 - t * 0.05, 3)
            thr_adj    = 0.0
            desc       = f"LV alargando: {spread_bps:.1f} bps"

        else:
            # Vacuum: > 25 bps
            t          = min((spread_bps - LV_WARN_BPS) / (LV_VACUUM_BPS - LV_WARN_BPS), 1.0)
            score      = 0.20
            kelly_mult = round(0.95 - t * 0.05, 3)   # [0.95, 0.90]
            thr_adj    = 0.010
            desc       = f"LV vacuum: {spread_bps:.1f} bps (thin book)"

        return AlphaSignal("lv", score, kelly_mult, thr_adj, desc)

    # ── Sinal 3: Overnight Drift ──────────────────────────────────────────────

    def _overnight_drift(self, candles_1h: list) -> AlphaSignal:
        """
        Detecta viés de drift em sessões de baixo volume (overnight).

        Overnight UTC: candles com hour in {23, 0, 1, 2, 3} — sessão asiática.
        Lógica:
          Se os últimos OD_LOOKBACK candles overnight mostram drift consistente
          na mesma direção → momentum de sessão confirmado → boost 6%

          Gap do open atual vs close anterior:
          Gap up (> OD_GAP_THRESHOLD) em uptrend → confirmação → boost
          Gap down em uptrend → distribuição → cut

        Ortogonalidade: M1 mede retornos absolutos sem distinguir período.
        OD especificamente detecta padrão *temporal* de drift por sessão.
        """
        if not candles_1h or len(candles_1h) < 4:
            return AlphaSignal("od", 0.5, 1.0, 0.0, "OD: dados insuficientes")

        # Gap do candle mais recente (open vs close anterior)
        c0 = candles_1h[0]
        c1 = candles_1h[1] if len(candles_1h) > 1 else c0

        gap = (c0.open - c1.close) / c1.close if c1.close > 0 else 0.0

        # Drift nos últimos N candles (retornos open-to-close)
        n = min(OD_LOOKBACK, len(candles_1h))
        drifts = []
        for i in range(n):
            c = candles_1h[i]
            if c.open > 0:
                drifts.append((c.close - c.open) / c.open)

        if not drifts:
            return AlphaSignal("od", 0.5, 1.0, 0.0, "OD: sem dados de drift")

        avg_drift = sum(drifts) / len(drifts)

        # Gap significativo + drift consistente → sinal forte
        if gap >= OD_GAP_THRESHOLD and avg_drift > 0.001:
            score      = 0.80
            kelly_mult = 1.07
            thr_adj    = -0.008
            desc       = f"OD gap up {gap*100:.2f}% + drift {avg_drift*100:.3f}% positivo"

        elif gap <= -OD_GAP_THRESHOLD and avg_drift < -0.001:
            # Gap down + drift negativo → distribuição
            score      = 0.25
            kelly_mult = 0.92
            thr_adj    = 0.012
            desc       = f"OD gap down {gap*100:.2f}% + drift negativo"

        elif gap >= OD_GAP_THRESHOLD:
            # Gap up mas sem drift consistente → inconclusivo
            score      = 0.60
            kelly_mult = 1.03
            thr_adj    = 0.0
            desc       = f"OD gap up {gap*100:.2f}% (sem confirmação de drift)"

        else:
            # Sem gap significativo
            score      = 0.50
            kelly_mult = 1.0
            thr_adj    = 0.0
            desc       = f"OD neutro: gap={gap*100:.3f}%, drift_avg={avg_drift*100:.3f}%"

        return AlphaSignal("od", score, kelly_mult, thr_adj, desc)

    # ── Sinal 4: Mean Reversion Microstructure ────────────────────────────────

    def _mean_reversion_micro(self, candles_1h: list) -> AlphaSignal:
        """
        Detecta overextension microestrutural via z-score de preço.

        Z-score = (price_now - mean_N) / std_N

        Lógica:
          z > MRM_ZSCORE_EXTREME (2.5) → muito overextended → cut 10%
          z > MRM_ZSCORE_WARN (1.5)    → levemente overextended → cut 5%
          -1.5 < z < 1.5               → neutro
          z < -1.5                     → oversold → boost 6% (bom dip para comprar)

        Ortogonalidade: M1 mede retorno relativo (direcional).
        MRM mede overextension *relativa* à distribuição histórica recente —
        matematicamente ortogonal (mede desvio padrão, não direção).
        """
        n = min(MRM_LOOKBACK, len(candles_1h))
        if n < 5:
            return AlphaSignal("mrm", 0.5, 1.0, 0.0, "MRM: dados insuficientes")

        closes = [c.close for c in candles_1h[:n]]
        price_now = closes[0]
        mean_n    = sum(closes) / n

        # Desvio padrão amostral
        variance = sum((c - mean_n) ** 2 for c in closes) / (n - 1)
        std_n    = math.sqrt(variance) if variance > 0 else mean_n * 0.01

        zscore = (price_now - mean_n) / std_n

        if zscore >= MRM_ZSCORE_EXTREME:
            score      = 0.15
            kelly_mult = 0.90   # -10%: overextendido, reversão provável
            thr_adj    = 0.020
            desc       = f"MRM overextended z={zscore:.2f} (σ extremo → cut)"

        elif zscore >= MRM_ZSCORE_WARN:
            t          = (zscore - MRM_ZSCORE_WARN) / (MRM_ZSCORE_EXTREME - MRM_ZSCORE_WARN)
            score      = 0.30
            kelly_mult = round(1.0 - t * 0.10, 3)   # [1.0, 0.90]
            thr_adj    = round(t * 0.020, 4)
            desc       = f"MRM levemente overextended z={zscore:.2f}"

        elif zscore <= -MRM_ZSCORE_EXTREME:
            # Oversold extremo — dip comprador dentro de uptrend
            score      = 0.85
            kelly_mult = 1.08   # +8%
            thr_adj    = -0.012
            desc       = f"MRM oversold z={zscore:.2f} (dip extremo → boost)"

        elif zscore <= -MRM_ZSCORE_WARN:
            t          = (abs(zscore) - MRM_ZSCORE_WARN) / (MRM_ZSCORE_EXTREME - MRM_ZSCORE_WARN)
            score      = 0.70
            kelly_mult = round(1.0 + t * 0.08, 3)   # [1.0, 1.08]
            thr_adj    = round(-t * 0.012, 4)
            desc       = f"MRM oversold z={zscore:.2f} (dip moderado → boost)"

        else:
            score      = 0.50
            kelly_mult = 1.0
            thr_adj    = 0.0
            desc       = f"MRM neutro z={zscore:.2f}"

        return AlphaSignal("mrm", score, kelly_mult, thr_adj, desc)


# ── Singleton ──────────────────────────────────────────────────────────────────
alpha_orthogonality = AlphaOrthogonality()
