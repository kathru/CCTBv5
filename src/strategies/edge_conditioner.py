"""
EdgeConditioner — Edge Conditioning gates para o sistema operar apenas
quando as condições de qualidade de edge estão satisfeitas.

Filosofia: o sistema deve aprender a NÃO operar.
Operar em condições ruins destrói o edge acumulado com condições boas.

Gates implementados (todos como threshold multipliers — nunca bloqueio total,
exceto em situações extremas verificadas):

  Gate 1 — PSI drift (model_health):
    Features drifando do baseline → modelo menos confiável.
    PSI < 0.10 → normal.    PSI 0.10–0.20 → eleva threshold ×1.05.
    PSI 0.20–0.35 → eleva threshold ×1.15.
    PSI > 0.35  → BLOCK (drift extremo — features mudaram muito).

  Gate 2 — Model health geral (health_score):
    Score abaixo de 60 (ATENCAO/DEGRADANDO/CRITICO) → eleva threshold.
    Score < 40 (CRITICO) → eleva threshold ×1.20.
    Score < 60 (ATENCAO) → eleva threshold ×1.08.

  Gate 3 — Liquidez (volume ratio dos candles 1H):
    Volume atual vs média recente — proxy de liquidez.
    vol_ratio < 0.25 → BLOCK (mercado seco — slippage inaceitável).
    vol_ratio 0.25–0.50 → eleva threshold ×1.10.
    vol_ratio 0.50–0.70 → eleva threshold ×1.04.
    vol_ratio ≥ 0.70 → normal.

  Gate 4 — WR calibration drift (model_health):
    WR live muito abaixo do calibrado → modelo superestimando edge.
    diff < -0.15 → BLOCK (divergência extrema).
    diff -0.10 a -0.15 → eleva threshold ×1.12.
    diff -0.05 a -0.10 → eleva threshold ×1.06.
    diff > -0.05 → normal (ou model underestimating → ok).

Resultado final:
  threshold_mult: multiplicador combinado (produto dos 4 gates).
  should_block:   True se qualquer gate extremo foi ativado.
  block_reason:   nome do gate que bloqueou.
  conditions:     breakdown completo para diagnóstico.

Integração:
  Em momentum_strategy.evaluate(), após o ATR threshold adjustment:
    edge = edge_conditioner.evaluate(ctx, candles)
    if edge.should_block:
        _log("GATE_CLOSED", ...)
        return None
    threshold = round(min(threshold * edge.threshold_mult, 0.99), 4)
"""

import logging
from dataclasses import dataclass, field

from ..core.models import Candle

logger = logging.getLogger(__name__)

# ── Thresholds dos gates ──────────────────────────────────────────────────────

# Gate 1 — PSI
PSI_BLOCK      = 0.35   # acima disso → BLOCK
PSI_HIGH       = 0.20   # 0.20–0.35 → ×1.15
PSI_MODERATE   = 0.10   # 0.10–0.20 → ×1.05
PSI_MULT_HIGH  = 1.15
PSI_MULT_MOD   = 1.05

# Gate 2 — Health score
HEALTH_CRITICO   = 40   # < 40 → ×1.20
HEALTH_ATENCAO   = 60   # < 60 → ×1.08
HEALTH_MULT_CRIT = 1.20
HEALTH_MULT_ATEN = 1.08

# Gate 3 — Liquidity (volume ratio)
LIQ_BLOCK      = 0.25   # < 0.25 → BLOCK
LIQ_LOW        = 0.50   # 0.25–0.50 → ×1.10
LIQ_MODERATE   = 0.70   # 0.50–0.70 → ×1.04
LIQ_MULT_LOW   = 1.10
LIQ_MULT_MOD   = 1.04

# Gate 4 — WR calibration drift
WR_BLOCK         = -0.15   # < -0.15 → BLOCK
WR_HIGH_DRIFT    = -0.10   # -0.15 a -0.10 → ×1.12
WR_MOD_DRIFT     = -0.05   # -0.10 a -0.05 → ×1.06
WR_MULT_HIGH     = 1.12
WR_MULT_MOD      = 1.06


# ── Resultado ─────────────────────────────────────────────────────────────────

@dataclass
class EdgeCondition:
    """Resultado da avaliação dos gates de edge conditioning."""
    threshold_mult: float          # multiplicador combinado (produto dos gates)
    should_block:   bool           # True = gate extremo ativado → não entrar
    block_reason:   str | None     # qual gate bloqueou
    conditions:     dict = field(default_factory=dict)  # breakdown para diagnóstico

    def to_factors(self) -> dict:
        """Breakdown como dict para incluir nos factors do sinal."""
        return {
            "ec_thr_mult":    round(self.threshold_mult, 3),
            "ec_blocked":     1.0 if self.should_block else 0.0,
            "ec_psi_mult":    round(self.conditions.get("psi_mult", 1.0), 3),
            "ec_health_mult": round(self.conditions.get("health_mult", 1.0), 3),
            "ec_liq_mult":    round(self.conditions.get("liq_mult", 1.0), 3),
            "ec_wr_mult":     round(self.conditions.get("wr_mult", 1.0), 3),
        }


# ── EdgeConditioner ───────────────────────────────────────────────────────────

class EdgeConditioner:
    """
    Avalia as condições de edge antes de cada entrada.
    Stateless — pode ser chamado a cada ciclo de avaliação.
    """

    def evaluate(
        self,
        candles_1h:        list[Candle],
        model_health_data: dict | None,
    ) -> EdgeCondition:
        """
        Avalia todos os gates e retorna o EdgeCondition combinado.

        Args:
            candles_1h:        Candles 1H do símbolo (para gate de liquidez)
            model_health_data: Dado do ModelHealthMonitor (ctx.extra["model_health"])
        """
        conditions: dict = {}

        # ── Gate 1: PSI drift ─────────────────────────────────────────────────
        psi_mult, psi_block, max_psi = self._psi_gate(model_health_data)
        conditions["psi_mult"]  = psi_mult
        conditions["max_psi"]   = max_psi
        if psi_block:
            logger.warning(
                "EdgeConditioner: BLOCK psi_gate max_psi=%.3f > %.2f (drift extremo)",
                max_psi or 0, PSI_BLOCK,
            )
            return EdgeCondition(
                threshold_mult=psi_mult,
                should_block=True,
                block_reason=f"psi_extreme_{max_psi:.3f}",
                conditions=conditions,
            )

        # ── Gate 2: Model health geral ────────────────────────────────────────
        health_mult, health_score = self._health_gate(model_health_data)
        conditions["health_mult"]  = health_mult
        conditions["health_score"] = health_score

        # ── Gate 3: Liquidez ──────────────────────────────────────────────────
        liq_mult, liq_block, vol_ratio = self._liquidity_gate(candles_1h)
        conditions["liq_mult"]  = liq_mult
        conditions["vol_ratio"] = vol_ratio
        if liq_block:
            logger.warning(
                "EdgeConditioner: BLOCK liquidity_gate vol_ratio=%.2f < %.2f (mercado seco)",
                vol_ratio, LIQ_BLOCK,
            )
            return EdgeCondition(
                threshold_mult=liq_mult,
                should_block=True,
                block_reason=f"liquidity_dry_{vol_ratio:.2f}",
                conditions=conditions,
            )

        # ── Gate 4: WR calibration drift ─────────────────────────────────────
        wr_mult, wr_block, wr_diff = self._wr_gate(model_health_data)
        conditions["wr_mult"] = wr_mult
        conditions["wr_diff"] = wr_diff
        if wr_block:
            logger.warning(
                "EdgeConditioner: BLOCK wr_gate wr_diff=%.3f < %.2f (modelo superestimando)",
                wr_diff or 0, WR_BLOCK,
            )
            return EdgeCondition(
                threshold_mult=wr_mult,
                should_block=True,
                block_reason=f"wr_extreme_drift_{wr_diff:.3f}",
                conditions=conditions,
            )

        # ── Combinação final dos multiplicadores ──────────────────────────────
        combined = psi_mult * health_mult * liq_mult * wr_mult
        combined = round(min(combined, 1.50), 4)   # cap: threshold não passa de 1.5×

        if combined > 1.02:
            logger.info(
                "EdgeConditioner: threshold ×%.3f "
                "(psi=×%.2f health=×%.2f liq=×%.2f wr=×%.2f)",
                combined, psi_mult, health_mult, liq_mult, wr_mult,
            )

        return EdgeCondition(
            threshold_mult=combined,
            should_block=False,
            block_reason=None,
            conditions=conditions,
        )

    # ── Gate 1: PSI drift ─────────────────────────────────────────────────────

    def _psi_gate(
        self, model_health: dict | None
    ) -> tuple[float, bool, float | None]:
        """Retorna (mult, should_block, max_psi)."""
        max_psi = _extract_max_psi(model_health)
        if max_psi is None:
            return 1.0, False, None   # sem dados = neutro
        if max_psi > PSI_BLOCK:
            return PSI_MULT_HIGH, True, max_psi
        if max_psi > PSI_HIGH:
            return PSI_MULT_HIGH, False, max_psi
        if max_psi > PSI_MODERATE:
            return PSI_MULT_MOD, False, max_psi
        return 1.0, False, max_psi

    # ── Gate 2: Model health geral ────────────────────────────────────────────

    def _health_gate(
        self, model_health: dict | None
    ) -> tuple[float, float | None]:
        """Retorna (mult, health_score)."""
        if not model_health:
            return 1.0, None
        score = model_health.get("health_score")
        if score is None:
            return 1.0, None
        score = float(score)
        if score < HEALTH_CRITICO:
            return HEALTH_MULT_CRIT, score
        if score < HEALTH_ATENCAO:
            return HEALTH_MULT_ATEN, score
        return 1.0, score

    # ── Gate 3: Liquidez ──────────────────────────────────────────────────────

    def _liquidity_gate(
        self, candles: list[Candle]
    ) -> tuple[float, bool, float]:
        """
        Compara volume recente com média dos últimos 20 candles.
        vol_ratio = vol_atual / avg_vol_20.
        Retorna (mult, should_block, vol_ratio).
        """
        if len(candles) < 5:
            return 1.0, False, 1.0   # sem dados = neutro

        vols     = [c.volume for c in candles[:20]]
        avg_vol  = sum(vols[1:]) / max(len(vols) - 1, 1)
        curr_vol = vols[0]

        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
        vol_ratio = round(vol_ratio, 4)

        if vol_ratio < LIQ_BLOCK:
            return LIQ_MULT_LOW, True, vol_ratio
        if vol_ratio < LIQ_LOW:
            return LIQ_MULT_LOW, False, vol_ratio
        if vol_ratio < LIQ_MODERATE:
            return LIQ_MULT_MOD, False, vol_ratio
        return 1.0, False, vol_ratio

    # ── Gate 4: WR calibration drift ─────────────────────────────────────────

    def _wr_gate(
        self, model_health: dict | None
    ) -> tuple[float, bool, float | None]:
        """Retorna (mult, should_block, wr_diff)."""
        wr_diff = _extract_wr_diff(model_health)
        if wr_diff is None:
            return 1.0, False, None
        if wr_diff < WR_BLOCK:
            return WR_MULT_HIGH, True, wr_diff
        if wr_diff < WR_HIGH_DRIFT:
            return WR_MULT_HIGH, False, wr_diff
        if wr_diff < WR_MOD_DRIFT:
            return WR_MULT_MOD, False, wr_diff
        return 1.0, False, wr_diff


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_max_psi(model_health: dict | None) -> float | None:
    if not model_health:
        return None
    try:
        v = model_health.get("dimensions", {}).get("psi", {}).get("max_psi")
        return float(v) if v is not None else None
    except Exception:
        return None


def _extract_wr_diff(model_health: dict | None) -> float | None:
    if not model_health:
        return None
    try:
        wr = model_health.get("dimensions", {}).get("win_rate", {})
        live  = wr.get("live_wr")
        calib = wr.get("calibrated_wr")
        if live is None or calib is None:
            return None
        return round(float(live) - float(calib), 4)
    except Exception:
        return None
