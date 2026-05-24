"""
EdgeConditioner — Edge Conditioning gates para o sistema operar apenas
quando as condições de qualidade de edge estão satisfeitas.

Filosofia: o sistema deve aprender a NÃO operar.
Operar em condições ruins destrói o edge acumulado com condições boas.

Gates implementados (threshold multipliers + sizing reduction graduados):

  Gate 1 — PSI drift (model_health):
    PSI < 0.10 → normal.
    PSI 0.10–0.20 → threshold ×1.05.
    PSI 0.20–0.35 → threshold ×1.15.
    PSI > 0.35  → BLOCK (drift extremo — features mudaram muito).

  Gate 2 — Model health geral (health_score):
    Score < 40 (CRITICO) → threshold ×1.20.
    Score < 60 (ATENCAO) → threshold ×1.08.

  Gate 3 — Liquidez (volume ratio dos candles 1H):
    vol_ratio < 0.25 → BLOCK (mercado seco — slippage inaceitável).
    vol_ratio 0.25–0.50 → threshold ×1.10.
    vol_ratio 0.50–0.70 → threshold ×1.04.
    vol_ratio ≥ 0.70 → normal.

  Gate 4 — WR calibration drift (model_health):
    Anti-deadlock: bloquear totalmente impede recuperação do WR live.
    Abordagem graduada — permite micro-trades para re-calibração:

    diff > -0.05          → normal (100% sizing).
    diff -0.05 a -0.10    → threshold ×1.06 | sizing 90%.
    diff -0.10 a -0.15    → threshold ×1.12 | sizing 75%.
    diff -0.15 a -0.25    → threshold ×1.20 | sizing 40% (micro-trades).
    diff -0.25 a -0.35    → threshold ×1.25 | sizing 20% (micro-trades).
    diff < -0.35           → BLOCK (divergência catastrófica).

    Lógica: com diff < -0.15 ainda permitimos entrada com sizing mínimo.
    Isso evita o deadlock (sem trades → WR não se recupera → gate não abre).
    Sizing de 20-40% do Kelly limita risco enquanto coleta dados reais.

Resultado final:
  threshold_mult: multiplicador combinado (produto dos 4 gates).
  sizing_mult:    multiplicador de kelly (1.0=normal, 0.2=micro-trade).
  should_block:   True apenas em situações verdadeiramente catastróficas.
  block_reason:   nome do gate que bloqueou.
  conditions:     breakdown completo para diagnóstico.

Integração:
  edge = edge_conditioner.evaluate(candles_1h, model_health_data)
  if edge.should_block:
      return None
  threshold = round(min(threshold * edge.threshold_mult, 0.99), 4)
  kelly = kelly * edge.sizing_mult   # reduz sizing em condições ruins
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

# Gate 4 — WR calibration drift (graduado anti-deadlock)
WR_BLOCK              = -0.35   # < -0.35 → BLOCK catastrófico
WR_MICRO_SEVERE       = -0.25   # -0.35 a -0.25 → sizing 20% | thr ×1.25
WR_MICRO_HIGH         = -0.15   # -0.25 a -0.15 → sizing 40% | thr ×1.20
WR_HIGH_DRIFT         = -0.10   # -0.15 a -0.10 → sizing 75% | thr ×1.12
WR_MOD_DRIFT          = -0.05   # -0.10 a -0.05 → sizing 90% | thr ×1.06

WR_MULT_CATASTROPHIC  = 1.25
WR_MULT_SEVERE        = 1.25
WR_MULT_HIGH_         = 1.20
WR_MULT_HIGH          = 1.12
WR_MULT_MOD           = 1.06

# Sizing mult por nível de drift (aplicado ao kelly_fraction)
WR_SIZING_CATASTROPHIC = 0.20   # micro-trade extremo
WR_SIZING_SEVERE       = 0.20   # micro-trade severo
WR_SIZING_HIGH         = 0.40   # micro-trade padrão
WR_SIZING_MOD          = 0.75   # sizing reduzido
WR_SIZING_LOW          = 0.90   # sizing levemente reduzido


# ── Resultado ─────────────────────────────────────────────────────────────────

@dataclass
class EdgeCondition:
    """Resultado da avaliação dos gates de edge conditioning."""
    threshold_mult: float          # multiplicador combinado (produto dos gates)
    sizing_mult:    float          # multiplicador de kelly (anti-deadlock WR gate)
    should_block:   bool           # True = gate catastrófico → não entrar
    block_reason:   str | None     # qual gate bloqueou
    conditions:     dict = field(default_factory=dict)  # breakdown para diagnóstico

    def to_factors(self) -> dict:
        """Breakdown como dict para incluir nos factors do sinal."""
        return {
            "ec_thr_mult":    round(self.threshold_mult, 3),
            "ec_sizing_mult": round(self.sizing_mult, 3),
            "ec_blocked":     1.0 if self.should_block else 0.0,
            "ec_psi_mult":    round(self.conditions.get("psi_mult", 1.0), 3),
            "ec_health_mult": round(self.conditions.get("health_mult", 1.0), 3),
            "ec_liq_mult":    round(self.conditions.get("liq_mult", 1.0), 3),
            "ec_wr_mult":     round(self.conditions.get("wr_mult", 1.0), 3),
            "ec_wr_sizing":   round(self.conditions.get("wr_sizing", 1.0), 3),
            "ec_wr_diff":     round(self.conditions.get("wr_diff", 0.0), 4),
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
                sizing_mult=0.0,
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
                sizing_mult=0.0,
                should_block=True,
                block_reason=f"liquidity_dry_{vol_ratio:.2f}",
                conditions=conditions,
            )

        # ── Gate 4: WR calibration drift (graduado — anti-deadlock) ──────────
        wr_mult, wr_sizing, wr_block, wr_diff = self._wr_gate(model_health_data)
        conditions["wr_mult"]   = wr_mult
        conditions["wr_sizing"] = wr_sizing
        conditions["wr_diff"]   = wr_diff or 0.0
        if wr_block:
            logger.warning(
                "EdgeConditioner: BLOCK wr_gate wr_diff=%.3f < %.2f (divergência catastrófica)",
                wr_diff or 0, WR_BLOCK,
            )
            return EdgeCondition(
                threshold_mult=wr_mult,
                sizing_mult=0.0,
                should_block=True,
                block_reason=f"wr_catastrophic_{wr_diff:.3f}",
                conditions=conditions,
            )

        # Micro-trade: log informativo quando sizing está reduzido por WR drift
        if wr_sizing < 0.99:
            logger.info(
                "EdgeConditioner: WR drift=%.3f → micro-trade sizing=%.0f%% thr×%.2f",
                wr_diff or 0, wr_sizing * 100, wr_mult,
            )

        # ── Combinação final dos multiplicadores ──────────────────────────────
        combined = psi_mult * health_mult * liq_mult * wr_mult
        combined = round(min(combined, 1.50), 4)   # cap: threshold não passa de 1.5×

        if combined > 1.02:
            logger.info(
                "EdgeConditioner: threshold ×%.3f sizing=%.0f%% "
                "(psi=×%.2f health=×%.2f liq=×%.2f wr=×%.2f)",
                combined, wr_sizing * 100, psi_mult, health_mult, liq_mult, wr_mult,
            )

        return EdgeCondition(
            threshold_mult=combined,
            sizing_mult=wr_sizing,
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

    # ── Gate 4: WR calibration drift (graduado — anti-deadlock) ─────────────

    def _wr_gate(
        self, model_health: dict | None
    ) -> tuple[float, float, bool, float | None]:
        """
        Retorna (thr_mult, sizing_mult, should_block, wr_diff).

        Abordagem graduada para evitar deadlock:
        - Bloqueio total apenas em divergência catastrófica (< -0.35)
        - Entre -0.15 e -0.35: micro-trades com sizing reduzido
        - Isso permite re-calibração do WR live sem exposição total
        """
        wr_diff = _extract_wr_diff(model_health)
        if wr_diff is None:
            return 1.0, 1.0, False, None

        if wr_diff < WR_BLOCK:                   # < -0.35: catastrófico
            return WR_MULT_CATASTROPHIC, 0.0, True, wr_diff

        if wr_diff < WR_MICRO_SEVERE:            # -0.35 a -0.25: micro severo
            return WR_MULT_SEVERE, WR_SIZING_SEVERE, False, wr_diff

        if wr_diff < WR_MICRO_HIGH:              # -0.25 a -0.15: micro padrão
            return WR_MULT_HIGH_, WR_SIZING_HIGH, False, wr_diff

        if wr_diff < WR_HIGH_DRIFT:              # -0.15 a -0.10: sizing reduzido
            return WR_MULT_HIGH, WR_SIZING_MOD, False, wr_diff

        if wr_diff < WR_MOD_DRIFT:               # -0.10 a -0.05: sizing leve
            return WR_MULT_MOD, WR_SIZING_LOW, False, wr_diff

        return 1.0, 1.0, False, wr_diff          # > -0.05: normal


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
