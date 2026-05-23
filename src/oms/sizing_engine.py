"""
SizingEngine — Kelly Composto para Position Sizing Dinâmico.

Computa o multiplicador final do Kelly com base em 5 dimensões independentes.
Módulo puro — sem estado, sem Redis, sem I/O. Recebe dados já coletados.

Fórmula:
  final_kelly = base_kelly
              × regime_mult       (qualidade do regime — já existe)
              × drift_mult        (PSI das features — do model_health)
              × vol_state_mult    (estado de volatilidade M8)
              × calibration_mult  (WR live vs WR calibrado)
              × score_mult        (força do sinal — calibrated score)

  cap: [MIN_KELLY=0.01, MAX_KELLY=0.15]

Filosofia:
  Aumentar sizing quando edge sobe.
  Reduzir AGRESSIVAMENTE quando edge degrada.
  O sistema aprende a expressar alta convicção com tamanho de posição maior.

Cada dimensão é independente e tem fallback neutro (1.0) quando sem dados.
Isso garante que a ausência de dados nunca force uma saída ou bloqueio —
apenas mantém o sizing no nível base.
"""

import logging

logger = logging.getLogger(__name__)

# ── Limites do Kelly final ────────────────────────────────────────────────────

MIN_KELLY = 0.01   # 1%  — nunca zera (se passou todos os gates, entra pequeno)
MAX_KELLY = 0.15   # 15% — nunca alavanca além do cap institucional

# ── Vol State → multiplicador ─────────────────────────────────────────────────
# EXPANDING/TREND = volatilidade saudável para momentum → size normal/cheio
# CHAOTIC = volatilidade explosiva sem direção → size mínimo

VOL_STATE_MULT: dict[str, float] = {
    "EXPANDING":      1.00,   # breakout ativo — tamanho cheio
    "TREND":          0.90,   # tendência direcional — quase cheio
    "COMPRESSED":     0.75,   # pré-breakout — moderado (direção incerta)
    "MEAN_REVERTING": 0.50,   # lateralização — half size
    "CHAOTIC":        0.25,   # explosivo sem direção — size mínimo
    "UNKNOWN":        0.70,   # sem dados — conservador
}


# ── Resultado do SizingEngine ─────────────────────────────────────────────────

class SizingResult:
    """Resultado do Kelly composto — kelly final e breakdown de cada multiplicador."""

    def __init__(
        self,
        final_kelly:      float,
        base_kelly:       float,
        regime_mult:      float,
        drift_mult:       float,
        vol_state_mult:   float,
        calibration_mult: float,
        score_mult:       float,
        vol_state:        str,
        max_psi:          float | None,
        wr_diff:          float | None,
    ) -> None:
        self.final_kelly      = final_kelly
        self.base_kelly       = base_kelly
        self.regime_mult      = regime_mult
        self.drift_mult       = drift_mult
        self.vol_state_mult   = vol_state_mult
        self.calibration_mult = calibration_mult
        self.score_mult       = score_mult
        self.vol_state        = vol_state
        self.max_psi          = max_psi
        self.wr_diff          = wr_diff

    def to_factors(self) -> dict:
        """Retorna breakdown como dict para incluir nos factors do sinal."""
        return {
            "sz_final_kelly":      round(self.final_kelly, 4),
            "sz_regime_mult":      round(self.regime_mult, 3),
            "sz_drift_mult":       round(self.drift_mult, 3),
            "sz_vol_state_mult":   round(self.vol_state_mult, 3),
            "sz_calibration_mult": round(self.calibration_mult, 3),
            "sz_score_mult":       round(self.score_mult, 3),
            "sz_vol_state":        self.vol_state,
        }

    def summary(self) -> str:
        return (
            f"kelly={self.final_kelly:.3f} "
            f"(base={self.base_kelly:.3f} × "
            f"regime={self.regime_mult:.2f} × "
            f"drift={self.drift_mult:.2f} × "
            f"vol={self.vol_state_mult:.2f} × "
            f"calib={self.calibration_mult:.2f} × "
            f"score={self.score_mult:.2f})"
        )


# ── SizingEngine ──────────────────────────────────────────────────────────────

class SizingEngine:
    """
    Compõe o Kelly final a partir de múltiplas dimensões de edge quality.
    Instanciar uma vez na estratégia — é stateless (pode ser chamado a cada ciclo).
    """

    def compute(
        self,
        base_kelly:       float,
        regime_mult:      float,
        calibrated_score: float,
        vol_state_data:   dict | None,
        model_health_data: dict | None,
    ) -> SizingResult:
        """
        Computa o Kelly final.

        Args:
            base_kelly:        Kelly base = min(calibrated × 0.25, 0.15)
            regime_mult:       Multiplicador do regime (0.50–1.00)
            calibrated_score:  Score calibrado pelo Platt (0–1)
            vol_state_data:    Dado do M8 (ctx.extra["vol_state"])
            model_health_data: Dado do ModelHealthMonitor (ctx.extra["model_health"])
        """
        drift_mult        = self._drift_mult(model_health_data)
        vol_state_mult, vol_state = self._vol_state_mult(vol_state_data)
        calibration_mult  = self._calibration_mult(model_health_data)
        score_mult        = self._score_mult(calibrated_score)

        # PSI e WR diff para logging e breakdown
        max_psi = _extract_max_psi(model_health_data)
        wr_diff = _extract_wr_diff(model_health_data)

        raw = base_kelly * regime_mult * drift_mult * vol_state_mult * calibration_mult * score_mult
        final_kelly = round(max(MIN_KELLY, min(MAX_KELLY, raw)), 4)

        result = SizingResult(
            final_kelly=final_kelly,
            base_kelly=base_kelly,
            regime_mult=regime_mult,
            drift_mult=drift_mult,
            vol_state_mult=vol_state_mult,
            calibration_mult=calibration_mult,
            score_mult=score_mult,
            vol_state=vol_state,
            max_psi=max_psi,
            wr_diff=wr_diff,
        )

        logger.debug("SizingEngine: %s", result.summary())

        # Loga quando o sizing diverge significativamente do base
        base_final = base_kelly * regime_mult
        if final_kelly < base_final * 0.70:
            logger.info(
                "SizingEngine: sizing REDUZIDO %.3f→%.3f "
                "(psi=%.3f wr_diff=%.3f vol=%s score=%.2f)",
                base_final, final_kelly,
                max_psi or 0, wr_diff or 0, vol_state, calibrated_score,
            )
        elif final_kelly > base_final * 1.05:
            logger.info(
                "SizingEngine: sizing AMPLIADO %.3f→%.3f "
                "(wr_diff=%.3f score=%.2f)",
                base_final, final_kelly, wr_diff or 0, calibrated_score,
            )

        return result

    # ── Dimensão 1: Drift (PSI) ───────────────────────────────────────────────

    def _drift_mult(self, model_health: dict | None) -> float:
        """
        Reduz sizing quando features drifam do baseline.
        PSI alto = distribuição das features mudou = modelo menos confiável.
        """
        max_psi = _extract_max_psi(model_health)
        if max_psi is None:
            return 1.0   # sem dados = neutro
        if max_psi < 0.10:
            return 1.00  # estável
        if max_psi < 0.15:
            return 0.80  # drift leve
        if max_psi < 0.20:
            return 0.60  # drift moderado — atenção
        return 0.40      # drift crítico — reduz agressivamente

    # ── Dimensão 2: Volatility State (M8) ────────────────────────────────────

    def _vol_state_mult(self, vol_state_data: dict | None) -> tuple[float, str]:
        """
        Ajusta sizing pelo estado de volatilidade M8.
        EXPANDING/TREND = ambiente favorável para momentum.
        CHAOTIC = explosivo sem direção = risco assimétrico.
        """
        if not vol_state_data:
            return 0.70, "UNKNOWN"
        state = vol_state_data.get("state", "UNKNOWN")
        return VOL_STATE_MULT.get(state, 0.70), state

    # ── Dimensão 3: Calibration drift (WR live vs calibrado) ─────────────────

    def _calibration_mult(self, model_health: dict | None) -> float:
        """
        Se o WR live está acima do calibrado → modelo subestima o edge → size up leve.
        Se o WR live está muito abaixo → modelo superestimou → size down agressivo.
        """
        wr_diff = _extract_wr_diff(model_health)
        if wr_diff is None:
            return 1.0   # sem dados = neutro
        if wr_diff > 0.10:
            return 1.10  # live bem melhor: modelo subestimou — aumenta levemente
        if wr_diff > 0.05:
            return 1.05
        if wr_diff >= -0.05:
            return 1.00  # alinhado — mantém
        if wr_diff >= -0.10:
            return 0.75  # live pior: modelo superestimou — reduz
        return 0.50      # divergência grande — reduz agressivamente

    # ── Dimensão 4: Score mult (força do sinal) ───────────────────────────────

    def _score_mult(self, calibrated: float) -> float:
        """
        Score calibrado como proxy de convicção no momento da entrada.
        Score alto = sinal forte = mais capital justificado.
        Score mínimo (acima do threshold) = entra com menor exposição.
        """
        if calibrated >= 0.70:
            return 1.00
        if calibrated >= 0.60:
            return 0.90
        if calibrated >= 0.55:
            return 0.80
        return 0.70   # score mínimo aceitável — posição menor


# ── Helpers internos ──────────────────────────────────────────────────────────

def _extract_max_psi(model_health: dict | None) -> float | None:
    """Extrai max_psi do model_health dict (pode vir em estruturas diferentes)."""
    if not model_health:
        return None
    try:
        psi_dim = model_health.get("dimensions", {}).get("psi", {})
        v = psi_dim.get("max_psi")
        return float(v) if v is not None else None
    except Exception:
        return None


def _extract_wr_diff(model_health: dict | None) -> float | None:
    """Extrai diferença WR live - WR calibrado do model_health."""
    if not model_health:
        return None
    try:
        wr_dim = model_health.get("dimensions", {}).get("win_rate", {})
        live   = wr_dim.get("live_wr")
        calib  = wr_dim.get("calibrated_wr")
        if live is None or calib is None:
            return None
        return round(float(live) - float(calib), 4)
    except Exception:
        return None
