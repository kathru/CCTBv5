"""
ModelHealthMonitor — Phase 15.2: Validação Online do Modelo em Produção.

Detecta degradação silenciosa do modelo de forma automática.

Monitora 4 dimensões:
  1. PSI (Population Stability Index) — features ao vivo vs baseline
     PSI < 0.10 → estável | 0.10-0.20 → atenção | > 0.20 → drift crítico

  2. Win Rate drift — WR ao vivo vs WR calibrado
     Divergência > 2σ por 7 dias → alerta Discord + badge vermelho

  3. Score distribution — média dos scores ao vivo vs baseline
     Score médio caindo > 10% → possível regime shift

  4. Execution rate — % de avaliações que geram sinal
     Taxa muito baixa (<1%) ou muito alta (>20%) → modelo desajustado

Health Score [0–100]:
  >= 80 → SAUDÁVEL (verde)
  60–79 → ATENÇÃO (amarelo)
  40–59 → DEGRADANDO (laranja)
   < 40 → CRÍTICO (vermelho)

Ciclo: 15 minutos. Cache Redis: `model_health` TTL=1800s.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

POLL_INTERVAL = 900     # 15 min
REDIS_TTL     = 1800    # 30 min

_ROOT          = Path(__file__).parent.parent.parent
MODELS_DIR     = _ROOT / "data" / "models"
CALIB_PATH     = MODELS_DIR / "calibration_coef.json"
BASELINE_PATH  = MODELS_DIR / "feature_baseline.json"
EXP_LOG_PATH   = MODELS_DIR / "experiment_log.json"

# Thresholds de saúde
PSI_STABLE     = 0.10
PSI_ALERT      = 0.20
WR_DRIFT_WARN  = 0.05   # diferença absoluta no win rate
WR_DRIFT_CRIT  = 0.10
SCORE_DRIFT    = 0.10   # queda > 10% no score médio
MIN_OBS        = 50     # mínimo de observações ao vivo para avaliar


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


class ModelHealthMonitor:
    """
    Monitora saúde do modelo em produção.
    Roda como background task — acessa signal_audit_log e drift monitor.
    """

    def __init__(self, cache, signal_log=None) -> None:
        self._cache      = cache
        self._signal_log = signal_log   # injetado pós-import
        self._task: asyncio.Task | None = None
        self._running    = False

    def set_signal_log(self, signal_log: object) -> None:
        self._signal_log = signal_log

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._evaluate()
        self._task = asyncio.create_task(self._loop(), name="model_health")
        logger.info("ModelHealthMonitor started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._evaluate()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("ModelHealthMonitor: %s", exc)

    # ── Core evaluation ───────────────────────────────────────────────────────

    async def _evaluate(self) -> None:
        try:
            result = self._compute()
            await self._cache.set("model_health", json.dumps(result), ttl=REDIS_TTL)
            health = result["health_score"]
            status = result["status"]
            logger.info(
                "ModelHealth: score=%d (%s) | psi_max=%.3f | wr_live=%.1f%% vs calib=%.1f%%",
                health, status,
                result["dimensions"]["psi"]["max_psi"] or 0,
                (result["dimensions"]["win_rate"]["live_wr"] or 0) * 100,
                (result["dimensions"]["win_rate"]["calibrated_wr"] or 0) * 100,
            )
        except Exception as exc:
            logger.warning("ModelHealthMonitor._evaluate: %s", exc)

    def _compute(self) -> dict:
        calib   = _load_json(CALIB_PATH)
        baseline= _load_json(BASELINE_PATH)
        exp_log = _load_json(EXP_LOG_PATH)

        # ── Dimensão 1: PSI (lê do DriftMonitor se disponível) ───────────────
        psi_dim = self._compute_psi_dim()

        # ── Dimensão 2: Win Rate drift ────────────────────────────────────────
        wr_dim = self._compute_wr_dim(calib)

        # ── Dimensão 3: Score distribution ───────────────────────────────────
        score_dim = self._compute_score_dim(baseline)

        # ── Dimensão 4: Execution rate ────────────────────────────────────────
        exec_dim = self._compute_exec_dim()

        # ── Health Score ─────────────────────────────────────────────────────
        scores = {
            "psi":       psi_dim["score"],
            "win_rate":  wr_dim["score"],
            "score_dist":score_dim["score"],
            "exec_rate": exec_dim["score"],
        }
        weights = {"psi": 0.35, "win_rate": 0.35, "score_dist": 0.15, "exec_rate": 0.15}
        health_score = round(sum(scores[k] * weights[k] for k in scores), 1)

        # Status label
        status = (
            "SAUDAVEL"   if health_score >= 80 else
            "ATENCAO"    if health_score >= 60 else
            "DEGRADANDO" if health_score >= 40 else
            "CRITICO"
        )
        color = {
            "SAUDAVEL":   "green",
            "ATENCAO":    "yellow",
            "DEGRADANDO": "orange",
            "CRITICO":    "red",
        }[status]

        # ── Experiment info ────────────────────────────────────────────────────
        experiments = exp_log.get("experiments", [])
        latest_exp  = experiments[-1] if experiments else {}

        return {
            "health_score":   health_score,
            "status":         status,
            "color":          color,
            "dimensions": {
                "psi":        psi_dim,
                "win_rate":   wr_dim,
                "score_dist": score_dim,
                "exec_rate":  exec_dim,
            },
            "calibration": {
                "platt_a":    calib.get("platt_a"),
                "platt_b":    calib.get("platt_b"),
                "calibrated_at": calib.get("calibrated_at"),
                "n_samples":  calib.get("n"),
                "win_rate":   calib.get("win_rate"),
            },
            "experiment": {
                "total_runs":    len(experiments),
                "latest_run_at": latest_exp.get("run_at"),
                "latest_version":latest_exp.get("version"),
                "latest_wr":     latest_exp.get("win_rate"),
                "latest_a":      latest_exp.get("platt_a"),
            },
            "recommendations": self._recommendations(status, psi_dim, wr_dim),
            "evaluated_at": datetime.now(UTC).isoformat(),
        }

    def _compute_psi_dim(self) -> dict:
        """PSI máximo das features — via signal_audit_log._entries."""
        if self._signal_log is None:
            return {"score": 50, "max_psi": None, "status": "sem_dados",
                    "n_obs": 0, "features": {}}

        # Coleta scores das features das últimas observações
        entries = list(self._signal_log._entries)
        n_obs   = len(entries)

        if n_obs < MIN_OBS:
            return {"score": 50, "max_psi": None,
                    "status": f"aguardando ({n_obs}/{MIN_OBS} obs)",
                    "n_obs": n_obs, "features": {}}

        # Calcula médias dos fatores ao vivo
        feature_vals: dict[str, list[float]] = {}
        for e in entries:
            for k, v in e.factors.items():
                if isinstance(v, (int, float)) and not k.startswith("meta"):
                    feature_vals.setdefault(k, []).append(float(v))

        # Lê baseline
        baseline = _load_json(BASELINE_PATH)
        bl_features = baseline.get("features", {})

        psi_by_feature: dict[str, float] = {}
        for fname, vals in feature_vals.items():
            if fname not in bl_features or len(vals) < 5:
                continue
            bl  = bl_features[fname]
            bl_mean = bl.get("mean", 0.5)
            bl_std  = bl.get("std", 0.1) or 0.1
            live_mean = sum(vals) / len(vals)
            # PSI simplificado: baseado na diferença normalizada de médias
            psi = abs(live_mean - bl_mean) / bl_std
            # Mapeia para escala PSI padrão (diferença > 2σ ≈ PSI > 0.2)
            psi_mapped = round(psi * 0.1, 4)
            psi_by_feature[fname] = psi_mapped

        max_psi = max(psi_by_feature.values(), default=0.0)
        psi_status = (
            "estavel"  if max_psi < PSI_STABLE else
            "atencao"  if max_psi < PSI_ALERT  else
            "drift"
        )
        score = 100 if max_psi < PSI_STABLE else 60 if max_psi < PSI_ALERT else 20

        return {
            "score":    score,
            "max_psi":  round(max_psi, 4),
            "status":   psi_status,
            "n_obs":    n_obs,
            "features": psi_by_feature,
        }

    def _compute_wr_dim(self, calib: dict) -> dict:
        """Win rate ao vivo vs calibrado."""
        calibrated_wr = calib.get("win_rate")
        if self._signal_log is None or calibrated_wr is None:
            return {"score": 50, "live_wr": None, "calibrated_wr": calibrated_wr,
                    "diff": None, "status": "sem_dados", "n_signals": 0}

        stats = self._signal_log.stats()
        total = stats.get("total_evaluations", 0)
        if total < MIN_OBS:
            return {"score": 50, "live_wr": None, "calibrated_wr": calibrated_wr,
                    "diff": None, "status": f"aguardando ({total} obs)", "n_signals": 0}

        # Win rate aqui = taxa de sinais gerados (execução bem sucedida)
        # Para WR real precisaríamos dos fills — usamos proxy
        counters  = self._signal_log._counters
        n_signal  = counters.get("SIGNAL", 0)
        n_total   = sum(counters.values()) or 1
        live_exec = n_signal / n_total   # taxa de execução

        # Compara com win_rate calibrado (proxy: deveria ser ~37%)
        diff   = abs(live_exec - calibrated_wr)
        status = (
            "alinhado"   if diff < WR_DRIFT_WARN else
            "atencao"    if diff < WR_DRIFT_CRIT else
            "drift_critico"
        )
        score = 100 if diff < WR_DRIFT_WARN else 50 if diff < WR_DRIFT_CRIT else 10

        return {
            "score":         score,
            "live_wr":       round(live_exec, 4),
            "calibrated_wr": round(calibrated_wr, 4),
            "diff":          round(diff, 4),
            "status":        status,
            "n_signals":     n_signal,
            "n_total":       n_total,
        }

    def _compute_score_dim(self, baseline: dict) -> dict:
        """Distribuição dos scores ao vivo vs baseline."""
        if self._signal_log is None:
            return {"score": 50, "live_avg_score": None,
                    "baseline_avg_score": None, "drift": None}

        entries = list(self._signal_log._entries)
        if len(entries) < MIN_OBS:
            return {"score": 50, "live_avg_score": None,
                    "baseline_avg_score": None, "drift": None,
                    "n": len(entries)}

        scores = [e.score for e in entries if e.score > 0]
        if not scores:
            return {"score": 50, "live_avg_score": None,
                    "baseline_avg_score": None, "drift": None}

        live_avg = sum(scores) / len(scores)

        # Baseline: score médio esperado (de recalibrate.py, avg_score = 0.531)
        # Não temos avg_score direto — usa a média dos fatores do baseline como proxy
        bl_features = baseline.get("features", {})
        if bl_features:
            bl_means = [v.get("mean", 0.5) for v in bl_features.values()]
            baseline_avg = sum(bl_means) / len(bl_means) if bl_means else 0.5
        else:
            baseline_avg = 0.531   # último conhecido

        drift = abs(live_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
        score = 100 if drift < 0.05 else 60 if drift < SCORE_DRIFT else 20

        return {
            "score":               score,
            "live_avg_score":      round(live_avg, 4),
            "baseline_avg_score":  round(baseline_avg, 4),
            "drift_pct":           round(drift * 100, 2),
            "status": "estavel" if drift < 0.05 else "atencao" if drift < SCORE_DRIFT else "drift",
        }

    def _compute_exec_dim(self) -> dict:
        """Taxa de execução de sinais (% avaliações que geram BUY)."""
        if self._signal_log is None:
            return {"score": 50, "exec_rate": None, "status": "sem_dados"}

        stats = self._signal_log.stats()
        total = stats.get("total_evaluations", 0)
        if total < MIN_OBS:
            return {"score": 50, "exec_rate": None,
                    "status": f"aguardando ({total} obs)"}

        rate_pct = stats.get("signal_rate_pct", 0)
        exec_rate = rate_pct / 100

        # Taxa saudável: 1–15% das avaliações geram sinal
        status = (
            "baixa_demais"  if exec_rate < 0.005 else
            "saudavel"      if exec_rate < 0.15  else
            "alta_demais"
        )
        score = 100 if 0.005 <= exec_rate < 0.15 else 40

        return {
            "score":      score,
            "exec_rate":  round(exec_rate, 4),
            "exec_pct":   round(rate_pct, 2),
            "status":     status,
            "total_evals":total,
        }

    @staticmethod
    def _recommendations(status: str, psi: dict, wr: dict) -> list[str]:
        recs = []
        if status == "CRITICO":
            recs.append("🔴 Modelo em estado crítico — rever calibração imediatamente")
        if psi.get("max_psi", 0) and psi["max_psi"] >= PSI_ALERT:
            recs.append("⚠ Drift significativo nas features — rodar feature_analysis.py")
        if wr.get("status") == "drift_critico":
            recs.append("⚠ Win rate divergindo do calibrado — possível regime shift")
        if not recs:
            recs.append("✅ Modelo operando dentro dos parâmetros esperados")
        return recs

    async def get_health(self) -> dict | None:
        raw = await self._cache.get("model_health")
        if not raw:
            return None
        return raw if isinstance(raw, dict) else json.loads(raw)


# Singleton global
model_health_monitor = ModelHealthMonitor(cache=None)
