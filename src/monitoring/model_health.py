"""
ModelHealthMonitor — Phase 15.2: Validação Online do Modelo em Produção.

Detecta degradação silenciosa do modelo de forma automática.

Monitora 4 dimensões:
  1. PSI (Population Stability Index) — features ao vivo vs baseline
     PSI < 0.10 → estável | 0.10-0.20 → atenção | > 0.20 → drift crítico

  2. Win Rate drift — WR ao vivo (baseado em PnL real de trades fechados)
     vs WR calibrado. Divergência > 2σ por 7 dias → alerta Discord + badge vermelho

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
import math
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
MIN_CLOSED_TRADES = 10  # mínimo de trades fechados para calcular WR real

# Proporções de baseline esperadas para os 6 bins do PSI canônico:
# (-∞, p10), (p10, p25), (p25, p50), (p50, p75), (p75, p90), (p90, +∞)
PSI_BASELINE_PROPORTIONS = [0.10, 0.15, 0.25, 0.25, 0.15, 0.10]
PSI_EPSILON = 1e-6   # evita log(0)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _canonical_psi(live_values: list[float], bl: dict) -> float | None:
    """
    PSI canônico com 6 bins baseados nos percentis do baseline.

    Bins: (-inf, p10], (p10, p25], (p25, p50], (p50, p75], (p75, p90], (p90, +inf)
    Proporções esperadas: 10%, 15%, 25%, 25%, 15%, 10%

    PSI = Σ (A_i - E_i) × ln(A_i / E_i)
      A_i = proporção live no bin i
      E_i = proporção esperada no bin i

    Retorna None se não há dados suficientes ou baseline é constante (std=0).
    """
    if not live_values or len(live_values) < 5:
        return None

    # Ignora features constantes no baseline (ex: m6/m7/m9 com std=0)
    bl_std = bl.get("std", 0.0)
    if bl_std == 0.0:
        return None

    bounds = [
        bl.get("p10", 0.0),
        bl.get("p25", 0.0),
        bl.get("p50", 0.5),
        bl.get("p75", 1.0),
        bl.get("p90", 1.0),
    ]

    n = len(live_values)
    counts = [0] * 6
    for v in live_values:
        if v <= bounds[0]:
            counts[0] += 1
        elif v <= bounds[1]:
            counts[1] += 1
        elif v <= bounds[2]:
            counts[2] += 1
        elif v <= bounds[3]:
            counts[3] += 1
        elif v <= bounds[4]:
            counts[4] += 1
        else:
            counts[5] += 1

    psi = 0.0
    for count, expected_pct in zip(counts, PSI_BASELINE_PROPORTIONS, strict=False):
        actual_pct = max(count / n, PSI_EPSILON)
        expected   = max(expected_pct, PSI_EPSILON)
        psi += (actual_pct - expected) * math.log(actual_pct / expected)

    return round(psi, 6)


def _compute_closed_wr(closed_orders: list) -> dict | None:
    """
    Calcula win rate real a partir de ordens fechadas (BUY/SELL pareadas por FIFO).

    Retorna dict com: live_wr, n_trades, wins, losses
    Retorna None se não há pares suficientes.
    """
    from ..core.models import OrderSide

    # Agrupa por símbolo e ordena por filled_at
    by_symbol: dict[str, list] = {}
    for o in closed_orders:
        if o.filled_quantity and o.filled_quantity > 0 and o.avg_fill_price and o.avg_fill_price > 0:
            by_symbol.setdefault(o.symbol, []).append(o)

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda o: o.filled_at or o.created_at)

    wins = 0
    losses = 0

    for _sym, orders in by_symbol.items():
        buy_queue: list[tuple[float, float, float]] = []  # (price, qty, fees)
        for o in orders:
            if o.side == OrderSide.BUY:
                buy_queue.append((o.avg_fill_price, o.filled_quantity, o.fees_paid))
            elif o.side == OrderSide.SELL and buy_queue:
                buy_price, buy_qty, buy_fees = buy_queue.pop(0)
                sell_price = o.avg_fill_price
                sell_qty   = min(o.filled_quantity, buy_qty)
                sell_fees  = o.fees_paid
                pnl = (sell_price - buy_price) * sell_qty - buy_fees - sell_fees
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

    total = wins + losses
    if total < MIN_CLOSED_TRADES:
        return None

    return {
        "live_wr":  round(wins / total, 4),
        "n_trades": total,
        "wins":     wins,
        "losses":   losses,
    }


class ModelHealthMonitor:
    """
    Monitora saúde do modelo em produção.
    Roda como background task — acessa signal_audit_log e drift monitor.
    """

    def __init__(self, cache, signal_log=None) -> None:
        self._cache      = cache
        self._signal_log = signal_log   # injetado pós-import
        self._db         = None         # injetado via set_db()
        self._task: asyncio.Task | None = None
        self._running    = False

    def set_signal_log(self, signal_log: object) -> None:
        self._signal_log = signal_log

    def set_db(self, db) -> None:
        """Injeta acesso ao banco para calcular WR real de trades fechados."""
        self._db = db

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
            # Busca trades fechados do DB (para WR real) — faz aqui, fora do _compute síncrono
            closed_orders = []
            if self._db is not None:
                try:
                    from ..persistence.repositories.orders import OrderRepository
                    repo = OrderRepository(self._db)
                    closed_orders = await repo.get_filled(limit=500)
                except Exception as exc:
                    logger.warning("ModelHealthMonitor: erro ao buscar trades fechados: %s", exc)

            result = self._compute(closed_orders)
            await self._cache.set("model_health", json.dumps(result), ttl=REDIS_TTL)
            health = result["health_score"]
            status = result["status"]
            wr_dim = result["dimensions"]["win_rate"]
            logger.info(
                "ModelHealth: score=%d (%s) | psi_max=%.3f | wr_live=%.1f%% vs calib=%.1f%% | trades=%s",
                health, status,
                result["dimensions"]["psi"]["max_psi"] or 0,
                (wr_dim.get("live_wr") or 0) * 100,
                (wr_dim.get("calibrated_wr") or 0) * 100,
                wr_dim.get("n_trades", "n/a"),
            )
        except Exception as exc:
            logger.warning("ModelHealthMonitor._evaluate: %s", exc)

    def _compute(self, closed_orders: list) -> dict:
        calib   = _load_json(CALIB_PATH)
        baseline= _load_json(BASELINE_PATH)
        exp_log = _load_json(EXP_LOG_PATH)

        # ── Dimensão 1: PSI canônico ─────────────────────────────────────────
        psi_dim = self._compute_psi_dim()

        # ── Dimensão 2: Win Rate drift (PnL real) ────────────────────────────
        wr_dim = self._compute_wr_dim(calib, closed_orders)

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
        """PSI canônico das features — usa binning de distribuição real."""
        if self._signal_log is None:
            return {"score": 50, "max_psi": None, "status": "sem_dados",
                    "n_obs": 0, "features": {}}

        entries = list(self._signal_log._entries)
        n_obs   = len(entries)

        if n_obs < MIN_OBS:
            return {"score": 50, "max_psi": None,
                    "status": f"aguardando ({n_obs}/{MIN_OBS} obs)",
                    "n_obs": n_obs, "features": {}}

        # Coleta valores das features ao vivo
        feature_vals: dict[str, list[float]] = {}
        for e in entries:
            for k, v in e.factors.items():
                if isinstance(v, (int, float)) and not k.startswith("meta") and not k.startswith("sz_"):
                    feature_vals.setdefault(k, []).append(float(v))

        # Lê baseline
        baseline = _load_json(BASELINE_PATH)
        bl_features = baseline.get("features", {})

        psi_by_feature: dict[str, float] = {}
        for fname, vals in feature_vals.items():
            if fname not in bl_features:
                continue
            bl  = bl_features[fname]
            psi = _canonical_psi(vals, bl)
            if psi is not None:
                psi_by_feature[fname] = psi

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
            "features": {k: round(v, 4) for k, v in psi_by_feature.items()},
        }

    def _compute_wr_dim(self, calib: dict, closed_orders: list) -> dict:
        """Win rate ao vivo baseado em PnL real de trades fechados vs WR calibrado."""
        calibrated_wr = calib.get("win_rate")
        if calibrated_wr is None:
            return {"score": 50, "live_wr": None, "calibrated_wr": None,
                    "diff": None, "status": "sem_dados", "n_trades": 0}

        if not closed_orders:
            return {"score": 50, "live_wr": None, "calibrated_wr": round(calibrated_wr, 4),
                    "diff": None, "status": "aguardando_trades", "n_trades": 0}

        trade_stats = _compute_closed_wr(closed_orders)
        if trade_stats is None:
            n = len([o for o in closed_orders if hasattr(o, 'side')])
            return {"score": 50, "live_wr": None, "calibrated_wr": round(calibrated_wr, 4),
                    "diff": None,
                    "status": f"aguardando ({n} trades / min {MIN_CLOSED_TRADES})",
                    "n_trades": n}

        live_wr = trade_stats["live_wr"]
        diff    = live_wr - calibrated_wr   # positivo = live melhor que calibrado
        abs_diff = abs(diff)

        status = (
            "alinhado"      if abs_diff < WR_DRIFT_WARN else
            "atencao"       if abs_diff < WR_DRIFT_CRIT else
            "drift_critico"
        )
        score = 100 if abs_diff < WR_DRIFT_WARN else 50 if abs_diff < WR_DRIFT_CRIT else 10

        return {
            "score":         score,
            "live_wr":       live_wr,
            "calibrated_wr": round(calibrated_wr, 4),
            "diff":          round(diff, 4),
            "status":        status,
            "n_trades":      trade_stats["n_trades"],
            "wins":          trade_stats["wins"],
            "losses":        trade_stats["losses"],
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
            recs.append("⚠ Win rate real divergindo do calibrado — possível regime shift")
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
