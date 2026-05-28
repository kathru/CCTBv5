"""
Feature Governance — controle de qualidade das features de ML.

Componentes:
  1. FeatureSchema    — versiona features com definições formais
  2. LeakageGuard     — valida que nenhuma feature usa dados futuros
  3. DriftMonitor     — detecta quando features se distanciam da distribuição de treino
  4. FeatureImportance — mede contribuição de cada feature para a qualidade do sinal

Princípio: toda feature que entra no modelo deve ser auditável, rastreável e monitorada.

Uso:
  from src.monitoring.feature_governance import governance, LeakageGuard

  # Registrar features
  governance.record_live(symbol, features_dict)

  # Verificar drift
  report = governance.drift_report()
"""

import json
import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT         = Path(__file__).parent.parent.parent   # raiz do projeto
SCHEMA_PATH   = _ROOT / "data" / "models" / "feature_schema.json"
BASELINE_PATH = _ROOT / "data" / "models" / "feature_baseline.json"

# ── 1. Feature Schema (Versionamento) ────────────────────────────────────────

@dataclass
class FeatureDef:
    """Definição formal de uma feature."""
    weight:           float          # peso no score final (0–1)
    description:      str            # descrição legível
    lookback:         int            # candles necessários para calcular
    range_min:        float = 0.0    # valor mínimo esperado
    range_max:        float = 1.0    # valor máximo esperado
    is_normalized:    bool  = True   # já está em [0,1]?
    leakage_safe:     bool  = True   # confirmado sem leakage?


@dataclass
class FeatureSchema:
    """Schema versionado das features do modelo."""
    version:          str
    model_id:         str
    features:         dict[str, FeatureDef]
    created_at:       str
    lookback_candles: int
    notes:            str = ""

    def to_dict(self) -> dict:
        return {
            "version":          self.version,
            "model_id":         self.model_id,
            "created_at":       self.created_at,
            "lookback_candles": self.lookback_candles,
            "notes":            self.notes,
            "features": {
                name: {
                    "weight":       f.weight,
                    "description":  f.description,
                    "lookback":     f.lookback,
                    "range":        [f.range_min, f.range_max],
                    "normalized":   f.is_normalized,
                    "leakage_safe": f.leakage_safe,
                }
                for name, f in self.features.items()
            },
        }

    def save(self, path: Path = SCHEMA_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("FeatureSchema v%s salvo em %s", self.version, path)

    @classmethod
    def load(cls, path: Path = SCHEMA_PATH) -> "FeatureSchema | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            features = {
                name: FeatureDef(
                    weight=fd["weight"],
                    description=fd["description"],
                    lookback=fd["lookback"],
                    range_min=fd["range"][0],
                    range_max=fd["range"][1],
                    is_normalized=fd.get("normalized", True),
                    leakage_safe=fd.get("leakage_safe", True),
                )
                for name, fd in d["features"].items()
            }
            return cls(
                version=d["version"],
                model_id=d["model_id"],
                features=features,
                created_at=d["created_at"],
                lookback_candles=d["lookback_candles"],
                notes=d.get("notes", ""),
            )
        except Exception as exc:
            logger.error("Erro ao carregar FeatureSchema: %s", exc)
            return None


# Schema atual (v3.0.0 — reforma quant 2026-05-28)
CURRENT_SCHEMA = FeatureSchema(
    version="3.0.0",
    model_id="momentum_scoring_v3",
    created_at="2026-05-28",
    lookback_candles=25,
    notes=(
        "v3.0.0 2026-05-28: Reforma quant — M1/M2/M5 desativados (perm_imp negativa)."
        " M3 reformulado para directional volume (buy_vol/total_vol 12 candles)."
        " ADX(10)+DI+/DI- substituiu SMA crossover para regime detection."
        " Novos pesos: M3=40%, M6=22%, M7=18%, M9=14%, M8=6%."
        " M1 mantido como RSI(14) diagnóstico (peso=0%)."
    ),
    features={
        # Pesos v3.0.0 — sync com momentum_strategy.py
        # M1/M2/M5: DESATIVADOS (perm_imp negativa). M1 = diagnóstico RSI(14).
        # M6/M7/M9: não mensurável no IS (dado externo) — Spearman=0.52 em live
        "m1_momentum": FeatureDef(
            weight=0.00,   # v3.0.0: DESATIVADO. RSI(14) diagnóstico apenas (perm_imp=-0.0013)
            description=(
                "RSI(14) 1H — zona ótima BUY: 40-65. Penaliza overbought >70."
                " Peso=0%: diagnóstico sem impacto no score."
            ),
            lookback=16,
            leakage_safe=True,
        ),
        "m2_consistency": FeatureDef(
            weight=0.00,   # v3.0.0: DESATIVADO (perm_imp=-0.0033 negativa)
            description=(
                "Consistência de tendência: % candles bullish"
                " + higher-highs E higher-lows graduais. Monitorado no dashboard."
            ),
            lookback=10,
            leakage_safe=True,
        ),
        "m3_volume": FeatureDef(
            weight=0.40,   # v3.0.0: 35%→40% — Directional Volume, único edge real
            description=(
                "Directional Volume v3.0: buy_vol/(buy_vol+sell_vol) 12 candles"
                " + volume relativo + momentum de volume."
                " Formula: directional_ratio*0.50 + vol_ratio*0.30 + vol_trend_score*0.20"
            ),
            lookback=20,
            leakage_safe=True,
        ),
        "m4_regime_str": FeatureDef(
            weight=0.00,   # DESATIVADO: regime detection migrou para ADX(10)+DI+/DI-
            description=(
                "Força do regime SMA: substituído por ADX(10) no detect_regime()."
            ),
            lookback=20,
            leakage_safe=True,
        ),
        "m5_candle": FeatureDef(
            weight=0.00,   # v3.0.0: DESATIVADO (perm_imp=-0.001 negativa)
            description=(
                "Estrutura do candle: posição do close no range dos últimos 3 candles"
                " (Williams %R style). Monitorado no dashboard."
            ),
            lookback=3,
            leakage_safe=True,
        ),
        "m6_futures": FeatureDef(
            weight=0.22,   # v3.0.0: 16%→22% (Spearman=0.52 em live, não mensurável IS)
            description=(
                "Futures Flow: funding rate (perp) + variação de Open Interest."
                " Lê sinal do mercado de derivativos sem operar futuros."
            ),
            lookback=1,
            leakage_safe=True,
        ),
        "m7_rel_strength": FeatureDef(
            weight=0.18,   # v3.0.0: 14%→18% (Spearman=0.52 em live, não mensurável IS)
            description=(
                "Relative Strength vs BTC: RS 1h/5h/24h ponderado"
                " + BTC leadership score + tendência de RS"
            ),
            lookback=25,
            leakage_safe=True,
        ),
        "m8_vol_state": FeatureDef(
            weight=0.06,   # v3.0.0: mantido 6% (estrutural, perm_imp=-0.0028)
            description=(
                "Volatility State Machine: 5 estados"
                " (EXPANDING/TREND/COMPRESSED/MEAN_REVERTING/CHAOTIC)"
                " via ATR + BB + dir_consistency"
            ),
            lookback=20,
            leakage_safe=True,
        ),
        "m9_sentiment": FeatureDef(
            weight=0.14,   # v3.0.0: 12%→14% (Spearman=0.52 em live, não mensurável IS)
            description=(
                "News Sentiment: Fear & Greed Index (alternative.me)"
                " + CoinGecko social sentiment + price momentum 24h"
            ),
            lookback=1,
            leakage_safe=True,
        ),
    },
)


# ── 2. Leakage Guard ──────────────────────────────────────────────────────────

class LeakageGuard:
    """
    Valida que nenhuma feature usa dados do futuro.

    Regras de leakage:
      - Feature computada no instante T deve usar APENAS candles[0..T-1]
      - O label (forward return) usa candles[T..T+N] — NUNCA pode entrar na feature
      - Volume futuro nunca pode aparecer em features de volume
    """

    @staticmethod
    def validate_feature_window(
        feature_name:  str,
        feature_value: float,
        candle_index:  int,
        total_candles: int,
        forward_candles: int = 5,
    ) -> tuple[bool, str]:
        """
        Valida que uma feature computada no candle_index não usa dados futuros.
        Retorna (is_safe, reason).
        """
        # Regra 1: feature não pode usar candles além do índice atual
        if candle_index + forward_candles >= total_candles:
            return False, (
                f"Sem espaço para label forward: {candle_index} + {forward_candles}"
                f" >= {total_candles}"
            )

        # Regra 2: feature value deve estar no range esperado
        if feature_name in CURRENT_SCHEMA.features:
            fdef = CURRENT_SCHEMA.features[feature_name]
            if not (fdef.range_min - 0.01 <= feature_value <= fdef.range_max + 0.01):
                return False, (
                    f"{feature_name}={feature_value:.4f} fora do range "
                    f"[{fdef.range_min}, {fdef.range_max}]"
                )

        return True, "ok"

    @staticmethod
    def validate_schema(schema: FeatureSchema) -> list[str]:
        """Valida o schema completo. Retorna lista de violações."""
        violations = []

        total_weight = sum(f.weight for f in schema.features.values())
        if abs(total_weight - 1.0) > 0.01:
            violations.append(f"Pesos não somam 1.0: {total_weight:.4f}")

        for name, fdef in schema.features.items():
            if not fdef.leakage_safe:
                violations.append(f"{name}: marcado como NÃO leakage-safe")
            if fdef.lookback > schema.lookback_candles:
                violations.append(
                    f"{name}: lookback={fdef.lookback} > schema.lookback={schema.lookback_candles}"
                )

        return violations

    @staticmethod
    def check_temporal_order(candle_timestamps: list) -> tuple[bool, str]:
        """Verifica que os candles estão em ordem cronológica correta (mais recente primeiro)."""
        if len(candle_timestamps) < 2:
            return True, "ok"

        # Candles devem estar em ordem decrescente (newest first)
        for i in range(len(candle_timestamps) - 1):
            if candle_timestamps[i] < candle_timestamps[i + 1]:
                return False, (
                    f"Candles fora de ordem no índice {i}: "
                    f"{candle_timestamps[i]} < {candle_timestamps[i+1]}"
                )
        return True, "ok"


# ── 3. Drift Monitor ──────────────────────────────────────────────────────────

@dataclass
class FeatureStats:
    """Estatísticas de distribuição de uma feature."""
    mean:    float
    std:     float
    p10:     float   # percentil 10
    p25:     float   # percentil 25
    p50:     float   # mediana
    p75:     float   # percentil 75
    p90:     float   # percentil 90
    n:       int     # número de amostras

    @classmethod
    def from_values(cls, values: list[float]) -> "FeatureStats":
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        s = sorted(values)
        n = len(s)
        def pct(p): return s[int(p * n / 100)]
        return cls(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0.0,
            p10=pct(10), p25=pct(25), p50=pct(50),
            p75=pct(75), p90=pct(90),
            n=n,
        )

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4), "std": round(self.std, 4),
            "p10":  round(self.p10, 4),  "p25": round(self.p25, 4),
            "p50":  round(self.p50, 4),  "p75": round(self.p75, 4),
            "p90":  round(self.p90, 4),  "n":   self.n,
        }


def _psi(expected_stats: FeatureStats, actual_stats: FeatureStats) -> float:
    """
    Population Stability Index (standard formula using percentile bins).
    Bins defined by baseline percentiles — each bin gets approx 1/6 of expected mass.
    PSI < 0.10: stable | 0.10–0.25: watch | > 0.25: significant drift → recalibrate

    NOTE: Previous implementation used raw percentile values in the PSI formula
    instead of bin proportions, which inflated PSI results significantly (e.g.
    reporting PSI=0.8 for moderate drift that actually scores ~0.15 with the
    standard formula). This corrected version uses the standard bin-proportion
    approach with CDF interpolation from percentile landmarks.
    """
    # Constant features (fixed neutral = 0.5) have no spread — PSI is undefined; return 0.
    if expected_stats.std == 0.0 or expected_stats.p10 == expected_stats.p90:
        return 0.0

    def _cdf_at(x: float, stats: FeatureStats) -> float:
        """
        Estimate the CDF P(X <= x) for a distribution described by percentile stats.
        Uses piecewise-linear interpolation between known percentile landmarks.
        Handles degenerate cases (tied percentiles) gracefully.
        """
        xs = [stats.p10, stats.p25, stats.p50, stats.p75, stats.p90]
        cs = [0.10,       0.25,       0.50,       0.75,       0.90]
        if x <= xs[0]:
            return cs[0]   # P10 is a hard lower-bound; everything at or below gets 0.10
        if x >= xs[-1]:
            return cs[-1]  # P90 is a hard upper-bound; everything at or above gets 0.90
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                span = xs[i + 1] - xs[i]
                if span == 0.0:
                    # Tied percentiles: use the left CDF value (all mass before the tie)
                    return cs[i]
                t = (x - xs[i]) / span
                return cs[i] + t * (cs[i + 1] - cs[i])
        return cs[-1]

    finite_edges = [
        expected_stats.p10,
        expected_stats.p25,
        expected_stats.p50,
        expected_stats.p75,
        expected_stats.p90,
    ]

    # Expected proportions: apply baseline CDF to baseline edges.
    # Because cdf_at(p10, baseline) = 0.10 exactly, etc., this gives the true bin masses
    # for the baseline distribution (including degenerate tie cases like m8_vol_state).
    exp_cdf = [_cdf_at(e, expected_stats) for e in finite_edges]
    exp_props = (
        [exp_cdf[0]]
        + [max(exp_cdf[i + 1] - exp_cdf[i], 0.0) for i in range(len(finite_edges) - 1)]
        + [max(1.0 - exp_cdf[-1], 0.0)]
    )
    # Normalise (they should already sum to 1.0 if cdf_at returns exactly 0.10/0.90 at p10/p90)
    exp_total = sum(exp_props) or 1.0
    exp_props = [p / exp_total for p in exp_props]

    # Actual proportions: apply live CDF to baseline edges.
    act_cdf = [_cdf_at(e, actual_stats) for e in finite_edges]
    act_props = (
        [act_cdf[0]]
        + [max(act_cdf[i + 1] - act_cdf[i], 0.0) for i in range(len(finite_edges) - 1)]
        + [max(1.0 - act_cdf[-1], 0.0)]
    )
    act_total = sum(act_props) or 1.0
    act_props = [p / act_total for p in act_props]

    # PSI = Σ (actual - expected) × ln(actual / expected)
    psi_total = 0.0
    eps = 1e-4  # avoid log(0)
    for exp_p, act_p in zip(exp_props, act_props, strict=False):
        act_p = max(act_p, eps)
        exp_p = max(exp_p, eps)
        psi_total += (act_p - exp_p) * math.log(act_p / exp_p)

    return round(abs(psi_total), 4)


class DriftMonitor:
    """
    Monitora se as features ao vivo estão se desviando da distribuição de treino.
    Guarda janela deslizante de 500 observações recentes e compara com baseline.
    """

    WINDOW_SIZE     = 500    # amostras recentes para comparar
    DRIFT_WARN      = 0.10   # PSI que gera warning
    DRIFT_ALERT     = 0.25   # PSI que gera alerta crítico (recalibrar)
    MIN_SAMPLES     = 30     # mínimo para calcular drift

    def __init__(self) -> None:
        # Baseline: estatísticas da distribuição de treino
        self._baseline:    dict[str, FeatureStats] = {}
        # Janela viva: últimas N observações por feature (persistida no Redis)
        self._live_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.WINDOW_SIZE))
        self._obs_count:   int = 0
        self._cache = None   # injetado via load_from_redis() ou set_cache()

    def set_baseline(self, feature_stats: dict[str, FeatureStats]) -> None:
        """Define a distribuição de referência (calculada no treino)."""
        self._baseline = feature_stats
        logger.info(
            "DriftMonitor baseline definido: %d features, %s amostras",
            len(feature_stats),
            {k: v.n for k, v in feature_stats.items()},
        )

    def load_baseline_from_file(self, path: Path = BASELINE_PATH) -> bool:
        """
        Carrega baseline de distribuição do arquivo JSON gerado por feature_analysis.py.
        Retorna True se carregado com sucesso.
        """
        if not path.exists():
            logger.debug("Baseline não encontrado em %s", path)
            return False
        try:
            data = json.loads(path.read_text())
            features_raw = data.get("features", {})
            baseline: dict[str, FeatureStats] = {}
            for name, s in features_raw.items():
                baseline[name] = FeatureStats(
                    mean=s["mean"], std=s["std"],
                    p10=s["p10"],   p25=s["p25"],
                    p50=s["p50"],   p75=s["p75"],
                    p90=s["p90"],   n=s["n"],
                )
            self.set_baseline(baseline)
            logger.info(
                "Baseline carregado de %s (schema=%s, n=%d, computed_at=%s)",
                path, data.get("schema_version","?"),
                data.get("n_samples", 0), data.get("computed_at","?"),
            )
            return True
        except Exception as exc:
            logger.error("Erro ao carregar baseline: %s", exc)
            return False

    def record(self, features: dict[str, float]) -> None:
        """Registra uma observação ao vivo e agenda persistência no Redis."""
        for name, value in features.items():
            self._live_window[name].append(value)
        self._obs_count += 1
        # Persiste no Redis a cada 10 observações (fire-and-forget, não bloqueia)
        if self._obs_count % 10 == 0 and self._cache:
            import asyncio
            try:
                asyncio.get_running_loop().create_task(
                    self._save_to_redis(),
                    name="drift_save_redis",
                )
            except RuntimeError:
                pass  # fora de um event loop — ignora

    async def _save_to_redis(self) -> None:
        """Persiste a janela deslizante inteira no Redis (TTL = 7 dias)."""
        if not self._cache:
            return
        try:
            data = {
                name: list(window)
                for name, window in self._live_window.items()
                if window
            }
            data["__obs_count__"] = self._obs_count
            await self._cache.set(
                "governance:live_window",
                json.dumps(data),
                ttl=604800,   # 7 dias
            )
            logger.debug(
                "DriftMonitor: janela salva no Redis (%d obs, %d features)",
                self._obs_count, len(data) - 1,
            )
        except Exception as exc:
            logger.debug("DriftMonitor: falha ao salvar Redis: %s", exc)

    async def load_from_redis(self, cache) -> bool:
        """
        Restaura a janela deslizante do Redis no boot.
        Retorna True se restaurado com sucesso.
        """
        self._cache = cache
        try:
            raw = await cache.get("governance:live_window")
            if not raw:
                logger.info("DriftMonitor: sem histórico no Redis — janela começa do zero")
                return False
            data = json.loads(raw)
            obs_count = data.pop("__obs_count__", 0)
            restored = 0
            for name, values in data.items():
                if isinstance(values, list) and values:
                    self._live_window[name] = deque(values, maxlen=self.WINDOW_SIZE)
                    restored += 1
            self._obs_count = obs_count
            logger.info(
                "DriftMonitor: histórico restaurado do Redis — "
                "%d obs, %d features (janela até %d amostras por feature)",
                obs_count, restored, self.WINDOW_SIZE,
            )
            return True
        except Exception as exc:
            logger.warning("DriftMonitor: falha ao restaurar Redis: %s", exc)
            return False

    def set_cache(self, cache) -> None:
        """Define o cache Redis para persistência automática."""
        self._cache = cache

    def drift_report(self) -> dict[str, Any]:
        """
        Calcula PSI para cada feature e retorna relatório de drift.
        """
        report: dict[str, Any] = {
            "timestamp":    datetime.now(UTC).isoformat(),
            "obs_count":    self._obs_count,
            "features":     {},
            "overall_status": "ok",
            "alerts":       [],
        }

        if not self._baseline:
            report["overall_status"] = "no_baseline"
            return report

        # Sempre inclui todas as features do baseline no report,
        # mesmo sem observações ao vivo suficientes.
        # Isso permite o dashboard mostrar o baseline como referência desde o boot.
        worst_psi = 0.0
        for name, baseline in self._baseline.items():
            window = self._live_window.get(name, [])

            if len(window) < self.MIN_SAMPLES:
                # Ainda sem dados ao vivo — mostra baseline como referência
                report["features"][name] = {
                    "status":   "awaiting_data",
                    "psi":      None,
                    "n_live":   len(window),
                    "baseline": baseline.to_dict(),
                    "delta_mean": None,
                }
                continue

            live_values = list(window)
            live_stats  = FeatureStats.from_values(live_values)

            psi = _psi(baseline, live_stats)
            worst_psi = max(worst_psi, psi)

            if psi >= self.DRIFT_ALERT:
                status = "critical"
                report["alerts"].append(f"{name}: drift crítico PSI={psi:.3f} (recalibrar!)")
            elif psi >= self.DRIFT_WARN:
                status = "warning"
                report["alerts"].append(f"{name}: drift detectado PSI={psi:.3f}")
            else:
                status = "ok"

            report["features"][name] = {
                "status":     status,
                "psi":        round(psi, 4),
                "n_live":     len(window),
                "live":       live_stats.to_dict(),
                "baseline":   baseline.to_dict(),
                "delta_mean": round(live_stats.mean - baseline.mean, 4),
            }

        if worst_psi >= self.DRIFT_ALERT:
            report["overall_status"] = "critical"
        elif worst_psi >= self.DRIFT_WARN:
            report["overall_status"] = "warning"

        return report

    def live_stats(self) -> dict[str, FeatureStats]:
        """Retorna estatísticas das features ao vivo."""
        return {
            name: FeatureStats.from_values(list(window))
            for name, window in self._live_window.items()
            if len(window) >= self.MIN_SAMPLES
        }


# ── 4. Feature Importance ─────────────────────────────────────────────────────

class FeatureImportance:
    """
    Mede a importância de cada feature por permutação.

    Método:
      1. Calcula correlação de Spearman de cada feature com o label (win/loss)
      2. Permuta cada feature aleatoriamente e mede degradação do score
      3. Calcula importância relativa normalizada
    """

    @staticmethod
    def from_samples(
        features_list: list[dict[str, float]],
        labels:        list[int],              # 0 ou 1
        n_permutations: int = 100,
        seed: int = 42,
    ) -> dict[str, dict]:
        """
        Calcula importância por permutação.

        Args:
          features_list: lista de dicts {feature_name: value} por amostra
          labels:        lista de 0/1 (0=loss, 1=win)
          n_permutations: quantas permutações por feature
        """
        import random as rng
        rng.seed(seed)

        if len(features_list) < 10:
            return {}

        feature_names = list(features_list[0].keys())

        # ── Baseline: acurácia com score original ──────────────
        def predict_win(f: dict) -> float:
            """Score bruto como proxy de probabilidade."""
            schema = CURRENT_SCHEMA
            total = sum(
                f.get(name, 0.5) * fdef.weight
                for name, fdef in schema.features.items()
            )
            return total

        def accuracy(feat_list, lbls) -> float:
            """Acurácia simples: score > median → prediz win."""
            scores = [predict_win(f) for f in feat_list]
            median = sorted(scores)[len(scores)//2]
            correct = sum(
                1 for s, lbl in zip(scores, lbls, strict=False) if (s > median) == (lbl == 1)
            )
            return correct / len(lbls)

        baseline_acc = accuracy(features_list, labels)

        # ── Correlação de Spearman (feature ↔ label) ───────────
        def spearman(vals, lbls) -> float:
            n_ = len(vals)
            if n_ < 5:
                return 0.0
            r_vals = sorted(range(n_), key=lambda i: vals[i])
            r_lbls = sorted(range(n_), key=lambda i: lbls[i])
            rank_v = [0] * n_
            rank_l = [0] * n_
            for rank, idx in enumerate(r_vals):
                rank_v[idx] = rank
            for rank, idx in enumerate(r_lbls):
                rank_l[idx] = rank
            d2 = sum((rank_v[i] - rank_l[i])**2 for i in range(n_))
            return 1 - 6 * d2 / (n_ * (n_**2 - 1))

        # ── Importância por permutação ──────────────────────────
        results = {}
        for feat in feature_names:
            feat_vals = [f.get(feat, 0.5) for f in features_list]

            # Correlação com label
            corr = spearman(feat_vals, labels)

            # Degradação de acurácia após permutação
            degradations = []
            for _ in range(n_permutations):
                permuted_vals = feat_vals[:]
                rng.shuffle(permuted_vals)
                permuted_features = [
                    {**f, feat: permuted_vals[i]}
                    for i, f in enumerate(features_list)
                ]
                perm_acc = accuracy(permuted_features, labels)
                degradations.append(baseline_acc - perm_acc)

            avg_degradation = sum(degradations) / len(degradations)
            std_degradation = statistics.stdev(degradations) if len(degradations) > 1 else 0

            results[feat] = {
                "spearman_corr":    round(corr, 4),
                "perm_importance":  round(avg_degradation, 4),
                "perm_std":         round(std_degradation, 4),
                "baseline_acc":     round(baseline_acc, 4),
                "schema_weight":    CURRENT_SCHEMA.features.get(feat, FeatureDef(0,"",-1)).weight,
            }

        # ── Normaliza importâncias relativas ────────────────────
        total_imp = sum(max(v["perm_importance"], 0) for v in results.values())
        for feat in results:
            imp = max(results[feat]["perm_importance"], 0)
            results[feat]["relative_importance"] = round(imp / total_imp if total_imp > 0 else 0, 4)

        return dict(sorted(
            results.items(),
            key=lambda x: x[1]["perm_importance"],
            reverse=True,
        ))

    @staticmethod
    def format_report(importance: dict[str, dict]) -> str:
        lines = ["Feature Importance Report", "=" * 60]
        lines.append(f"  {'Feature':<20} {'Corr':>6} {'PermImp':>8} {'RelImp':>8} {'Weight':>7}")
        lines.append("  " + "-" * 56)
        for feat, metrics in importance.items():
            lines.append(
                f"  {feat:<20} {metrics['spearman_corr']:>6.3f} "
                f"{metrics['perm_importance']:>8.4f} "
                f"{metrics['relative_importance']:>7.1%} "
                f"{metrics['schema_weight']:>7.0%}"
            )
        return "\n".join(lines)


# ── Governance singleton ──────────────────────────────────────────────────────

class FeatureGovernance:
    """
    Ponto central de acesso ao sistema de governance.
    Singleton — use `governance` exportado abaixo.
    """

    def __init__(self) -> None:
        self.schema   = CURRENT_SCHEMA
        self.leakage  = LeakageGuard()
        self.drift    = DriftMonitor()
        self._feature_history: deque = deque(maxlen=10000)
        # Auto-load baseline gerado por feature_analysis.py (se disponível)
        self.drift.load_baseline_from_file()

    def record_live(self, symbol: str, features: dict[str, float]) -> None:
        """Registra features ao vivo para monitoramento de drift."""
        self.drift.record(features)
        self._feature_history.append({
            "ts":      datetime.now(UTC).isoformat(),
            "symbol":  symbol,
            **features,
        })

    def validate_schema(self) -> list[str]:
        """Valida o schema atual. Retorna violações encontradas."""
        violations = self.leakage.validate_schema(self.schema)
        if not violations:
            logger.info("FeatureSchema v%s: VÁLIDO (sem violações)", self.schema.version)
        else:
            for v in violations:
                logger.warning("Schema violation: %s", v)
        return violations

    def status(self) -> dict:
        """Status resumido do governance."""
        drift_report = self.drift.drift_report()
        violations   = self.validate_schema()
        return {
            "schema_version":  self.schema.version,
            "schema_valid":    len(violations) == 0,
            "violations":      violations,
            "drift_status":    drift_report["overall_status"],
            "drift_alerts":    drift_report["alerts"],
            "obs_count":       drift_report["obs_count"],
            "features":        list(self.schema.features.keys()),
        }


# Singleton global
governance = FeatureGovernance()
