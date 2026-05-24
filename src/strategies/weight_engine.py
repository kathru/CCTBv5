"""
StrategyWeightEngine — Capital Allocator por Regime com Online Learning.

Arquitetura de convergência simulação → realidade:

  1. PRIOR (simulação):
     Calibrado uma vez por scripts/regime_backtest.py com 6+ meses de dados.
     Representa o "conhecimento inicial" sobre qual estratégia performa
     melhor em cada regime, baseado em backtests históricos.

  2. LIKELIHOOD (trades reais):
     A cada trade fechado, atualiza stats reais para o par (strategy, regime).
     Com poucos trades: dados reais têm pouca influência (prior domina).
     Com muitos trades: dados reais dominam (prior torna-se irrelevante).

  3. POSTERIOR (peso blendado):
     weight = (1 - alpha) × sim_weight + alpha × real_weight
     alpha  = min(n_real_trades / CONVERGENCE_N, 1.0)

  Convergência por par (strategy, regime):
    0 trades   → 100% simulação
    10 trades  → 33% real + 67% simulação
    20 trades  → 67% real + 33% simulação
    30+ trades → 100% real  (prior descartado)

Uso operacional:
  O weight é aplicado como multiplicador no kelly_fraction do sinal:
    final_kelly = signal.kelly_fraction × strategy_weight(strategy_id, regime)

  Pesos são relativos (0.1 → mínimo confiança, 1.0 → confiança total,
  >1.0 não é usado — o limite é o kelly base).

  Isso cria assimetria por regime:
    TREND_EXPANSION   → momentum × 1.0, reversal × 0.20
    MEAN_REVERTING_CHOP → momentum × 0.25, reversal × 1.0
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuração de convergência ──────────────────────────────────────────────

CONVERGENCE_N = 30    # trades reais para 100% peso real por par (strategy, regime)
MIN_INFLUENCE  = 5    # mínimo de trades reais para ter qualquer influência

# Arquivo de persistência
_ROOT         = Path(__file__).parent.parent.parent
WEIGHTS_PATH  = _ROOT / "data" / "models" / "regime_weights.json"

# ── Pesos padrão por simulação (prior de domain knowledge) ───────────────────
# Derivados de backtest Jan/2025–Mai/2026 + conhecimento teórico.
# Serão sobrescritos por regime_backtest.py quando rodar a simulação completa.
# Valores: 1.0 = confiança plena, 0.1 = mínima confiança (não bloqueia, apenas reduz)

SIM_WEIGHTS_DEFAULT: dict[str, dict[str, float]] = {
    # Tendência clara + volume confirmando → momentum domina
    "TREND_EXPANSION": {
        "momentum_v2":    1.00,
        "trend_v1":       0.80,
        "reversal_v1":    0.15,
    },
    # Pré-breakout, lateralização estreita → trend e momentum razoáveis
    "VOLATILITY_COMPRESSION": {
        "momentum_v2":    0.70,
        "trend_v1":       0.90,
        "reversal_v1":    0.20,
    },
    # Lateralização → reversal domina, momentum sofre
    "MEAN_REVERTING_CHOP": {
        "momentum_v2":    0.25,
        "trend_v1":       0.20,
        "reversal_v1":    1.00,
    },
    # Trend fraco, volume baixo → apenas trend sobrevive
    "TREND_EXHAUSTION": {
        "momentum_v2":    0.15,
        "trend_v1":       0.60,
        "reversal_v1":    0.30,
    },
    # Alta correlação / risco sistêmico → todos reduzidos
    "HIGH_CORRELATION_RISK": {
        "momentum_v2":    0.20,
        "trend_v1":       0.15,
        "reversal_v1":    0.20,
    },
    # Bear / Panic → bloqueados pelo regime, mas se chegarem aqui: mínimo
    "BEAR_TREND": {
        "momentum_v2":    0.05,
        "trend_v1":       0.05,
        "reversal_v1":    0.10,
    },
    "PANIC_LIQUIDATION": {
        "momentum_v2":    0.05,
        "trend_v1":       0.05,
        "reversal_v1":    0.05,
    },
    # Fallback para regimes desconhecidos
    "UNKNOWN": {
        "momentum_v2":    0.50,
        "trend_v1":       0.50,
        "reversal_v1":    0.50,
    },
}


class WeightEngine:
    """
    Gerencia pesos de estratégia por regime com online learning.
    Thread-safe para leitura — escritas ocorrem apenas em _exit() do PositionMonitor.
    """

    def __init__(self, weights_path: Path = WEIGHTS_PATH) -> None:
        self._path = weights_path
        self._data = self._load()
        logger.info(
            "WeightEngine inicializado: %d regimes, %d pares (strategy, regime) com dados reais",
            len(self._data.get("weights", {})),
            self._count_real_pairs(),
        )

    # ── API pública ───────────────────────────────────────────────────────────

    def get_weight(self, strategy_id: str, regime: str) -> float:
        """
        Retorna o peso blendado para (strategy_id, regime).
        1.0 = confiança total (kelly inalterado)
        0.x = confiança reduzida (kelly × x)

        Nunca retorna 0.0 — floor em 0.05 para não silenciar estratégias.
        """
        weights = self._data.get("weights", {})
        regime_data = weights.get(regime) or weights.get("UNKNOWN", {})
        entry = regime_data.get(strategy_id)

        if not entry:
            # Estratégia nova — usa sim_default ou 0.5
            defaults = SIM_WEIGHTS_DEFAULT.get(regime, SIM_WEIGHTS_DEFAULT["UNKNOWN"])
            return max(defaults.get(strategy_id, 0.50), 0.05)

        sim_w  = entry.get("sim_weight", 0.50)
        n      = entry.get("n_real", 0)

        if n < MIN_INFLUENCE:
            # Poucos trades — usa apenas simulação
            return max(sim_w, 0.05)

        real_w = entry.get("real_weight", sim_w)
        alpha  = min(n / CONVERGENCE_N, 1.0)
        blended = (1 - alpha) * sim_w + alpha * real_w

        return max(round(blended, 4), 0.05)

    def record_trade(
        self,
        strategy_id: str,
        regime:      str,
        pnl:         float,
        entry_price: float = 0.0,
    ) -> None:
        """
        Registra um trade fechado para o par (strategy_id, regime).
        Atualiza win_rate, expectancy e real_weight em tempo real.
        Persiste no JSON após cada update.
        """
        weights = self._data.setdefault("weights", {})
        regime_data = weights.setdefault(
            regime,
            {s: self._default_entry(s, regime) for s in SIM_WEIGHTS_DEFAULT.get(regime, {})}
        )

        if strategy_id not in regime_data:
            regime_data[strategy_id] = self._default_entry(strategy_id, regime)

        entry = regime_data[strategy_id]

        # Atualiza stats online com EMA (Exponential Moving Average)
        # Window efetivo: últimos ~20 trades têm maior peso
        alpha_ema = 0.05   # suavização: ~1/alpha = 20 trades de janela efetiva
        n = entry.get("n_real", 0) + 1
        entry["n_real"] = n

        won = pnl > 0
        old_wr = entry.get("real_win_rate", 0.5 if n == 1 else entry.get("sim_weight", 0.5))
        new_wr = old_wr + alpha_ema * (float(won) - old_wr)   # EMA win rate

        old_exp = entry.get("real_expectancy", 0.0)
        norm_pnl = pnl / max(abs(entry_price), 1.0) if entry_price else pnl
        new_exp = old_exp + alpha_ema * (norm_pnl - old_exp)   # EMA expectancy

        entry["real_win_rate"]    = round(new_wr, 4)
        entry["real_expectancy"]  = round(new_exp, 6)

        # real_weight: combinação de win_rate e expectancy normalizada
        # floor 0.05, cap 1.20 (permite leve boost para estratégias excepcionais)
        wr_component  = new_wr * 2.0   # 0→0, 0.5→1.0, 0.6→1.2
        exp_component = min(max(new_exp * 50 + 1.0, 0.0), 1.5)
        real_w = min(max((wr_component + exp_component) / 2, 0.05), 1.20)
        entry["real_weight"] = round(real_w, 4)

        # Recalcula blended
        entry["blended_weight"] = round(self.get_weight(strategy_id, regime), 4)

        entry["last_updated"] = datetime.now(UTC).isoformat()
        entry["confidence"] = (
            "real"       if n >= CONVERGENCE_N else
            "mixed"      if n >= MIN_INFLUENCE else
            "simulation"
        )

        logger.info(
            "WeightEngine update %s/%s: trade #%d pnl=%.4f won=%s "
            "→ wr=%.2f%% exp=%.4f real_w=%.2f blended=%.2f confidence=%s",
            strategy_id, regime, n, pnl, won,
            new_wr * 100, new_exp, real_w, entry["blended_weight"],
            entry["confidence"],
        )

        self._data["updated_at"] = datetime.now(UTC).isoformat()
        self._save()

    def update_from_simulation(
        self,
        regime: str,
        strategy_id: str,
        sim_weight: float,
        sim_win_rate: float | None = None,
        sim_expectancy: float | None = None,
        n_sim_trades: int = 0,
    ) -> None:
        """
        Chamado por regime_backtest.py para atualizar os pesos de simulação.
        Preserva dados reais existentes (não sobrescreve n_real, real_win_rate, etc.).
        """
        weights = self._data.setdefault("weights", {})
        regime_data = weights.setdefault(regime, {})

        existing = regime_data.get(strategy_id, {})
        existing["sim_weight"]       = round(sim_weight, 4)
        existing["sim_win_rate"]     = sim_win_rate
        existing["sim_expectancy"]   = sim_expectancy
        existing["n_sim_trades"]     = n_sim_trades
        existing["sim_updated_at"]   = datetime.now(UTC).isoformat()

        # Recalcula blended preservando dados reais
        n_real = existing.get("n_real", 0)
        if n_real < MIN_INFLUENCE:
            existing["blended_weight"] = round(sim_weight, 4)
            existing["confidence"]     = "simulation"
        else:
            existing["blended_weight"] = round(self.get_weight(strategy_id, regime), 4)
            existing["confidence"]     = "mixed" if n_real < CONVERGENCE_N else "real"

        regime_data[strategy_id] = existing
        self._data["updated_at"] = datetime.now(UTC).isoformat()
        self._save()
        logger.info(
            "WeightEngine sim update %s/%s: sim_w=%.2f wr=%.1f%% n=%d",
            strategy_id, regime, sim_weight,
            (sim_win_rate or 0) * 100, n_sim_trades,
        )

    def summary(self) -> dict:
        """Retorna resumo dos pesos atuais para o dashboard/API."""
        result = {}
        for regime, strats in self._data.get("weights", {}).items():
            result[regime] = {}
            for sid, entry in strats.items():
                result[regime][sid] = {
                    "blended":    entry.get("blended_weight", 0.5),
                    "sim":        entry.get("sim_weight", 0.5),
                    "real":       entry.get("real_weight"),
                    "n_real":     entry.get("n_real", 0),
                    "confidence": entry.get("confidence", "simulation"),
                    "win_rate":   entry.get("real_win_rate"),
                }
        return result

    # ── Internos ──────────────────────────────────────────────────────────────

    def _default_entry(self, strategy_id: str, regime: str) -> dict:
        """Cria entrada padrão com sim_weight do prior."""
        sim_w = SIM_WEIGHTS_DEFAULT.get(regime, {}).get(strategy_id, 0.50)
        return {
            "sim_weight":    sim_w,
            "real_weight":   None,
            "blended_weight": sim_w,
            "n_real":        0,
            "real_win_rate": None,
            "real_expectancy": None,
            "confidence":    "simulation",
            "last_updated":  None,
        }

    def _count_real_pairs(self) -> int:
        count = 0
        for strats in self._data.get("weights", {}).values():
            for entry in strats.values():
                if entry.get("n_real", 0) >= MIN_INFLUENCE:
                    count += 1
        return count

    def _load(self) -> dict:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.debug("WeightEngine: carregado de %s", self._path)
                return data
            except Exception as exc:
                logger.warning("WeightEngine: falha ao carregar %s: %s", self._path, exc)

        # Inicializa com defaults de simulação
        logger.info("WeightEngine: inicializando com pesos de domain knowledge")
        data: dict = {
            "version":    "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "weights":    {},
        }
        for regime, strats in SIM_WEIGHTS_DEFAULT.items():
            data["weights"][regime] = {
                sid: {
                    "sim_weight":     w,
                    "real_weight":    None,
                    "blended_weight": w,
                    "n_real":         0,
                    "real_win_rate":  None,
                    "real_expectancy": None,
                    "confidence":     "simulation",
                    "last_updated":   None,
                }
                for sid, w in strats.items()
            }
        self._save(data)
        return data

    def _save(self, data: dict | None = None) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            (data or self._data)["updated_at"] = datetime.now(UTC).isoformat()
            self._path.write_text(
                json.dumps(data or self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("WeightEngine: falha ao salvar: %s", exc)


# ── Singleton global ──────────────────────────────────────────────────────────
weight_engine = WeightEngine()
