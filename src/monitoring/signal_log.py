"""
Signal Audit Log — registra cada avaliação de sinal com resultado completo.

Singleton acessível por qualquer estratégia. Ring buffer de 500 entradas.
Exposto via GET /api/signals/log para o dashboard.

Resultado de cada avaliação:
  SIGNAL          → passou todos os filtros, sinal gerado
  NO_CANDLES      → candles insuficientes
  REGIME_BLOCKED  → regime PANIC ou VACUUM
  SCORE_LOW       → score calibrado abaixo do threshold
  EV_LOW          → expected value abaixo do mínimo
  DIRECTION_FLAT  → sem direção clara (preço lateral)
  GATE_CLOSED     → OMS gate fechado (reconciliação ou kill switch)
  RISK_BLOCKED    → bloqueado pelo RiskEngine
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

MAX_ENTRIES = 500

# Ícones por resultado
RESULT_ICON = {
    "SIGNAL":         "🟢",
    "NO_CANDLES":     "⚪",
    "REGIME_BLOCKED": "🔴",
    "SCORE_LOW":      "🟡",
    "EV_LOW":         "🟠",
    "DIRECTION_FLAT": "⚪",
    "GATE_CLOSED":    "🔴",
    "RISK_BLOCKED":   "🔴",
}


@dataclass
class SignalAuditEntry:
    timestamp:   datetime
    symbol:      str
    regime:      str
    score:       float        # score bruto (0–1)
    calibrated:  float        # score calibrado (probabilidade)
    threshold:   float        # threshold do regime
    ev:          float        # expected value
    direction:   str          # LONG | FLAT | N/A
    result:      str          # ver constantes acima
    detail:      str          # mensagem legível do motivo
    factors:     dict = field(default_factory=dict)   # m1, m2, m3, m4

    def to_dict(self) -> dict:
        return {
            "ts":          self.timestamp.isoformat(),
            "symbol":      self.symbol,
            "regime":      self.regime,
            "score":       round(self.score, 4),
            "calibrated":  round(self.calibrated, 4),
            "threshold":   round(self.threshold, 4),
            "ev":          round(self.ev, 4),
            "direction":   self.direction,
            "result":      self.result,
            "detail":      self.detail,
            "icon":        RESULT_ICON.get(self.result, "⚪"),
            "factors":     {k: round(v, 3) for k, v in self.factors.items()},
        }


class SignalAuditLog:
    """
    Ring buffer thread-safe (asyncio single-thread) de avaliações de sinal.
    Singleton — use `signal_audit_log` exportado abaixo.
    """

    def __init__(self, maxlen: int = MAX_ENTRIES) -> None:
        self._entries: deque[SignalAuditEntry] = deque(maxlen=maxlen)
        self._counters: dict[str, int] = {}
        self._started_at: datetime = datetime.now(UTC)

    def record(self, entry: SignalAuditEntry) -> None:
        self._entries.appendleft(entry)   # mais recente primeiro
        self._counters[entry.result] = self._counters.get(entry.result, 0) + 1

    def recent(self, limit: int = 100) -> list[dict]:
        return [e.to_dict() for e in list(self._entries)[:limit]]

    def stats(self) -> dict:
        total = sum(self._counters.values())
        signals = self._counters.get("SIGNAL", 0)
        uptime_min = (datetime.now(UTC) - self._started_at).total_seconds() / 60
        return {
            "total_evaluations": total,
            "signals_generated": signals,
            "signal_rate_pct":   round(100 * signals / total, 1) if total else 0,
            "uptime_minutes":    round(uptime_min, 1),
            "evals_per_minute":  round(total / uptime_min, 1) if uptime_min > 0 else 0,
            "by_result":         dict(self._counters),
            "started_at":        self._started_at.isoformat(),
        }

    def symbol_stats(self) -> dict:
        """Contagem de sinais gerados por símbolo."""
        by_sym: dict[str, dict] = {}
        for e in self._entries:
            s = by_sym.setdefault(e.symbol, {"total": 0, "signals": 0})
            s["total"] += 1
            if e.result == "SIGNAL":
                s["signals"] += 1
        return by_sym

    def funnel(self, window_minutes: int | None = None) -> dict:
        """
        Funil de filtragem: mostra onde cada sinal é bloqueado.

        window_minutes=None → todos os dados históricos (contadores)
        window_minutes=60   → apenas últimas 60 min (a partir das entries)
        """
        # Ordem lógica do funil (mais cedo para mais tarde no pipeline)
        FUNNEL_ORDER = [
            ("NO_CANDLES",     "Sem candles",          "var(--muted)"),
            ("REGIME_BLOCKED", "Regime bloqueado",     "var(--red)"),
            ("SCORE_LOW",      "Score baixo",          "var(--yellow)"),
            ("EV_LOW",         "EV negativo",          "var(--orange)"),
            ("RISK_BLOCKED",   "Risk Engine",          "var(--red)"),
            ("GATE_CLOSED",    "Gate fechado",         "var(--red)"),
            ("DIRECTION_FLAT", "Direção plana",        "var(--muted)"),
            ("SIGNAL",         "Sinal executado",      "var(--green)"),
        ]

        if window_minutes is None:
            # Usa contadores acumulados (toda a sessão)
            counts = dict(self._counters)
        else:
            # Filtra entries pela janela de tempo
            cutoff = datetime.now(UTC).timestamp() - window_minutes * 60
            counts: dict[str, int] = {}
            for e in self._entries:
                if e.timestamp.timestamp() >= cutoff:
                    counts[e.result] = counts.get(e.result, 0) + 1

        total = sum(counts.values()) or 1
        signals = counts.get("SIGNAL", 0)
        blocked = total - signals

        steps = []
        for result_key, label, color in FUNNEL_ORDER:
            n = counts.get(result_key, 0)
            if n == 0 and result_key not in ("SIGNAL",):
                continue  # omite etapas sem ocorrências
            steps.append({
                "result":  result_key,
                "label":   label,
                "color":   color,
                "count":   n,
                "pct":     round(100 * n / total, 1),
                "bar_pct": round(100 * n / total, 1),
            })

        # Por símbolo
        by_sym: dict[str, dict] = {}
        entries_src = self._entries if window_minutes is None else [
            e for e in self._entries
            if e.timestamp.timestamp() >= (datetime.now(UTC).timestamp() - (window_minutes or 0) * 60)
        ]
        for e in entries_src:
            s = by_sym.setdefault(e.symbol, {})
            s[e.result] = s.get(e.result, 0) + 1

        return {
            "total":        total,
            "signals":      signals,
            "blocked":      blocked,
            "execution_rate": round(100 * signals / total, 2),
            "window_minutes": window_minutes,
            "steps":        steps,
            "by_symbol":    by_sym,
        }


# ── Singleton global ──────────────────────────────────────────────────────────
signal_audit_log = SignalAuditLog()
