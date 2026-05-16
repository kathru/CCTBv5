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
from typing import Deque

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
        self._entries: Deque[SignalAuditEntry] = deque(maxlen=maxlen)
        self._counters: dict[str, int] = {}

    def record(self, entry: SignalAuditEntry) -> None:
        self._entries.appendleft(entry)   # mais recente primeiro
        self._counters[entry.result] = self._counters.get(entry.result, 0) + 1

    def recent(self, limit: int = 100) -> list[dict]:
        return [e.to_dict() for e in list(self._entries)[:limit]]

    def stats(self) -> dict:
        total = sum(self._counters.values())
        signals = self._counters.get("SIGNAL", 0)
        return {
            "total_evaluations": total,
            "signals_generated": signals,
            "signal_rate_pct":   round(100 * signals / total, 1) if total else 0,
            "by_result":         dict(self._counters),
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


# ── Singleton global ──────────────────────────────────────────────────────────
signal_audit_log = SignalAuditLog()
