"""
Signal Audit Log — registra cada avaliação de sinal com resultado completo.

Singleton acessível por qualquer estratégia. Ring buffer de 500 entradas em memória,
persistido no PostgreSQL para sobreviver a restarts.

Exposto via GET /api/signals/log para o dashboard.
"""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

MAX_ENTRIES = 500

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
    score:       float
    calibrated:  float
    threshold:   float
    ev:          float
    direction:   str
    result:      str
    detail:      str
    factors:     dict = field(default_factory=dict)
    strategy_id: str  = "momentum_v2"

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
            "strategy_id": self.strategy_id,
            "factors":     {
                k: round(v, 3) if isinstance(v, (int, float)) else v
                for k, v in self.factors.items()
            },
        }


class SignalAuditLog:
    """
    Ring buffer de avaliações de sinal — persistido no PostgreSQL.
    """

    def __init__(self, maxlen: int = MAX_ENTRIES) -> None:
        self._entries: deque[SignalAuditEntry] = deque(maxlen=maxlen)
        self._counters: dict[str, int] = {}
        self._started_at: datetime = datetime.now(UTC)
        self._db = None   # injetado via set_db() no boot

    def set_db(self, db: object) -> None:
        """Injeta o Database após inicialização assíncrona."""
        self._db = db

    def record(self, entry: SignalAuditEntry) -> None:
        self._entries.appendleft(entry)
        self._counters[entry.result] = self._counters.get(entry.result, 0) + 1
        # Persiste no PostgreSQL sem bloquear
        if self._db is not None:
            asyncio.create_task(self._persist(entry))

    async def _persist(self, entry: SignalAuditEntry) -> None:
        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO signal_evaluations
                        (ts, symbol, regime, score, calibrated, threshold,
                         ev, direction, result, detail, factors)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """,
                    entry.timestamp,
                    entry.symbol,
                    entry.regime,
                    entry.score,
                    entry.calibrated,
                    entry.threshold,
                    entry.ev,
                    entry.direction,
                    entry.result,
                    entry.detail,
                    json.dumps({k: round(v, 4) if isinstance(v, (int, float)) else v
                                for k, v in entry.factors.items()}),
                )
        except Exception as exc:
            logger.debug("signal_log persist error: %s", exc)

    async def restore_from_db(self, db: object) -> None:
        """
        Chamado no boot — carrega os últimos MAX_ENTRIES do PostgreSQL
        para restaurar o histórico após restart.
        """
        self.set_db(db)
        try:
            async with db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ts, symbol, regime, score, calibrated, threshold,
                           ev, direction, result, detail, factors
                    FROM signal_evaluations
                    ORDER BY ts DESC
                    LIMIT $1
                    """,
                    MAX_ENTRIES,
                )
            loaded = 0
            for row in reversed(rows):   # oldest-first para appendleft ficar certo
                factors = json.loads(row["factors"]) if row["factors"] else {}
                entry = SignalAuditEntry(
                    timestamp=row["ts"],
                    symbol=row["symbol"],
                    regime=row["regime"],
                    score=float(row["score"]),
                    calibrated=float(row["calibrated"]),
                    threshold=float(row["threshold"]),
                    ev=float(row["ev"]),
                    direction=row["direction"],
                    result=row["result"],
                    detail=row["detail"],
                    factors=factors,
                    strategy_id=factors.pop("_strategy_id", "momentum_v2"),
                )
                self._entries.appendleft(entry)
                self._counters[entry.result] = self._counters.get(entry.result, 0) + 1
                loaded += 1
            logger.info("SignalAuditLog: %d entradas restauradas do PostgreSQL", loaded)
        except Exception as exc:
            logger.warning("SignalAuditLog: falha ao restaurar do DB: %s", exc)

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
        by_sym: dict[str, dict] = {}
        for e in self._entries:
            s = by_sym.setdefault(e.symbol, {"total": 0, "signals": 0})
            s["total"] += 1
            if e.result == "SIGNAL":
                s["signals"] += 1
        return by_sym

    def funnel(self, window_minutes: int | None = None) -> dict:
        FUNNEL_ORDER = [
            ("NO_CANDLES",     "Sem candles",       "var(--muted)"),
            ("REGIME_BLOCKED", "Regime bloqueado",  "var(--red)"),
            ("SCORE_LOW",      "Score baixo",       "var(--yellow)"),
            ("EV_LOW",         "EV negativo",       "var(--orange)"),
            ("RISK_BLOCKED",   "Risk Engine",       "var(--red)"),
            ("GATE_CLOSED",    "Gate fechado",      "var(--red)"),
            ("DIRECTION_FLAT", "Direção plana",     "var(--muted)"),
            ("SIGNAL",         "Sinal executado",   "var(--green)"),
        ]

        if window_minutes is None:
            counts = dict(self._counters)
        else:
            cutoff = datetime.now(UTC).timestamp() - window_minutes * 60
            counts: dict[str, int] = {}
            for e in self._entries:
                if e.timestamp.timestamp() >= cutoff:
                    counts[e.result] = counts.get(e.result, 0) + 1

        total = sum(counts.values()) or 1
        signals = counts.get("SIGNAL", 0)

        steps = []
        for result_key, label, color in FUNNEL_ORDER:
            n = counts.get(result_key, 0)
            if n == 0 and result_key not in ("SIGNAL",):
                continue
            steps.append({
                "result":  result_key,
                "label":   label,
                "color":   color,
                "count":   n,
                "pct":     round(100 * n / total, 1),
                "bar_pct": round(100 * n / total, 1),
            })

        by_sym: dict[str, dict] = {}
        entries_src = self._entries if window_minutes is None else [
            e for e in self._entries
            if e.timestamp.timestamp() >= (
                datetime.now(UTC).timestamp() - (window_minutes or 0) * 60
            )
        ]
        for e in entries_src:
            s = by_sym.setdefault(e.symbol, {})
            s[e.result] = s.get(e.result, 0) + 1

        return {
            "total":            total,
            "signals":          signals,
            "blocked":          total - signals,
            "execution_rate":   round(100 * signals / total, 2),
            "window_minutes":   window_minutes,
            "steps":            steps,
            "by_symbol":        by_sym,
        }


# ── Singleton global ──────────────────────────────────────────────────────────
signal_audit_log = SignalAuditLog()
