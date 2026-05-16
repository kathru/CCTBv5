"""Signals router — expõe o log de auditoria de sinais."""

from fastapi import APIRouter, Query

from ...monitoring.signal_log import signal_audit_log

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/log")
async def get_signal_log(limit: int = Query(default=100, le=500)) -> dict:
    """Últimas N avaliações de sinal com resultado e motivo."""
    return {
        "entries":   signal_audit_log.recent(limit),
        "stats":     signal_audit_log.stats(),
        "by_symbol": signal_audit_log.symbol_stats(),
    }


@router.get("/symbols")
async def get_symbol_analysis() -> dict:
    """Análise mais recente por símbolo (regime, score, threshold, factors)."""
    latest: dict = {}
    for entry in signal_audit_log._entries:
        if entry.symbol not in latest:
            latest[entry.symbol] = entry.to_dict()
    return latest
