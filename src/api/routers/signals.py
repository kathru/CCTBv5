"""Signals router — expõe o log de auditoria de sinais."""

import json
from pathlib import Path

from fastapi import APIRouter, Query

from ...monitoring.signal_log import signal_audit_log

router = APIRouter(prefix="/api/signals", tags=["signals"])

# Cache injetado no startup via dependency
_cache = None

def set_cache(cache) -> None:  # noqa: ANN001
    global _cache  # noqa: PLW0603
    _cache = cache
CALIB_PATH = Path("data") / "models" / "calibration_coef.json"


@router.get("/log")
async def get_signal_log(limit: int = Query(default=100, le=500)) -> dict:
    """Últimas N avaliações de sinal com resultado e motivo."""
    return {
        "entries":   signal_audit_log.recent(limit),
        "stats":     signal_audit_log.stats(),
        "by_symbol": signal_audit_log.symbol_stats(),
    }


@router.get("/calibration")
async def get_calibration() -> dict:
    """Dados de calibração Platt (do arquivo calibration_coef.json)."""
    if not CALIB_PATH.exists():
        return {"available": False}
    try:
        data = json.loads(CALIB_PATH.read_text())
        return {"available": True, **data}
    except Exception:
        return {"available": False}


@router.get("/funnel")
async def get_signal_funnel(  # noqa: ARG001
    window: int = Query(default=0, description="Minutos (0=todos)"),
) -> dict:
    """Funil de filtragem: onde cada sinal é bloqueado."""
    return {
        "all_time": signal_audit_log.funnel(window_minutes=None),
        "last_1h":  signal_audit_log.funnel(window_minutes=60),
        "last_6h":  signal_audit_log.funnel(window_minutes=360),
    }


@router.get("/symbols")
async def get_symbol_analysis() -> dict:
    """Análise mais recente por símbolo (regime, score, threshold, factors, ticker 24h)."""
    latest: dict = {}
    for entry in signal_audit_log._entries:
        if entry.symbol not in latest:
            latest[entry.symbol] = entry.to_dict()

    # Enriquece com dados de ticker 24h do cache
    if _cache:
        for symbol in list(latest.keys()):
            ticker = await _cache.get_ticker(symbol)
            if ticker:
                latest[symbol]["change_pct"] = ticker.get("change_pct", 0.0)
                latest[symbol]["volume_24h"] = ticker.get("volume_24h", 0.0)

    return latest
