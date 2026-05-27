"""Reconciler status endpoint."""
import json

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/reconciler", tags=["reconciler"])


@router.get("/status")
async def get_reconciler_status(request: Request) -> dict:
    """Último relatório de conciliação OKX ↔ DB."""
    cache = getattr(request.app.state, "cache", None)
    if not cache:
        return {"available": False}

    raw = await cache.get("reconciler:last_report")
    if not raw:
        return {
            "available":   False,
            "message":     "Nenhum ciclo de conciliação executado ainda.",
        }

    try:
        report = raw if isinstance(raw, dict) else json.loads(raw)
        report["available"] = True
        return report
    except Exception:
        return {"available": False, "error": "parse error"}
