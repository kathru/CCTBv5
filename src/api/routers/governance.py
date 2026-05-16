"""Feature Governance router — schema, drift, importância e WFO."""

import json
from pathlib import Path

from fastapi import APIRouter

from ...monitoring.feature_governance import CURRENT_SCHEMA, governance

router = APIRouter(prefix="/api/governance", tags=["governance"])

_ROOT         = Path(__file__).parent.parent.parent.parent
_MODELS_DIR   = _ROOT / "data" / "models"


@router.get("/status")
async def get_governance_status() -> dict:
    """Status geral do feature governance."""
    return governance.status()


@router.get("/schema")
async def get_schema() -> dict:
    """Schema atual das features com definições e pesos."""
    return CURRENT_SCHEMA.to_dict()


@router.get("/drift")
async def get_drift_report() -> dict:
    """Relatório de drift das features ao vivo vs baseline de treino."""
    return governance.drift.drift_report()


@router.get("/live-stats")
async def get_live_stats() -> dict:
    """Estatísticas das features ao vivo (últimas 500 observações)."""
    stats = governance.drift.live_stats()
    return {k: v.to_dict() for k, v in stats.items()}


@router.get("/importance")
async def get_feature_importance() -> dict:
    """Feature importance calculada por permutação (via feature_analysis.py)."""
    path = _MODELS_DIR / "feature_importance.json"
    if not path.exists():
        return {"available": False, "message": "Rode: python scripts/feature_analysis.py"}
    try:
        return {"available": True, **json.loads(path.read_text())}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@router.get("/wfo")
async def get_wfo_results() -> dict:
    """Resultados do Walk-Forward Optimization por símbolo."""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    results = {}
    for sym in symbols:
        path = _MODELS_DIR / f"wfo_{sym.replace('-','_')}_latest.json"
        if path.exists():
            try:
                results[sym] = json.loads(path.read_text())
            except Exception:
                pass
    if not results:
        return {"available": False, "message": "Rode: python scripts/walk_forward.py"}
    return {"available": True, "symbols": results}
