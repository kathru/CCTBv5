"""Feature Governance router — schema, drift e importância de features."""

from fastapi import APIRouter

from ...monitoring.feature_governance import CURRENT_SCHEMA, governance

router = APIRouter(prefix="/api/governance", tags=["governance"])


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
