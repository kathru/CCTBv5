"""Cross-asset engine status endpoint."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/cross_asset", tags=["cross_asset"])


@router.get("/status")
async def get_cross_asset_status(request: Request) -> dict:
    """Retorna estado atual do CrossAssetEngine (posição market-neutral ativa ou não)."""
    tl = getattr(request.app.state, "trading_loop", None)
    if tl is None:
        return {"active": False, "error": "trading_loop not available"}

    engine = getattr(tl, "_cross_asset", None)
    if engine is None:
        return {"active": False, "error": "CrossAssetEngine not initialized"}

    try:
        status = await engine.get_status()
        return status
    except Exception as exc:
        return {"active": False, "error": str(exc)}
