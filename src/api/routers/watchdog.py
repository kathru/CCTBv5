"""Watchdog status + kill switch management."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/watchdog", tags=["watchdog"])


@router.get("/status")
async def get_watchdog_status(request: Request) -> dict:
    """Return status of all registered watchdogs."""
    state = request.app.state
    result = {}

    if hasattr(state, "ws_watchdog") and state.ws_watchdog:
        result["websocket"] = state.ws_watchdog.status()

    if hasattr(state, "heartbeat_watchdog") and state.heartbeat_watchdog:
        result["heartbeat"] = state.heartbeat_watchdog.status()

    if hasattr(state, "resource_watchdog") and state.resource_watchdog:
        result["resources"] = state.resource_watchdog.status()

    return result


@router.post("/reset-kill-switch")
async def reset_kill_switch(request: Request) -> dict:
    """Manually reset soft kill switch when system is healthy."""
    loop = request.app.state.trading_loop
    ks = getattr(loop, "_kill_switch", None) if loop else None

    if not ks:
        return {"ok": False, "message": "Kill switch não encontrado"}

    if not ks.is_armed:
        return {"ok": True, "message": "Kill switch já estava inativo"}

    ks_reason = getattr(getattr(ks, "_current_event", None), "reason", "")
    reset_ok = ks.reset_soft(reason="manual_reset_api")

    # Re-open OMS gate
    oms = getattr(loop, "_oms", None)
    if oms:
        oms.open_gate()

    return {
        "ok": reset_ok,
        "previous_reason": ks_reason,
        "message": "Kill switch resetado manualmente" if reset_ok else "Falhou (pode ser HARD)",
    }
