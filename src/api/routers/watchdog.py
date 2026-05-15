"""Watchdog status endpoint — read-only."""
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
