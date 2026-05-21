"""Watchdog status + kill switch management."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/watchdog", tags=["watchdog"])


@router.post("/sync-exchange")
async def sync_with_exchange(request: Request) -> dict:
    """Sincroniza estado local com OKX: saldo, posições e histórico de ordens."""
    loop = getattr(request.app.state, "trading_loop", None)
    if not loop:
        return {"ok": False, "message": "TradingLoop não encontrado"}
    try:
        from ...recovery.exchange_sync import ExchangeSync
        sync = ExchangeSync(
            exchange=loop._okx,
            db=loop._db,
            cache=loop._cache,
            portfolio=loop._portfolio,
        )
        result = await sync.run()
        # Atualiza runner com equity total
        total_eq = result["total_equity_usd"]
        if total_eq > 0:
            loop._cash = result["usdt_balance"]
            loop._runner.update_portfolio_value(total_eq)
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


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

    if ks.allows_new_entries:
        return {"ok": True, "message": "Sistema já está ativo (sem kill switch)"}

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
