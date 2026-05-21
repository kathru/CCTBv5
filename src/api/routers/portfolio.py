"""
Portfolio router — expõe estado do PortfolioEngine e métricas de performance.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def _serialize(v):
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {k: _serialize(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_serialize(i) for i in v]
    return str(v)


@router.get("/exits")
async def exit_plans(request: Request) -> dict:
    """Status dos planos de saída ativos (PositionMonitor)."""
    monitor = getattr(request.app.state, "position_monitor", None)
    if monitor is None:
        return {"available": False, "active_plans": 0, "plans": {}}
    return monitor.status()


@router.get("/summary")
async def portfolio_summary(request: Request) -> dict:
    """
    Estado completo do portfolio: capital, P&L, risco, exposição.
    Retorna zeros se o TradingLoop ainda não iniciou.
    """
    portfolio = getattr(request.app.state, "portfolio", None)
    if portfolio is None:
        return {
            "available": False,
            "initial_capital":    85_000.0,
            "total_value":        85_000.0,
            "cash_available":     85_000.0,
            "total_exposure_pct": 0.0,
            "open_position_count": 0,
            "realized_pnl":       0.0,
            "unrealized_pnl":     0.0,
            "daily_pnl":          0.0,
            "total_return_pct":   0.0,
            "drawdown_pct":       0.0,
            "portfolio_beta":     0.0,
            "avg_correlation":    0.0,
            "concentration_risk": 0.0,
            "updated_at":         datetime.now(UTC).isoformat(),
        }

    s = portfolio.state

    db = getattr(request.app.state, "db", None)
    cache = getattr(request.app.state, "cache", None)
    initial = s.initial_capital if s.initial_capital > 1000.0 else 85_000.0

    # ── Posições abertas do DB (para open_count e exposição da exchange_sync) ──
    from ...persistence.repositories.positions import PositionRepository
    db_open_positions = []
    if db:
        try:
            repo = PositionRepository(db)
            db_open_positions = await repo.get_open()
        except Exception:
            pass
    open_count = max(s.open_position_count, len(db_open_positions))
    # será recalculado após derivar positions_data das ordens

    # ── P&L Realizado: cost-basis por símbolo ───────────────────────────────────
    # Fórmula correta: realized = sell_notional - (sell_qty × avg_buy_price)
    # Evita contar como "custo" o capital ainda investido em posições abertas.
    realized_pnl = 0.0
    order_stats: dict[str, dict] = {}  # symbol → {buy_qty, buy_notional, sell_qty, sell_notional}
    if db:
        try:
            rows = await db.fetch(
                """
                SELECT symbol, side,
                       SUM(filled_quantity)                        AS qty,
                       SUM(filled_quantity * avg_fill_price)       AS notional,
                       SUM(COALESCE(fees_paid, 0))                 AS fees
                FROM orders
                WHERE status='filled' AND strategy_id != 'exchange_sync'
                GROUP BY symbol, side
                """
            )
            for r in rows:
                sym  = r["symbol"]
                side = str(r["side"]).upper()
                if sym not in order_stats:
                    order_stats[sym] = {"buy_qty": 0.0, "buy_notional": 0.0,
                                        "sell_qty": 0.0, "sell_notional": 0.0, "fees": 0.0}
                if side in ("BUY", "LONG"):
                    order_stats[sym]["buy_qty"]      += float(r["qty"] or 0)
                    order_stats[sym]["buy_notional"] += float(r["notional"] or 0)
                elif side in ("SELL", "SHORT"):
                    order_stats[sym]["sell_qty"]      += float(r["qty"] or 0)
                    order_stats[sym]["sell_notional"] += float(r["notional"] or 0)
                order_stats[sym]["fees"] += float(r["fees"] or 0)

            for _sym, st in order_stats.items():
                if st["buy_qty"] > 0 and st["sell_qty"] > 0:
                    avg_buy_px = st["buy_notional"] / st["buy_qty"]
                    # P&L das unidades já vendidas = recebido - custo das vendas
                    realized_pnl += st["sell_notional"] - (st["sell_qty"] * avg_buy_px)
            realized_pnl = round(realized_pnl, 4)
        except Exception:
            pass

    # ── Posições abertas do bot (ordens filled: buy_qty - sell_qty) ────────────
    positions_data: dict[str, dict] = {}
    notional_bot = 0.0
    unrealized_bot = 0.0
    if cache and order_stats:
        for sym, st in order_stats.items():
            open_qty = st["buy_qty"] - st["sell_qty"]
            if open_qty <= 1e-8 or st["buy_qty"] <= 0:
                continue
            avg_buy_px = st["buy_notional"] / st["buy_qty"]
            price_raw  = await cache.get_price(sym)
            price      = float(price_raw) if price_raw else avg_buy_px
            notional   = open_qty * price
            unreal     = (price - avg_buy_px) * open_qty
            notional_bot   += notional
            unrealized_bot += unreal
            positions_data[sym] = {
                "quantity":       round(open_qty, 6),
                "avg_entry":      round(avg_buy_px, 4),
                "current_price":  round(price, 4),
                "notional":       round(notional, 4),
                "unrealized_pnl": round(unreal, 4),
                "strategy_id":    "momentum_v2",
            }

    # ── Posições exchange_sync (BTC/ETH/SOL pré-existentes) ────────────────────
    # Incluídas no portfolio total, no P&L não realizado e no saldo disponível.
    notional_sync  = 0.0
    unrealized_sync = 0.0
    for pos in db_open_positions:
        strat = pos.get("strategy_id", "") if isinstance(pos, dict) else getattr(pos, "strategy_id", "")
        if str(strat) != "exchange_sync":
            continue
        sym   = pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", "")
        qty   = float(pos.get("quantity", 0) if isinstance(pos, dict) else getattr(pos, "quantity", 0))
        entry = float(pos.get("avg_entry_price", 0) if isinstance(pos, dict) else getattr(pos, "avg_entry_price", 0))
        if not cache or not sym or qty <= 0:
            continue
        price_raw = await cache.get_price(sym)
        price     = float(price_raw) if price_raw else (entry or 0.0)
        notional  = qty * price
        unreal    = (price - entry) * qty if entry > 0 else 0.0
        notional_sync   += notional
        unrealized_sync += unreal
        # Mescla com posição do bot (se houver) — soma quantidades e notionals
        if sym in positions_data:
            existing = positions_data[sym]
            merged_qty      = existing["quantity"] + qty
            merged_notional = existing["notional"] + notional
            merged_unreal   = existing["unrealized_pnl"] + unreal
            # Preço médio ponderado
            merged_entry    = (existing["avg_entry"] * existing["quantity"] + entry * qty) / merged_qty if merged_qty > 0 else entry
            positions_data[sym] = {
                "quantity":       round(merged_qty, 6),
                "avg_entry":      round(merged_entry, 4),
                "current_price":  round(price, 4),
                "notional":       round(merged_notional, 4),
                "unrealized_pnl": round(merged_unreal, 4),
                "strategy_id":    "combined",
            }
        else:
            positions_data[sym] = {
                "quantity":       round(qty, 6),
                "avg_entry":      round(entry, 4),
                "current_price":  round(price, 4),
                "notional":       round(notional, 4),
                "unrealized_pnl": round(unreal, 4),
                "strategy_id":    "exchange_sync",
            }

    # ── Portfolio total = USDT + BTC + ETH + SOL (tudo a preço atual) ──────────
    cash_value    = s.cash_available
    notional_all  = notional_bot + notional_sync          # toda crypto
    unrealized    = unrealized_bot + unrealized_sync       # P&L não realizado total
    total_value   = cash_value + notional_all             # portfolio completo

    # Exposição = crypto / total (incluindo holdings pré-existentes)
    exposure_pct     = notional_all / total_value if total_value > 0 and notional_all > 0 else 0.0
    total_return_pct = (total_value - initial) / initial if initial > 0 else 0.0

    # Saldo disponível para o dashboard = USDT + toda crypto (pode vender tudo)
    liquid_total = total_value

    open_count = len(positions_data)

    return {
        "available":           True,
        "initial_capital":     initial,
        "total_value":         round(total_value, 2),
        "cash_available":      round(cash_value, 2),
        "liquid_total":        round(liquid_total, 2),   # USDT + toda crypto
        "notional_crypto":     round(notional_all, 2),   # valor total da crypto
        "total_exposure_pct":  round(exposure_pct, 4),
        "open_position_count": open_count,
        "positions":           positions_data,
        "realized_pnl":        realized_pnl,
        "unrealized_pnl":      round(unrealized, 4),
        "unrealized_bot":      round(unrealized_bot, 4),   # só posições do bot
        "unrealized_sync":     round(unrealized_sync, 4),  # só holdings pré-existentes
        "daily_pnl":           round(unrealized, 4),
        "total_return_pct":    round(total_return_pct, 4),
        "drawdown_pct":        round(s.drawdown_pct, 4),
        "portfolio_beta":      round(s.portfolio_beta, 3),
        "avg_correlation":     round(s.avg_correlation, 3),
        "concentration_risk":  round(s.concentration_risk, 3),
        "is_overexposed":      s.is_overexposed,
        "updated_at":          _serialize(s.updated_at),
    }
