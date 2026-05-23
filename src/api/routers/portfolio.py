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
    initial = s.initial_capital if s.initial_capital > 1000.0 else 96_592.87

    # ── Lê saldos OKX do Redis (atualizados pelo sync periódico) ───────────────
    # Inclui TODOS os ativos: USDT, BTC, ETH, SOL, OKB, BRL
    ALL_OKX_CCYS = ["USDT", "BTC", "ETH", "SOL", "OKB", "BRL"]
    okx_balances: dict[str, dict] = {}
    if cache:
        for ccy in ALL_OKX_CCYS:
            raw = await cache.get(f"okx:balance:{ccy}")
            if raw:
                try:
                    import ast
                    okx_balances[ccy] = ast.literal_eval(raw)
                except Exception:
                    pass

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
            from src.persistence.repositories.orders import EXCLUDED_STRATEGY_IDS
            rows = await db.fetch(
                """
                SELECT symbol, side,
                       SUM(filled_quantity)                        AS qty,
                       SUM(filled_quantity * avg_fill_price)       AS notional,
                       SUM(COALESCE(fees_paid, 0))                 AS fees
                FROM orders
                WHERE status='filled' AND strategy_id != ALL($1)
                GROUP BY symbol, side
                """,
                list(EXCLUDED_STRATEGY_IDS),
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

    # ── Posições reais: derivadas dos saldos OKX do Redis ─────────────────────
    # FONTE DA VERDADE: okx_balances (Redis, atualizado a cada 5 min da OKX real).
    # Nunca usa order_stats para calcular posições abertas — esses dados históricos
    # não refletem o saldo real (podem ter ordens de outros períodos ou bots).
    positions_data: dict[str, dict] = {}
    notional_bot    = 0.0
    notional_sync   = 0.0
    unrealized_bot  = 0.0
    unrealized_sync = 0.0

    TRADING_CCYS = ["BTC", "ETH", "SOL"]
    for ccy in TRADING_CCYS:
        sym = f"{ccy}-USDT"
        bal = okx_balances.get(ccy, {})
        qty = float(bal.get("cashBal", 0.0))
        usd = float(bal.get("usdValue", 0.0))

        if qty <= 1e-8 or usd < 1.0:  # ignora dust < $1
            continue  # sem saldo real nesse ativo

        # Preço atual = usdValue / qty (direto do OKX, mais preciso que cache)
        price = usd / qty if qty > 0 else 0.0
        # Tenta melhorar com preço do cache de mercado se disponível
        if cache:
            price_raw = await cache.get_price(sym)
            if price_raw:
                price = float(price_raw)

        # Preço médio de entrada: busca do order_stats (bot) ou usa preço atual
        avg_entry = price  # fallback: preço atual = sem P&L não realizado
        unreal = 0.0
        strategy = "exchange_sync"

        if sym in order_stats:
            st = order_stats[sym]
            bot_open_qty = st["buy_qty"] - st["sell_qty"]
            if bot_open_qty > 1e-8 and st["buy_qty"] > 0:
                avg_entry = st["buy_notional"] / st["buy_qty"]
                unreal = (price - avg_entry) * qty
                strategy = "momentum_v2"
                notional_bot += usd
                unrealized_bot += unreal
            else:
                # Posição exchange_sync — sem custo base definido
                notional_sync += usd
                unrealized_sync += unreal
        else:
            notional_sync += usd

        positions_data[sym] = {
            "quantity":       round(qty, 6),
            "avg_entry":      round(avg_entry, 4),
            "current_price":  round(price, 4),
            "notional":       round(usd, 4),     # usa usdValue do OKX diretamente
            "unrealized_pnl": round(unreal, 4),
            "strategy_id":    strategy,
        }

    # ── Portfolio total = soma de TODOS os ativos OKX via Redis ─────────────────
    # Se Redis tem dados frescos do sync periódico, usa eles (mais preciso)
    # Fallback: recalcula com o que temos localmente
    okx_total_raw = await cache.get("okx:portfolio_total_usd") if cache else None
    if okx_total_raw:
        total_value = float(okx_total_raw)
        cash_value  = float(okx_balances.get("USDT", {}).get("cashBal", s.cash_available))
    else:
        cash_value  = s.cash_available
        total_value = cash_value + notional_bot + notional_sync

    # Adiciona OKB e BRL ao total (se não incluídos no notional_sync)
    extra_usd = 0.0
    okx_extra_assets = {}   # OKB, BRL para exibição
    for ccy in ["OKB", "BRL"]:
        bal = okx_balances.get(ccy, {})
        usd_val = bal.get("usdValue", 0.0)
        if usd_val > 0:
            extra_usd += usd_val
            okx_extra_assets[ccy] = {
                "cashBal":  bal.get("cashBal", 0.0),
                "usdValue": usd_val,
                "ccy":      ccy,
            }

    # Se o total do Redis não inclui OKB/BRL, adiciona
    if not okx_total_raw:
        total_value += extra_usd

    unrealized    = unrealized_bot + unrealized_sync
    # Exposição = apenas posições de trading (BTC/ETH/SOL), NÃO inclui OKB/BRL
    # OKB e BRL são ativos watch-only, não representam risco de trading
    notional_trading = notional_bot + notional_sync   # só BTC/ETH/SOL
    notional_all     = total_value - cash_value       # total incl. OKB/BRL (para donut)
    exposure_pct     = notional_trading / total_value if total_value > 0 and notional_trading > 0 else 0.0
    total_return_pct = (total_value - initial) / initial if initial > 0 else 0.0
    liquid_total  = total_value

    open_count = len(positions_data)

    return {
        "available":           True,
        "initial_capital":     initial,
        "total_value":         round(total_value, 2),
        "cash_available":      round(cash_value, 2),
        "liquid_total":        round(liquid_total, 2),
        "notional_crypto":     round(notional_all, 2),
        "okx_balances":        okx_balances,        # todos os saldos OKX brutos
        "okx_extra_assets":    okx_extra_assets,    # OKB e BRL para display
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
