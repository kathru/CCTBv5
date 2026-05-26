"""
reconcile_trades.py — Conciliação completa BUY↔SELL entre OKX e banco local.

Fluxo:
  1. Busca TODOS os fills do OKX (paper trading demo)
  2. Busca todas as ordens do banco local (todos os strategy_ids)
  3. Identifica ordens OKX que estão ausentes ou com status errado no banco
  4. Insere/corrige SELLs faltantes com strategy_id='momentum_v2'
  5. Exibe relatório completo de casamento BUY↔SELL e P&L

Uso:
  docker exec cctb_app python3 scripts/reconcile_trades.py
  docker exec cctb_app python3 scripts/reconcile_trades.py --dry-run
"""

import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, "/app")

DRY_RUN = "--dry-run" in sys.argv
EXCLUDED_STRAT = {"okx_import", "exchange_sync", "okx_adjustment"}


async def fetch_okx_fills(settings) -> list[dict]:
    """Busca todos os fills do OKX paper trading (paginado)."""
    from src.exchange.okx.auth import build_headers
    import httpx

    all_data: list[dict] = []
    after = ""
    page = 0
    while True:
        page += 1
        params = "instType=SPOT&state=filled&limit=100" + (f"&after={after}" if after else "")
        path = f"/api/v5/trade/orders-history?{params}"
        headers = build_headers(
            settings.okx_demo_api_key,
            settings.okx_demo_secret_key,
            settings.okx_demo_passphrase,
            "GET", path, paper=True,
        )
        async with httpx.AsyncClient(base_url="https://www.okx.com", timeout=20) as c:
            r = await c.get(path, headers=headers)
            r.raise_for_status()
            data = r.json().get("data", [])
            all_data.extend(data)
            print(f"  OKX página {page}: {len(data)} ordens (total: {len(all_data)})")
            if len(data) < 100:
                break
            after = data[-1].get("ordId", "")
            if not after:
                break
    return all_data


def parse_okx_order(o: dict) -> dict:
    """Normaliza um registro OKX para dict padronizado."""
    fill_ts = int(o.get("fillTime") or o.get("uTime") or 0)
    ctime   = int(o.get("cTime") or fill_ts)
    return {
        "ord_id":    o.get("ordId", ""),
        "cl_ord_id": o.get("clOrdId", "") or o.get("ordId", ""),
        "symbol":    o.get("instId", ""),
        "side":      o.get("side", "").lower(),
        "qty":       float(o.get("accFillSz") or 0),
        "price":     float(o.get("avgPx") or 0),
        "fee":       abs(float(o.get("fee") or 0)),
        "filled_at": datetime.fromtimestamp(fill_ts / 1000, tz=timezone.utc) if fill_ts else None,
        "created_at":datetime.fromtimestamp(ctime   / 1000, tz=timezone.utc) if ctime   else None,
        "state":     o.get("state", ""),
    }


def pair_trades(orders: list[dict]) -> list[dict]:
    """Casa BUY→SELL por símbolo (FIFO). Retorna trades completos."""
    by_sym: dict[str, list] = defaultdict(list)
    for o in orders:
        by_sym[o["symbol"]].append(o)

    trades = []
    unmatched_buys: dict[str, list] = {}

    for sym, sym_orders in by_sym.items():
        sym_orders.sort(key=lambda o: o["filled_at"] or datetime.min.replace(tzinfo=timezone.utc))
        buy_q: list[dict] = []
        for o in sym_orders:
            if o["side"] == "buy":
                buy_q.append(o)
            elif o["side"] == "sell" and buy_q:
                entry = buy_q.pop(0)
                qty   = min(entry["qty"], o["qty"])
                fee   = entry["fee"] + o["fee"]
                pnl   = (o["price"] - entry["price"]) * qty - fee
                pnl_pct = (o["price"] - entry["price"]) / entry["price"] if entry["price"] else 0
                trades.append({
                    "symbol":    sym,
                    "buy":       entry,
                    "sell":      o,
                    "qty":       qty,
                    "fee":       fee,
                    "pnl":       pnl,
                    "pnl_pct":   pnl_pct,
                    "entry_px":  entry["price"],
                    "exit_px":   o["price"],
                    "entry_ts":  entry["filled_at"],
                    "exit_ts":   o["filled_at"],
                })
            elif o["side"] == "sell" and not buy_q:
                print(f"  AVISO: SELL sem BUY correspondente: {sym} ord_id={o['ord_id']}")

        if buy_q:
            unmatched_buys[sym] = buy_q

    return trades, unmatched_buys


async def main() -> None:
    from src.core.config import settings
    from src.persistence.postgres import Database

    print("=" * 65)
    print("  CONCILIAÇÃO DE TRADES — CCTBv5")
    print(f"  Modo: {'DRY-RUN (sem alterações)' if DRY_RUN else 'LIVE (aplica correções)'}")
    print("=" * 65)

    db = Database(dsn=settings.database_url)
    await db.connect()

    # ── 1. Busca OKX ─────────────────────────────────────────────────────────
    print("\n[1/5] Buscando fills do OKX paper trading...")
    okx_raw = await fetch_okx_fills(settings)
    okx_filled = [parse_okx_order(o) for o in okx_raw if o.get("state") == "filled"]
    print(f"  Total fills OKX: {len(okx_filled)}")

    # ── 2. Busca banco local ──────────────────────────────────────────────────
    print("\n[2/5] Buscando ordens no banco local...")
    db_rows = await db.fetch(
        "SELECT client_order_id, exchange_order_id, symbol, side, status, "
        "filled_quantity, avg_fill_price, fees_paid, strategy_id, filled_at "
        "FROM orders ORDER BY filled_at ASC NULLS LAST"
    )
    db_by_eid = {r["exchange_order_id"]: r for r in db_rows if r["exchange_order_id"]}
    db_by_cid = {r["client_order_id"]: r for r in db_rows}
    print(f"  Total ordens no banco: {len(db_rows)}")

    bot_orders = [r for r in db_rows if r["strategy_id"] not in EXCLUDED_STRAT]
    print(f"  Ordens do bot (momentum_v2 etc): {len(bot_orders)}")
    bot_buys  = [r for r in bot_orders if r["side"] == "buy"  and r["status"] == "filled"]
    bot_sells = [r for r in bot_orders if r["side"] == "sell" and r["status"] == "filled"]
    print(f"  Bot BUYs  preenchidas: {len(bot_buys)}")
    print(f"  Bot SELLs preenchidas: {len(bot_sells)}")

    # ── 3. Identifica lacunas ─────────────────────────────────────────────────
    print("\n[3/5] Identificando ordens OKX ausentes no banco do bot...")

    missing: list[dict] = []
    stale_submitted: list[dict] = []

    for okx_order in okx_filled:
        eid = okx_order["ord_id"]
        cid = okx_order["cl_ord_id"]

        db_row = db_by_eid.get(eid) or db_by_cid.get(cid)

        if db_row is None:
            # Completamente ausente
            missing.append(okx_order)
            tag = "AUSENTE"
        elif db_row["status"] != "filled":
            # Está no banco mas não marcada como filled
            stale_submitted.append({**okx_order, "db_row": db_row})
            tag = f"SUBMITTED→FILLED (strategy={db_row['strategy_id']})"
        elif db_row["strategy_id"] in EXCLUDED_STRAT:
            # Foi importada como okx_import — precisa atualizar strategy_id se for do bot
            tag = f"okx_import (ignorada no dashboard)"
        else:
            tag = "OK"

        sym_tag = f"{okx_order['side'].upper():<5} {okx_order['symbol']:<12}"
        ts_tag  = okx_order["filled_at"].strftime("%d/%m %H:%M") if okx_order["filled_at"] else "?"
        print(f"  {sym_tag} {ts_tag}  px={okx_order['price']:>10.2f}  [{tag}]")

    print(f"\n  Ausentes totalmente : {len(missing)}")
    print(f"  Presas como submitted: {len(stale_submitted)}")

    # ── 4. Casa BUY↔SELL para encontrar os round-trips faltantes ──────────────
    print("\n[4/5] Casamento BUY↔SELL (todos os fills OKX)...")
    trades, unmatched_buys = pair_trades(okx_filled)

    print(f"\n  Round-trips completos: {len(trades)}")
    print(f"  BUYs sem SELL ainda : {sum(len(v) for v in unmatched_buys.values())}")
    print()
    print(f"  {'SÍMBOLO':<12} {'ENTRADA':>10} {'SAÍDA':>10} {'QTY':>10}  {'P&L':>10}  {'%':>7}  {'ABERTURA':>14}  {'FECHAMENTO':>14}")
    print("  " + "-" * 100)

    total_pnl = 0.0
    total_fee = 0.0
    for t in sorted(trades, key=lambda x: x["exit_ts"] or datetime.min.replace(tzinfo=timezone.utc)):
        ets = t["entry_ts"].strftime("%d/%m %H:%M") if t["entry_ts"] else "?"
        xts = t["exit_ts"].strftime("%d/%m %H:%M")  if t["exit_ts"]  else "?"
        sign = "+" if t["pnl"] >= 0 else ""
        print(f"  {t['symbol']:<12} {t['entry_px']:>10.2f} {t['exit_px']:>10.2f} "
              f"{t['qty']:>10.4f}  {sign}{t['pnl']:>9.2f}  {t['pnl_pct']*100:>+6.2f}%  {ets:>14}  {xts:>14}")
        total_pnl += t["pnl"]
        total_fee += t["fee"]

    print("  " + "-" * 100)
    print(f"  {'TOTAL':>12}  {'':>10}  {'':>10}  {'':>10}  {total_pnl:>+10.2f}  {'':>7}  fees: ${total_fee:.4f}")

    if unmatched_buys:
        print("\n  BUYs ainda abertas (sem SELL correspondente):")
        for sym, buys in unmatched_buys.items():
            for b in buys:
                ts = b["filled_at"].strftime("%d/%m %H:%M") if b["filled_at"] else "?"
                print(f"    BUY  {sym:<12} {b['price']:>10.2f}  qty={b['qty']:.4f}  {ts}")

    # ── 4b. Identifica okx_import SELLs que casam com bot BUYs ──────────────
    # Bot BUYs (momentum_v2) sem SELL correspondente na memória do bot
    bot_buy_rows = await db.fetch(
        "SELECT symbol, filled_quantity, avg_fill_price, filled_at, exchange_order_id "
        "FROM orders WHERE status='filled' AND side='buy' "
        "AND strategy_id NOT IN ('okx_import','exchange_sync','okx_adjustment') "
        "ORDER BY filled_at ASC"
    )
    bot_sell_rows = await db.fetch(
        "SELECT symbol, filled_at FROM orders WHERE status='filled' AND side='sell' "
        "AND strategy_id NOT IN ('okx_import','exchange_sync','okx_adjustment') "
        "ORDER BY filled_at ASC"
    )

    # Para cada symbol, faz o casamento FIFO de bot BUYs → bot SELLs
    # Qualquer BUY sem SELL no bot precisa ter seu SELL encontrado no OKX
    bot_sells_by_sym: dict[str, list] = defaultdict(list)
    for r in bot_sell_rows:
        bot_sells_by_sym[r["symbol"]].append(r["filled_at"])

    orphan_buys: list[dict] = []  # BUYs sem SELL no bot
    consumed_sells: dict[str, int] = defaultdict(int)
    for r in bot_buy_rows:
        sym = r["symbol"]
        available_sells = bot_sells_by_sym[sym][consumed_sells[sym]:]
        buy_ts = r["filled_at"]
        # Encontra o primeiro SELL após este BUY
        matched = next((s for s in available_sells if s > buy_ts), None)
        if matched:
            consumed_sells[sym] += 1
        else:
            orphan_buys.append(dict(r))

    print(f"\n  BUYs do bot sem SELL correspondente: {len(orphan_buys)}")
    for b in orphan_buys:
        ts = b["filled_at"].strftime("%d/%m %H:%M") if b["filled_at"] else "?"
        print(f"    BUY {b['symbol']:<12} @ {float(b['avg_fill_price']):.2f}  {ts}")

    # Para cada orphan BUY, acha o SELL mais próximo no OKX (mesmo símbolo, após o BUY)
    to_promote: list[dict] = []   # okx_import SELLs a promover
    for b in orphan_buys:
        sym     = b["symbol"]
        buy_ts  = b["filled_at"]
        # Busca no OKX o primeiro SELL deste símbolo após o BUY
        candidates = [
            o for o in okx_filled
            if o["symbol"] == sym and o["side"] == "sell"
            and o["filled_at"] and o["filled_at"] > buy_ts
        ]
        if not candidates:
            print(f"  ⚠ Nenhum SELL OKX encontrado para BUY {sym} @ {float(b['avg_fill_price']):.2f}")
            continue
        # Ordena por data e pega o mais próximo
        best = sorted(candidates, key=lambda x: x["filled_at"])[0]
        # Verifica se já está no banco como okx_import
        db_match = db_by_eid.get(best["ord_id"]) or db_by_cid.get(best["cl_ord_id"])
        if db_match and db_match["strategy_id"] in EXCLUDED_STRAT:
            to_promote.append({**best, "db_cid": db_match["client_order_id"]})
            print(f"  → PROMOVER okx_import→momentum_v2: SELL {sym} @ {best['price']:.2f} "
                  f"em {best['filled_at'].strftime('%d/%m %H:%M')}")
        elif db_match and db_match["strategy_id"] not in EXCLUDED_STRAT:
            print(f"  → Já em momentum_v2: SELL {sym} @ {best['price']:.2f}")
        else:
            # Não está no banco — vai para missing
            if best not in missing:
                missing.append(best)
                print(f"  → INSERIR AUSENTE: SELL {sym} @ {best['price']:.2f} "
                      f"em {best['filled_at'].strftime('%d/%m %H:%M')}")

    # ── 5. Aplica correções ────────────────────────────────────────────────────
    print("\n[5/5] Aplicando correções no banco...")

    if DRY_RUN:
        print(f"  [DRY-RUN] Seriam inseridas {len(missing)} ordens e promovidas {len(to_promote)}.")
        print("  Execute sem --dry-run para aplicar.")
    else:
        inserted = updated = promoted = 0

        # 5a. Insere ordens ausentes
        for o in missing:
            strat = "momentum_v2"
            cid   = o["cl_ord_id"] or o["ord_id"]
            try:
                await db.execute("""
                    INSERT INTO orders (
                        client_order_id, exchange_order_id, symbol, side,
                        order_type, mode, status, quantity, filled_quantity,
                        avg_fill_price, limit_price, stop_loss, take_profit,
                        fees_paid, strategy_id, signal_id, retry_count,
                        last_error, created_at, submitted_at, filled_at, cancelled_at
                    ) VALUES (
                        $1,$2,$3,$4,'market','market','filled',
                        $5,$5,$6,NULL,NULL,NULL,$7,$8,'reconciled',0,
                        NULL,$9,$9,$9,NULL
                    )
                    ON CONFLICT (client_order_id) DO UPDATE SET
                        status         = 'filled',
                        filled_quantity= EXCLUDED.filled_quantity,
                        avg_fill_price = EXCLUDED.avg_fill_price,
                        fees_paid      = EXCLUDED.fees_paid,
                        filled_at      = EXCLUDED.filled_at,
                        strategy_id    = EXCLUDED.strategy_id
                """,
                    cid, o["ord_id"], o["symbol"], o["side"],
                    o["qty"], o["price"], o["fee"],
                    strat,
                    o["filled_at"],
                )
                inserted += 1
                print(f"  ✓ INSERIDA: {o['side'].upper():<5} {o['symbol']:<12} "
                      f"qty={o['qty']:.4f} px={o['price']:.2f} fee={o['fee']:.4f}")
            except Exception as exc:
                print(f"  ✗ ERRO ao inserir {o['ord_id']}: {exc}")

        # 5b. Promove okx_import → momentum_v2
        for o in to_promote:
            try:
                await db.execute("""
                    UPDATE orders SET
                        strategy_id     = 'momentum_v2',
                        filled_quantity = $1,
                        avg_fill_price  = $2,
                        fees_paid       = $3,
                        status          = 'filled',
                        filled_at       = COALESCE(filled_at, $4)
                    WHERE client_order_id = $5
                """,
                    o["qty"], o["price"], o["fee"], o["filled_at"],
                    o["db_cid"],
                )
                promoted += 1
                print(f"  ✓ PROMOVIDA: SELL {o['symbol']:<12} @ {o['price']:.2f} "
                      f"okx_import→momentum_v2")
            except Exception as exc:
                print(f"  ✗ ERRO ao promover {o.get('db_cid','?')}: {exc}")

        # 5c. Corrige ordens presas como SUBMITTED
        for item in stale_submitted:
            o   = item
            row = item["db_row"]
            cid = row["client_order_id"]
            try:
                await db.execute("""
                    UPDATE orders SET
                        status          = 'filled',
                        filled_quantity = $1,
                        avg_fill_price  = $2,
                        fees_paid       = $3,
                        filled_at       = $4,
                        strategy_id     = CASE
                            WHEN strategy_id = ANY($5::text[]) THEN 'momentum_v2'
                            ELSE strategy_id
                        END
                    WHERE client_order_id = $6
                """,
                    o["qty"], o["price"], o["fee"], o["filled_at"],
                    list(EXCLUDED_STRAT),
                    cid,
                )
                updated += 1
                print(f"  ✓ CORRIGIDA: {o['side'].upper():<5} {o['symbol']:<12} "
                      f"submitted→filled  px={o['price']:.2f}")
            except Exception as exc:
                print(f"  ✗ ERRO ao corrigir {cid}: {exc}")

        print(f"\n  Inseridas: {inserted} | Promovidas: {promoted} | Corrigidas: {updated}")

    # ── Resumo final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMO FINAL DO BANCO (após conciliação)")
    print("=" * 65)
    rows = await db.fetch(
        "SELECT strategy_id, side, COUNT(*) n, "
        "SUM(filled_quantity*avg_fill_price) vol, SUM(fees_paid) fees "
        "FROM orders WHERE status='filled' "
        "GROUP BY strategy_id, side ORDER BY strategy_id, side"
    )
    for r in rows:
        print(f"  {r['strategy_id']:<22} {r['side']:<5}  n={r['n']:>3}  "
              f"vol=${float(r['vol'] or 0):>12,.2f}  fees=${float(r['fees'] or 0):.4f}")

    # P&L dos trades casados do bot
    bot_filled = await db.fetch(
        "SELECT symbol, side, filled_quantity, avg_fill_price, fees_paid, filled_at "
        "FROM orders WHERE status='filled' AND strategy_id NOT IN ('okx_import','exchange_sync','okx_adjustment') "
        "ORDER BY filled_at ASC NULLS LAST"
    )
    bot_norm = [{"symbol": r["symbol"], "side": r["side"],
                 "qty": float(r["filled_quantity"]), "price": float(r["avg_fill_price"]),
                 "fee": float(r["fees_paid"]), "filled_at": r["filled_at"]} for r in bot_filled]
    bot_trades, bot_open = pair_trades(bot_norm)

    total_pnl = sum(t["pnl"] for t in bot_trades)
    total_fee = sum(t["fee"] for t in bot_trades)
    wins = [t for t in bot_trades if t["pnl"] > 0]
    losses = [t for t in bot_trades if t["pnl"] <= 0]

    print(f"\n  Trades completos (bot): {len(bot_trades)}")
    print(f"  Wins / Losses         : {len(wins)} / {len(losses)}")
    print(f"  Win Rate              : {len(wins)/len(bot_trades)*100:.1f}%" if bot_trades else "  Win Rate: N/A")
    print(f"  P&L Total             : ${total_pnl:+.2f}")
    print(f"  Fees Total            : ${total_fee:.4f}")
    print(f"  P&L líquido           : ${total_pnl - total_fee:+.2f}")
    print(f"  BUYs abertas (sem venda): {sum(len(v) for v in bot_open.values())}")
    if bot_open:
        for sym, buys in bot_open.items():
            for b in buys:
                ts = b["filled_at"].strftime("%d/%m %H:%M") if b["filled_at"] else "?"
                print(f"    → {sym} BUY @ {b['price']:.2f} em {ts} (ainda aberta)")

    print("\n  ✅ Conciliação concluída. Atualize o dashboard para ver os novos dados.")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
