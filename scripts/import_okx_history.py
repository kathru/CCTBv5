"""
Script para importar histórico completo de ordens do OKX para o banco local.
Uso: docker exec cctb_app python3 scripts/import_okx_history.py
"""
import asyncio
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/app")


async def main() -> None:
    from src.core.config import settings
    from src.exchange.okx.auth import build_headers
    from src.persistence.postgres import Database
    import httpx

    db = Database(dsn=settings.database_url)
    await db.connect()

    async def fetch_all() -> list[dict]:
        all_data: list[dict] = []
        after = ""
        while True:
            params = "instType=SPOT&limit=100" + (f"&after={after}" if after else "")
            path = f"/api/v5/trade/orders-history?{params}"
            headers = build_headers(
                settings.okx_demo_api_key,
                settings.okx_demo_secret_key,
                settings.okx_demo_passphrase,
                "GET", path, paper=True,
            )
            async with httpx.AsyncClient(base_url="https://www.okx.com", timeout=15) as c:
                r = await c.get(path, headers=headers)
                data = r.json().get("data", [])
                all_data.extend(data)
                if len(data) < 100:
                    break
                after = data[-1].get("ordId", "")
                if not after:
                    break
        return all_data

    orders = await fetch_all()
    print(f"OKX orders encontrados: {len(orders)}")

    imported = skipped = 0
    for o in orders:
        if o.get("state") != "filled":
            skipped += 1
            continue
        try:
            fill_ts   = int(o.get("fillTime") or o.get("uTime") or 0)
            ctime     = int(o.get("cTime") or fill_ts)
            fa = datetime.fromtimestamp(fill_ts / 1000, tz=timezone.utc) if fill_ts else datetime.now(timezone.utc)
            ca = datetime.fromtimestamp(ctime   / 1000, tz=timezone.utc) if ctime   else fa

            client_id = o.get("clOrdId") or o.get("ordId")
            qty  = float(o.get("accFillSz") or 0)
            px   = float(o.get("avgPx") or 0)
            fee  = abs(float(o.get("fee") or 0))
            sym  = o.get("instId", "")
            side = o.get("side", "")

            if not sym or qty <= 0:
                skipped += 1
                continue

            await db.execute(
                """
                INSERT INTO orders (
                    client_order_id, exchange_order_id, symbol, side,
                    order_type, mode, status, quantity, filled_quantity,
                    avg_fill_price, fees_paid, strategy_id, signal_id,
                    submitted_at, filled_at, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                ON CONFLICT (client_order_id) DO UPDATE SET
                    fees_paid  = EXCLUDED.fees_paid,
                    filled_at  = EXCLUDED.filled_at
                """,
                client_id, o.get("ordId", ""), sym, side,
                "market", "market", "filled",
                qty, qty, px, fee,
                "okx_import", "okx-import",
                fa, fa, ca,
            )
            imported += 1
            print(f"  OK  {side.upper():<5} {sym:<12} qty={qty:.6f}  px={px:.2f}  fee={fee:.4f}")

        except Exception as exc:
            print(f"  ERR {o.get('ordId','?')}: {exc}")
            skipped += 1

    # Resumo
    rows = await db.fetch(
        "SELECT strategy_id, side, COUNT(*) n, "
        "SUM(filled_quantity*avg_fill_price) vol, SUM(fees_paid) fees "
        "FROM orders GROUP BY strategy_id, side ORDER BY strategy_id, side"
    )
    print(f"\nResultado: {imported} importadas | {skipped} ignoradas")
    print("\n=== ORDERS TABLE ===")
    for r in rows:
        print(f"  {r['strategy_id']:<20} {r['side']:<5} n={r['n']}  vol=${float(r['vol'] or 0):,.2f}  fees=${float(r['fees'] or 0):.4f}")

    total_fees = await db.fetchrow("SELECT SUM(fees_paid) as f FROM orders WHERE strategy_id='okx_import'")
    print(f"\nTotal fees importadas: ${float(total_fees['f'] or 0):.4f}")

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
