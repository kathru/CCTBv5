"""
ExchangeSync — Sincronização completa com o estado real do OKX.

Ao iniciar, lê da exchange:
  1. Saldo de todos os ativos (USDT + crypto)
  2. Histórico de ordens preenchidas
  3. Reconstrói posições a partir do histórico

Garante que, após um restart, o bot reflete exatamente o estado da exchange.
"""

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Símbolos que o bot negocia — usados para filtrar posições relevantes
TRADING_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT"}
TRADING_CCYS    = {"BTC", "ETH", "SOL"}


class ExchangeSync:
    """
    Lê o estado real do OKX e atualiza o banco de dados local,
    Redis e o PortfolioEngine para ficarem em sincronia.
    """

    def __init__(self, exchange, db, cache, portfolio) -> None:
        self._okx       = exchange
        self._db        = db
        self._cache     = cache
        self._portfolio = portfolio

    async def run(self) -> dict:
        """
        Executa sincronização completa.
        Retorna resumo do que foi sincronizado.
        """
        summary = {
            "assets":           [],
            "usdt_balance":     0.0,
            "crypto_positions": {},
            "orders_imported":  0,
            "total_equity_usd": 0.0,
        }

        try:
            # ── 1. Saldo de todos os ativos ──────────────────────────────────
            logger.info("ExchangeSync: lendo saldo da conta OKX...")
            details = await self._okx.get_account_details()
            summary["assets"] = details

            usdt = next((d for d in details if d["ccy"] == "USDT"), {})
            summary["usdt_balance"] = usdt.get("cashBal", 0.0)
            summary["total_equity_usd"] = sum(d["usdValue"] for d in details)

            # ── 2. Posições crypto ───────────────────────────────────────────
            for d in details:
                ccy = d["ccy"]
                if ccy in TRADING_CCYS and d["cashBal"] > 0:
                    symbol = f"{ccy}-USDT"
                    summary["crypto_positions"][symbol] = d["cashBal"]

            # ── 3. Histórico de ordens preenchidas ───────────────────────────
            logger.info("ExchangeSync: importando histórico de ordens OKX...")
            okx_orders = await self._okx.get_filled_orders(limit=100)
            imported = await self._import_orders(okx_orders)
            summary["orders_imported"] = imported

            # ── 4. Atualiza portfolio com saldo real ─────────────────────────
            await self._sync_portfolio(details, summary["crypto_positions"])

            logger.info(
                "ExchangeSync concluído: USDT=%.2f equity=%.2f "
                "crypto_positions=%s orders_imported=%d",
                summary["usdt_balance"],
                summary["total_equity_usd"],
                list(summary["crypto_positions"].keys()),
                summary["orders_imported"],
            )

        except Exception as exc:
            logger.error("ExchangeSync falhou: %s", exc, exc_info=True)

        return summary

    async def _import_orders(self, okx_orders: list[dict]) -> int:
        """Salva ordens OKX no PostgreSQL (upsert — não duplica)."""
        if not okx_orders or not self._db:
            return 0

        imported = 0
        for o in okx_orders:
            if not o.get("ordId") or not o.get("symbol"):
                continue
            # Só importa símbolos que o bot negocia
            if o["symbol"] not in TRADING_SYMBOLS:
                continue
            try:
                filled_at = datetime.fromtimestamp(o["uTime"] / 1000, tz=UTC) if o["uTime"] else datetime.now(UTC)
                created_at = datetime.fromtimestamp(o["cTime"] / 1000, tz=UTC) if o["cTime"] else filled_at
                fee = abs(o.get("fee", 0))

                async with self._db.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO orders (
                            client_order_id, exchange_order_id, symbol, side,
                            order_type, mode, status, quantity, filled_quantity,
                            avg_fill_price, fees_paid, strategy_id,
                            submitted_at, filled_at, created_at
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                        ON CONFLICT (client_order_id) DO UPDATE SET
                            status          = EXCLUDED.status,
                            filled_quantity = EXCLUDED.filled_quantity,
                            avg_fill_price  = EXCLUDED.avg_fill_price,
                            fees_paid       = EXCLUDED.fees_paid,
                            filled_at       = EXCLUDED.filled_at
                        """,
                        # Usa clOrdId se disponível, caso contrário ordId
                        o["clOrdId"] or o["ordId"],
                        o["ordId"],
                        o["symbol"],
                        o["side"],
                        o.get("ordType", "market"),
                        "market",
                        "filled",
                        o["sz"],
                        o["fillSz"],
                        o["avgPx"],
                        fee,
                        "momentum_v2",
                        filled_at,
                        filled_at,
                        created_at,
                    )
                imported += 1
            except Exception as exc:
                logger.debug("ExchangeSync: skip order %s — %s", o.get("ordId"), exc)

        return imported

    async def _sync_portfolio(self, details: list[dict], crypto_positions: dict) -> None:
        """
        Atualiza o PortfolioEngine e o Redis com o estado real da exchange.
        Salva posições crypto como positions no Redis para o dashboard.
        """
        usdt = next((d for d in details if d["ccy"] == "USDT"), {})
        cash = usdt.get("cashBal", 0.0)
        total_eq = sum(d["usdValue"] for d in details)

        # Atualiza PortfolioEngine com equity total
        self._portfolio._state.cash_available = cash
        self._portfolio._state.total_value    = total_eq
        if self._portfolio._state.initial_capital <= 10.0:
            self._portfolio._state.initial_capital = total_eq

        # Salva posições crypto no Redis para o dashboard
        for symbol, qty in crypto_positions.items():
            price = await self._cache.get_price(symbol)
            price = float(price) if price else 0.0
            notional = qty * price

            ccy = symbol.replace("-USDT", "")
            ccy_data = next((d for d in details if d["ccy"] == ccy), {})
            usd_value = ccy_data.get("usdValue", notional)

            await self._cache.set_position(symbol, {
                "symbol":        symbol,
                "side":          "long",
                "quantity":      qty,
                "avg_entry":     price,   # sem histórico, usa preço atual
                "current_price": price,
                "notional":      usd_value,
                "unrealized_pnl": 0.0,
                "realized_pnl":   0.0,
                "strategy_id":    "exchange_sync",
                "source":         "okx_balance",
            })
            logger.info(
                "ExchangeSync: posição %s qty=%.6f value=%.2f USD",
                symbol, qty, usd_value,
            )

        # Remove posições do Redis que não existem mais na exchange
        for symbol in TRADING_SYMBOLS - set(crypto_positions.keys()):
            existing = await self._cache.get_position(symbol)
            if existing and existing.get("source") == "exchange_sync":
                await self._cache.delete_position(symbol)
