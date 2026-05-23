"""
ExchangeSync — Sincronização completa com o estado real do OKX.

Ao iniciar (e a cada SYNC_INTERVAL_SECS), lê da exchange:
  1. Saldo de TODOS os ativos (USDT, BTC, ETH, SOL, OKB, BRL, etc.)
  2. Histórico de ordens preenchidas
  3. Reconstrói posições a partir do histórico

Ativos monitorados:
  USDT — stablecoin operacional (base para trades)
  BTC, ETH, SOL — cryptos negociadas pelo bot
  OKB  — token nativo OKX (apenas monitorado, não negociado)
  BRL  — fiat brasileiro (apenas monitorado, não negociado)

Garante que, após um restart, o bot reflete exatamente o estado da exchange.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Ativos monitorados ────────────────────────────────────────────────────────
# Todos incluídos no portfolio total e exibidos no dashboard.
VALID_CCYS = {"USDT", "BTC", "ETH", "SOL", "OKB", "BRL"}

# Cryptos que o bot negocia ativamente (pares com USDT)
TRADING_CCYS    = {"BTC", "ETH", "SOL"}
TRADING_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT"}

# Ativos monitorados mas NÃO negociados pelo bot
WATCH_ONLY_CCYS = {"OKB", "BRL"}

# Ativos fiduciários (fiat) — valor em moeda local, convertido para USD
FIAT_CCYS = {"BRL"}

# Capital inicial do demo — sobrescrito pelo Redis na 1ª execução
INITIAL_CAPITAL_USD = 96_592.87

# Chave Redis para capital inicial persistente
INITIAL_CAPITAL_KEY = "portfolio:initial_capital_usdt"

# TTL dos dados de saldo no Redis (segundos)
BALANCE_TTL = 600   # 10 minutos


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
        Executa sincronização completa (boot + posições + ordens).
        Retorna resumo do que foi sincronizado.
        """
        summary = {
            "assets":           [],
            "usdt_balance":     0.0,
            "crypto_positions": {},
            "watch_balances":   {},   # OKB, BRL etc.
            "orders_imported":  0,
            "total_equity_usd": 0.0,
        }

        try:
            # ── 1. Saldo de todos os ativos ──────────────────────────────────
            logger.info("ExchangeSync: lendo saldo da conta OKX...")
            all_details = await self._okx.get_account_details()
            summary["assets"] = all_details

            # Separa válidos vs ignorados
            valid   = [d for d in all_details if d["ccy"] in VALID_CCYS]
            ignored = [d for d in all_details if d["ccy"] not in VALID_CCYS]
            if ignored:
                logger.warning(
                    "ExchangeSync: ativos ignorados: %s",
                    [f"{d['ccy']}={d['cashBal']:.4f}" for d in ignored],
                )

            details = valid

            usdt = next((d for d in details if d["ccy"] == "USDT"), {})
            summary["usdt_balance"]     = usdt.get("cashBal", 0.0)
            summary["total_equity_usd"] = sum(d["usdValue"] for d in details)

            logger.info(
                "ExchangeSync: ativos válidos — %s | Total=%.2f USD",
                " | ".join(f"{d['ccy']}={d['cashBal']:.4f}(~${d['usdValue']:.2f})" for d in details),
                summary["total_equity_usd"],
            )

            # ── 2. Separa crypto tradeable vs watch-only ─────────────────────
            # Threshold de dust: ignora saldos < $1 USD para evitar posições fantasma
            DUST_USD = 1.0
            for d in details:
                ccy = d["ccy"]
                if ccy in TRADING_CCYS and d["cashBal"] > 0 and d["usdValue"] >= DUST_USD:
                    symbol = f"{ccy}-USDT"
                    summary["crypto_positions"][symbol] = d["cashBal"]
                elif ccy in WATCH_ONLY_CCYS or ccy == "USDT":
                    summary["watch_balances"][ccy] = {
                        "cashBal":  d["cashBal"],
                        "usdValue": d["usdValue"],
                        "ccy":      ccy,
                    }

            # ── 3. Atualiza Redis com todos os saldos ───────────────────────
            await self._store_balances(details)

            # ── 4. Histórico de ordens (paginado) ────────────────────────────
            logger.info("ExchangeSync: importando histórico de ordens OKX...")
            try:
                all_okx_orders: list[dict] = []
                after = ""
                while True:
                    batch = await self._okx.get_filled_orders(limit=100, after=after)
                    if not batch:
                        break
                    all_okx_orders.extend(batch)
                    if len(batch) < 100:
                        break
                    after = batch[-1].get("ordId", "")
                    if not after:
                        break
                imported = await self._import_orders(all_okx_orders)
                summary["orders_imported"] = imported
                if imported > 0:
                    logger.info("ExchangeSync: %d novas ordens importadas do OKX", imported)
            except Exception as orders_exc:
                logger.warning(
                    "ExchangeSync: orders-history indisponível (%s) — usando DB local",
                    str(orders_exc)[:80],
                )

            # ── 5. Atualiza portfolio com saldo real ─────────────────────────
            await self._sync_portfolio(details, summary["crypto_positions"])

            logger.info(
                "ExchangeSync concluído: equity=%.2f USD  crypto=%s  orders=%d",
                summary["total_equity_usd"],
                list(summary["crypto_positions"].keys()),
                summary["orders_imported"],
            )

        except Exception as exc:
            logger.error("ExchangeSync falhou: %s", exc, exc_info=True)

        return summary

    async def sync_balances(self) -> float:
        """
        Sincronização leve (periódica): lê saldos, atualiza Redis/portfolio
        e executa conciliação de trades OKX ↔ DB.
        Roda a cada 5 minutos via TradingLoop._periodic_balance_sync().
        Retorna o portfolio total em USD.
        """
        try:
            all_details = await self._okx.get_account_details()
            details = [d for d in all_details if d["ccy"] in VALID_CCYS]

            await self._store_balances(details)

            total = sum(d["usdValue"] for d in details)
            usdt  = next((d for d in details if d["ccy"] == "USDT"), {})
            cash  = usdt.get("cashBal", 0.0)

            self._portfolio._state.cash_available = cash
            self._portfolio._state.total_value    = total

            # Atualiza preços implícitos derivados dos saldos OKX
            for d in details:
                ccy = d["ccy"]
                if ccy in TRADING_CCYS and d["cashBal"] > 0 and d["usdValue"] > 0:
                    implied_px = d["usdValue"] / d["cashBal"]
                    await self._cache.set_price(f"{ccy}-USDT", implied_px)

            logger.debug(
                "ExchangeSync (periódico): total=%.2f USD  USDT=%.2f",
                total, cash,
            )

            # ── Conciliação de trades OKX ↔ DB ────────────────────────────
            # Detecta e corrige divergências entre saldo OKX e registros do DB
            if self._db:
                from .trade_reconciler import TradeReconciler
                recon = TradeReconciler(
                    db=self._db,
                    cache=self._cache,
                    exchange=self._okx,
                )
                report = await recon.run()
                if report.get("divergences", 0) > 0:
                    logger.warning(
                        "TradeReconciler: %d divergência(s) detectada(s), "
                        "%d corrigida(s). Detalhes: %s",
                        report["divergences"],
                        report["fixed"],
                        report.get("warnings", []),
                    )

            return total

        except Exception as exc:
            logger.warning("ExchangeSync periódico falhou: %s", exc)
            return 0.0

    async def _store_balances(self, details: list[dict]) -> None:
        """
        Persiste saldos individuais no Redis para consulta rápida.
        Chaves: okx:balance:{CCY} = {cashBal, usdValue, ccy, updated_at}
        """
        now = datetime.now(UTC).isoformat()
        for d in details:
            key = f"okx:balance:{d['ccy']}"
            payload = {
                "ccy":        d["ccy"],
                "cashBal":    d["cashBal"],
                "availBal":   d.get("availBal", d["cashBal"]),
                "frozenBal":  d.get("frozenBal", 0.0),
                "usdValue":   d["usdValue"],
                "updated_at": now,
            }
            await self._cache.set(key, str(payload), ttl=BALANCE_TTL)

        # Armazena total também
        total_usd = sum(d["usdValue"] for d in details)
        await self._cache.set("okx:portfolio_total_usd", str(round(total_usd, 4)), ttl=BALANCE_TTL)

    async def _import_orders(self, okx_orders: list[dict]) -> int:
        """Salva ordens OKX no PostgreSQL (upsert — não duplica)."""
        if not okx_orders or not self._db:
            return 0

        imported = 0
        for o in okx_orders:
            if not o.get("ordId") or not o.get("symbol"):
                continue
            if o["symbol"] not in TRADING_SYMBOLS:
                continue
            try:
                filled_at  = datetime.fromtimestamp(o["uTime"] / 1000, tz=UTC) if o["uTime"] else datetime.now(UTC)
                created_at = datetime.fromtimestamp(o["cTime"] / 1000, tz=UTC) if o["cTime"] else filled_at
                fee = abs(o.get("fee", 0))

                async with self._db.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO orders (
                            client_order_id, exchange_order_id, symbol, side,
                            order_type, mode, status, quantity, filled_quantity,
                            avg_fill_price, fees_paid, strategy_id, signal_id,
                            submitted_at, filled_at, created_at
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                        ON CONFLICT (client_order_id) DO UPDATE SET
                            status          = EXCLUDED.status,
                            filled_quantity = EXCLUDED.filled_quantity,
                            avg_fill_price  = EXCLUDED.avg_fill_price,
                            fees_paid       = EXCLUDED.fees_paid,
                            filled_at       = EXCLUDED.filled_at
                        """,
                        o["clOrdId"] or o["ordId"],
                        o["ordId"],
                        o["symbol"],
                        o["side"],
                        o.get("ordType", "market"),
                        "market",
                        "filled",
                        float(o.get("accFillSz") or o.get("sz") or 0),
                        float(o.get("accFillSz") or o.get("fillSz") or 0),
                        float(o.get("avgPx") or 0),
                        fee,
                        "okx_import",   # identifica origem: importado do OKX
                        "okx-import",   # signal_id placeholder
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
        Atualiza PortfolioEngine, Redis e PostgreSQL com o estado real da exchange.
        Portfolio total = soma dos usdValues de TODOS os ativos válidos.
        """
        import uuid as _uuid

        from ..core.models import Position, PositionSide, PositionStatus
        from ..persistence.repositories.positions import PositionRepository

        # Portfolio total = soma dos usdValues de todos os ativos válidos
        # (USDT + BTC + ETH + SOL + OKB + BRL)
        total_portfolio = sum(d["usdValue"] for d in details)

        usdt = next((d for d in details if d["ccy"] == "USDT"), {})
        cash = usdt.get("cashBal", 0.0)

        self._portfolio._state.cash_available = cash
        self._portfolio._state.total_value    = total_portfolio

        # Capital inicial: persistido no Redis, restaurado em reboots
        stored_initial = await self._cache.get(INITIAL_CAPITAL_KEY)
        if stored_initial:
            initial_capital = float(stored_initial)
        else:
            initial_capital = INITIAL_CAPITAL_USD
            await self._cache.set(INITIAL_CAPITAL_KEY, str(initial_capital), ttl=0)
            logger.info("Capital inicial demo registrado: $%.2f", initial_capital)

        self._portfolio._state.initial_capital = initial_capital

        logger.info(
            "ExchangeSync portfolio: total=%.2f  inicial=%.2f  retorno=%.2f%%\n"
            "  Detalhes: %s",
            total_portfolio,
            initial_capital,
            (total_portfolio - initial_capital) / initial_capital * 100 if initial_capital else 0,
            " | ".join(f"{d['ccy']}=${d['usdValue']:.2f}" for d in details),
        )

        pos_repo = PositionRepository(self._db) if self._db else None

        # Salva posições crypto negociáveis no Redis e PostgreSQL
        for symbol, qty in crypto_positions.items():
            price = await self._cache.get_price(symbol)
            price = float(price) if price else 0.0

            ccy = symbol.replace("-USDT", "")
            ccy_data  = next((d for d in details if d["ccy"] == ccy), {})
            usd_value = ccy_data.get("usdValue", qty * price)

            # Preço implícito = usdValue / qty (melhor proxy se ticker não disponível)
            if qty > 0 and usd_value > 0 and price == 0:
                price = usd_value / qty
                await self._cache.set_price(symbol, price)

            await self._cache.set_position(symbol, {
                "symbol":         symbol,
                "side":           "long",
                "quantity":       qty,
                "avg_entry":      price,
                "current_price":  price,
                "notional":       usd_value,
                "unrealized_pnl": 0.0,
                "realized_pnl":   0.0,
                "strategy_id":    "exchange_sync",
                "source":         "okx_balance",
            })

            if pos_repo:
                try:
                    existing = await pos_repo.get_open(symbol)
                    if not existing:
                        pos = Position(
                            symbol=symbol,
                            side=PositionSide.LONG,
                            strategy_id="exchange_sync",
                            quantity=qty,
                            avg_entry_price=price,
                            total_fees=0.0,
                            status=PositionStatus.OPEN,
                        )
                        await pos_repo.save(pos, str(_uuid.uuid4()))
                        logger.info("ExchangeSync: posição %s qty=%.6f → DB", symbol, qty)
                    else:
                        logger.info("ExchangeSync: posição %s já existe no DB — mantida", symbol)
                except Exception as pos_exc:
                    logger.debug("ExchangeSync: erro ao salvar posição %s: %s", symbol, pos_exc)

        # Remove posições do Redis e DB que não existem mais na exchange
        for symbol in TRADING_SYMBOLS - set(crypto_positions.keys()):
            # Remove do Redis (qualquer source)
            existing = await self._cache.get_position(symbol)
            if existing:
                await self._cache.delete_position(symbol)
                logger.info("ExchangeSync: posição removida do Redis (saiu da OKX): %s", symbol)
            # Fecha no PostgreSQL também — evita posições "fantasma" no dashboard
            if pos_repo:
                try:
                    closed = await pos_repo.close_stale_by_symbol(symbol)
                    if closed:
                        logger.info(
                            "ExchangeSync: %d posição(ões) %s fechada(s) no DB "
                            "(saldo OKX zerado)", closed, symbol
                        )
                except Exception as e:
                    logger.debug("ExchangeSync: erro ao fechar posição %s no DB: %s", symbol, e)
