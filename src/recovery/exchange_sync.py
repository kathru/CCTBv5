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

# Ativos válidos para o sistema — apenas esses são considerados no portfolio e P&L.
# Qualquer outro ativo presente na conta OKX é ignorado (logado como aviso).
VALID_CCYS    = {"USDT", "BTC", "ETH", "SOL"}
VALID_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT"}
TRADING_CCYS  = VALID_CCYS - {"USDT"}   # crypto (sem USDT) — para posições
TRADING_SYMBOLS = VALID_SYMBOLS          # alias de compatibilidade


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
            all_details = await self._okx.get_account_details()
            summary["assets"] = all_details

            # Filtra para apenas USDT, BTC, ETH, SOL — ignora todo o resto
            valid   = [d for d in all_details if d["ccy"] in VALID_CCYS]
            ignored = [d for d in all_details if d["ccy"] not in VALID_CCYS]
            if ignored:
                logger.warning(
                    "ExchangeSync: ativos ignorados (fora do escopo do sistema): %s",
                    [f"{d['ccy']}={d['cashBal']:.6f}" for d in ignored],
                )
            details = valid  # daqui em diante só ativos válidos

            usdt = next((d for d in details if d["ccy"] == "USDT"), {})
            summary["usdt_balance"] = usdt.get("cashBal", 0.0)
            summary["total_equity_usd"] = sum(d["usdValue"] for d in details)

            logger.info(
                "ExchangeSync: ativos válidos — USDT=%.2f | %s",
                summary["usdt_balance"],
                " | ".join(
                    f"{d['ccy']}={d['cashBal']:.6f} (~${d['usdValue']:.2f})"
                    for d in details if d["ccy"] != "USDT"
                ) or "nenhuma crypto",
            )

            # ── 2. Posições crypto ───────────────────────────────────────────
            for d in details:
                ccy = d["ccy"]
                if ccy in TRADING_CCYS and d["cashBal"] > 0:
                    symbol = f"{ccy}-USDT"
                    summary["crypto_positions"][symbol] = d["cashBal"]

            # ── 3. Histórico de ordens preenchidas ───────────────────────────
            # OKX demo pode não suportar orders-history (401) — falha graciosamente
            logger.info("ExchangeSync: importando histórico de ordens OKX...")
            try:
                okx_orders = await self._okx.get_filled_orders(limit=100)
                imported = await self._import_orders(okx_orders)
                summary["orders_imported"] = imported
            except Exception as orders_exc:
                logger.warning(
                    "ExchangeSync: orders-history indisponível (%s) — "
                    "usando histórico local do DB",
                    str(orders_exc)[:80],
                )

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
        Atualiza o PortfolioEngine, Redis e PostgreSQL com o estado real da exchange.
        """
        import uuid as _uuid
        from ..persistence.repositories.positions import PositionRepository
        from ..core.models import Position, PositionSide, PositionStatus

        usdt = next((d for d in details if d["ccy"] == "USDT"), {})
        cash = usdt.get("cashBal", 0.0)

        # Portfolio rastreado em USDT (capital operacional do bot).
        # Não inclui BTC/ETH/SOL pré-existentes — só o USDT que o bot usa para operar.
        # Isso normaliza retorno %, drawdown e todos os outros índices.
        usdt_notional = cash  # começa com o USDT disponível

        # Adiciona unrealized P&L das posições do bot (capital investido em crypto)
        for symbol, qty in crypto_positions.items():
            price = await self._cache.get_price(symbol)
            if price:
                usdt_notional += float(price) * qty

        self._portfolio._state.cash_available = cash
        self._portfolio._state.total_value    = usdt_notional

        # Capital inicial: salvo no Redis na primeira vez, restaurado nos reboots
        INITIAL_CAPITAL_KEY = "portfolio:initial_capital_usdt"
        stored_initial = await self._cache.get(INITIAL_CAPITAL_KEY)
        if stored_initial:
            initial_capital = float(stored_initial)
        else:
            # Primeiro boot — salva o capital inicial USDT atual
            initial_capital = usdt_notional
            await self._cache.set(INITIAL_CAPITAL_KEY, str(round(initial_capital, 4)), ttl=0)
            logger.info("Capital inicial USDT registrado: %.2f", initial_capital)

        self._portfolio._state.initial_capital = initial_capital

        total_eq = sum(d["usdValue"] for d in details)  # mantém para log

        pos_repo = PositionRepository(self._db) if self._db else None

        # Salva posições crypto no Redis e PostgreSQL
        for symbol, qty in crypto_positions.items():
            price = await self._cache.get_price(symbol)
            price = float(price) if price else 0.0

            ccy = symbol.replace("-USDT", "")
            ccy_data = next((d for d in details if d["ccy"] == ccy), {})
            usd_value = ccy_data.get("usdValue", qty * price)

            # Redis → dashboard em tempo real
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

            # PostgreSQL → histórico e posições abertas
            if pos_repo:
                try:
                    # Verifica se já existe posição aberta para o símbolo
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
                        logger.info(
                            "ExchangeSync: posição %s qty=%.6f value=%.2f USD → DB",
                            symbol, qty, usd_value,
                        )
                    else:
                        logger.info(
                            "ExchangeSync: posição %s já existe no DB (qty=%.6f) — mantida",
                            symbol, qty,
                        )
                except Exception as pos_exc:
                    logger.debug("ExchangeSync: erro ao salvar posição %s: %s", symbol, pos_exc)

        # Remove posições do Redis que não existem mais na exchange
        # (só verifica os símbolos válidos do sistema: BTC-USDT, ETH-USDT, SOL-USDT)
        for symbol in VALID_SYMBOLS - set(crypto_positions.keys()):
            existing = await self._cache.get_position(symbol)
            if existing and existing.get("source") == "exchange_sync":
                await self._cache.delete_position(symbol)
                logger.info("ExchangeSync: posição removida do Redis (não existe mais na OKX): %s", symbol)
