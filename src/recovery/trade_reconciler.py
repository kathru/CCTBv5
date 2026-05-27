"""
TradeReconciler — Conciliação OKX ↔ banco de dados local.

Problema: OKX pode ter executado trades que o bot não capturou:
  - Fills durante restarts
  - Trades manuais feitos diretamente na OKX
  - Ordens parcialmente preenchidas não reportadas
  - orders-history 401 no demo mode (não importável pela API)

Solução: Comparação baseada em SALDO REAL da OKX.
  Para cada ativo negociável (BTC, ETH, SOL):
    okx_qty  = saldo real na conta OKX
    db_qty   = sum(filled buy qty) - sum(filled sell qty) no DB

  Se okx_qty ≠ db_qty → divergência detectada.

Estratégia de correção:
  1. Divergência pequena (< DRIFT_THRESHOLD): cria ajuste sintético no DB
  2. Divergência grande: alerta + atualiza posição no DB
  3. Posição zerada na OKX mas aberta no DB: fecha posição no DB

Roda a cada RECONCILE_INTERVAL via ExchangeSync.sync_balances().
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Thresholds de tolerância (ruído de arredondamento OKX)
DRIFT_THRESHOLD = {
    "BTC-USDT": 0.000_01,    # 0.00001 BTC ≈ $0.77
    "ETH-USDT": 0.000_1,     # 0.0001 ETH ≈ $0.21
    "SOL-USDT": 0.01,        # 0.01 SOL ≈ $0.86
}
DEFAULT_DRIFT = 0.001


class TradeReconciler:
    """
    Compara saldo real OKX com estado do DB e corrige divergências.
    """

    def __init__(self, db, cache, exchange) -> None:
        self._db       = db
        self._cache    = cache
        self._exchange = exchange

    async def run(self) -> dict:
        """
        Executa ciclo de conciliação. Retorna relatório.
        """
        report = {
            "checked_at":    datetime.now(UTC).isoformat(),
            "symbols":       {},
            "divergences":   0,
            "fixed":         0,
            "warnings":      [],
        }

        try:
            # ── 1. Lê saldo real da OKX ────────────────────────────────────
            details = await self._exchange.get_account_details()
            okx_qtys: dict[str, float] = {}
            for d in details:
                ccy = d["ccy"]
                if ccy in ("BTC", "ETH", "SOL"):
                    okx_qtys[f"{ccy}-USDT"] = float(d["cashBal"])

            if not okx_qtys:
                logger.debug("TradeReconciler: sem crypto na conta OKX")
                return report

            # ── 2. Lê state do DB por símbolo ──────────────────────────────
            db_state = await self._get_db_state()

            # ── 3. Compara e corrige ────────────────────────────────────────
            for symbol, okx_qty in okx_qtys.items():
                db_qty    = db_state.get(symbol, {}).get("net_qty", 0.0)
                threshold = DRIFT_THRESHOLD.get(symbol, DEFAULT_DRIFT)
                drift     = okx_qty - db_qty

                sym_report = {
                    "okx_qty":  round(okx_qty, 8),
                    "db_qty":   round(db_qty, 8),
                    "drift":    round(drift, 8),
                    "status":   "ok",
                }

                if abs(drift) <= threshold:
                    # Dentro da tolerância — OK
                    sym_report["status"] = "ok"
                    logger.debug(
                        "TradeReconciler %s: OK (okx=%.6f db=%.6f drift=%.8f)",
                        symbol, okx_qty, db_qty, drift,
                    )

                elif abs(drift) > threshold:
                    report["divergences"] += 1
                    sym_report["status"] = "divergence"

                    logger.warning(
                        "TradeReconciler %s: DIVERGÊNCIA "
                        "okx=%.6f db=%.6f drift=%+.6f",
                        symbol, okx_qty, db_qty, drift,
                    )

                    # Tenta corrigir automaticamente
                    fixed = await self._fix_divergence(symbol, okx_qty, db_qty, drift, db_state)
                    if fixed:
                        report["fixed"] += 1
                        sym_report["status"] = "fixed"
                        sym_report["fix_applied"] = True

                    warn = (
                        f"{symbol}: OKX={okx_qty:.6f} DB={db_qty:.6f} "
                        f"drift={drift:+.6f} → {'CORRIGIDO' if fixed else 'PENDENTE'}"
                    )
                    report["warnings"].append(warn)

                report["symbols"][symbol] = sym_report

            if report["divergences"] == 0:
                logger.debug(
                    "TradeReconciler: tudo conciliado (%d símbolos)", len(okx_qtys)
                )
            else:
                logger.warning(
                    "TradeReconciler: %d divergência(s), %d corrigida(s)",
                    report["divergences"], report["fixed"],
                )

        except Exception as exc:
            logger.error("TradeReconciler falhou: %s", exc, exc_info=True)
            report["error"] = str(exc)

        # Persiste último relatório no Redis para o dashboard
        if self._cache:
            try:
                import json
                await self._cache.set(
                    "reconciler:last_report",
                    json.dumps(report),
                    ttl=600,
                )
            except Exception:
                pass

        return report

    async def _get_db_state(self) -> dict:
        """
        Lê do DB a quantidade líquida de cada símbolo:
          net_qty = sum(buy filled_qty) - sum(sell filled_qty)
        Representa o que o bot acredita ter de cada ativo.
        """
        result: dict[str, dict] = {}
        if not self._db:
            return result

        try:
            from ..persistence.repositories.orders import EXCLUDED_STRATEGY_IDS
            # Use NOT IN (excluded) instead of IN (whitelist) so new strategies are
            # automatically included without requiring a code change here.
            # 'reconciler' is included here (NOT excluded) because synthetic orders
            # inserted by TradeReconciler represent real balance adjustments and must
            # be counted to prevent the same divergence from being re-reported.
            excluded_without_reconciler = [
                sid for sid in EXCLUDED_STRATEGY_IDS if sid != "reconciler"
            ]
            rows = await self._db.fetch(
                """
                SELECT symbol, side,
                       SUM(filled_quantity) AS qty,
                       SUM(filled_quantity * avg_fill_price) AS notional
                FROM orders
                WHERE status = 'filled'
                  AND strategy_id != ALL($1)
                GROUP BY symbol, side
                """,
                excluded_without_reconciler,
            )

            by_sym: dict[str, dict] = {}
            for r in rows:
                sym  = r["symbol"]
                side = str(r["side"]).upper()
                qty  = float(r["qty"] or 0)
                not_ = float(r["notional"] or 0)

                if sym not in by_sym:
                    by_sym[sym] = {"buy_qty": 0.0, "sell_qty": 0.0,
                                   "buy_notional": 0.0, "sell_notional": 0.0}

                if side in ("BUY", "LONG"):
                    by_sym[sym]["buy_qty"]      += qty
                    by_sym[sym]["buy_notional"] += not_
                elif side in ("SELL", "SHORT"):
                    by_sym[sym]["sell_qty"]      += qty
                    by_sym[sym]["sell_notional"] += not_

            for sym, st in by_sym.items():
                net_qty   = st["buy_qty"] - st["sell_qty"]
                avg_entry = (st["buy_notional"] / st["buy_qty"]
                             if st["buy_qty"] > 0 else 0.0)
                result[sym] = {
                    "net_qty":       max(net_qty, 0.0),
                    "buy_qty":       st["buy_qty"],
                    "sell_qty":      st["sell_qty"],
                    "avg_entry":     avg_entry,
                }

        except Exception as exc:
            logger.error("TradeReconciler _get_db_state falhou: %s", exc)

        return result

    async def _fix_divergence(
        self,
        symbol: str,
        okx_qty: float,
        db_qty: float,
        drift: float,
        db_state: dict,
    ) -> bool:
        """
        Tenta corrigir automaticamente a divergência inserindo um ajuste sintético.

        drift > 0: OKX tem mais que o DB → bot perdeu um BUY ou trade externo
        drift < 0: OKX tem menos que o DB → bot perdeu um SELL ou trade externo
        okx_qty ≈ 0: posição zerada na OKX mas aberta no DB → fecha no DB
        """
        if not self._db:
            return False

        import uuid
        now = datetime.now(UTC)

        try:
            price_raw = await self._cache.get_price(symbol) if self._cache else None
            price     = float(price_raw) if price_raw else 0.0

            if okx_qty < DRIFT_THRESHOLD.get(symbol, DEFAULT_DRIFT):
                # Posição zerada na OKX mas ainda aberta no DB.
                # Isso indica que o bot executou um SELL (ex: durante restart)
                # mas o fill não foi gravado. Inserimos um SELL sintético rastreável
                # para que o trade apareça no histórico — "o bot fez, tem que aparecer".
                db_net_qty = db_state.get(symbol, {}).get("net_qty", 0.0)

                if db_net_qty > DRIFT_THRESHOLD.get(symbol, DEFAULT_DRIFT) and price > 0:
                    sell_client_id = f"RECON-SELL-{symbol[:3]}-{int(now.timestamp())}"
                    await self._db.execute(
                        """
                        INSERT INTO orders (
                            client_order_id, exchange_order_id, symbol, side,
                            order_type, mode, status, quantity, filled_quantity,
                            avg_fill_price, fees_paid, strategy_id,
                            submitted_at, filled_at, created_at
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                        ON CONFLICT (client_order_id) DO NOTHING
                        """,
                        sell_client_id,
                        f"OKX-RECON-SELL-{uuid.uuid4().hex[:8]}",
                        symbol,
                        "sell",
                        "market", "market", "filled",
                        db_net_qty, db_net_qty,
                        price,
                        0.0,          # fee não capturada — melhor que não registrar
                        "reconciler", # strategy_id especial — identificável no histórico
                        now, now, now,
                    )
                    logger.warning(
                        "TradeReconciler %s: SELL faltante detectado "
                        "(posição zerou no OKX sem SELL no DB) → "
                        "SELL sintético inserido qty=%.6f @ %.2f [%s]",
                        symbol, db_net_qty, price, sell_client_id,
                    )

                # Fecha posição no DB
                await self._db.execute(
                    "UPDATE positions SET status='closed', closed_at=$1 "
                    "WHERE symbol=$2 AND status='open'",
                    now, symbol,
                )
                logger.info(
                    "TradeReconciler %s: posição zerada na OKX → fechada no DB",
                    symbol,
                )
                return True

            if abs(drift) > 0 and price > 0:
                # Insere ordem sintética de ajuste para reconciliar o saldo
                side       = "buy" if drift > 0 else "sell"
                adj_qty    = abs(drift)
                client_id  = f"RECON-{symbol[:3]}-{int(now.timestamp())}"

                await self._db.execute(
                    """
                    INSERT INTO orders (
                        client_order_id, exchange_order_id, symbol, side,
                        order_type, mode, status, quantity, filled_quantity,
                        avg_fill_price, fees_paid, strategy_id,
                        submitted_at, filled_at, created_at
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                    ON CONFLICT (client_order_id) DO NOTHING
                    """,
                    client_id,
                    f"OKX-RECON-{uuid.uuid4().hex[:8]}",
                    symbol,
                    side,
                    "market", "market", "filled",
                    adj_qty, adj_qty,
                    price,
                    0.0,
                    "reconciler",   # strategy_id especial — identificável
                    now, now, now,
                )

                logger.info(
                    "TradeReconciler %s: ajuste sintético %s qty=%.6f @ %.2f "
                    "(drift=%+.6f corrigido)",
                    symbol, side.upper(), adj_qty, price, drift,
                )
                return True

        except Exception as exc:
            logger.error(
                "TradeReconciler _fix_divergence %s falhou: %s", symbol, exc
            )

        return False
