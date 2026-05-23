"""
AdvancedRiskManager — Phase 14: Gestão de Risco Avançada.

Cinco módulos integrados:

  14.1 Correlação Cross-Asset
       Matriz rolling 30 dias BTC/ETH/SOL. Sizing reduzido quando
       correlação > 0.85 (risco concentrado).

  14.2 Portfolio VaR/CVaR
       VaR 95% e CVaR 95% históricos a partir dos trade P&Ls reais.
       Alerta quando VaR diário > 2% do portfolio.

  14.3 Stress Scenarios
       Impacto simulado de: crash -20% 24h, lateralidade, vol explosion,
       exchange outage. Calculado periodicamente e exposto via API.

  14.4 Circuit Breakers Avançados
       Bloqueiam novas entradas automaticamente:
         • 5+ losses consecutivos → pause 24h por símbolo
         • Weekly drawdown > 5%   → modo defensivo (kelly ×0.5)
         • Todos ativos -3% em 1h → halt imediato (kill switch)

  14.5 Liquidity Risk
       Volume corrente vs média 20 candles. Se ratio < 0.5 → sizing ×0.5.
       Slippage tracking: esperado vs executado.

Estado dos circuit breakers persiste no Redis (sobrevive restart).
Coleta e análise rodam a cada 15 minutos em background.
"""

import asyncio
import json
import logging
import math
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

POLL_INTERVAL = 900     # 15 min
REDIS_TTL     = 86400   # 24h para estado de CB
SYMBOLS       = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

# ── Circuit Breaker thresholds ────────────────────────────────────────────────
CB_MAX_CONSEC_LOSSES   = 5      # losses consecutivos → pause 24h por símbolo
CB_WEEKLY_DD_PCT       = 0.05   # -5% na semana → modo defensivo
CB_ALL_CRASH_PCT       = 0.03   # todos os ativos -3% em 1H → halt imediato
CB_PAUSE_HOURS         = 24     # duração do pause após CB ativado

# ── Correlation thresholds ────────────────────────────────────────────────────
CORR_REDUCE_THRESHOLD  = 0.85   # correlação > 0.85 → reduzir sizing
CORR_MIN_MULT          = 0.50   # multiplicador mínimo de sizing por correlação

# ── Liquidity threshold ───────────────────────────────────────────────────────
LIQUIDITY_MIN_RATIO    = 0.50   # volume < 50% da média → sizing ×0.5
LIQUIDITY_MIN_MULT     = 0.50

# ── VaR parameters ───────────────────────────────────────────────────────────
VAR_CONFIDENCE         = 0.95   # 95% confidence
VAR_ALERT_PCT          = 0.02   # alerta se VaR diário > 2%
VAR_MIN_TRADES         = 10     # mínimo de trades para calcular VaR


# ── Helpers matemáticos ───────────────────────────────────────────────────────

def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    sx = sum(xs[:n])
    sy = sum(ys[:n])
    sxy = sum(xs[i] * ys[i] for i in range(n))
    sx2 = sum(x * x for x in xs[:n])
    sy2 = sum(y * y for y in ys[:n])
    num = n * sxy - sx * sy
    den = math.sqrt(max(0.0, (n * sx2 - sx * sx) * (n * sy2 - sy * sy)))
    return round(num / den, 4) if den > 0 else 0.0


def _returns(closes: list[float], horizon: int = 720) -> list[float]:
    """Retornos 1H, até `horizon` períodos."""
    n = min(horizon, len(closes) - 1)
    return [(closes[i] - closes[i + 1]) / closes[i + 1]
            for i in range(n) if closes[i + 1] > 0]


def _var_cvar(returns_pct: list[float], confidence: float = VAR_CONFIDENCE) -> dict:
    """VaR e CVaR histórico. returns_pct em fração (não %)."""
    if len(returns_pct) < VAR_MIN_TRADES:
        return {"var_95": None, "cvar_95": None, "n": len(returns_pct)}
    s = sorted(returns_pct)
    idx = int((1 - confidence) * len(s))
    var = abs(s[idx])
    cvar = abs(sum(s[:idx + 1]) / (idx + 1)) if idx >= 0 else var
    return {
        "var_95":  round(var, 6),
        "cvar_95": round(cvar, 6),
        "n":       len(returns_pct),
        "worst":   round(s[0], 6),
        "best":    round(s[-1], 6),
    }


# ── AdvancedRiskManager ───────────────────────────────────────────────────────

class AdvancedRiskManager:
    """
    Gestão de risco avançada — coletor background + gates de sizing/bloqueio.
    Integrado no signal consumer do TradingLoop.
    """

    def __init__(self, market, cache, db=None) -> None:
        self._market  = market
        self._cache   = cache
        self._db      = db        # injetado pós-boot
        self._task: asyncio.Task | None = None
        self._running = False

    def set_db(self, db: object) -> None:
        self._db = db

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._collect()
        self._task = asyncio.create_task(self._loop(), name="advanced_risk")
        logger.info("AdvancedRiskManager started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._collect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("AdvancedRiskManager loop: %s", exc)

    async def _collect(self) -> None:
        """Coleta e persiste todas as métricas de risco avançado."""
        try:
            corr  = self._compute_correlation()
            var   = await self._compute_var()
            stress = self._compute_stress(corr)
            liq   = self._compute_liquidity()
            cb    = await self._evaluate_circuit_breakers(corr)

            payload = {
                "correlation":       corr,
                "var":               var,
                "stress":            stress,
                "liquidity":         liq,
                "circuit_breakers":  cb,
                "updated_at":        datetime.now(UTC).isoformat(),
            }
            await self._cache.set("advanced_risk", json.dumps(payload), ttl=REDIS_TTL)
            logger.debug("AdvancedRisk: corr_max=%.3f var95=%.4f cb_active=%s",
                         corr.get("max_pairwise", 0),
                         var.get("portfolio_var_95", 0) or 0,
                         cb.get("any_active", False))
        except Exception as exc:
            logger.warning("AdvancedRisk._collect: %s", exc)

    # ── 14.1 Correlação Cross-Asset ───────────────────────────────────────────

    def _compute_correlation(self) -> dict:
        """Matriz de correlação rolling 30 dias (720 candles 1H)."""
        returns_map: dict[str, list[float]] = {}
        for sym in SYMBOLS:
            candles = self._market.get_candles(sym, "1H")
            if len(candles) >= 30:
                closes = [c.close for c in candles[:721]]
                returns_map[sym] = _returns(closes, 720)

        matrix: dict[str, dict[str, float]] = {}
        pairs: list[tuple[str, str, float]] = []

        for i, s1 in enumerate(SYMBOLS):
            matrix[s1] = {}
            for s2 in SYMBOLS:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                elif s2 in returns_map and s1 in returns_map:
                    r = _pearson(returns_map[s1], returns_map[s2])
                    matrix[s1][s2] = r
                    if i < SYMBOLS.index(s2):
                        pairs.append((s1, s2, r))
                else:
                    matrix[s1][s2] = None

        max_corr = max((abs(r) for _, _, r in pairs), default=0.0)
        avg_corr = round(sum(abs(r) for _, _, r in pairs) / len(pairs), 4) if pairs else 0.0
        corr_regime = (
            "HIGH"   if max_corr >= 0.85 else
            "MEDIUM" if max_corr >= 0.60 else
            "LOW"
        )

        # Sizing multiplier por correlação
        # Se correlação > 0.85 → sizing do portfolio reduzido
        corr_sizing_mult = 1.0
        if max_corr >= CORR_REDUCE_THRESHOLD:
            corr_sizing_mult = max(CORR_MIN_MULT,
                                   1.0 - (max_corr - CORR_REDUCE_THRESHOLD) / 0.15)

        return {
            "matrix":           matrix,
            "pairs":            [{"sym1": p[0], "sym2": p[1], "corr": p[2]} for p in pairs],
            "max_pairwise":     round(max_corr, 4),
            "avg_pairwise":     avg_corr,
            "regime":           corr_regime,
            "sizing_mult":      round(corr_sizing_mult, 3),
            "high_corr_active": max_corr >= CORR_REDUCE_THRESHOLD,
        }

    # ── 14.2 VaR / CVaR ───────────────────────────────────────────────────────

    async def _compute_var(self) -> dict:
        """VaR 95% e CVaR 95% histórico dos trade P&Ls reais."""
        if self._db is None:
            return {"available": False, "reason": "DB não conectado"}

        try:
            from src.persistence.repositories.orders import EXCLUDED_STRATEGY_IDS
            rows = await self._db.fetch(
                """
                SELECT avg_fill_price, filled_quantity, fees_paid, side, filled_at, symbol
                FROM orders
                WHERE status='filled' AND strategy_id != ALL($1)
                ORDER BY filled_at ASC NULLS LAST
                LIMIT 200
                """,
                list(EXCLUDED_STRATEGY_IDS),
            )
            # Emparelha BUY→SELL por símbolo (FIFO simples)
            buys: dict[str, list] = {}
            pnl_pcts: list[float] = []
            for r in rows:
                sym  = r["symbol"] if "symbol" in r.keys() else "?"
                side = str(r["side"]).upper()
                px   = float(r["avg_fill_price"] or 0)
                qty  = float(r["filled_quantity"] or 0)
                fee  = float(r["fees_paid"] or 0)
                if side in ("BUY", "LONG"):
                    buys.setdefault(sym, []).append((px, qty, fee))
                elif side in ("SELL", "SHORT") and buys.get(sym):
                    entry_px, entry_qty, entry_fee = buys[sym].pop(0)
                    if entry_px > 0:
                        pnl_pct = (px - entry_px) / entry_px - fee / (entry_px * entry_qty + 1e-9)
                        pnl_pcts.append(pnl_pct)

            stats = _var_cvar(pnl_pcts)
            portfolio_var = None
            var_alert = False
            if stats["var_95"] is not None:
                # VaR diário estimado = VaR por trade × sqrt(avg trades/dia)
                # Aproximação conservadora: 1 trade/dia
                portfolio_var = round(stats["var_95"], 6)
                var_alert = portfolio_var > VAR_ALERT_PCT

            return {
                "available":       len(pnl_pcts) >= VAR_MIN_TRADES,
                "n_trades":        len(pnl_pcts),
                "var_95":          stats.get("var_95"),
                "cvar_95":         stats.get("cvar_95"),
                "portfolio_var_95": portfolio_var,
                "var_alert":       var_alert,
                "worst_trade_pct": stats.get("worst"),
                "best_trade_pct":  stats.get("best"),
                "var_alert_threshold": VAR_ALERT_PCT,
            }
        except Exception as exc:
            logger.debug("VaR compute error: %s", exc)
            return {"available": False, "reason": str(exc)}

    # ── 14.3 Stress Scenarios ─────────────────────────────────────────────────

    def _compute_stress(self, corr: dict) -> dict:
        """
        Impacto de cenários adversos no portfolio atual.
        Usa preços recentes e correlação calculada.
        """
        scenarios: list[dict] = []

        def _scenario(name: str, shock_pct: float, description: str,
                       corr_amplifier: float = 1.0) -> dict:
            """Calcula P&L estimado para um choque de mercado."""
            # Em alta correlação, choque afeta todos os ativos simultaneamente
            effective_shock = shock_pct * (1.0 + corr.get("max_pairwise", 0) * corr_amplifier * 0.5)
            # Assume exposição máxima de 30% do portfolio (kelly cap)
            max_exposure = 0.30
            portfolio_impact_pct = effective_shock * max_exposure
            return {
                "name":               name,
                "description":        description,
                "market_shock_pct":   round(shock_pct * 100, 1),
                "effective_shock_pct":round(effective_shock * 100, 1),
                "portfolio_impact_pct":round(portfolio_impact_pct * 100, 2),
                "severity":           "CRÍTICO" if portfolio_impact_pct < -0.05
                                      else "ALTO" if portfolio_impact_pct < -0.02
                                      else "MÉDIO",
            }

        scenarios = [
            _scenario("Crash -20% (FTX/LUNA)",     -0.20,
                      "Queda de 20% em 24h — histórico Nov/22, Jun/22", 1.5),
            _scenario("Crash -10% (correção)",      -0.10,
                      "Correção moderada — comum em mercados de crypto"),
            _scenario("Rally -5% (pullback)",       -0.05,
                      "Pullback saudável dentro de tendência"),
            _scenario("Volatility explosion ×2",    -0.08,
                      "ATR dobra em < 4h — forçando saídas por SL", 2.0),
            _scenario("Exchange outage 4h",         -0.04,
                      "Posição presa sem poder sair durante outage"),
        ]

        worst = min(s["portfolio_impact_pct"] for s in scenarios)
        return {
            "scenarios":         scenarios,
            "worst_case_pct":    round(worst, 2),
            "worst_case_label":  next(s["name"] for s in scenarios
                                      if s["portfolio_impact_pct"] == worst),
            "corr_amplifier_active": corr.get("high_corr_active", False),
        }

    # ── 14.4 Circuit Breakers ─────────────────────────────────────────────────

    async def _evaluate_circuit_breakers(self, corr: dict) -> dict:
        """
        Avalia todos os circuit breakers e persiste estado no Redis.
        Retorna status de cada CB e flag any_active.
        """
        results: dict[str, dict] = {}
        any_active = False

        # CB 1: 5+ losses consecutivos por símbolo (pausa 24h)
        if self._db is not None:
            for sym in SYMBOLS:
                cb_key = f"circuit_breaker:consec_loss:{sym}"
                try:
                    from src.persistence.repositories.orders import EXCLUDED_STRATEGY_IDS
                    rows = await self._db.fetch(
                        """
                        SELECT side, avg_fill_price, filled_quantity, filled_at
                        FROM orders
                        WHERE status='filled' AND symbol=$1
                          AND strategy_id != ALL($2)
                        ORDER BY filled_at DESC NULLS LAST
                        LIMIT 20
                        """,
                        sym, list(EXCLUDED_STRATEGY_IDS),
                    )
                    # Conta losses consecutivos
                    consec = 0
                    prev_buy: tuple | None = None
                    for r in reversed(rows):
                        side = str(r["side"]).upper()
                        px   = float(r["avg_fill_price"] or 0)
                        if side in ("BUY", "LONG"):
                            prev_buy = (px,)
                        elif side in ("SELL", "SHORT") and prev_buy:
                            if px < prev_buy[0]:
                                consec += 1
                            else:
                                consec = 0
                            prev_buy = None

                    triggered = consec >= CB_MAX_CONSEC_LOSSES
                    # Verifica se já estava em pausa
                    paused_until_raw = await self._cache.get(cb_key)
                    paused_until = None
                    if paused_until_raw:
                        try:
                            paused_until = datetime.fromisoformat(
                                paused_until_raw if isinstance(paused_until_raw, str)
                                else str(paused_until_raw)
                            )
                        except (ValueError, TypeError):
                            pass

                    now = datetime.now(UTC)
                    still_paused = paused_until and paused_until > now

                    if triggered and not still_paused:
                        pause_until = now + timedelta(hours=CB_PAUSE_HOURS)
                        await self._cache.set(cb_key, pause_until.isoformat(), ttl=REDIS_TTL)
                        paused_until = pause_until
                        still_paused = True
                        logger.warning(
                            "CIRCUIT BREAKER: %s — %d losses consecutivos → pausa %dh",
                            sym, consec, CB_PAUSE_HOURS,
                        )

                    cb_sym = {
                        "type":            "CONSEC_LOSS",
                        "symbol":          sym,
                        "consec_losses":   consec,
                        "threshold":       CB_MAX_CONSEC_LOSSES,
                        "triggered":       triggered or bool(still_paused),
                        "paused_until":    paused_until.isoformat() if paused_until else None,
                        "pause_remaining_h": round(
                            (paused_until - now).total_seconds() / 3600, 1
                        ) if still_paused else 0,
                    }
                    results[f"consec_loss_{sym}"] = cb_sym
                    if cb_sym["triggered"]:
                        any_active = True
                except Exception as exc:
                    logger.debug("CB consec_loss %s: %s", sym, exc)

        # CB 2: Weekly drawdown > 5% → modo defensivo (kelly ×0.5)
        try:
            hwm_raw  = await self._cache.get("analytics:hwm")
            curr_raw = await self._cache.get("portfolio:equity")
            if hwm_raw and curr_raw:
                hwm  = float(hwm_raw)
                curr = float(curr_raw) if isinstance(curr_raw, (int, float)) else float(str(curr_raw))
                weekly_dd = (curr - hwm) / hwm if hwm > 0 else 0.0
                dd_triggered = weekly_dd < -CB_WEEKLY_DD_PCT
                results["weekly_drawdown"] = {
                    "type":              "WEEKLY_DD",
                    "current_dd_pct":    round(weekly_dd * 100, 3),
                    "threshold_pct":     CB_WEEKLY_DD_PCT * 100,
                    "triggered":         dd_triggered,
                    "kelly_mult":        0.5 if dd_triggered else 1.0,
                    "mode":              "DEFENSIVE" if dd_triggered else "NORMAL",
                }
                if dd_triggered:
                    any_active = True
                    logger.warning("CIRCUIT BREAKER: weekly drawdown %.2f%% > threshold %.0f%%",
                                   weekly_dd * 100, CB_WEEKLY_DD_PCT * 100)
        except Exception as exc:
            logger.debug("CB weekly_dd: %s", exc)

        # CB 3: Todos os ativos -3% em 1H → halt imediato
        try:
            crash_count = 0
            crash_returns: dict[str, float] = {}
            for sym in SYMBOLS:
                candles = self._market.get_candles(sym, "1H")
                if len(candles) >= 2:
                    ret = (candles[0].close - candles[1].close) / candles[1].close
                    crash_returns[sym] = round(ret, 5)
                    if ret < -CB_ALL_CRASH_PCT:
                        crash_count += 1

            all_crash = crash_count == len(SYMBOLS)
            results["all_assets_crash"] = {
                "type":        "ALL_CRASH",
                "returns_1h":  crash_returns,
                "crash_count": crash_count,
                "threshold_pct": CB_ALL_CRASH_PCT * 100,
                "triggered":   all_crash,
                "action":      "HALT" if all_crash else "OK",
            }
            if all_crash:
                any_active = True
                logger.error("CIRCUIT BREAKER: todos os ativos caindo > %.0f%% em 1H — HALT",
                             CB_ALL_CRASH_PCT * 100)
        except Exception as exc:
            logger.debug("CB all_crash: %s", exc)

        # CB 4: Alta correlação + sizing reduction (da 14.1)
        results["high_correlation"] = {
            "type":        "HIGH_CORR",
            "max_corr":    corr.get("max_pairwise", 0),
            "threshold":   CORR_REDUCE_THRESHOLD,
            "triggered":   corr.get("high_corr_active", False),
            "sizing_mult": corr.get("sizing_mult", 1.0),
            "action":      "REDUCE_SIZING" if corr.get("high_corr_active") else "OK",
        }
        if corr.get("high_corr_active"):
            any_active = True

        return {
            "any_active":  any_active,
            "checklist":   results,
            "evaluated_at": datetime.now(UTC).isoformat(),
        }

    # ── 14.5 Liquidity Risk ────────────────────────────────────────────────────

    def _compute_liquidity(self) -> dict:
        """Volume corrente vs média 20 candles por símbolo."""
        by_sym: dict[str, dict] = {}
        for sym in SYMBOLS:
            candles = self._market.get_candles(sym, "1H")
            if len(candles) < 5:
                by_sym[sym] = {"available": False}
                continue
            vols     = [c.volume for c in candles[:21]]
            curr_vol = vols[0]
            avg20    = sum(vols[1:21]) / len(vols[1:21]) if len(vols) > 1 else curr_vol
            ratio    = curr_vol / avg20 if avg20 > 0 else 1.0
            iliquid  = ratio < LIQUIDITY_MIN_RATIO
            sizing_mult = LIQUIDITY_MIN_MULT if iliquid else min(1.0, 0.5 + ratio * 0.5)
            by_sym[sym] = {
                "current_vol":  round(curr_vol, 2),
                "avg20_vol":    round(avg20, 2),
                "ratio":        round(ratio, 3),
                "illiquid":     iliquid,
                "sizing_mult":  round(sizing_mult, 3),
            }

        any_illiquid = any(v.get("illiquid", False) for v in by_sym.values())
        return {
            "by_symbol":    by_sym,
            "any_illiquid": any_illiquid,
        }

    # ── Public interface — chamados pelo signal consumer ──────────────────────

    async def get_snapshot(self) -> dict | None:
        """Lê snapshot completo do Redis."""
        raw = await self._cache.get("advanced_risk")
        if not raw:
            return None
        return raw if isinstance(raw, dict) else json.loads(raw)

    async def check_circuit_breakers(self, symbol: str) -> dict:
        """
        Verifica todos os circuit breakers para um símbolo.
        Retorna: {allowed: bool, reason: str, kelly_mult: float}
        """
        snap = await self.get_snapshot()
        if not snap:
            return {"allowed": True, "reason": "no_data", "kelly_mult": 1.0}

        cb = snap.get("circuit_breakers", {})
        checklist = cb.get("checklist", {})

        # CB 3 → halt imediato (bloqueia todos)
        all_crash = checklist.get("all_assets_crash", {})
        if all_crash.get("triggered"):
            return {
                "allowed":    False,
                "reason":     "ALL_CRASH — todos os ativos caindo > 3% em 1H",
                "kelly_mult": 0.0,
                "cb_type":    "ALL_CRASH",
            }

        # CB 1 → pausa por símbolo
        cb_sym = checklist.get(f"consec_loss_{symbol}", {})
        if cb_sym.get("triggered"):
            h = cb_sym.get("pause_remaining_h", 0)
            return {
                "allowed":    False,
                "reason":     f"CONSEC_LOSS — {cb_sym.get('consec_losses')} losses consecutivos (pausa {h:.1f}h)",
                "kelly_mult": 0.0,
                "cb_type":    "CONSEC_LOSS",
            }

        # CB 2 → modo defensivo (permite mas reduz kelly)
        kelly_mult = 1.0
        weekly_dd  = checklist.get("weekly_drawdown", {})
        if weekly_dd.get("triggered"):
            kelly_mult = min(kelly_mult, weekly_dd.get("kelly_mult", 0.5))

        # CB 4 → reduz sizing por correlação
        high_corr = checklist.get("high_correlation", {})
        if high_corr.get("triggered"):
            kelly_mult = min(kelly_mult, high_corr.get("sizing_mult", 0.5))

        # 14.5 → reduz sizing por liquidez
        liq = snap.get("liquidity", {})
        sym_liq = liq.get("by_symbol", {}).get(symbol, {})
        if sym_liq.get("illiquid"):
            kelly_mult = min(kelly_mult, sym_liq.get("sizing_mult", 0.5))

        return {
            "allowed":    True,
            "reason":     "ok" if kelly_mult == 1.0 else "sizing_reduced",
            "kelly_mult": round(kelly_mult, 3),
            "cb_type":    None,
        }

    async def get_var_alert(self) -> dict | None:
        """Retorna alerta de VaR se ativo."""
        snap = await self.get_snapshot()
        if not snap:
            return None
        var = snap.get("var", {})
        if var.get("var_alert"):
            return {
                "alert":    True,
                "var_95":   var.get("portfolio_var_95"),
                "threshold": VAR_ALERT_PCT,
            }
        return None
