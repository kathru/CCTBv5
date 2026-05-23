"""
Trading Alerts Manager — CCTBv5

Centraliza todos os alertas de trading enviados ao Discord:
  - Trade executado (buy/sell, paper + live)
  - Posição fechada com P&L
  - Mudança de regime
  - Drawdown warning (2%) e crítico (3%)
  - Win rate degradado abaixo de 30%
  - Circuit breaker ativado

Design: stateful mínimo — só armazena o necessário para detectar mudanças.
"""

import logging
from datetime import UTC, datetime

from .base import Alert, AlertChannel, AlertLevel

logger = logging.getLogger(__name__)


class TradingAlertsManager:
    """
    Gerencia todos os alertas de trading.
    Integrado ao TradingLoop via injeção de dependência.
    """

    def __init__(self, channel: AlertChannel) -> None:
        self._ch = channel

        # Estado para detecção de mudanças
        self._last_regime:       str   = ""
        self._last_dd_warning:   bool  = False   # flag 2% disparado
        self._last_dd_critical:  bool  = False   # flag 3% disparado
        self._last_wr_warning:   bool  = False   # flag WR < 30%
        self._trades_today:      int   = 0
        self._pnl_today:         float = 0.0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _regime_emoji(self, regime: str) -> str:
        r = regime.upper()
        if "EXPANSION" in r:
            return "🟢"
        if "COMPRESSION" in r:
            return "🟡"
        if "EXHAUSTION" in r:
            return "🔵"
        if "CHOP" in r:
            return "🟡"
        if "PANIC" in r:
            return "🚨"
        if "BEAR" in r:
            return "🔴"
        if "CORR" in r:
            return "🟠"
        return "⚪"

    # ── Alertas de Trade ───────────────────────────────────────────────────────

    async def on_order_filled(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        notional: float,
        fees: float,
        strategy_id: str,
        regime: str = "",
        score: float = 0.0,
    ) -> None:
        """Dispara quando uma ordem é preenchida (buy ou sell)."""
        side_up  = side.upper()
        is_buy   = side_up in ("BUY", "LONG")
        emoji    = "🟢 COMPRA" if is_buy else "🔴 VENDA"
        sym      = symbol.replace("-USDT", "")
        reg_txt  = f"{self._regime_emoji(regime)} {regime.replace('_',' ')}" if regime else "–"

        fields = {
            "Símbolo":   sym,
            "Quantidade": f"{quantity:.4f}",
            "Preço":     f"${price:,.2f}",
            "Notional":  f"${notional:,.0f}",
            "Fees":      f"${fees:.2f}",
            "Regime":    reg_txt,
        }
        if score > 0:
            fields["Score"] = f"{score*100:.1f}%"

        await self._ch.send(Alert(
            level=AlertLevel.INFO,
            title=f"{emoji} — {symbol}",
            message=f"Ordem preenchida | Estratégia: `{strategy_id}`",
            fields=fields,
        ))

    async def on_position_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        hold_time_h: float,
        entry_price: float,
        exit_price: float,
    ) -> None:
        """Dispara quando uma posição é fechada com P&L calculado."""
        win    = pnl >= 0
        emoji  = "✅ WIN" if win else "❌ LOSS"
        sign   = "+" if pnl >= 0 else ""
        sym    = symbol.replace("-USDT", "")

        self._trades_today += 1
        self._pnl_today    += pnl

        await self._ch.send(Alert(
            level=AlertLevel.INFO if win else AlertLevel.WARNING,
            title=f"{emoji} — {sym}",
            message=f"P&L: **{sign}${pnl:,.2f}** ({sign}{pnl_pct:.2f}%)",
            fields={
                "Entrada":    f"${entry_price:,.2f}",
                "Saída":      f"${exit_price:,.2f}",
                "Hold Time":  f"{hold_time_h:.1f}h",
                "Hoje":       f"{sign}${self._pnl_today:,.2f} em {self._trades_today}t",
            },
        ))

    # ── Alertas de Regime ──────────────────────────────────────────────────────

    async def on_regime_check(self, regime: str) -> None:
        """
        Verifica se o regime mudou e dispara alerta apenas na mudança.
        Chame isso a cada ciclo do loop principal.
        """
        if not regime or regime == self._last_regime:
            return

        prev = self._last_regime
        self._last_regime = regime

        if not prev:
            return   # primeira detecção — não é mudança

        # Determina se é upgrade ou downgrade
        BULLISH = {"TREND_EXPANSION", "VOLATILITY_COMPRESSION"}
        BEARISH = {"BEAR_TREND", "PANIC_LIQUIDATION"}

        prev_class = ("bull" if prev in BULLISH else
                      "bear" if prev in BEARISH else "neutral")
        new_class  = ("bull" if regime in BULLISH else
                      "bear" if regime in BEARISH else "neutral")

        if new_class == "bear":
            level   = AlertLevel.WARNING
            headline = "⛔ Regime BEAR — entradas bloqueadas"
        elif new_class == "bull" and prev_class in ("bear", "neutral"):
            level   = AlertLevel.INFO
            headline = "🚀 Regime BULLISH — entradas liberadas!"
        else:
            level   = AlertLevel.INFO
            headline = "🔄 Mudança de Regime"

        await self._ch.send(Alert(
            level=level,
            title=headline,
            message=f"`{prev.replace('_',' ')}` → `{regime.replace('_',' ')}`",
            fields={
                "Anterior": f"{self._regime_emoji(prev)} {prev.replace('_',' ')}",
                "Novo":     f"{self._regime_emoji(regime)} {regime.replace('_',' ')}",
                "Hora":     datetime.now(UTC).strftime("%H:%M UTC"),
            },
        ))

    # ── Alertas de Drawdown ────────────────────────────────────────────────────

    async def on_drawdown_check(
        self, daily_dd_pct: float, total_value: float
    ) -> None:
        """
        Monitora drawdown diário. Dispara warning em 2% e crítico em 3%.
        Reseta flags ao início do dia.
        """
        now = datetime.now(UTC)
        # Reset diário às 00:00 UTC
        if now.hour == 0 and now.minute < 16:
            self._last_dd_warning  = False
            self._last_dd_critical = False

        if daily_dd_pct >= 0.03 and not self._last_dd_critical:
            self._last_dd_critical = True
            await self._ch.send(Alert(
                level=AlertLevel.CRITICAL,
                title="🚨 Drawdown Crítico — 3% atingido",
                message="Circuit breaker de drawdown diário ativado. **Novas entradas bloqueadas.**",
                fields={
                    "Drawdown":  f"{daily_dd_pct*100:.2f}%",
                    "Portfolio": f"${total_value:,.0f}",
                    "Hora":      now.strftime("%H:%M UTC"),
                },
            ))

        elif daily_dd_pct >= 0.02 and not self._last_dd_warning:
            self._last_dd_warning = True
            await self._ch.send(Alert(
                level=AlertLevel.WARNING,
                title="⚠️ Drawdown Warning — 2% atingido",
                message="Drawdown diário em zona de atenção. Circuit breaker em 3%.",
                fields={
                    "Drawdown":  f"{daily_dd_pct*100:.2f}%",
                    "Portfolio": f"${total_value:,.0f}",
                    "Margem":    f"{(0.03 - daily_dd_pct)*100:.2f}% até circuit breaker",
                },
            ))

        # Reset flags quando DD se recupera
        if daily_dd_pct < 0.015:
            self._last_dd_warning  = False
            self._last_dd_critical = False

    # ── Alerta de Win Rate ────────────────────────────────────────────────────

    async def on_win_rate_check(
        self, win_rate: float, n_trades: int
    ) -> None:
        """
        Monitora win rate rolling. Dispara se cair abaixo de 30% com ≥5 trades.
        """
        if n_trades < 5:
            return

        if win_rate < 0.30 and not self._last_wr_warning:
            self._last_wr_warning = True
            await self._ch.send(Alert(
                level=AlertLevel.WARNING,
                title="⚠️ Win Rate Degradado",
                message=f"Win rate das últimas {n_trades} operações caiu abaixo de 30%. Modelo pode precisar de recalibração.",
                fields={
                    "Win Rate":  f"{win_rate*100:.1f}%",
                    "Trades":    str(n_trades),
                    "Esperado":  "37.5% (calibrado)",
                    "Ação":      "Monitorar + recalibrar se persistir",
                },
            ))
        elif win_rate >= 0.35:
            self._last_wr_warning = False

    # ── Reset diário ──────────────────────────────────────────────────────────

    def reset_daily_stats(self) -> None:
        """Reseta contadores diários (chamar em _maybe_send_daily_summary)."""
        self._trades_today = 0
        self._pnl_today    = 0.0
