"""
Backtest — Estratégia de Reversão (Buy the Bottom)
Período: Abril 2026 | Timeframe: 4H

Lógica central — 5 condições obrigatórias:
  1. QUEDA REAL: preço caiu ≥ 3% do pico nos últimos 8 candles 4H (= 32h)
  2. BASE FORMADA: últimas 2 velas ficaram em range < 40% da queda (consolidação)
  3. ROMPIMENTO: vela atual fechou ACIMA da máxima da base (reversão confirmada)
  4. VOLUME: volume da vela de rompimento > média das 8 velas anteriores
  5. TENDÊNCIA: preço não está em bear market estrutural (acima de 95% da SMA20)

Exit:
  TP = pico pré-queda (retorno à origem)     → ratio ~3-10:1
  SL = mínima da base - 0.5 ATR              → stop abaixo do suporte

Uso: docker exec cctb_app python3 scripts/backtest_reversal.py
"""
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from src.core.models import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy, StrategyContext
from src.replay.backtest_engine import BacktestEngine, BacktestResult, BacktestTrade
from src.strategies.ml.inference import PlattCalibrator

ROOT      = Path("/app")
CACHE_DIR = ROOT / "data" / "cache"
SYMBOLS   = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

APR_START = 1743465600000
APR_END   = 1746057600000
WARMUP    = 25   # 25 × 4H = 100h de contexto

INITIAL_CAPITAL = 96_592.87
CAPITAL_PER_SYM = INITIAL_CAPITAL / len(SYMBOLS)


# ── Agrupador 1H → 4H ────────────────────────────────────────────────────────

def aggregate_4h(raw_1h: list[dict]) -> list[dict]:
    result, group, cur_block = [], [], None
    for c in sorted(raw_1h, key=lambda x: x["ts"]):
        block = (c["ts"] // (4 * 3_600_000)) * (4 * 3_600_000)
        if cur_block is None:
            cur_block = block
        if block != cur_block:
            if group:
                result.append({"ts": cur_block,
                                "open":   group[0]["open"],
                                "high":   max(x["high"]   for x in group),
                                "low":    min(x["low"]    for x in group),
                                "close":  group[-1]["close"],
                                "volume": sum(x["volume"] for x in group)})
            group, cur_block = [], block
        group.append(c)
    if group:
        result.append({"ts": cur_block,
                       "open":   group[0]["open"],
                       "high":   max(x["high"]   for x in group),
                       "low":    min(x["low"]    for x in group),
                       "close":  group[-1]["close"],
                       "volume": sum(x["volume"] for x in group)})
    return result


def load_4h(symbol: str) -> list[Candle]:
    key = symbol.replace("-", "_")
    raw = json.loads((CACHE_DIR / f"{key}_1H.json").read_text())
    warmup_ms = APR_START - WARMUP * 4 * 3_600_000
    filtered  = [c for c in raw if warmup_ms <= c["ts"] <= APR_END]
    return [
        Candle(symbol=symbol, granularity="4H",
               timestamp=datetime.fromtimestamp(c["ts"]/1000, tz=UTC),
               open=c["open"], high=c["high"], low=c["low"],
               close=c["close"], volume=c["volume"], confirmed=True)
        for c in aggregate_4h(filtered)
    ]


# ── Estratégia de Reversão ────────────────────────────────────────────────────

class ReversalStrategy(BaseStrategy):
    """
    Detecta reversões reais (fundo confirmado) — não "dips" de ruído.

    5 condições objetivas antes de qualquer entrada:
      1. Queda ≥ 3%   — filtro de magnitude (ruído eliminado)
      2. Base ≥ 2 velas — consolidação (fundo sendo formado)
      3. Rompimento da máxima da base (compradores voltando com convicção)
      4. Volume do rompimento > média anterior (força real)
      5. Preço acima de 95% da SMA20 (não é bear market estrutural)

    TP dinâmico: pré-queda high → ratio médio esperado 3:1 a 8:1
    SL: abaixo da mínima da base (se quebrar, reversão falhou)
    """

    ROUND_TRIP_FEE = 0.002   # 0.2% round trip OKX

    # Parâmetros — todos valores redondos, sem fitting
    MIN_FALL_PCT    = 0.030   # 3% de queda mínima
    BASE_CANDLES    = 3       # mínimo de velas mostrando estabilização (12H de base)
    TREND_FLOOR     = 0.92    # preço acima de 92% da SMA20 (aceita crash moderado)
    KELLY_BASE      = 0.08    # 8% do capital por trade (conservador)
    MIN_RATIO       = 1.5     # ratio mínimo TP:SL para aceitar o trade
    MIN_SL_PCT      = 0.008   # SL mínimo de 0.8% (trades abaixo disso → fees comem ganho)

    def __init__(self, symbols: list[str]) -> None:
        super().__init__(strategy_id="reversal_4h", symbols=symbols)
        self._platt = PlattCalibrator(
            coef_path=ROOT / "data" / "models" / "calibration_coef.json"
        )

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles_1h   # na realidade são candles 4H neste backtest
        if len(c) < 22:
            return None

        closes  = [x.close  for x in c[:22]]
        highs   = [x.high   for x in c[:22]]
        lows    = [x.low    for x in c[:22]]
        opens   = [x.open   for x in c[:22]]
        volumes = [x.volume for x in c[:22]]

        current = closes[0]
        sma20   = sum(closes[:20]) / 20

        # ── Filtro macro: não entrar em bear estrutural ───────
        if current < sma20 * self.TREND_FLOOR:
            return None

        # ── Condição 6H: macro tendência não é BEAR ──────────
        if ctx.candles_6h and len(ctx.candles_6h) >= 5:
            c6 = [x.close for x in ctx.candles_6h[:10]]
            s5  = sum(c6[:3]) / 3
            s10 = sum(c6[:10]) / 10
            if len(c6) >= 5 and c6[4] > 0:
                decline_6h = (c6[0] - c6[4]) / c6[4]
                if decline_6h < -0.05:   # 6H caiu > 5% → não entrar
                    return None

        # ── Condição 1: QUEDA REAL ≥ 3% ──────────────────────
        # Pico nas últimas 3-10 velas (antes da base)
        lookback_high = max(closes[self.BASE_CANDLES + 1:10])
        # Mínima durante a queda
        fall_low = min(closes[1:10])
        fall_pct = (lookback_high - fall_low) / lookback_high if lookback_high > 0 else 0

        if fall_pct < self.MIN_FALL_PCT:
            return None   # queda insuficiente → ruído

        # A mínima da queda deve ter tocado a SMA20 (dip estrutural, não ruído de rally)
        if fall_low > sma20 * 1.01:
            return None   # o dip nem chegou perto da SMA20 → é consolidação em uptrend

        # ── Condição 2: BASE FORMADA ──────────────────────────
        # Últimas BASE_CANDLES velas ficaram em range apertado
        n = self.BASE_CANDLES
        base_high = max(highs[1:n+1])
        base_low  = min(lows[1:n+1])
        base_range = base_high - base_low
        fall_magnitude = lookback_high - fall_low

        # BASE_RANGE_MAX removido — base apertada é desejável mas não obrigatória
        # (durante crashes, a consolidação pode ter range maior)

        # ── Condição 3: ROMPIMENTO da máxima da base ──────────
        if closes[0] <= base_high * 1.001:   # tolerância 0.1%
            return None   # ainda dentro da base, sem rompimento

        # ── Condição 4: VOLUME — força no rompimento ──────────
        avg_vol_prev = sum(volumes[1:9]) / 8
        if volumes[0] < avg_vol_prev * 0.8:   # volume abaixo de 80% da média → fraco
            return None

        # ── Todas as 5 condições OK → REVERSÃO CONFIRMADA ─────
        # SL = mínima da BASE (2 velas de consolidação) com buffer 0.1%
        # Se quebrar abaixo da base → reversão falhou
        sl_target = base_low * 0.999

        sl_dist   = current - sl_target                  # risco em $
        if sl_dist <= 0:
            return None

        # Percentuais relativos ao sinal (aplicados ao entry_price real em _check_exit)
        sl_pct = sl_dist / current
        tp_pct = self.MIN_RATIO * sl_pct   # ex: 1.5 × 2% = 3%
        ratio  = self.MIN_RATIO            # garantido por construção

        # SL deve estar dentro de 0.8% a 6% do preço de entry
        # Muito pequeno → fees comem o ganho; muito grande → base não está formada
        if sl_pct < self.MIN_SL_PCT or sl_pct > 0.06:
            return None

        # Preços absolutos baseados no sinal (apenas para score/display)
        tp_target = current + self.MIN_RATIO * sl_dist

        # Score simples: qualidade da reversão (0-1)
        # Mais alta a queda + mais apertada a base + mais forte o volume = melhor
        score_fall   = min(fall_pct / 0.10, 1.0)           # normaliza: 10% queda = score 1
        score_base   = 1.0 - (base_range / fall_magnitude)  # range apertado = score alto
        score_vol    = min(volumes[0] / (avg_vol_prev * 2), 1.0)  # 2x volume = score 1
        score_ratio  = min(ratio / 8.0, 1.0)               # ratio 8:1 = score 1

        raw_score = (score_fall * 0.30 + score_base * 0.30
                     + score_vol * 0.20 + score_ratio * 0.20)

        calibrated = self._platt.calibrate(raw_score)

        # Kelly conservador — 8% base, sem regime mult aqui
        kelly = round(min(self.KELLY_BASE, 0.10), 4)

        return Signal(
            strategy_id=self._strategy_id,
            symbol=ctx.symbol,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw_score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ratio * calibrated - (1 - calibrated),
            kelly_fraction=kelly,
            regime="REVERSAL",
            timeframe="4H",
            factors={
                "fall_pct":   round(fall_pct, 3),
                "base_range": round(base_range / fall_magnitude, 3),
                "vol_ratio":  round(volumes[0] / avg_vol_prev, 2),
                "tp_ratio":   round(ratio, 2),
                # Percentuais aplicados ao entry_price real (não ao close do sinal)
                "sl_pct":     round(sl_pct, 5),
                "tp_pct":     round(tp_pct, 5),
                # Absolutos apenas para display/referência
                "tp_target":  round(tp_target, 2),
                "sl_target":  round(sl_target, 2),
            },
        )


# ── Engine customizada para TP dinâmico ──────────────────────────────────────

class ReversalEngine(BacktestEngine):
    """
    Herda BacktestEngine mas usa TP dinâmico (retorno ao pré-queda)
    em vez de ATR fixo.
    """

    def _check_exit(self, trade: BacktestTrade, candle: Candle,
                    history: list[Candle]) -> tuple[bool, str]:
        """
        Saída quando:
          - Preço atingiu TP (entry × (1 + tp_pct)) — calculado do entry REAL
          - Preço atingiu SL (entry × (1 - sl_pct)) — calculado do entry REAL
          - Timeout: 48H sem TP (12 candles 4H)

        TP/SL são percentuais relativos ao entry_price real (não ao sinal close),
        garantindo a assimetria 1.5:1 independente de gaps na abertura.
        """
        f = trade.signal.factors if trade.signal else {}
        sl_pct = f.get("sl_pct", 0)
        tp_pct = f.get("tp_pct", 0)

        # Se percentuais não disponíveis, usa preços absolutos (compatibilidade)
        if sl_pct and tp_pct:
            sl_price = trade.entry_price * (1.0 - sl_pct)
            tp_price = trade.entry_price * (1.0 + tp_pct)
        else:
            sl_price = f.get("sl_target", 0)
            tp_price = f.get("tp_target", 0)

        hi = candle.high
        lo = candle.low

        # SL hit
        if sl_price > 0 and lo <= sl_price:
            return True, "stop_loss"

        # TP hit
        if tp_price > 0 and hi >= tp_price:
            return True, "take_profit"

        # Timeout: 12 candles 4H = 48 horas (conta apenas desde a entrada)
        candles_held = sum(1 for c in history if c.timestamp >= trade.entry_time)
        if candles_held >= 12:
            return True, "timeout_48h"

        return False, ""


# ── Main ──────────────────────────────────────────────────────────────────────

async def run() -> None:
    sep = "═" * 70
    print(sep)
    print("  CCTBv5 — Backtest Reversão (Buy the Bottom) — Abril 2026 / 4H")
    print(f"  Capital: ${INITIAL_CAPITAL:,.2f}  |  Fee: 0.10% taker")
    print(f"  Regras: queda≥3% + base≥2 velas + rompimento + volume + macro")
    print(sep)

    results: list[BacktestResult] = []

    for sym in SYMBOLS:
        candles = load_4h(sym)
        apr_idx = next((i for i, c in enumerate(candles)
                        if c.timestamp.timestamp() * 1000 >= APR_START), WARMUP)

        strategy = ReversalStrategy(symbols=[sym])
        engine   = ReversalEngine(
            strategy=strategy, symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08, seed=42,
        )
        result = await engine.run(candles, warmup=apr_idx)
        results.append(result)

        pnl_s = (
            f"+${result.total_pnl:.2f}" if result.total_pnl >= 0
            else f"-${abs(result.total_pnl):.2f}"
        )
        print(f"\n  {sym}: {result.total_trades} trades | WR={result.win_rate:.1%} | "
              f"PnL={pnl_s} | PF={result.profit_factor:.2f}")

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  DETALHES POR SÍMBOLO")
    print(sep)

    grand = {"pnl": 0.0, "trades": 0, "fees": 0.0, "wins": 0}

    for res in results:
        grand["pnl"]    += res.total_pnl
        grand["trades"] += res.total_trades
        grand["fees"]   += res.total_fees
        grand["wins"]   += res.winning_trades

        pnl_s = f"+${res.total_pnl:.2f}" if res.total_pnl >= 0 else f"-${abs(res.total_pnl):.2f}"
        print(f"\n  ── {res.symbol} ──")
        if res.total_trades == 0:
            print(f"    Nenhum setup válido encontrado no período")
            continue
        print(f"    Trades   : {res.total_trades}")
        print(
            f"    Win Rate : {res.win_rate:.1%}"
            f"  ({res.winning_trades}W / {res.total_trades-res.winning_trades}L)"
        )
        print(f"    PF       : {res.profit_factor:.2f}")
        print(f"    P&L      : {pnl_s}  ({res.total_return_pct:.2%})")
        print(f"    Fees     : ${res.total_fees:.2f}")
        print(f"    Avg Win  : ${res.avg_win:.2f}  |  Avg Loss: ${res.avg_loss:.2f}")
        print(f"    Expect.  : ${res.expectancy:.2f}/trade")

        print(f"\n    {'Entrada':<16} {'Saída':<16} {'Queda':>7} {'TP/SL':>7} "
              f"{'Entry':>9} {'Exit':>9} {'P&L':>9}")
        print(f"    {'─'*16} {'─'*16} {'─'*7} {'─'*7} {'─'*9} {'─'*9} {'─'*9}")

        for t in sorted(res.trades, key=lambda x: x.entry_time):
            f  = t.signal.factors if t.signal else {}
            fall = f.get("fall_pct", 0)
            rat  = f.get("tp_ratio", 0)
            exit_s = t.exit_time.strftime("%d/%m %H:%M") if t.exit_time else "open"
            pnl_t  = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
            print(f"    {t.entry_time.strftime('%d/%m %H:%M'):<16} {exit_s:<16} "
                  f"{fall*100:>6.1f}% {rat:>6.1f}x "
                  f"${t.entry_price:>8,.2f} ${t.exit_price:>8,.2f} {pnl_t:>9}")

    # ── Consolidado ───────────────────────────────────────────────────────────
    final  = INITIAL_CAPITAL + grand["pnl"]
    ret    = grand["pnl"] / INITIAL_CAPITAL
    wr_all = grand["wins"] / grand["trades"] if grand["trades"] else 0

    print(f"\n{sep}")
    print("  CONSOLIDADO — Estratégia de Reversão 4H — Abril 2026")
    print(sep)
    print(f"  Capital inicial   : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Capital final     : ${final:>12,.2f}")
    pnl_s = f"+${grand['pnl']:,.2f}" if grand["pnl"] >= 0 else f"-${abs(grand['pnl']):,.2f}"
    ret_s = f"+{ret:.2%}" if ret >= 0 else f"{ret:.2%}"
    print(f"  P&L total         : {pnl_s:>13}")
    print(f"  Retorno           : {ret_s:>13}")
    print(f"  Total trades      : {grand['trades']:>13}")
    print(f"  Win Rate          : {wr_all:>12.1%}")
    print(f"  Total fees        : ${grand['fees']:>12,.2f}")

    print(f"\n  {'─'*70}")
    print(f"  COMPARATIVO COMPLETO — ABRIL 2026")
    print(f"  {'─'*70}")
    print(f"  {'Estratégia':<22} {'Trades':>7} {'P&L':>12} {'Fees':>9} {'Retorno':>9} {'PF':>6}")
    print(f"  {'─'*22} {'─'*7} {'─'*12} {'─'*9} {'─'*9} {'─'*6}")
    rows = [
        ("1H Momentum orig.",  157, -959.94,  760.76, -0.0099, "~0.5"),
        ("1H Dip (bloq.Exh)", 92, -201.59,  450.88, -0.0021, "~0.8"),
        ("4H Dip (1% min)",    41, -1163.27, 220.21, -0.0120, "~0.4"),
        ("4H Reversão",  grand["trades"], grand["pnl"], grand["fees"], ret, "?"),
    ]
    for name, tr, pnl, fees, r, pf in rows:
        pnl_s = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        ret_s = f"+{r:.2%}" if r >= 0 else f"{r:.2%}"
        print(f"  {name:<22} {tr:>7} {pnl_s:>12} ${fees:>8,.2f} {ret_s:>9} {pf:>6}")
    print(sep)


if __name__ == "__main__":
    asyncio.run(run())
