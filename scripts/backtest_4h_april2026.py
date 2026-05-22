"""
Backtest 4H — Abril 2026
Valida a estratégia buy-the-dip em candles de 4 horas.

Candles 4H são derivados dos candles 1H do cache (agrupamento OHLCV).
Uso: docker exec cctb_app python3 scripts/backtest_4h_april2026.py
"""
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from src.core.models import Candle
from src.replay.backtest_engine import BacktestEngine, BacktestResult

ROOT      = Path("/app")
CACHE_DIR = ROOT / "data" / "cache"

SYMBOLS   = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
APR_START = 1743465600000  # 2026-04-01 00:00 UTC
APR_END   = 1746057600000  # 2026-04-30 23:59 UTC
WARMUP_4H = 20             # 20 × 4H = 80h de contexto

INITIAL_CAPITAL = 96_592.87
CAPITAL_PER_SYM = INITIAL_CAPITAL / len(SYMBOLS)


def aggregate_4h(candles_1h: list[dict]) -> list[dict]:
    """Agrupa candles 1H em 4H (OHLCV correto)."""
    result = []
    # Alinha para múltiplos de 4H (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
    group: list[dict] = []
    current_4h_ts: int | None = None

    for c in sorted(candles_1h, key=lambda x: x["ts"]):
        # Timestamp do início do bloco 4H
        ts_sec   = c["ts"] // 1000
        block_ts = (ts_sec // (4 * 3600)) * (4 * 3600) * 1000

        if current_4h_ts is None:
            current_4h_ts = block_ts

        if block_ts != current_4h_ts:
            if group:
                result.append(_merge(group, current_4h_ts))
            group = []
            current_4h_ts = block_ts

        group.append(c)

    if group:
        result.append(_merge(group, current_4h_ts))

    return result


def _merge(group: list[dict], ts: int) -> dict:
    return {
        "ts":     ts,
        "open":   group[0]["open"],
        "high":   max(c["high"]   for c in group),
        "low":    min(c["low"]    for c in group),
        "close":  group[-1]["close"],
        "volume": sum(c["volume"] for c in group),
    }


def load_candles_4h(symbol: str) -> list[Candle]:
    key  = symbol.replace("-", "_")
    path = CACHE_DIR / f"{key}_1H.json"
    raw_1h = json.loads(path.read_text())

    # Inclui warmup antes de abril
    warmup_start = APR_START - WARMUP_4H * 4 * 3_600_000
    filtered_1h  = [c for c in raw_1h if warmup_start <= c["ts"] <= APR_END]

    raw_4h = aggregate_4h(filtered_1h)

    return [
        Candle(
            symbol=symbol,
            granularity="4H",
            timestamp=datetime.fromtimestamp(c["ts"] / 1000, tz=UTC),
            open=c["open"], high=c["high"],
            low=c["low"],   close=c["close"],
            volume=c["volume"],
            confirmed=True,
        )
        for c in raw_4h
    ]


async def run_backtest() -> None:
    sep = "═" * 68
    print(sep)
    print("  CCTBv5 — Backtest Abril 2026 (4H)")
    print(f"  Capital inicial : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Período         : 01/04/2026 → 30/04/2026")
    print(f"  Timeframe       : 4H (derivado de 1H)")
    print(f"  Fee OKX demo    : 0.10% (taker)")
    print(sep)

    # ── Importa a estratégia MODIFICADA para 4H ───────────────────────────────
    # Aplica as 4 mudanças diretamente para o backtest:
    #   1. Bloqueia TREND_EXHAUSTION + HIGH_CORRELATION_RISK
    #   2. Thresholds mais altos
    #   3. _direction = dip detector (mínimo 1% em 4H)
    #   4. SL/TP ratio 1:4
    from src.strategies.momentum.momentum_strategy import MomentumStrategy
    from src.strategies.base import StrategyContext, BaseStrategy
    from src.core.models import Signal, SignalDirection
    from src.strategies.momentum.momentum_strategy import (
        REGIME_THRESHOLDS, REGIME_KELLY_MULT, REGIME_M4,
        REGIME_MIN_EV_MULT,
    )
    from src.strategies.ml.inference import PlattCalibrator

    MODELS_DIR = ROOT / "data" / "models"

    # Thresholds 4H — mais conservadores que 1H (menos ruído, exige mais convicção)
    THRESH_4H = {
        "TREND_EXPANSION":        0.50,
        "VOLATILITY_COMPRESSION": 0.52,
        "MEAN_REVERTING_CHOP":    0.60,
        "TREND_EXHAUSTION":       0.99,   # bloqueado
        "HIGH_CORRELATION_RISK":  0.99,   # bloqueado
        "BEAR_TREND":             0.99,
        "PANIC_LIQUIDATION":      0.99,
    }
    BLOCKED_4H = {"BEAR_TREND", "PANIC_LIQUIDATION",
                  "TREND_EXHAUSTION", "HIGH_CORRELATION_RISK"}

    class DipBuyStrategy(BaseStrategy):
        """
        Estratégia buy-the-dip em 4H.
        Princípio: comprar fraqueza dentro de força — nunca comprar força.

        Entrada quando:
          1. Regime válido (EXPANSION ou COMPRESSION)
          2. Score ≥ threshold
          3. DIP real ≥ 1% das últimas 3 velas 4H + recuperação
        """

        ROUND_TRIP_FEE = 0.002   # 0.2% round trip OKX

        def __init__(self, symbols: list[str]) -> None:
            super().__init__(strategy_id="dip_buy_4h", symbols=symbols)
            self._platt = PlattCalibrator(
                coef_path=MODELS_DIR / "calibration_coef.json"
            )

        async def evaluate(self, ctx: StrategyContext) -> Signal | None:
            from datetime import UTC, datetime
            from src.monitoring.signal_log import SignalAuditEntry, signal_audit_log

            ts     = datetime.now(UTC)
            symbol = ctx.symbol

            if len(ctx.candles_1h) < 20:
                return None

            # ── Regime ───────────────────────────────────────────
            closes  = [c.close  for c in ctx.candles_1h[:20]]
            volumes = [c.volume for c in ctx.candles_1h[:20]]
            sma5    = sum(closes[:5])  / 5
            sma20   = sum(closes[:20]) / 20
            avg_vol = sum(volumes) / len(volumes)
            last_vol = volumes[0]

            if len(closes) >= 2 and closes[1] > 0:
                if (closes[0] - closes[1]) / closes[1] < -0.04:
                    regime_1h = "PANIC_LIQUIDATION"
                elif sma5 > sma20:
                    regime_1h = ("TREND_EXPANSION" if last_vol > avg_vol * 1.2
                                 else "VOLATILITY_COMPRESSION")
                elif len(closes) >= 11 and closes[10] > 0:
                    regime_1h = ("BEAR_TREND"
                                 if (closes[0] - closes[10]) / closes[10] < -0.02
                                 else "MEAN_REVERTING_CHOP")
                else:
                    regime_1h = "MEAN_REVERTING_CHOP"
            else:
                regime_1h = "MEAN_REVERTING_CHOP"

            # Confirmação 6H
            regime = regime_1h
            if ctx.candles_6h and len(ctx.candles_6h) >= 5:
                c6h = [c.close for c in ctx.candles_6h[:10]]
                s5  = sum(c6h[:3]) / 3
                s10 = sum(c6h[:10]) / 10
                trend_6h = ("BULL" if s5 > s10
                            else "BEAR" if len(c6h) >= 5 and c6h[4] > 0
                            and (c6h[0] - c6h[4]) / c6h[4] < -0.03
                            else "CHOP")
                fam = ("BULL" if regime_1h in {"TREND_EXPANSION","VOLATILITY_COMPRESSION","TREND_EXHAUSTION"}
                       else "BEAR" if regime_1h in {"BEAR_TREND","PANIC_LIQUIDATION"}
                       else "CHOP")
                if fam == "BEAR" and trend_6h == "BEAR":
                    regime = "BEAR_TREND"
                elif fam == "BULL" and trend_6h == "BEAR":
                    regime = "MEAN_REVERTING_CHOP"
                elif fam == "CHOP" and trend_6h == "BEAR":
                    regime = "HIGH_CORRELATION_RISK"
                elif fam == "CHOP" and trend_6h == "BULL":
                    regime = "VOLATILITY_COMPRESSION"

            if regime in BLOCKED_4H:
                return None

            threshold = THRESH_4H.get(regime, 0.99)

            # ── Score (mesmos M1-M5) ──────────────────────────────
            highs = [c.high   for c in ctx.candles_1h[:10]]
            lows  = [c.low    for c in ctx.candles_1h[:10]]
            opens = [c.open   for c in ctx.candles_1h[:10]]
            vols  = [c.volume for c in ctx.candles_1h[:20]]

            atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
            norm = max(atr*2, closes[0]*0.005)

            r1  = (closes[0]-closes[1])/closes[1]  if closes[1]>0 else 0
            r5  = (closes[0]-closes[5])/closes[5]  if len(closes)>5 and closes[5]>0 else 0
            r10 = (closes[0]-closes[10])/closes[10] if len(closes)>10 and closes[10]>0 else 0
            r20 = (closes[0]-closes[20])/closes[20] if len(closes)>20 and closes[20]>0 else 0
            mw  = r1*0.30 + r5*0.30 + r10*0.25 + r20*0.15
            m1  = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

            n    = min(6, len(closes)-1)
            bull = sum(1 for i in range(n) if closes[i]>opens[i])/n if n>0 else 0.5
            hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
            hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
            m2   = bull*0.5 + (hh+hl)/2*0.5

            avg5  = sum(vols[:5])/5   if len(vols)>=5  else vols[0] if vols else 1
            avg20 = sum(vols[:20])/20 if len(vols)>=20 else avg5
            vr    = min(vols[0]/avg5,3.0)/3.0 if avg5>0 else 0.5
            vt    = min(max(sum(vols[:3])/sum(vols[3:6]),0.3),2.0) if len(vols)>=6 and sum(vols[3:6])>0 else 1.0
            cc    = 1.0 if closes[0]>opens[0] and vols[0]>avg20 else 0.4
            m3    = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

            sma5v  = sum(closes[:5])/5
            sma20v = sum(closes[:20])/20 if len(closes)>=20 else sma5v
            dist   = (sma5v-sma20v)/sma20v if sma20v>0 else 0
            m4r    = min(max((dist+0.02)/0.04, 0.0), 1.0)
            m4     = m4r*0.6 + REGIME_M4.get(regime, 0.45)*0.4

            cs = [(closes[i]-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0.5
                  for i in range(min(3,len(closes)))]
            m5 = sum(cs)/len(cs) if cs else 0.5

            score = round(min(max(m1*0.25+m2*0.25+m3*0.20+m4*0.20+m5*0.10, 0.0), 1.0), 4)

            if score < threshold:
                return None

            # ── EV ───────────────────────────────────────────────
            calibrated = self._platt.calibrate(score)
            tp_r = {"TREND_EXPANSION": 4.0, "VOLATILITY_COMPRESSION": 4.0,
                    "MEAN_REVERTING_CHOP": 4.0}.get(regime, 4.0)
            ev = calibrated * tp_r - (1 - calibrated) * 1.0
            min_ev = REGIME_MIN_EV_MULT.get(regime, 0.5) * self.ROUND_TRIP_FEE
            if ev < min_ev:
                return None

            # ── DIP DETECTOR 4H ─────────────────────────────────
            # Condição A: dip real ≥ 1% das últimas 3 velas 4H
            # (cada "candle" no contexto 4H = 4 candles 1H)
            # Usando os últimos 12 candles 1H = últimas 3 velas 4H
            recent_high = max(closes[1:13]) if len(closes) >= 13 else max(closes[1:])
            recent_low  = min(closes[1:13]) if len(closes) >= 13 else min(closes[1:])
            dip_pct = (recent_high - recent_low) / recent_high if recent_high > 0 else 0

            if dip_pct < 0.010:   # mínimo 1% de dip no período 4H
                return None

            # Condição B: preço atual acima da SMA20 (uptrend intacto)
            if closes[0] < sma20 * 0.998:
                return None

            # Condição C: vela atual de recuperação (close > open)
            if closes[0] <= opens[0]:
                return None

            kelly_mult = REGIME_KELLY_MULT.get(regime, 0.5)
            base_kelly = min(calibrated * 0.25, 0.10)   # cap 10% em 4H
            kelly = round(base_kelly * kelly_mult, 4)

            return Signal(
                strategy_id=self._strategy_id,
                symbol=symbol,
                direction=SignalDirection.LONG,
                timestamp=ts,
                score=score,
                calibrated_score=calibrated,
                confidence=calibrated,
                expected_value=ev,
                kelly_fraction=kelly,
                regime=regime,
                timeframe="4H",
                factors={"m1": round(m1,3), "m2": round(m2,3),
                         "m3": round(m3,3), "m4": round(m4,3), "m5": round(m5,3)},
            )

    # ── Roda backtest por símbolo ─────────────────────────────────────────────
    results: list[BacktestResult] = []

    for sym in SYMBOLS:
        candles = load_candles_4h(sym)
        apr_idx = next(
            (i for i, c in enumerate(candles)
             if c.timestamp.timestamp() * 1000 >= APR_START),
            WARMUP_4H,
        )
        print(f"\n  {sym}: {len(candles)} candles 4H | abril começa idx={apr_idx} "
              f"({candles[apr_idx].timestamp:%d/%m %H:%M})")

        strategy = DipBuyStrategy(symbols=[sym])
        engine   = BacktestEngine(
            strategy=strategy,
            symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08,
            seed=42,
            fee_tier="standard",
        )
        result = await engine.run(candles, warmup=apr_idx)
        results.append(result)

        pnl_s = f"+${result.total_pnl:.2f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):.2f}"
        print(f"  → {result.total_trades} trades | WR={result.win_rate:.1%} | "
              f"PnL={pnl_s} | PF={result.profit_factor:.2f}")

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTADOS DETALHADOS POR SÍMBOLO (4H)")
    print(sep)

    grand_pnl = grand_trades = grand_fees = grand_wins = 0

    for res in results:
        grand_pnl    += res.total_pnl
        grand_trades += res.total_trades
        grand_fees   += res.total_fees
        grand_wins   += res.winning_trades

        pnl_s = f"+${res.total_pnl:.2f}" if res.total_pnl >= 0 else f"-${abs(res.total_pnl):.2f}"
        print(f"\n  ── {res.symbol} ──")
        print(f"    Trades      : {res.total_trades}")
        if res.total_trades > 0:
            print(f"    Win Rate    : {res.win_rate:.1%}  ({res.winning_trades}W / {res.total_trades - res.winning_trades}L)")
            print(f"    Profit Fac. : {res.profit_factor:.2f}")
            print(f"    P&L         : {pnl_s}  ({res.total_return_pct:.2%})")
            print(f"    Fees totais : ${res.total_fees:.2f}")
            print(f"    Expectancy  : ${res.expectancy:.2f}/trade")
            print(f"    Avg Win     : ${res.avg_win:.2f}")
            print(f"    Avg Loss    : ${res.avg_loss:.2f}")
            print(f"\n    {'Entrada':<16} {'Saída':<16} {'Regime':<26} {'Entry':>9} {'Exit':>9} {'P&L':>9}")
            print(f"    {'─'*16} {'─'*16} {'─'*26} {'─'*9} {'─'*9} {'─'*9}")
            for t in sorted(res.trades, key=lambda x: x.entry_time):
                pnl_t = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
                exit_s = t.exit_time.strftime("%d/%m %H:%M") if t.exit_time else "timeout"
                print(f"    {t.entry_time.strftime('%d/%m %H:%M'):<16} {exit_s:<16} "
                      f"{(t.signal.regime if t.signal else '–'):<26} "
                      f"${t.entry_price:>8,.2f} ${t.exit_price:>8,.2f} {pnl_t:>9}")

    final_capital = INITIAL_CAPITAL + grand_pnl
    total_return  = grand_pnl / INITIAL_CAPITAL
    overall_wr    = grand_wins / grand_trades if grand_trades else 0

    print(f"\n{sep}")
    print("  CONSOLIDADO — ABRIL 2026 (4H)")
    print(sep)
    print(f"  Capital inicial   : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Capital final     : ${final_capital:>12,.2f}")
    pnl_str = f"+${grand_pnl:,.2f}" if grand_pnl >= 0 else f"-${abs(grand_pnl):,.2f}"
    ret_str = f"+{total_return:.2%}" if total_return >= 0 else f"{total_return:.2%}"
    print(f"  P&L total         : {pnl_str:>13}")
    print(f"  Retorno total     : {ret_str:>13}")
    print(f"  Total de trades   : {grand_trades:>13}")
    print(f"  Win Rate global   : {overall_wr:>12.1%}")
    print(f"  Total fees        : ${grand_fees:>12,.2f}")
    print(f"\n  Comparativo:")
    print(f"  {'Timeframe':<12} {'Trades':>8} {'P&L':>12} {'Fees':>10} {'Retorno':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*10} {'─'*10}")
    print(f"  {'1H original':<12} {'157':>8} {'-$959.94':>12} {'$760.76':>10} {'-0.99%':>10}")
    print(f"  {'1H nova':12} {'92':>8} {'-$201.59':>12} {'$450.88':>10} {'-0.21%':>10}")
    pnl_4h = f"+${grand_pnl:,.2f}" if grand_pnl >= 0 else f"-${abs(grand_pnl):,.2f}"
    fees_4h = f"${grand_fees:,.2f}"
    ret_4h  = f"+{total_return:.2%}" if total_return >= 0 else f"{total_return:.2%}"
    print(f"  {'4H buy-dip':<12} {grand_trades:>8} {pnl_4h:>12} {fees_4h:>10} {ret_4h:>10}")
    print(sep)


if __name__ == "__main__":
    asyncio.run(run_backtest())
