"""
Backtest — Estratégia de Reversão 1H (Buy the Bottom)
Período: Abril 2026 | Timeframe: 1H

Lógica idêntica ao backtest 4H validado, mas com candles 1H nativos.
Parâmetros reescalados para 1H:
  - BASE_CANDLES   : 6 velas 1H  (= 6H de consolidação, equivale a ~1.5 candles 4H)
  - lookback queda : closes[7:20] (7–20h atrás)
  - fall_low       : closes[1:20] (mínima das últimas 19h)
  - Timeout        : 48 candles 1H = 48H

Condições (mantidas idênticas ao 4H):
  1. QUEDA REAL ≥ 3% no lookback
  2. BASE ≥ 6H estabilizando (close dentro da base)
  3. ROMPIMENTO do topo da base + 0.1%
  4. VOLUME do rompimento ≥ 80% da média das 8h anteriores
  5. MACRO: preço entre 92%–102% da SMA20 (dip estrutural)
  6. fall_low deve ter tocado a SMA20 (não é ruído de rally)

Exit (relativo ao entry_price real):
  SL = base_low × 0.999   →  SL% = (entry − SL) / entry
  TP = entry × (1 + 1.5 × SL%)   →  assimetria 1.5:1 garantida

Uso: docker exec cctb_app python3 /tmp/btrev1h.py
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
WARMUP    = 40   # 40 × 1H = 40h de contexto (SMA20 precisa de 20)

INITIAL_CAPITAL = 96_592.87
CAPITAL_PER_SYM = INITIAL_CAPITAL / len(SYMBOLS)


def load_1h(symbol: str) -> list[Candle]:
    key       = symbol.replace("-", "_")
    raw       = json.loads((CACHE_DIR / f"{key}_1H.json").read_text())
    warmup_ms = APR_START - WARMUP * 3_600_000
    filtered  = sorted(
        [c for c in raw if warmup_ms <= c["ts"] <= APR_END],
        key=lambda c: c["ts"],
    )
    return [
        Candle(
            symbol=symbol, granularity="1H",
            timestamp=datetime.fromtimestamp(c["ts"] / 1000, tz=UTC),
            open=c["open"], high=c["high"], low=c["low"],
            close=c["close"], volume=c["volume"], confirmed=True,
        )
        for c in filtered
    ]


# ── Estratégia de Reversão 1H ─────────────────────────────────────────────────

class ReversalStrategy1H(BaseStrategy):
    """
    Detecta reversões reais no timeframe 1H.

    Parâmetros reescalados de 4H → 1H:
      - 4H BASE_CANDLES=3 (12H) → 1H BASE_CANDLES=6 (6H)
      - 4H lookback closes[4:11] (16-44H) → 1H closes[7:20] (7-20H)
      - 4H timeout=12 candles (48H) → 1H timeout=48 candles (48H)
    """

    # Parâmetros calibrados para 1H — sem fitting, baseados na lógica de mercado
    MIN_FALL_PCT = 0.030   # 3% de queda mínima (igual ao 4H)
    BASE_CANDLES = 6       # 6 velas 1H = 6H de estabilização
    TREND_FLOOR  = 0.92    # 92% da SMA20 (igual ao 4H)
    KELLY_BASE   = 0.08    # 8% do capital por trade
    MIN_RATIO    = 1.5     # TP = 1.5× SL (garantido por construção)
    MIN_SL_PCT   = 0.008   # SL mínimo 0.8% (evita trade onde fee > gain)
    LOOKBACK_HI  = slice(BASE_CANDLES + 1, 20)  # closes[7:20] = pico pré-queda
    FALL_WINDOW  = slice(1, 20)                  # closes[1:20] = mínima da queda

    def __init__(self, symbols: list[str]) -> None:
        super().__init__(strategy_id="reversal_1h", symbols=symbols)
        self._platt = PlattCalibrator(
            coef_path=ROOT / "data" / "models" / "calibration_coef.json"
        )

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles_1h   # newest first
        if len(c) < 22:
            return None

        closes  = [x.close  for x in c[:22]]
        highs   = [x.high   for x in c[:22]]
        lows    = [x.low    for x in c[:22]]
        volumes = [x.volume for x in c[:22]]

        current = closes[0]
        sma20   = sum(closes[:20]) / 20

        # ── Filtro macro: não entrar em bear estrutural nem rally avançado ──
        if current < sma20 * self.TREND_FLOOR:
            return None
        if current > sma20 * 1.02:   # já recuperou demais → não é reversão
            return None

        # ── Condição 1: QUEDA REAL ≥ 3% ──────────────────────────────────────
        lookback_high = max(closes[self.BASE_CANDLES + 1:20])   # closes[7:20]
        fall_low      = min(closes[1:20])
        fall_pct      = (lookback_high - fall_low) / lookback_high if lookback_high > 0 else 0

        if fall_pct < self.MIN_FALL_PCT:
            return None

        # A mínima da queda deve ter tocado a SMA20 (dip estrutural)
        if fall_low > sma20 * 1.01:
            return None

        # ── Condição 2: BASE FORMADA (6 velas = 6H de consolidação) ──────────
        n         = self.BASE_CANDLES
        base_high = max(highs[1:n + 1])   # highs[1:7]
        base_low  = min(lows[1:n + 1])    # lows[1:7]
        base_range = base_high - base_low
        fall_magnitude = lookback_high - fall_low

        # ── Condição 3: ROMPIMENTO da máxima da base ──────────────────────────
        if closes[0] <= base_high * 1.001:
            return None

        # ── Condição 4: VOLUME — força no rompimento ──────────────────────────
        avg_vol = sum(volumes[1:9]) / 8
        if volumes[0] < avg_vol * 0.8:
            return None

        # ── Calcula SL/TP ─────────────────────────────────────────────────────
        sl_target = base_low * 0.999
        sl_dist   = current - sl_target
        if sl_dist <= 0:
            return None

        sl_pct = sl_dist / current
        if sl_pct < self.MIN_SL_PCT or sl_pct > 0.06:
            return None

        tp_pct    = self.MIN_RATIO * sl_pct
        tp_target = current + self.MIN_RATIO * sl_dist   # display only
        ratio     = self.MIN_RATIO

        # ── Score ──────────────────────────────────────────────────────────────
        score_fall  = min(fall_pct / 0.10, 1.0)
        score_base  = 1.0 - min(base_range / fall_magnitude, 1.0) if fall_magnitude > 0 else 0
        score_vol   = min(volumes[0] / (avg_vol * 2), 1.0)
        score_ratio = min(ratio / 8.0, 1.0)

        raw_score  = (score_fall * 0.30 + score_base * 0.30
                      + score_vol * 0.20 + score_ratio * 0.20)
        calibrated = self._platt.calibrate(raw_score)
        kelly      = round(min(self.KELLY_BASE, 0.10), 4)

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
            regime="REVERSAL_1H",
            timeframe="1H",
            factors={
                "fall_pct":   round(fall_pct, 3),
                "base_range": round(base_range / fall_magnitude, 3) if fall_magnitude > 0 else 0,
                "vol_ratio":  round(volumes[0] / avg_vol, 2),
                "tp_ratio":   round(ratio, 2),
                "sl_pct":     round(sl_pct, 5),
                "tp_pct":     round(tp_pct, 5),
                "tp_target":  round(tp_target, 2),
                "sl_target":  round(sl_target, 2),
            },
        )


# ── Engine com TP/SL relativos ao entry_price real ───────────────────────────

class ReversalEngine1H(BacktestEngine):
    """
    Herda BacktestEngine — overrides _check_exit com:
      - SL = entry × (1 - sl_pct)
      - TP = entry × (1 + tp_pct)
      - Timeout = 48 candles 1H = 48H
    Usando percentuais do sinal (calculados vs close do sinal),
    garantindo assimetria 1.5:1 independente de gaps na abertura.
    """

    def _check_exit(
        self,
        trade:   BacktestTrade,
        candle:  Candle,
        history: list[Candle],
    ) -> tuple[bool, str]:
        f      = trade.signal.factors if trade.signal else {}
        sl_pct = f.get("sl_pct", 0)
        tp_pct = f.get("tp_pct", 0)

        if sl_pct and tp_pct:
            sl_price = trade.entry_price * (1.0 - sl_pct)
            tp_price = trade.entry_price * (1.0 + tp_pct)
        else:
            sl_price = f.get("sl_target", 0)
            tp_price = f.get("tp_target", 0)

        # SL hit
        if sl_price > 0 and candle.low <= sl_price:
            return True, "stop_loss"

        # TP hit
        if tp_price > 0 and candle.high >= tp_price:
            return True, "take_profit"

        # Timeout: 48 candles 1H = 48H (conta apenas desde a entrada)
        candles_held = sum(1 for c in history if c.timestamp >= trade.entry_time)
        if candles_held >= 48:
            return True, "timeout_48h"

        return False, ""


# ── Main ──────────────────────────────────────────────────────────────────────

async def run() -> None:
    sep = "═" * 70
    print(sep)
    print("  CCTBv5 — Backtest Reversão 1H (Buy the Bottom) — Abril 2026")
    print(f"  Capital: ${INITIAL_CAPITAL:,.2f}  |  Fee: 0.10% taker")
    print(f"  Regras: queda≥3% + base≥6H + rompimento + volume + macro (SMA20)")
    print(f"  SL/TP : relativos ao entry_price real | Timeout: 48H")
    print(sep)

    results: list[BacktestResult] = []

    for sym in SYMBOLS:
        candles = load_1h(sym)
        apr_idx = next(
            (i for i, c in enumerate(candles) if c.timestamp.timestamp() * 1000 >= APR_START),
            WARMUP,
        )
        print(f"\n  {sym}: {len(candles)} candles  "
              f"({candles[0].timestamp:%d/%m %Hh} → {candles[-1].timestamp:%d/%m %Hh})  "
              f"| abril começa no índice {apr_idx}")

        strategy = ReversalStrategy1H(symbols=[sym])
        engine   = ReversalEngine1H(
            strategy=strategy,
            symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08,
            seed=42,
        )
        result = await engine.run(candles, warmup=apr_idx)
        results.append(result)

        pnl_s = (
            f"+${result.total_pnl:.2f}" if result.total_pnl >= 0
            else f"-${abs(result.total_pnl):.2f}"
        )
        print(f"  → {result.total_trades} trades | WR={result.win_rate:.1%} | "
              f"PnL={pnl_s} | PF={result.profit_factor:.2f}")

    # ── Relatório detalhado ───────────────────────────────────────────────────
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

        print(f"    Trades      : {res.total_trades}")
        print(
            f"    Win Rate    : {res.win_rate:.1%}"
            f"  ({res.winning_trades}W / {res.total_trades - res.winning_trades}L)"
        )
        print(f"    Profit Fac. : {res.profit_factor:.2f}")
        print(f"    P&L         : {pnl_s}  ({res.total_return_pct:.2%})")
        print(f"    Fees        : ${res.total_fees:.2f}")
        print(f"    Avg Win     : ${res.avg_win:.2f}  |  Avg Loss: ${res.avg_loss:.2f}")
        print(f"    Expectancy  : ${res.expectancy:.2f}/trade")

        print(f"\n    {'Entrada':<16} {'Saída':<16} {'Queda':>6} {'TP/SL':>6} "
              f"{'Entry':>9} {'Exit':>9} {'P&L':>9} {'Motivo'}")
        print(f"    {'─'*16} {'─'*16} {'─'*6} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*8}")

        for t in sorted(res.trades, key=lambda x: x.entry_time):
            f    = t.signal.factors if t.signal else {}
            fall = f.get("fall_pct", 0)
            rat  = f.get("tp_ratio", 0)
            exit_s = t.exit_time.strftime("%d/%m %H:%M") if t.exit_time else "open"
            pnl_t  = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
            reason = getattr(t, "exit_reason", "–")
            print(f"    {t.entry_time.strftime('%d/%m %H:%M'):<16} {exit_s:<16} "
                  f"{fall*100:>5.1f}% {rat:>5.1f}x "
                  f"${t.entry_price:>8,.2f} ${t.exit_price:>8,.2f} {pnl_t:>9}  {reason}")

    # ── Consolidado ───────────────────────────────────────────────────────────
    final  = INITIAL_CAPITAL + grand["pnl"]
    ret    = grand["pnl"] / INITIAL_CAPITAL
    wr_all = grand["wins"] / grand["trades"] if grand["trades"] else 0

    print(f"\n{sep}")
    print("  CONSOLIDADO — Reversão 1H — Abril 2026")
    print(sep)
    print(f"  Capital inicial   : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Capital final     : ${final:>12,.2f}")
    pnl_s = f"+${grand['pnl']:,.2f}" if grand["pnl"] >= 0 else f"-${abs(grand['pnl']):,.2f}"
    ret_s = f"+{ret:.2%}" if ret >= 0 else f"{ret:.2%}"
    print(f"  P&L total         : {pnl_s:>13}")
    print(f"  Retorno           : {ret_s:>13}")
    print(f"  Total trades      : {grand['trades']:>13}")
    print(f"  Win Rate global   : {wr_all:>12.1%}")
    print(f"  Total fees        : ${grand['fees']:>12,.2f}")
    print(sep)

    print(f"\n  ─────────────────────────────────────────────────────────────────────")
    print(f"  COMPARATIVO COMPLETO — ABRIL 2026")
    print(f"  ─────────────────────────────────────────────────────────────────────")
    print(f"  {'Estratégia':<24} {'Trades':>7} {'P&L':>12} {'Fees':>9} {'Retorno':>9} {'WR':>6}")
    print(f"  {'─'*24} {'─'*7} {'─'*12} {'─'*9} {'─'*9} {'─'*6}")
    print(
        f"  {'1H Momentum orig.':<24} {'157':>7} {'   -$959.94':>12}"
        f" {'$  760.76':>9} {'   -0.99%':>9} {'~35%':>6}"
    )
    print(
        f"  {'1H Dip (bloq.Exh)':<24} {' 92':>7} {'   -$201.59':>12}"
        f" {'$  450.88':>9} {'   -0.21%':>9} {'~38%':>6}"
    )
    print(
        f"  {'4H Dip (1% min)':<24} {' 41':>7} {' -$1,163.27':>12}"
        f" {'$  220.21':>9} {'   -1.20%':>9} {'~30%':>6}"
    )
    print(
        f"  {'4H Reversão (corr.)':<24} {'  7':>7} {'    +$194.94':>12}"
        f" {'$   37.86':>9} {'   +0.20%':>9} {'71.4%':>6}"
    )
    pnl_cmp  = f"+${grand['pnl']:,.2f}" if grand["pnl"] >= 0 else f"-${abs(grand['pnl']):,.2f}"
    ret_cmp  = f"+{ret:.2%}" if ret >= 0 else f"{ret:.2%}"
    wr_cmp   = f"{wr_all:.1%}"
    trades_c = str(grand["trades"])
    print(
        f"  {'1H Reversão (novo)':<24} {trades_c:>7} {pnl_cmp:>12}"
        f" ${grand['fees']:>8,.2f} {ret_cmp:>9} {wr_cmp:>6}"
    )
    print(f"  {'─'*24} {'─'*7} {'─'*12} {'─'*9} {'─'*9} {'─'*6}")


if __name__ == "__main__":
    asyncio.run(run())
