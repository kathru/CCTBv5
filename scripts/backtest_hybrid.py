"""
Backtest Híbrido — HybridStrategy1H
Jan 2025 → Abr 2026 (16 meses)

Combina Reversal + Momentum com 3 filtros extras:
  1. MACRO FILTER: BTC > SMA100H × 0.85 → não é bear estrutural
  2. BTC ANCHOR: ETH/SOL só entram se BTC também em dip (≥2%)
  3. SL MÍNIMO: 1.5% (elimina micro-trades comidos por fees)

Regras por modo:
  REVERSAL (dip ≥3% detectado):
    BTC  : fall ≥ 3% + 6 condições + SL 1.5–6%
    ETH  : fall ≥ 4% + BTC em dip ≥2% + SL 1.5–6%
    SOL  : fall ≥ 5% + BTC em dip ≥2% + SL 1.5–6%
    TP = 1.5× SL | Timeout 48H

  MOMENTUM (sem dip, mercado em alta):
    BTC/ETH: 3 velas bullish + vol ↑ + above SMA20
    SOL: bloqueada (muito volátil para momentum)
    TP = 2× SL | SL = SMA20 × 0.998 | Timeout 12H

Uso: docker exec cctb_app python3 /tmp/bthybrid.py
"""
import asyncio
import json
import sys
from calendar import monthrange
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

INITIAL_CAPITAL = 96_592.87
CAPITAL_PER_SYM = INITIAL_CAPITAL / len(SYMBOLS)
WARMUP_H        = 110   # 110H de contexto (SMA100 precisa de 100 candles)

MONTHS = [
    (2025,  1), (2025,  2), (2025,  3), (2025,  4),
    (2025,  5), (2025,  6), (2025,  7), (2025,  8),
    (2025,  9), (2025, 10), (2025, 11), (2025, 12),
    (2026,  1), (2026,  2), (2026,  3), (2026,  4),
]


def month_bounds(year: int, month: int) -> tuple[int, int]:
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=UTC)
    last  = monthrange(year, month)[1]
    end   = datetime(year, month, last, 23, 59, 59, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def load_candles(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    key      = symbol.replace("-", "_")
    raw      = json.loads((CACHE_DIR / f"{key}_1H.json").read_text())
    warmup_start = start_ms - WARMUP_H * 3_600_000
    filtered = sorted(
        [c for c in raw if warmup_start <= c["ts"] <= end_ms],
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


# ── Hybrid Strategy ──────────────────────────────────────────────────────────

class HybridStrategy1H(BaseStrategy):
    """
    Estratégia híbrida: Reversal em crashes + Momentum em bull markets.
    Filtros macro impedem entradas em bear markets estruturais.
    """

    # ── Parâmetros Reversal ────────────────────────────────────────────────
    FALL_PCT   = {"BTC-USDT": 0.030, "ETH-USDT": 0.040, "SOL-USDT": 0.050}
    BASE_CANDLES = 6
    TREND_FLOOR  = 0.92
    TREND_CEIL   = 1.02
    MIN_SL_PCT   = 0.015   # ← 1.5% mínimo (era 0.8%)
    MAX_SL_PCT   = 0.060
    MIN_RATIO    = 1.5
    BTC_ANCHOR_FALL = 0.020  # BTC deve estar em dip ≥2% para ETH/SOL entrarem

    # ── Parâmetros Momentum ────────────────────────────────────────────────
    MOM_CANDLES      = 3      # últimas 3 velas bullish consecutivas
    MOM_VOL_MULT     = 1.2    # volume ≥ 1.2× média 8H
    MOM_MIN_SL_PCT   = 0.010  # SL mínimo 1% (abaixo SMA20)
    MOM_MAX_SL_PCT   = 0.030  # SL máximo 3%
    MOM_RATIO        = 2.0    # TP = 2× SL
    MOM_SYMBOLS      = {"BTC-USDT", "ETH-USDT"}  # SOL excluída do momentum

    # ── Filtro macro ───────────────────────────────────────────────────────
    SMA_MACRO = 100          # SMA 100H como filtro de bear market
    BEAR_FLOOR = 0.85        # preço < SMA100 × 85% → bear estrutural → skip

    # ── Estado compartilhado entre símbolos ────────────────────────────────
    _btc_fall_cache: float = 0.0    # fall_pct do BTC na última avaliação
    _btc_in_bull:    bool  = False  # BTC em tendência de alta

    def __init__(self, symbols):
        super().__init__(strategy_id="hybrid_1h", symbols=symbols)
        try:
            self._platt = PlattCalibrator(
                coef_path=ROOT / "data" / "models" / "calibration_coef.json"
            )
        except Exception:
            self._platt = None

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles_1h   # newest first
        if len(c) < self.SMA_MACRO + 5:
            return None

        closes  = [x.close  for x in c[:self.SMA_MACRO + 5]]
        highs   = [x.high   for x in c[:self.SMA_MACRO + 5]]
        lows    = [x.low    for x in c[:self.SMA_MACRO + 5]]
        volumes = [x.volume for x in c[:self.SMA_MACRO + 5]]

        current  = closes[0]
        sma20    = sum(closes[:20]) / 20
        sma100   = sum(closes[:self.SMA_MACRO]) / self.SMA_MACRO

        # ── Filtro 1: Bear market estrutural ────────────────────────────────
        # Preço muito abaixo da SMA100 → bear market → nenhuma entrada
        if current < sma100 * self.BEAR_FLOOR:
            if ctx.symbol == "BTC-USDT":
                HybridStrategy1H._btc_fall_cache = 0.0
                HybridStrategy1H._btc_in_bull    = False
            return None

        # ── Atualiza estado BTC (para âncora de ETH/SOL) ────────────────────
        if ctx.symbol == "BTC-USDT":
            lh = max(closes[self.BASE_CANDLES + 1:20])
            fl = min(closes[1:20])
            btc_fall = (lh - fl) / lh if lh > 0 else 0
            HybridStrategy1H._btc_fall_cache = btc_fall
            # BTC em bull: acima da SMA100 e SMA20
            HybridStrategy1H._btc_in_bull = (
                current > sma100 * 1.00 and
                current > sma20  * 0.99
            )

        # ── Filtro 2: Âncora BTC para ETH/SOL ────────────────────────────
        # ETH e SOL só operam se BTC também estiver em dip ou bull confirmado
        if ctx.symbol in ("ETH-USDT", "SOL-USDT"):
            btc_fall = HybridStrategy1H._btc_fall_cache
            btc_bull = HybridStrategy1H._btc_in_bull
            # Para reversal: BTC precisa estar em dip ≥2%
            # Para momentum: BTC precisa estar em bull
            if btc_fall < self.BTC_ANCHOR_FALL and not btc_bull:
                return None

        # ── Tenta REVERSAL ───────────────────────────────────────────────────
        sig = self._try_reversal(ctx.symbol, closes, highs, lows, volumes, current, sma20)
        if sig:
            return sig

        # ── Tenta MOMENTUM (apenas BTC e ETH, não em dip) ────────────────────
        if ctx.symbol in self.MOM_SYMBOLS:
            sig = self._try_momentum(ctx.symbol, closes, highs, lows, volumes, current, sma20)
            if sig:
                return sig

        return None

    def _try_reversal(self, symbol, closes, highs, lows, volumes, current, sma20):
        n = self.BASE_CANDLES
        min_fall = self.FALL_PCT.get(symbol, 0.030)

        # Faixa macro aceitável para reversão
        if not (sma20 * self.TREND_FLOOR <= current <= sma20 * self.TREND_CEIL):
            return None

        # Queda real
        lh       = max(closes[n + 1:20])
        fall_low = min(closes[1:20])
        fall_pct = (lh - fall_low) / lh if lh > 0 else 0
        if fall_pct < min_fall:
            return None

        # Dip tocou a SMA20
        if fall_low > sma20 * 1.01:
            return None

        # Base formada
        base_high  = max(highs[1:n + 1])
        base_low   = min(lows[1:n + 1])
        base_range = base_high - base_low
        fall_mag   = lh - fall_low

        # Rompimento
        if current <= base_high * 1.001:
            return None

        # Volume
        avg_vol   = sum(volumes[1:9]) / 8
        vol_ratio = volumes[0] / avg_vol if avg_vol > 0 else 0
        if avg_vol > 0 and vol_ratio < 0.8:
            return None

        # SL / TP
        sl_target = base_low * 0.999
        sl_dist   = current - sl_target
        if sl_dist <= 0:
            return None
        sl_pct = sl_dist / current
        if not (self.MIN_SL_PCT <= sl_pct <= self.MAX_SL_PCT):
            return None

        tp_pct = self.MIN_RATIO * sl_pct
        ratio  = self.MIN_RATIO

        # Score
        sf = min(fall_pct / 0.10, 1.0)
        sb = 1.0 - min(base_range / fall_mag, 1.0) if fall_mag > 0 else 0
        sv = min(vol_ratio / 2.0, 1.0)
        sr = min(ratio / 8.0, 1.0)
        raw = sf * 0.30 + sb * 0.30 + sv * 0.20 + sr * 0.20
        cal = self._platt.calibrate(raw) if self._platt else raw

        return Signal(
            strategy_id=self._strategy_id,
            symbol=symbol,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw, calibrated_score=cal,
            confidence=cal,
            expected_value=ratio * cal - (1 - cal),
            kelly_fraction=0.08,
            regime="REVERSAL",
            timeframe="1H",
            factors={
                "sl_pct": round(sl_pct, 5), "tp_pct": round(tp_pct, 5),
                "fall_pct": round(fall_pct, 3),
                "base_range": round(base_range / fall_mag, 3) if fall_mag > 0 else 0,
                "vol_ratio": round(vol_ratio, 2),
                "tp_ratio": round(ratio, 2),
                "mode": "reversal",
            },
        )

    def _try_momentum(self, symbol, closes, highs, lows, volumes, current, sma20):
        """Momentum simples: 3 velas consecutivas bullish + volume + above SMA20."""

        # Só em tendência de alta (acima SMA20)
        if current < sma20 * 1.005:
            return None

        # 3 velas consecutivas bullish
        n = self.MOM_CANDLES
        if len(closes) < n + 2:
            return None
        for i in range(n):
            if closes[i] <= closes[i + 1]:  # close deve ser maior que anterior
                return None

        # Candle atual deve ser bullish (close > open)
        # No contexto: closes[0]=current, highs/lows são das mesmas velas
        # Verificamos pelo range: se close está no terço superior
        c_range = highs[0] - lows[0]
        if c_range > 0 and (closes[0] - lows[0]) / c_range < 0.5:
            return None  # fechou no terço inferior → não bullish suficiente

        # Volume acima da média
        avg_vol = sum(volumes[1:9]) / 8
        if avg_vol > 0 and volumes[0] < avg_vol * self.MOM_VOL_MULT:
            return None

        # SL = SMA20 como suporte (0.2% abaixo)
        sl_price = sma20 * 0.998
        sl_dist  = current - sl_price
        if sl_dist <= 0:
            return None

        sl_pct = sl_dist / current
        if not (self.MOM_MIN_SL_PCT <= sl_pct <= self.MOM_MAX_SL_PCT):
            return None

        tp_pct = self.MOM_RATIO * sl_pct

        raw = 0.55  # score fixo moderado para momentum simples
        cal = self._platt.calibrate(raw) if self._platt else raw

        return Signal(
            strategy_id=self._strategy_id,
            symbol=symbol,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw, calibrated_score=cal,
            confidence=cal,
            expected_value=self.MOM_RATIO * cal - (1 - cal),
            kelly_fraction=0.06,   # menor que reversal (momentum menos confiável)
            regime="MOMENTUM",
            timeframe="1H",
            factors={
                "sl_pct": round(sl_pct, 5), "tp_pct": round(tp_pct, 5),
                "vol_ratio": round(volumes[0] / avg_vol if avg_vol > 0 else 0, 2),
                "tp_ratio": round(self.MOM_RATIO, 2),
                "mode": "momentum",
            },
        )


# ── Engine com TP/SL relativos ao entry_price ────────────────────────────────

class HybridEngine(BacktestEngine):
    def _check_exit(self, trade, candle, history):
        f      = trade.signal.factors if trade.signal else {}
        sl_pct = f.get("sl_pct", 0)
        tp_pct = f.get("tp_pct", 0)
        mode   = f.get("mode", "reversal")

        sl_price = trade.entry_price * (1.0 - sl_pct) if sl_pct else 0
        tp_price = trade.entry_price * (1.0 + tp_pct) if tp_pct else 0

        if sl_price > 0 and candle.low <= sl_price:
            return True, "stop_loss"
        if tp_price > 0 and candle.high >= tp_price:
            return True, "take_profit"

        # Timeout: 48H reversal, 12H momentum
        timeout = 48 if mode == "reversal" else 12
        held    = sum(1 for c in history if c.timestamp >= trade.entry_time)
        if held >= timeout:
            return True, f"timeout_{timeout}h"

        return False, ""


# ── Runner mensal ────────────────────────────────────────────────────────────

async def run_month(year: int, month: int) -> dict:
    start_ms, end_ms = month_bounds(year, month)
    totals = {"pnl": 0.0, "fees": 0.0, "trades": 0, "wins": 0,
              "rev_trades": 0, "mom_trades": 0, "results": []}

    # Reset estado compartilhado BTC para cada mês
    HybridStrategy1H._btc_fall_cache = 0.0
    HybridStrategy1H._btc_in_bull    = False

    for sym in SYMBOLS:
        candles = load_candles(sym, start_ms, end_ms)
        if len(candles) < WARMUP_H + 5:
            continue

        apr_idx = next(
            (i for i, c in enumerate(candles) if c.timestamp.timestamp() * 1000 >= start_ms),
            WARMUP_H,
        )

        strategy = HybridStrategy1H(symbols=SYMBOLS)
        engine   = HybridEngine(
            strategy=strategy, symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08, seed=42,
        )
        result = await engine.run(candles, warmup=apr_idx)

        totals["pnl"]    += result.total_pnl
        totals["fees"]   += result.total_fees
        totals["trades"] += result.total_trades
        totals["wins"]   += result.winning_trades
        totals["results"].append(result)

        for t in result.trades:
            mode = (t.signal.factors or {}).get("mode", "reversal") if t.signal else "?"
            if mode == "momentum":
                totals["mom_trades"] += 1
            else:
                totals["rev_trades"] += 1

    totals["wr"] = totals["wins"] / totals["trades"] if totals["trades"] > 0 else 0
    totals["year"]  = year
    totals["month"] = month
    return totals


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    sep  = "═" * 84
    sep2 = "─" * 84

    print(sep)
    print("  CCTBv5 — Backtest Híbrido | HybridStrategy1H | Jan 2025 → Abr 2026")
    print(f"  Capital: ${INITIAL_CAPITAL:,.2f} | Fee: 0.10% | 3 símbolos")
    print(f"  Reversal: SL≥1.5% | BTC anchor ETH/SOL | fall BTC=3% ETH=4% SOL=5%")
    print(f"  Momentum: 3 velas bullish + vol + SMA20 suporte | apenas BTC+ETH")
    print(f"  Macro: SMA100 filter (bloqueia bear market estrutural)")
    print(sep)
    print(f"  {'Mês':<8} {'P&L':>10} {'Trades':>7} {'Rev':>5} {'Mom':>5} {'WR':>7} {'Fees':>8} {'Status'}")
    print(sep2)

    monthly = []
    for year, month in MONTHS:
        m = await run_month(year, month)
        monthly.append(m)
        sign   = "+" if m["pnl"] >= 0 else ""
        icon   = "✅" if m["pnl"] > 0 else "❌" if m["pnl"] < -10 else "➖"
        print(
            f"  {icon} {year}/{month:02d}  "
            f"{sign}${m['pnl']:>8.2f}  "
            f"{m['trades']:>5}  "
            f"{m['rev_trades']:>4}r  "
            f"{m['mom_trades']:>4}m  "
            f"{m['wr']:>6.1%}  "
            f"${m['fees']:>6.2f}"
        )

    # ── Consolidado ──────────────────────────────────────────────────────────
    total_pnl    = sum(m["pnl"]       for m in monthly)
    total_fees   = sum(m["fees"]      for m in monthly)
    total_trades = sum(m["trades"]    for m in monthly)
    total_wins   = sum(m["wins"]      for m in monthly)
    total_rev    = sum(m["rev_trades"] for m in monthly)
    total_mom    = sum(m["mom_trades"] for m in monthly)
    pos_months   = sum(1 for m in monthly if m["pnl"] > 0)
    neg_months   = sum(1 for m in monthly if m["pnl"] < -10)
    zero_months  = sum(1 for m in monthly if m["trades"] == 0)
    overall_wr   = total_wins / total_trades if total_trades > 0 else 0
    monthly_avg  = total_pnl / len(monthly)

    print(sep)
    print("  CONSOLIDADO — 16 MESES")
    print(sep)
    print(f"  P&L total          : {'+'if total_pnl>=0 else ''}${total_pnl:,.2f}")
    print(f"  Retorno            : {'+'if total_pnl>=0 else ''}{total_pnl/INITIAL_CAPITAL:.2%}")
    print(f"  P&L médio/mês      : {'+'if monthly_avg>=0 else ''}${monthly_avg:,.2f}")
    print(f"  Total trades       : {total_trades}  (reversal: {total_rev} | momentum: {total_mom})")
    print(f"  Win Rate global    : {overall_wr:.1%}")
    print(f"  Total fees         : ${total_fees:,.2f}")
    print(sep2)
    print(f"  Meses positivos    : {pos_months} / {len(monthly)}  ({pos_months/len(monthly):.0%})")
    print(f"  Meses negativos    : {neg_months} / {len(monthly)}  ({neg_months/len(monthly):.0%})")
    print(f"  Meses sem trades   : {zero_months} / {len(monthly)}")
    print(sep2)
    best  = max(monthly, key=lambda m: m["pnl"])
    worst = min(monthly, key=lambda m: m["pnl"])
    print(f"  Melhor mês         : {best['year']}/{best['month']:02d}  +${best['pnl']:,.2f}")
    print(f"  Pior mês           : {worst['year']}/{worst['month']:02d}  ${worst['pnl']:,.2f}")
    print(sep)

    # ── Análise por símbolo ───────────────────────────────────────────────────
    print("  ANÁLISE POR SÍMBOLO")
    print(sep2)
    sym_pnl:    dict[str, float] = {s: 0.0 for s in SYMBOLS}
    sym_trades: dict[str, int]   = {s: 0    for s in SYMBOLS}
    sym_wins:   dict[str, int]   = {s: 0    for s in SYMBOLS}
    for m in monthly:
        for r in m["results"]:
            sym_pnl[r.symbol]    += r.total_pnl
            sym_trades[r.symbol] += r.total_trades
            sym_wins[r.symbol]   += r.winning_trades
    for sym in SYMBOLS:
        t  = sym_trades[sym]
        wr = sym_wins[sym] / t if t > 0 else 0
        p  = sym_pnl[sym]
        ag = p / t if t > 0 else 0
        sign = "+" if p >= 0 else ""
        print(f"  {sym:<12}  P&L: {sign}${p:>8.2f}  trades: {t:>3}  WR: {wr:>5.1%}  avg: {sign}${ag:>6.2f}")

    # ── Comparativo vs puro reversal ─────────────────────────────────────────
    print()
    print(sep)
    print("  COMPARATIVO vs REVERSAL PURO (16 meses)")
    print(sep)
    print(f"  {'Estratégia':<25} {'P&L':>10} {'Trades':>8} {'WR':>7} {'Fees':>9} {'Meses+':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*7} {'─'*9} {'─'*8}")
    print(f"  {'Reversal puro':<25} {'$-2,276':>10} {'197':>8} {'39.1%':>7} {'$1,088':>9} {'4/16':>8}")
    sign_h = "+" if total_pnl >= 0 else ""
    print(f"  {'Híbrido (novo)':<25} {sign_h+'$'+f'{total_pnl:,.0f}':>10} {str(total_trades):>8} {overall_wr:.1%:>7} {'$'+f'{total_fees:,.0f}':>9} {str(pos_months)+'/16':>8}")
    print(sep)

    # ── Veredicto ─────────────────────────────────────────────────────────────
    active = [m for m in monthly if m["trades"] > 0]
    pos_active = sum(1 for m in active if m["pnl"] > 0) / len(active) if active else 0

    print("  VEREDICTO")
    print(sep)
    if pos_active >= 0.60 and total_pnl > 0:
        print(f"  ✅ ESTRATÉGIA VÁLIDA — {pos_active:.0%} meses com trades foram positivos")
        print(f"     Edge real detectado em múltiplos regimes de mercado.")
    elif pos_active >= 0.50 and total_pnl > 0:
        print(f"  ⚠️  MARGINAL — Positivo total mas {pos_active:.0%} consistência")
        print(f"     Funciona mas precisa de mais validação (paper trading).")
    elif pos_active >= 0.50:
        print(f"  ⚠️  INCONSISTENTE — {pos_active:.0%} meses positivos, P&L total negativo")
        print(f"     Vence mais meses mas perde mais no total (outliers negativos grandes).")
    else:
        print(f"  ❌ INSUFICIENTE — apenas {pos_active:.0%} meses positivos")
        print(f"     Revisão de parâmetros necessária.")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
