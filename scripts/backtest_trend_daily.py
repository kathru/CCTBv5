"""
Backtest — EWMA Trend Daily + Vol Scaling
Jan 2025 → Abr 2026 (16 meses)

Estratégia Man AHL / Winton simplificada:
  SINAL  : close > EMA50D → LONG | close < EMA50D → FLAT
  SIZING : position = capital × min(target_vol / realized_vol, max_pos_pct)
           onde realized_vol = ATR20D% × sqrt(365)
  FILTRO : se vol_20D > 80% aa (crash mode) → reduz posição 50%

Uma regra. Sem fitting. Executada 1× por dia no fechamento.
Fee: 0.10% taker por lado.
"""
import json
import math
from pathlib import Path
from datetime import UTC, datetime
from calendar import monthrange
from collections import defaultdict

CACHE   = Path("/app/data/cache")
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

INITIAL_CAPITAL = 96_592.87
FEE_ONE_SIDE    = 0.001   # 0.10% por operação

EMA_PERIOD  = 50     # dias
ATR_PERIOD  = 20     # dias
VOL_TARGET  = 0.15   # 15% vol anualizada alvo
VOL_CAP     = 0.80   # se vol > 80% aa → reduz posição 50%
MAX_POS_PCT = 0.33   # máx 33% capital por símbolo
ANN         = math.sqrt(365)

REVERSAL_PNL = -2276.31
BH_BTC_RET   = -18.4

MONTHS = [
    (2025,1),(2025,2),(2025,3),(2025,4),(2025,5),(2025,6),
    (2025,7),(2025,8),(2025,9),(2025,10),(2025,11),(2025,12),
    (2026,1),(2026,2),(2026,3),(2026,4),
]


def month_bounds(y, m):
    s = datetime(y, m, 1, tzinfo=UTC)
    e = datetime(y, m, monthrange(y, m)[1], 23, 59, 59, tzinfo=UTC)
    return int(s.timestamp()*1000), int(e.timestamp()*1000)


def to_daily(raw_1h):
    by_day = defaultdict(list)
    for c in sorted(raw_1h, key=lambda x: x["ts"]):
        day = (c["ts"] // 86_400_000) * 86_400_000
        by_day[day].append(c)
    result = []
    for day in sorted(by_day):
        g = by_day[day]
        result.append({
            "ts": day, "open": g[0]["open"],
            "high": max(x["high"] for x in g),
            "low":  min(x["low"]  for x in g),
            "close": g[-1]["close"],
        })
    return result


def calc_ema(values, period):
    """EMA. Retorna lista com NaN nos primeiros (period-1) valores."""
    out = []
    k = 2.0 / (period + 1)
    sma_acc = []
    for v in values:
        if math.isnan(v):
            out.append(float("nan"))
            continue
        sma_acc.append(v)
        if len(sma_acc) < period:
            out.append(float("nan"))
        elif len(sma_acc) == period:
            out.append(sum(sma_acc) / period)
        else:
            out.append(v * k + out[-1] * (1 - k))
    return out


def calc_atr_pct(candles, period):
    """ATR como % do close (retorna lista alinhada com candles)."""
    trs = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = (c["high"] - c["low"]) / c["close"]
        else:
            prev = candles[i-1]["close"]
            tr = max(c["high"]-c["low"], abs(c["high"]-prev), abs(c["low"]-prev))
            tr /= c["close"]
        trs.append(tr)
    # Smooth com SMA então EMA
    out = []
    sma_acc = []
    k = 1.0 / period
    for tr in trs:
        sma_acc.append(tr)
        if len(sma_acc) < period:
            out.append(float("nan"))
        elif len(sma_acc) == period:
            out.append(sum(sma_acc) / period)
        else:
            out.append(tr * k + out[-1] * (1 - k))
    return out


def run_backtest():
    # Carrega com warmup de 80 dias (suficiente para EMA50 + ATR20)
    warmup_ms = month_bounds(2025, 1)[0] - 80 * 86_400_000
    end_ms    = month_bounds(2026, 4)[1]

    daily = {}
    for sym in SYMBOLS:
        key  = sym.replace("-", "_")
        raw  = json.loads((CACHE / f"{key}_1H.json").read_text())
        filt = [c for c in raw if warmup_ms <= c["ts"] <= end_ms]
        daily[sym] = to_daily(filt)

    # Pré-calcula indicadores
    ema50 = {}
    atr20 = {}
    for sym in SYMBOLS:
        closes      = [c["close"] for c in daily[sym]]
        ema50[sym]  = calc_ema(closes, EMA_PERIOD)
        atr20[sym]  = calc_atr_pct(daily[sym], ATR_PERIOD)

    # Estado da simulação
    capital  = INITIAL_CAPITAL
    qty      = {s: 0.0 for s in SYMBOLS}
    avg_px   = {s: 0.0 for s in SYMBOLS}
    monthly  = {(y,m): {"pnl":0.,"fees":0.,"trades":0,"in_long":{s:0 for s in SYMBOLS}}
                for y,m in MONTHS}

    def get_ym(ts):
        d = datetime.fromtimestamp(ts/1000, tz=UTC)
        return (d.year, d.month)

    # Itera dia a dia (usa BTC como calendário referência)
    btc_days = {c["ts"]: i for i, c in enumerate(daily["BTC-USDT"])}

    for day_ts in sorted(btc_days):
        ym = get_ym(day_ts)
        if ym not in monthly:
            continue

        for sym in SYMBOLS:
            sym_days = {c["ts"]: i for i, c in enumerate(daily[sym])}
            if day_ts not in sym_days:
                continue
            idx   = sym_days[day_ts]
            e50   = ema50[sym][idx]
            atr_p = atr20[sym][idx]
            price = daily[sym][idx]["close"]

            if math.isnan(e50) or math.isnan(atr_p) or price <= 0:
                continue

            # ── Sinal ──────────────────────────────────────────────────────
            trend_long = price > e50
            vol_ann    = atr_p * ANN          # vol realizada anualizada

            # ── Sizing ─────────────────────────────────────────────────────
            if trend_long and vol_ann > 0:
                raw_pct     = VOL_TARGET / vol_ann
                if vol_ann > VOL_CAP:          # vol muito alta → reduz 50%
                    raw_pct *= 0.5
                pos_pct     = min(raw_pct, MAX_POS_PCT)
                target_qty  = (capital + qty[sym] * price) * pos_pct / price
            else:
                target_qty  = 0.0

            delta = target_qty - qty[sym]

            # Threshold de rebalanceamento: só executa se mudança > 2% capital
            if abs(delta) * price < (capital + sum(qty[s]*price for s in SYMBOLS)) * 0.02:
                if trend_long:
                    monthly[ym]["in_long"][sym] += 1
                continue

            # ── Execução ───────────────────────────────────────────────────
            if delta > 0:
                cost = delta * price * (1 + FEE_ONE_SIDE)
                total_val = capital + sum(qty[s] * price for s in SYMBOLS)
                if cost > capital:
                    delta = capital / (price * (1 + FEE_ONE_SIDE))
                    cost  = delta * price * (1 + FEE_ONE_SIDE)
                if delta > 0.000001:
                    fee = delta * price * FEE_ONE_SIDE
                    avg_px[sym] = (
                        (avg_px[sym]*qty[sym] + price*delta) / (qty[sym]+delta)
                        if qty[sym] > 0 else price
                    )
                    qty[sym]    += delta
                    capital     -= cost
                    monthly[ym]["trades"] += 1
                    monthly[ym]["fees"]   += fee
            elif delta < 0:
                sell_qty = min(abs(delta), qty[sym])
                if sell_qty > 0.000001:
                    fee      = sell_qty * price * FEE_ONE_SIDE
                    pnl      = (price - avg_px[sym]) * sell_qty - fee
                    proceeds = sell_qty * price - fee
                    qty[sym]    -= sell_qty
                    capital     += proceeds
                    monthly[ym]["pnl"]    += pnl
                    monthly[ym]["trades"] += 1
                    monthly[ym]["fees"]   += fee

            if qty[sym] > 0:
                monthly[ym]["in_long"][sym] += 1

    # Fecha todas as posições no último dia de Apr 2026
    last_ym = (2026, 4)
    for sym in SYMBOLS:
        if qty[sym] > 0:
            sym_days = {c["ts"]: i for i, c in enumerate(daily[sym])}
            last_ts  = max(ts for ts in sym_days if get_ym(ts) == last_ym)
            idx      = sym_days[last_ts]
            price    = daily[sym][idx]["close"]
            fee      = qty[sym] * price * FEE_ONE_SIDE
            pnl      = (price - avg_px[sym]) * qty[sym] - fee
            capital += qty[sym] * price - fee
            monthly[last_ym]["pnl"]    += pnl
            monthly[last_ym]["trades"] += 1
            monthly[last_ym]["fees"]   += fee
            qty[sym] = 0.0

    return {"monthly": monthly, "final_capital": capital}


def main():
    sep  = "═" * 76
    sep2 = "─" * 76

    print(sep)
    print("  EWMA Trend Daily + Vol Scaling  |  Jan 2025 → Abr 2026")
    print(f"  Regra: close > EMA50D → LONG | Sizing: 15% vol target | máx 33%/sym")
    print(sep)

    res     = run_backtest()
    monthly = res["monthly"]
    final   = res["final_capital"]
    total_pnl   = final - INITIAL_CAPITAL
    ret_pct     = total_pnl / INITIAL_CAPITAL * 100

    total_fees = total_trades = pos_m = neg_m = 0

    print(f"  {'Mês':<8} {'P&L':>10} {'Trades':>7} {'Fees':>8}  Posição (dias long)")
    print(sep2)

    for y, m in MONTHS:
        ym = (y, m)
        d  = monthly[ym]
        pnl = d["pnl"]
        total_pnl_track = pnl
        total_fees   += d["fees"]
        total_trades += d["trades"]

        icon = "✅" if pnl > 5 else "❌" if pnl < -5 else "➖"
        if pnl > 5:  pos_m += 1
        if pnl < -5: neg_m += 1

        longs = " | ".join(f"{s.replace('-USDT','')}: {d['in_long'][s]}d"
                           for s in SYMBOLS)
        sign = "+" if pnl >= 0 else ""
        print(f"  {icon} {y}/{m:02d}  {sign}${pnl:>8.2f}  "
              f"{d['trades']:>5}  ${d['fees']:>6.2f}   {longs}")

    print(sep)
    print("  CONSOLIDADO")
    print(sep)
    print(f"  Capital inicial   : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Capital final     : ${final:>12,.2f}")
    sign = "+" if total_pnl >= 0 else ""
    print(f"  P&L total         : {sign}${abs(total_pnl):>11,.2f}  ({sign}{ret_pct:.2f}%)")
    print(f"  Total trades      : {total_trades:>13}  ({total_trades/16:.1f}/mês)")
    print(f"  Total fees        : ${total_fees:>12,.2f}")
    print(f"  Meses positivos   : {pos_m:>4} / 16  ({pos_m/16:.0%})")
    print(f"  Meses negativos   : {neg_m:>4} / 16  ({neg_m/16:.0%})")

    bh_final  = INITIAL_CAPITAL * (1 + BH_BTC_RET/100)
    rev_final = INITIAL_CAPITAL + REVERSAL_PNL

    print(sep)
    print("  COMPARATIVO — mesmos 16 meses")
    print(sep2)
    print(f"  {'Estratégia':<28} {'Capital Final':>14} {'Retorno':>9} {'Meses+':>7}")
    print(f"  {'─'*28} {'─'*14} {'─'*9} {'─'*7}")
    print(f"  {'Buy & Hold BTC':<28} ${bh_final:>12,.2f}   {BH_BTC_RET:>+6.1f}%      —")
    rev_ret = REVERSAL_PNL/INITIAL_CAPITAL*100
    print(f"  {'Reversal 1H':<28} ${rev_final:>12,.2f}   {rev_ret:>+6.1f}%   4/16")
    print(
        f"  {'EWMA Trend Daily (novo)':<28} ${final:>12,.2f}   {sign}{ret_pct:>5.1f}%  {pos_m}/16"
    )

    alpha_vs_bh = ret_pct - BH_BTC_RET
    print(sep)
    print("  VEREDICTO")
    print(sep2)

    if ret_pct > 0 and pos_m >= 9:
        print(f"  ✅ EDGE CONFIRMADO — +{ret_pct:.1f}% em 16 meses, {pos_m}/16 positivos")
        print(f"     Alpha vs buy&hold: +{alpha_vs_bh:.1f}pp. Implementar.")
    elif ret_pct > 0:
        print(f"  ✅ POSITIVO — +{ret_pct:.1f}% mas consistência moderada ({pos_m}/16)")
        print(f"     Alpha vs buy&hold: +{alpha_vs_bh:.1f}pp.")
        print(f"     Melhor que qualquer estratégia testada até agora.")
    elif ret_pct > REVERSAL_PNL/INITIAL_CAPITAL*100:
        print(f"  ⚠️  MELHOR QUE REVERSAL mas ainda negativo ({ret_pct:.1f}%)")
        print(f"     Alpha vs buy&hold: +{alpha_vs_bh:.1f}pp (preserva capital).")
    else:
        print(f"  ❌ RESULTADO ABAIXO DO ESPERADO — verificar implementação")

    print(sep)


if __name__ == "__main__":
    main()
