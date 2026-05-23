"""
Backtest — EWMA Trend Daily + SHORT (simulado)
Jan 2025 → Abr 2026 (16 meses)

Valida a hipótese: adicionando SHORT via perps, a estratégia ganha
em qualquer regime? Esta é a validação ANTES de implementar em live.

LONG  : close > EMA50D → compra spot
SHORT : close < EMA50D → vende perp (simulado com posição negativa)
FLAT  : transição, posição zerada

Custo SHORT: funding rate 0.01%/8h = 0.03%/dia (OKX média histórica BTC)
             + fee 0.02% taker (perp é mais barato que spot na OKX)

Sizing: vol-target 15% | máx 33%/símbolo | mesmo lado sempre
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
FEE_SPOT        = 0.001    # 0.10%/lado spot
FEE_PERP        = 0.0002   # 0.02%/lado perp (taker futures OKX)
FUNDING_DAY     = 0.0003   # 0.03%/dia custo de carregamento do short

EMA_PERIOD  = 50
ATR_PERIOD  = 20
VOL_TARGET  = 0.15
VOL_CAP     = 0.80
MAX_POS_PCT = 0.33
ANN         = math.sqrt(365)

MONTHS = [
    (2025,1),(2025,2),(2025,3),(2025,4),(2025,5),(2025,6),
    (2025,7),(2025,8),(2025,9),(2025,10),(2025,11),(2025,12),
    (2026,1),(2026,2),(2026,3),(2026,4),
]

PREV_RESULTS = {
    "buy_hold": -18.4,
    "reversal": -2.4,
    "trend_long_only": -6.0,
}


def month_bounds(y, m):
    s = datetime(y, m, 1, tzinfo=UTC)
    e = datetime(y, m, monthrange(y,m)[1], 23, 59, 59, tzinfo=UTC)
    return int(s.timestamp()*1000), int(e.timestamp()*1000)


def to_daily(raw_1h):
    by_day = defaultdict(list)
    for c in sorted(raw_1h, key=lambda x: x["ts"]):
        day = (c["ts"]//86_400_000)*86_400_000
        by_day[day].append(c)
    return [{"ts":d,"open":g[0]["open"],"high":max(x["high"]for x in g),
             "low":min(x["low"]for x in g),"close":g[-1]["close"]}
            for d in sorted(by_day) for g in [by_day[d]]]


def calc_ema(values, period):
    out, k, acc = [], 2/(period+1), []
    for v in values:
        if math.isnan(v): out.append(float("nan")); continue
        acc.append(v)
        if len(acc) < period: out.append(float("nan"))
        elif len(acc) == period: out.append(sum(acc)/period)
        else: out.append(v*k + out[-1]*(1-k))
    return out


def calc_atr_pct(candles, period):
    trs, out = [], []
    for i, c in enumerate(candles):
        tr = (c["high"]-c["low"])/c["close"] if i==0 else \
             max(c["high"]-c["low"],abs(c["high"]-candles[i-1]["close"]),
                 abs(c["low"]-candles[i-1]["close"]))/c["close"]
        trs.append(tr)
    k, acc = 1/period, []
    for tr in trs:
        acc.append(tr)
        if len(acc) < period: out.append(float("nan"))
        elif len(acc) == period: out.append(sum(acc)/period)
        else: out.append(tr*k + out[-1]*(1-k))
    return out


def run_backtest(allow_short: bool = True) -> dict:
    warmup_ms = month_bounds(2025,1)[0] - 80*86_400_000
    end_ms    = month_bounds(2026,4)[1]

    daily = {}
    for sym in SYMBOLS:
        key = sym.replace("-","_")
        raw = json.loads((CACHE/f"{key}_1H.json").read_text())
        daily[sym] = to_daily([c for c in raw if warmup_ms<=c["ts"]<=end_ms])

    ema50 = {s: calc_ema([c["close"] for c in daily[s]], EMA_PERIOD) for s in SYMBOLS}
    atr20 = {s: calc_atr_pct(daily[s], ATR_PERIOD) for s in SYMBOLS}

    capital = INITIAL_CAPITAL
    # Posição: >0 = long spot, <0 = short perp, 0 = flat
    pos      = {s: 0.0 for s in SYMBOLS}
    avg_px   = {s: 0.0 for s in SYMBOLS}
    monthly  = {(y,m): {"pnl":0.,"fees":0.,"funding":0.,"trades":0,
                         "long_days":{s:0 for s in SYMBOLS},
                         "short_days":{s:0 for s in SYMBOLS}}
                for y,m in MONTHS}

    def get_ym(ts):
        d = datetime.fromtimestamp(ts/1000, tz=UTC)
        return (d.year, d.month)

    sym_idx = {s: {c["ts"]:i for i,c in enumerate(daily[s])} for s in SYMBOLS}
    all_days = sorted({c["ts"] for c in daily["BTC-USDT"]})

    for day_ts in all_days:
        ym = get_ym(day_ts)
        if ym not in monthly:
            # Ainda no warmup — mantém posições mas não contabiliza
            # Aplica funding costs durante warmup? Não, simplifica.
            continue

        for sym in SYMBOLS:
            if day_ts not in sym_idx[sym]: continue
            idx   = sym_idx[sym][day_ts]
            e50   = ema50[sym][idx]
            atr_p = atr20[sym][idx]
            price = daily[sym][idx]["close"]

            if math.isnan(e50) or math.isnan(atr_p) or price <= 0:
                continue

            # ── Sinal ──────────────────────────────────────────────────────
            trend_long  = price > e50
            trend_short = price < e50 and allow_short
            vol_ann = atr_p * ANN
            vol_ok  = vol_ann < VOL_CAP

            # ── Funding cost daily para posições short ──────────────────────
            if pos[sym] < 0:
                funding = abs(pos[sym]) * price * FUNDING_DAY
                monthly[ym]["funding"] += funding
                monthly[ym]["pnl"]     -= funding

            # ── Contagem de dias ────────────────────────────────────────────
            if pos[sym] > 0: monthly[ym]["long_days"][sym]  += 1
            if pos[sym] < 0: monthly[ym]["short_days"][sym] += 1

            # ── Sizing ─────────────────────────────────────────────────────
            if (trend_long or trend_short) and vol_ann > 0:
                raw_pct    = VOL_TARGET / vol_ann
                if not vol_ok: raw_pct *= 0.5
                pos_pct    = min(raw_pct, MAX_POS_PCT)
                total_eq   = capital + sum(
                    pos[s] * daily[s][sym_idx[s].get(day_ts, -1)]["close"]
                    if day_ts in sym_idx[s] and sym_idx[s][day_ts] >= 0 else 0
                    for s in SYMBOLS if pos[s] != 0
                )
                notional   = max(capital, INITIAL_CAPITAL * 0.5) * pos_pct
                target_abs = notional / price
                target     =  target_abs if trend_long else -target_abs
            else:
                target = 0.0

            # Threshold de rebalanceamento: 2% do capital
            if abs(target - pos[sym]) * price < capital * 0.02:
                continue

            # ── Execução ───────────────────────────────────────────────────
            old_pos  = pos[sym]
            delta    = target - old_pos
            fee_rate = FEE_PERP if (target < 0 or old_pos < 0) else FEE_SPOT

            if old_pos > 0 and target <= 0:
                # Fecha long spot
                fee  = old_pos * price * FEE_SPOT
                pnl  = (price - avg_px[sym]) * old_pos - fee
                capital += old_pos * price - fee
                monthly[ym]["pnl"]    += pnl
                monthly[ym]["fees"]   += fee
                monthly[ym]["trades"] += 1
                pos[sym] = 0.0

            if old_pos < 0 and target >= 0:
                # Fecha short perp
                fee = abs(old_pos) * price * FEE_PERP
                pnl = (avg_px[sym] - price) * abs(old_pos) - fee
                capital += abs(old_pos) * price * 0 + pnl  # perp: lucro vai para capital
                monthly[ym]["pnl"]    += pnl
                monthly[ym]["fees"]   += fee
                monthly[ym]["trades"] += 1
                pos[sym] = 0.0

            # Abre nova posição
            if abs(target) > 0.000001 and pos[sym] == 0:
                fee = abs(target) * price * fee_rate
                if target > 0:
                    # Long spot: desembolsa capital
                    cost = target * price + fee
                    if cost > capital: target = (capital - fee) / price
                    if target > 0:
                        capital -= target * price + fee
                        avg_px[sym] = price
                        pos[sym]    = target
                else:
                    # Short perp: não desembolsa capital (margin simulada)
                    fee_cost = abs(target) * price * fee_rate
                    capital  -= fee_cost
                    avg_px[sym] = price
                    pos[sym]    = target  # negativo

                monthly[ym]["fees"]   += fee
                monthly[ym]["trades"] += 1

    # Fecha tudo no último dia
    last_ym = (2026, 4)
    for sym in SYMBOLS:
        if pos[sym] == 0: continue
        sym_days = sorted(ts for ts in sym_idx[sym] if get_ym(ts) == last_ym)
        if not sym_days: continue
        last_ts = sym_days[-1]
        price   = daily[sym][sym_idx[sym][last_ts]]["close"]
        if pos[sym] > 0:
            fee = pos[sym] * price * FEE_SPOT
            pnl = (price - avg_px[sym]) * pos[sym] - fee
            capital += pos[sym] * price - fee
            monthly[last_ym]["pnl"]    += pnl
            monthly[last_ym]["fees"]   += fee
        else:
            fee = abs(pos[sym]) * price * FEE_PERP
            pnl = (avg_px[sym] - price) * abs(pos[sym]) - fee
            capital += pnl
            monthly[last_ym]["pnl"]    += pnl
            monthly[last_ym]["fees"]   += fee
        monthly[last_ym]["trades"] += 1
        pos[sym] = 0.0

    return {"monthly": monthly, "final_capital": capital}


def print_results(label: str, res: dict, ref_pnl: float = 0.0) -> tuple:
    monthly = res["monthly"]
    final   = res["final_capital"]
    total_pnl  = final - INITIAL_CAPITAL
    ret_pct    = total_pnl / INITIAL_CAPITAL * 100
    total_fees = sum(d["fees"] for d in monthly.values())
    total_fund = sum(d.get("funding",0) for d in monthly.values())
    total_tr   = sum(d["trades"] for d in monthly.values())
    pos_m = sum(1 for d in monthly.values() if d["pnl"] > 5)
    neg_m = sum(1 for d in monthly.values() if d["pnl"] < -5)
    return total_pnl, ret_pct, pos_m, neg_m, total_fees, total_fund, total_tr


def main():
    sep  = "═"*76
    sep2 = "─"*76

    print(sep)
    print("  Backtest: EWMA Trend + SHORT via Perps  |  Jan 2025 → Abr 2026")
    print("  Valida hipótese: SHORT completa o sistema e gera alpha em qualquer regime")
    print(sep)

    # Sem short (controle)
    print("\n  Rodando: Trend Long Only...")
    r_long = run_backtest(allow_short=False)

    # Com short (hipótese)
    print("  Rodando: Trend Long + Short...")
    r_both = run_backtest(allow_short=True)

    # Resultados mensais lado a lado
    print()
    print(
        f"  {'Mês':<8} {'B&H BTC':>9} {'Rev 1H':>8}"
        f" {'Trend L':>8} {'Trend L+S':>10} {'Melhor':>8}"
    )
    print(sep2)

    rev_monthly = {
        (2025,1):-249.07,(2025,2):-316.05,(2025,3):-468.25,(2025,4):240.68,
        (2025,5):-219.25,(2025,6):167.20,(2025,7):-229.17,(2025,8):-399.72,
        (2025,9):-267.42,(2025,10):78.10,(2025,11):-303.07,(2025,12):66.17,
        (2026,1):-14.10,(2026,2):-108.63,(2026,3):-121.85,(2026,4):-131.89,
    }

    # Carrega BTC para B&H mensal
    raw_btc = json.loads((CACHE/"BTC_USDT_1H.json").read_text())
    daily_btc = to_daily(sorted(raw_btc, key=lambda c: c["ts"]))
    btc_idx = {c["ts"]:i for i,c in enumerate(daily_btc)}

    trend_long_monthly = {
        (2025,1):-874.04,(2025,2):-125.57,(2025,3):0,(2025,4):0,
        (2025,5):817.72,(2025,6):-545.57,(2025,7):-71.24,(2025,8):2564.04,
        (2025,9):950.74,(2025,10):-3995.55,(2025,11):0,(2025,12):-153.37,
        (2026,1):-3861.18,(2026,2):0,(2026,3):-975.61,(2026,4):973.13,
    }

    for y, m in MONTHS:
        ym = (y, m)
        s_ms, e_ms = month_bounds(y, m)
        btc_m = [c for c in daily_btc if s_ms <= c["ts"] <= e_ms]
        bh = (
            (btc_m[-1]["close"]-btc_m[0]["open"])/btc_m[0]["open"]*INITIAL_CAPITAL/100*100
            if btc_m else 0
        )
        bh_pct = (
            (btc_m[-1]["close"]-btc_m[0]["open"])/btc_m[0]["open"]*100 if btc_m else 0
        )

        rev_p   = rev_monthly.get(ym, 0)
        long_p  = trend_long_monthly.get(ym, 0)
        both_p  = r_both["monthly"][ym]["pnl"]

        vals = {"B&H": bh, "Rev": rev_p, "Long": long_p, "L+S": both_p}
        best = max(vals, key=vals.get)
        icon = "✅" if both_p > 5 else "❌" if both_p < -5 else "➖"

        print(f"  {icon} {y}/{m:02d}  "
              f"{bh_pct:>+7.1f}%  "
              f"{rev_p:>+7.0f}$  "
              f"{long_p:>+7.0f}$  "
              f"{both_p:>+8.0f}$  "
              f"{'⭐'+best if best=='L+S' else best:>8}")

    # Consolidado
    pnl_l, ret_l, pos_l, neg_l, fees_l, fund_l, tr_l = print_results("Long", r_long)
    pnl_b, ret_b, pos_b, neg_b, fees_b, fund_b, tr_b = print_results("L+S",  r_both)

    bh_total = INITIAL_CAPITAL * (1 + PREV_RESULTS["buy_hold"]/100)
    rev_final = INITIAL_CAPITAL + PREV_RESULTS["reversal"]/100 * INITIAL_CAPITAL

    print(sep)
    print("  COMPARATIVO FINAL — 16 meses")
    print(sep)
    print(f"  {'Estratégia':<28} {'Capital Final':>14} {'Retorno':>9} {'Meses+':>8} {'Fees':>9}")
    print(f"  {'─'*28} {'─'*14} {'─'*9} {'─'*8} {'─'*9}")
    print(
        f"  {'Buy & Hold BTC':<28} ${bh_total:>12,.0f}"
        f"   {PREV_RESULTS['buy_hold']:>+6.1f}%       —        —"
    )
    rev_cap = INITIAL_CAPITAL + pnl_l*0 - 2276.31
    print(
        f"  {'Reversal 1H':<28} ${rev_cap:>12,.0f}"
        f"   {PREV_RESULTS['reversal']:>+6.1f}%    4/16  $1,088"
    )

    final_l = r_long["final_capital"]
    final_b = r_both["final_capital"]
    long_ret = (final_l-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    print(
        f"  {'Trend Long Only':<28} ${final_l:>12,.0f}"
        f"   {long_ret:>+6.1f}%  {pos_l:>2}/16  ${fees_l:>6,.0f}"
    )
    print(
        f"  {'Trend Long + SHORT ★':<28} ${final_b:>12,.0f}"
        f"   {ret_b:>+6.1f}%  {pos_b:>2}/16  ${fees_b+fund_b:>6,.0f}"
    )
    print(sep)

    # Veredicto
    print("  VEREDICTO")
    print(sep2)
    alpha = ret_b - PREV_RESULTS["buy_hold"]
    alpha_rev = ret_b - PREV_RESULTS["reversal"]

    if ret_b > 0:
        print(f"  ✅ SHORT RESOLVE O PROBLEMA — Retorno POSITIVO em período de queda de BTC")
        print(f"     Retorno: +{ret_b:.1f}% vs Buy&Hold {PREV_RESULTS['buy_hold']:.1f}%")
        print(f"     Alpha vs Buy&Hold   : +{alpha:.1f}pp")
        print(f"     Alpha vs Reversal   : +{alpha_rev:.1f}pp")
        print(f"     Alpha vs Trend Long : +{ret_b - long_ret:.1f}pp")
        print(f"     Consistência        : {pos_b}/16 meses positivos")
        print()
        print(f"  ✅ IMPLEMENTAR v5.6: EMA50D + SHORT via OKX Perpetuals")
    elif ret_b > PREV_RESULTS["reversal"]:
        print(
            f"  ⚠️  SHORT MELHORA MAS NÃO RESOLVE"
            f" — {ret_b:.1f}% vs reversal {PREV_RESULTS['reversal']:.1f}%"
        )
        print(f"     Alpha vs Buy&Hold: +{alpha:.1f}pp")
        print(f"     Revisar parâmetros antes de implementar.")
    else:
        print(f"  ❌ SHORT NÃO AJUDOU — {ret_b:.1f}% — Revisar lógica")
    print(sep)

    # Funding costs
    print(f"\n  Nota: Funding costs SHORT simulados = ${fund_b:,.2f} (0.03%/dia × {tr_b} dias)")
    print(f"  Em live, funding rate varia (positivo ou negativo). Monitorar diariamente.")


if __name__ == "__main__":
    main()
