"""
Backtest — EMA Trend + Reversal Hybrid (melhor dos dois mundos)
Jan 2025 → Abr 2026 (16 meses)

Lógica:
  BTC > EMA50D → LONG (compra e segura, vol-target sizing)
  BTC < EMA50D → REVERSAL (usa ReversalStrategy1H para capturar bounces)

Vantagens vs abordagens anteriores:
  vs Trend Long: não fica flat no bear — tenta ganhar com reversões
  vs Reversal puro: captura os bull markets que o reversal perde
  vs EMA+SHORT: não precisa de derivativos (compatível com BR)

Uso: docker exec cctb_app python3 /tmp/btemarev.py
"""
import json, math, sys
from pathlib import Path
from datetime import UTC, datetime
from calendar import monthrange
from collections import defaultdict

sys.path.insert(0, "/app")

CACHE   = Path("/app/data/cache")
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
INITIAL_CAPITAL = 96_592.87
CAPITAL_PER_SYM = INITIAL_CAPITAL / len(SYMBOLS)
FEE = 0.001

EMA_PERIOD = 50
ATR_PERIOD = 20
VOL_TARGET = 0.15
MAX_POS    = 0.33
ANN        = math.sqrt(365)

# Parâmetros Reversal (idênticos ao backtest reversal validado)
MIN_FALL     = 0.030
BASE_CANDLES = 6
TREND_FLOOR  = 0.92
TREND_CEIL   = 1.02
MIN_SL_PCT   = 0.008
MAX_SL_PCT   = 0.060
MIN_RATIO    = 1.5
TIMEOUT_H    = 48

MONTHS = [
    (2025,1),(2025,2),(2025,3),(2025,4),(2025,5),(2025,6),
    (2025,7),(2025,8),(2025,9),(2025,10),(2025,11),(2025,12),
    (2026,1),(2026,2),(2026,3),(2026,4),
]

def month_bounds(y, m):
    s = datetime(y, m, 1, tzinfo=UTC)
    e = datetime(y, m, monthrange(y,m)[1], 23, 59, 59, tzinfo=UTC)
    return int(s.timestamp()*1000), int(e.timestamp()*1000)

def to_daily(raw_1h):
    by_day = defaultdict(list)
    for c in sorted(raw_1h, key=lambda x: x["ts"]):
        day = (c["ts"]//86_400_000)*86_400_000
        by_day[day].append(c)
    return [{"ts":d,"close":g[-1]["close"],"high":max(x["high"]for x in g),
             "low":min(x["low"]for x in g)} for d,g in sorted(by_day.items()) for g in [by_day[d]]]

def calc_ema(values, period):
    out, k, acc = [], 2/(period+1), []
    for v in values:
        if math.isnan(v): out.append(float("nan")); continue
        acc.append(v)
        if len(acc)<period: out.append(float("nan"))
        elif len(acc)==period: out.append(sum(acc)/period)
        else: out.append(v*k+out[-1]*(1-k))
    return out

def calc_atr_pct(candles, period):
    trs, out = [], []
    for i,c in enumerate(candles):
        tr=(c["high"]-c["low"])/c["close"] if i==0 else \
           max(c["high"]-c["low"],abs(c["high"]-candles[i-1]["close"]),abs(c["low"]-candles[i-1]["close"]))/c["close"]
        trs.append(tr)
    k, acc = 1/period, []
    for tr in trs:
        acc.append(tr)
        if len(acc)<period: out.append(float("nan"))
        elif len(acc)==period: out.append(sum(acc)/period)
        else: out.append(tr*k+out[-1]*(1-k))
    return out

def reversal_signal(closes, highs, lows, volumes, sma20, current):
    """Retorna (sl_pct, tp_pct) se há sinal de reversão, ou None."""
    if not (sma20*TREND_FLOOR <= current <= sma20*TREND_CEIL):
        return None
    lh = max(closes[BASE_CANDLES+1:20])
    fl = min(closes[1:20])
    fp = (lh-fl)/lh if lh>0 else 0
    if fp < MIN_FALL: return None
    if fl > sma20*1.01: return None
    bh = max(highs[1:BASE_CANDLES+1])
    bl = min(lows[1:BASE_CANDLES+1])
    if current <= bh*1.001: return None
    av = sum(volumes[1:9])/8
    if av>0 and volumes[0]<av*0.8: return None
    sl_target = bl*0.999
    sl_dist = current - sl_target
    if sl_dist <= 0: return None
    sl_pct = sl_dist/current
    if not (MIN_SL_PCT <= sl_pct <= MAX_SL_PCT): return None
    return sl_pct, MIN_RATIO*sl_pct

def run():
    # Carrega dados
    warmup_ms = month_bounds(2025,1)[0] - 80*86_400_000
    end_ms    = month_bounds(2026,4)[1]
    warmup_1h = month_bounds(2025,1)[0] - 50*3_600_000

    daily, hourly = {}, {}
    for sym in SYMBOLS:
        key = sym.replace("-","_")
        raw = json.loads((CACHE/f"{key}_1H.json").read_text())
        daily[sym]  = to_daily([c for c in raw if warmup_ms<=c["ts"]<=end_ms])
        hourly[sym] = sorted([c for c in raw if warmup_1h<=c["ts"]<=end_ms], key=lambda c: c["ts"])

    # Indicadores diários
    ema50  = {s: calc_ema([c["close"] for c in daily[s]], EMA_PERIOD) for s in SYMBOLS}
    atr20d = {s: calc_atr_pct(daily[s], ATR_PERIOD) for s in SYMBOLS}
    sym_idx = {s: {c["ts"]:i for i,c in enumerate(daily[s])} for s in SYMBOLS}

    # Estado
    capital = INITIAL_CAPITAL
    pos      = {s: 0.0 for s in SYMBOLS}   # >0 = long, 0 = flat
    avg_px   = {s: 0.0 for s in SYMBOLS}
    rev_pos  = {s: None for s in SYMBOLS}   # {entry, sl, tp, ts_open}
    monthly  = {(y,m): {"pnl":0.,"fees":0.,"trades":0,"trend":0,"rev":0} for y,m in MONTHS}

    def get_ym(ts):
        d = datetime.fromtimestamp(ts/1000, tz=UTC)
        return (d.year, d.month)

    all_days = sorted({c["ts"] for c in daily["BTC-USDT"]})

    for day_ts in all_days:
        ym = get_ym(day_ts)
        if ym not in monthly: continue

        for sym in SYMBOLS:
            if day_ts not in sym_idx[sym]: continue
            idx   = sym_idx[sym][day_ts]
            e50   = ema50[sym][idx]
            atr_p = atr20d[sym][idx]
            price = daily[sym][idx]["close"]
            if math.isnan(e50) or math.isnan(atr_p) or price<=0: continue

            vol_ann    = atr_p * ANN
            trend_up   = price > e50
            vol_ok     = vol_ann < 0.80

            # ── MODO TREND (BTC > EMA50D) ─────────────────────────────────
            if trend_up and vol_ok:
                # Fecha qualquer posição reversal se estiver aberta
                if rev_pos[sym]:
                    entry = rev_pos[sym]["entry"]
                    pnl   = (price - entry) * rev_pos[sym]["qty"] * rev_pos[sym]["fill"]
                    fee   = rev_pos[sym]["qty"] * price * FEE
                    capital += rev_pos[sym]["qty"]*price*rev_pos[sym]["fill"] - fee + (pnl - rev_pos[sym]["qty"]*price*rev_pos[sym]["fill"])
                    monthly[ym]["pnl"] += pnl - fee
                    monthly[ym]["fees"] += fee
                    monthly[ym]["trades"] += 1
                    rev_pos[sym] = None

                # Sizing vol-target
                pos_pct    = min(VOL_TARGET/vol_ann, MAX_POS)
                total_eq   = capital + sum(pos[s]*daily[s][sym_idx[s].get(day_ts,0)]["close"]
                                           if day_ts in sym_idx[s] else 0 for s in SYMBOLS if pos[s]>0)
                target_qty = max(total_eq, INITIAL_CAPITAL*0.3) * pos_pct / price
                delta      = target_qty - pos[sym]

                if abs(delta)*price > capital*0.02:
                    if delta > 0 and capital > delta*price*(1+FEE):
                        fee = delta*price*FEE
                        avg_px[sym] = (avg_px[sym]*pos[sym]+price*delta)/(pos[sym]+delta) if pos[sym]>0 else price
                        pos[sym]   += delta
                        capital    -= delta*price + fee
                        monthly[ym]["fees"]   += fee
                        monthly[ym]["trades"] += 1
                        monthly[ym]["trend"]  += 1
                    elif delta < 0 and pos[sym] > 0:
                        sq  = min(abs(delta), pos[sym])
                        fee = sq*price*FEE
                        pnl = (price-avg_px[sym])*sq - fee
                        capital += sq*price - fee
                        pos[sym] -= sq
                        monthly[ym]["pnl"]    += pnl
                        monthly[ym]["fees"]   += fee
                        monthly[ym]["trades"] += 1
                        monthly[ym]["trend"]  += 1

            # ── MODO REVERSAL (BTC < EMA50D) ──────────────────────────────
            else:
                # Fecha posição trend se existir
                if pos[sym] > 0:
                    fee = pos[sym]*price*FEE
                    pnl = (price-avg_px[sym])*pos[sym] - fee
                    capital += pos[sym]*price - fee
                    monthly[ym]["pnl"]    += pnl
                    monthly[ym]["fees"]   += fee
                    monthly[ym]["trades"] += 1
                    pos[sym] = 0.0

                # Verifica/gerencia posição reversal aberta
                if rev_pos[sym]:
                    rp     = rev_pos[sym]
                    sl_p   = rp["entry"]*(1-rp["sl_pct"])
                    tp_p   = rp["entry"]*(1+rp["tp_pct"])
                    hi     = daily[sym][idx]["high"]
                    lo     = daily[sym][idx]["low"]
                    held_h = (day_ts - rp["ts_open"]) / 3_600_000

                    exit_reason = None
                    exit_price  = price
                    if lo <= sl_p:   exit_reason="sl"; exit_price=sl_p
                    elif hi >= tp_p: exit_reason="tp"; exit_price=tp_p
                    elif held_h >= TIMEOUT_H: exit_reason="timeout"

                    if exit_reason:
                        qty = rp["qty"]; entry = rp["entry"]
                        pnl = (exit_price-entry)*qty - exit_price*qty*FEE
                        capital += qty*exit_price*(1-FEE)
                        monthly[ym]["pnl"]    += pnl
                        monthly[ym]["fees"]   += exit_price*qty*FEE
                        monthly[ym]["trades"] += 1
                        rev_pos[sym] = None

                # Tenta abrir novo reversal (usa últimas 22 horas de candles 1H)
                if rev_pos[sym] is None:
                    h_candles = [c for c in hourly[sym] if c["ts"] <= day_ts][-22:]
                    if len(h_candles) >= 22:
                        closes_h  = [c["close"]  for c in reversed(h_candles)]
                        highs_h   = [c["high"]   for c in reversed(h_candles)]
                        lows_h    = [c["low"]    for c in reversed(h_candles)]
                        volumes_h = [c["volume"] for c in reversed(h_candles)]
                        sma20_h   = sum(closes_h[:20])/20
                        current_h = closes_h[0]

                        sig = reversal_signal(closes_h, highs_h, lows_h, volumes_h, sma20_h, current_h)
                        if sig:
                            sl_pct, tp_pct = sig
                            entry   = price  # executa no open do dia seguinte (aproximado)
                            notional = min(capital, INITIAL_CAPITAL*0.08)
                            qty      = notional / entry
                            fee_e    = qty*entry*FEE
                            if fee_e < capital:
                                capital -= fee_e
                                rev_pos[sym] = {
                                    "entry": entry, "qty": qty, "fill": 1.0,
                                    "sl_pct": sl_pct, "tp_pct": tp_pct,
                                    "ts_open": day_ts,
                                }
                                monthly[ym]["fees"]   += fee_e
                                monthly[ym]["trades"] += 1
                                monthly[ym]["rev"]    += 1

    # Fecha tudo no último dia
    last_ym = (2026,4)
    for sym in SYMBOLS:
        last_days = sorted(ts for ts in sym_idx[sym] if get_ym(ts)==last_ym)
        if not last_days: continue
        price = daily[sym][sym_idx[sym][last_days[-1]]]["close"]
        if pos[sym] > 0:
            fee = pos[sym]*price*FEE
            pnl = (price-avg_px[sym])*pos[sym]-fee
            capital += pos[sym]*price-fee
            monthly[last_ym]["pnl"] += pnl
            monthly[last_ym]["fees"] += fee
            pos[sym] = 0.0
        if rev_pos[sym]:
            rp = rev_pos[sym]
            fee = rp["qty"]*price*FEE
            pnl = (price-rp["entry"])*rp["qty"]-fee
            capital += rp["qty"]*price*(1-FEE)
            monthly[last_ym]["pnl"] += pnl
            monthly[last_ym]["fees"] += fee
            rev_pos[sym] = None

    return {"monthly": monthly, "final": capital}

def main():
    sep = "═"*74
    sep2 = "─"*74
    print(sep)
    print("  EMA Trend + Reversal Hybrid  |  Jan 2025 → Abr 2026  |  16 meses")
    print("  BTC>EMA50D → LONG (trend) | BTC<EMA50D → REVERSAL (bounce)")
    print(sep)

    res     = run()
    monthly = res["monthly"]
    final   = res["final"]
    total   = final - INITIAL_CAPITAL
    ret     = total/INITIAL_CAPITAL*100
    fees    = sum(d["fees"] for d in monthly.values())
    trades  = sum(d["trades"] for d in monthly.values())
    trend_t = sum(d["trend"] for d in monthly.values())
    rev_t   = sum(d["rev"] for d in monthly.values())
    pos_m   = sum(1 for d in monthly.values() if d["pnl"]>5)
    neg_m   = sum(1 for d in monthly.values() if d["pnl"]<-5)

    print(f"  {'Mês':<8} {'P&L':>10} {'Trades':>7} {'Trend':>6} {'Reversal':>9} Status")
    print(sep2)
    for y,m in MONTHS:
        ym = (y,m); d = monthly[ym]
        icon = "✅" if d["pnl"]>5 else "❌" if d["pnl"]<-5 else "➖"
        sign = "+" if d["pnl"]>=0 else ""
        print(f"  {icon} {y}/{m:02d}  {sign}${d['pnl']:>8.2f}  "
              f"{d['trades']:>5}  {d['trend']:>5}t  {d['rev']:>7}r")

    print(sep)
    print(f"  Capital final : ${final:>12,.2f}")
    print(f"  P&L total     : {'+'if total>=0 else ''}${abs(total):>10,.2f}  ({'+'if ret>=0 else ''}{ret:.2f}%)")
    print(f"  Total trades  : {trades} (trend:{trend_t} | reversal:{rev_t})")
    print(f"  Total fees    : ${fees:>10,.2f}")
    print(f"  Meses +/-     : {pos_m}/{neg_m}")
    print(sep)

    # Comparativo
    print("  COMPARATIVO — 16 meses")
    print(sep2)
    results = [
        ("Buy & Hold BTC",         -18.4, "—"),
        ("Reversal 1H puro",        -2.4, "4/16"),
        ("EMA Trend Long+Flat",     -1.4, "5/16"),
        ("EMA+Reversal Hybrid",     ret,  f"{pos_m}/16"),
    ]
    for name, r, pos in results:
        sign = "+" if r>=0 else ""
        print(f"  {name:<28} {sign}{r:.1f}%   {pos}")
    print(sep)

    # Veredicto
    print("  VEREDICTO")
    print(sep2)
    best = max([(-18.4,"B&H"),(-2.4,"Reversal"),(-1.4,"Trend"),( ret,"Hybrid")], key=lambda x: x[0])
    if ret > 0:
        print(f"  ✅ MELHOR DE TODOS — único com retorno POSITIVO (+{ret:.1f}%)")
        print(f"     Combinar EMA Trend + Reversal funciona em qualquer regime.")
    elif ret > -1.4:
        print(f"  ✅ MELHOR QUE TREND E REVERSAL PUROS ({ret:.1f}%)")
        print(f"     Híbrido supera as duas estratégias individuais.")
    elif ret > -2.4:
        print(f"  ⚠️  MARGINALMENTE MELHOR QUE REVERSAL — diferença pequena ({ret:.1f}% vs -2.4%)")
    else:
        print(f"  ❌ PIOR QUE REVERSAL PURO — revisar parâmetros")
    print(sep)

if __name__ == "__main__":
    main()
