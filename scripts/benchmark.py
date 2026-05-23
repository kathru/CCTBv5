"""
Benchmark — Jan 2025 → Abr 2026

Compara 4 abordagens para entender onde está o alpha:
  1. Buy & Hold BTC  (baseline passivo)
  2. Buy & Hold ETH  (baseline passivo)
  3. Trend Following 50 dias em BTC (estratégia simples provada)
  4. Nossas estratégias (reversal / híbrido)

Se não batemos o Buy & Hold -> não temos alpha.
Se não batemos o Trend Following simples -> nossa complexidade não agrega valor.

Uso: docker exec cctb_app python3 /tmp/bench.py
"""
import json
from pathlib import Path
from datetime import UTC, datetime
from calendar import monthrange

CACHE = Path("/app/data/cache")

MONTHS = [
    (2025,  1), (2025,  2), (2025,  3), (2025,  4),
    (2025,  5), (2025,  6), (2025,  7), (2025,  8),
    (2025,  9), (2025, 10), (2025, 11), (2025, 12),
    (2026,  1), (2026,  2), (2026,  3), (2026,  4),
]

# Resultados das nossas estratégias (já calculados)
OUR_REVERSAL = {
    (2025, 1): -249.07, (2025, 2): -316.05, (2025, 3): -468.25,
    (2025, 4):  240.68, (2025, 5): -219.25, (2025, 6):  167.20,
    (2025, 7): -229.17, (2025, 8): -399.72, (2025, 9): -267.42,
    (2025,10):   78.10, (2025,11): -303.07, (2025,12):   66.17,
    (2026, 1):  -14.10, (2026, 2): -108.63, (2026, 3): -121.85,
    (2026, 4): -131.89,
}
INITIAL_CAPITAL = 96_592.87


def month_bounds(y, m):
    s = datetime(y, m, 1, tzinfo=UTC)
    e = datetime(y, m, monthrange(y, m)[1], 23, 59, 59, tzinfo=UTC)
    return int(s.timestamp() * 1000), int(e.timestamp() * 1000)


def load_candles(sym, s_ms, e_ms, warmup_ms=0):
    raw = json.loads((CACHE / f"{sym}_1H.json").read_text())
    start = warmup_ms if warmup_ms else s_ms
    return sorted([c for c in raw if start <= c["ts"] <= e_ms], key=lambda c: c["ts"])


def trend_follow_50d(all_candles, month_s, month_e):
    """
    Trend following com SMA 1200H (≈50 dias).
    Regra: preço > SMA → long. Preço < SMA → flat.
    Conta retorno percentual de posições abertas/fechadas no mês.
    """
    month_candles = [c for c in all_candles if month_s <= c["ts"] <= month_e]
    if not month_candles:
        return 0.0

    SMA_PERIOD = 1200
    in_trade   = False
    entry_px   = 0.0
    pnl_pct    = 0.0

    for c in all_candles:
        idx = all_candles.index(c)
        if idx < SMA_PERIOD:
            continue
        sma = sum(x["close"] for x in all_candles[idx - SMA_PERIOD:idx]) / SMA_PERIOD
        price = c["close"]
        in_month = month_s <= c["ts"] <= month_e

        if price > sma * 1.001 and not in_trade:
            in_trade = True
            entry_px = price
        elif price < sma * 0.999 and in_trade:
            gain = (price - entry_px) / entry_px * 100
            if in_month:
                pnl_pct += gain
            in_trade = False

    # Fecha posição aberta no final do mês
    if in_trade and month_candles:
        gain = (month_candles[-1]["close"] - entry_px) / entry_px * 100
        pnl_pct += gain

    return pnl_pct


def main():
    sep  = "=" * 76
    sep2 = "-" * 76

    # Carrega todos os candles com warmup para trend following
    warmup_start = month_bounds(2025, 1)[0] - 1200 * 3_600_000
    all_end      = month_bounds(2026, 4)[1]
    btc_all = load_candles("BTC_USDT", warmup_start, all_end, warmup_start)
    eth_all = load_candles("ETH_USDT", warmup_start, all_end, warmup_start)
    sol_all = load_candles("SOL_USDT", warmup_start, all_end, warmup_start)

    print(sep)
    print("  BENCHMARK — Jan 2025 → Abr 2026  |  16 meses")
    print("  Quanto cada abordagem renderia com $96.592 inicial?")
    print(sep)
    print(f"  {'Mês':<8} {'B&H BTC':>8} {'B&H ETH':>8} {'B&H SOL':>8} "
          f"{'TF-50D':>8} {'Reversal':>9} {'Diff':>8}")
    print(sep2)

    bh_btc_total  = 0.0
    bh_eth_total  = 0.0
    bh_sol_total  = 0.0
    tf_total      = 0.0
    rev_total     = 0.0
    capital_bh    = INITIAL_CAPITAL
    capital_tf    = INITIAL_CAPITAL
    capital_rev   = INITIAL_CAPITAL
    eq_bh         = []
    eq_tf         = []
    eq_rev        = []

    for y, m in MONTHS:
        s_ms, e_ms = month_bounds(y, m)

        # Buy & Hold: retorno mês = (close_final - open_inicial) / open_inicial
        btc_m = [c for c in btc_all if s_ms <= c["ts"] <= e_ms]
        eth_m = [c for c in eth_all if s_ms <= c["ts"] <= e_ms]
        sol_m = [c for c in sol_all if s_ms <= c["ts"] <= e_ms]

        if not btc_m:
            continue

        bh_btc = (btc_m[-1]["close"] - btc_m[0]["open"]) / btc_m[0]["open"] * 100
        bh_eth = (eth_m[-1]["close"] - eth_m[0]["open"]) / eth_m[0]["open"] * 100 if eth_m else 0
        bh_sol = (sol_m[-1]["close"] - sol_m[0]["open"]) / sol_m[0]["open"] * 100 if sol_m else 0
        # Portfolio B&H = 1/3 cada
        bh_port = (bh_btc + bh_eth + bh_sol) / 3

        # Trend Following
        tf_pct = trend_follow_50d(btc_all, s_ms, e_ms)

        # Reversal
        rev_pnl = OUR_REVERSAL.get((y, m), 0.0)
        rev_pct = rev_pnl / INITIAL_CAPITAL * 100

        bh_btc_total += bh_btc
        bh_eth_total += bh_eth
        bh_sol_total += bh_sol
        tf_total     += tf_pct
        rev_total    += rev_pct

        # Equity curves
        capital_bh  *= (1 + bh_port / 100)
        capital_tf  *= (1 + tf_pct  / 100)
        capital_rev += rev_pnl
        eq_bh.append(capital_bh)
        eq_tf.append(capital_tf)
        eq_rev.append(capital_rev)

        diff = rev_pct - bh_port
        icon = ("✅" if rev_pct > bh_port else "❌")
        print(f"  {icon} {y}/{m:02d}  "
              f"{bh_btc:>+7.1f}%  {bh_eth:>+7.1f}%  {bh_sol:>+7.1f}%  "
              f"{tf_pct:>+7.1f}%  "
              f"{rev_pnl:>+8.0f}$  "
              f"{diff:>+7.1f}%")

    # Capital final
    rev_final = INITIAL_CAPITAL + sum(OUR_REVERSAL.values())
    bh_final  = eq_bh[-1] if eq_bh else INITIAL_CAPITAL
    tf_final  = eq_tf[-1] if eq_tf else INITIAL_CAPITAL

    bh_btc_price_start = [c for c in btc_all if month_bounds(2025,1)[0] <= c["ts"]][0]["open"]
    bh_btc_price_end   = [c for c in btc_all if c["ts"] <= month_bounds(2026,4)[1]][-1]["close"]
    bh_btc_ret         = (bh_btc_price_end - bh_btc_price_start) / bh_btc_price_start * 100

    print(sep)
    print("  CAPITAL FINAL (partindo de $96.592,87)")
    print(sep2)
    bh_ret_pct = (bh_final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    tf_ret_pct = (tf_final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    rev_ret_pct = (rev_final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    print(
        f"  Buy & Hold BTC puro   : ${INITIAL_CAPITAL * (1 + bh_btc_ret/100):>12,.2f}"
        f"  ({bh_btc_ret:>+.1f}%)"
    )
    print(f"  Buy & Hold Portf. 1/3 : ${bh_final:>12,.2f}  ({bh_ret_pct:>+.1f}%)")
    print(f"  Trend Follow 50D BTC  : ${tf_final:>12,.2f}  ({tf_ret_pct:>+.1f}%)")
    print(f"  Reversal 1H (nosso)   : ${rev_final:>12,.2f}  ({rev_ret_pct:>+.1f}%)")
    print(sep)

    # Análise por tipo de mês
    print("  ANÁLISE: QUANDO O MERCADO GANHA, GANHAMOS TAMBÉM?")
    print(sep2)
    both_up = both_down = mkt_up_us_down = mkt_down_us_up = 0
    for i, (y, m) in enumerate(MONTHS):
        btc_m = [c for c in btc_all if month_bounds(y,m)[0] <= c["ts"] <= month_bounds(y,m)[1]]
        if not btc_m: continue
        mkt = (btc_m[-1]["close"] - btc_m[0]["open"]) / btc_m[0]["open"]
        ours = OUR_REVERSAL.get((y, m), 0) / INITIAL_CAPITAL
        if mkt > 0 and ours > 0: both_up += 1
        elif mkt < 0 and ours < 0: both_down += 1
        elif mkt > 0 and ours < 0: mkt_up_us_down += 1
        elif mkt < 0 and ours > 0: mkt_down_us_up += 1

    print(f"  Mercado sobe, nós subimos  : {both_up:>2} meses  (ideal: captura bull)")
    print(f"  Mercado cai, nós caímos    : {both_down:>2} meses  (esperado: sem short)")
    print(f"  Mercado sobe, nós caímos   : {mkt_up_us_down:>2} meses  ← PROBLEMA: perdemos em alta")
    print(f"  Mercado cai, nós subimos   : {mkt_down_us_up:>2} meses  ← EDGE: ganhamos em queda")
    print(sep)

    # Diagnóstico final
    print("  DIAGNÓSTICO")
    print(sep2)
    print(f"  BTC caiu {bh_btc_ret:.1f}% no período → qualquer estratégia long-only")
    print(f"  tem vento contrário. Mesmo buy&hold perdeu.")
    print()
    print(f"  Trend Following 50D:")
    tf_ret = (tf_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    if tf_ret > (rev_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:
        print(f"  → SUPERA nossa estratégia em {tf_ret:.1f}% vs {rev_ret_pct:.1f}%")
        print(f"  → Uma regra simples (preço > SMA50D = long) é melhor que nossa complexidade")
    else:
        print(f"  → Nossa estratégia bate o trend following")
    print()
    print(f"  CONCLUSÃO:")
    if bh_btc_ret < 0:
        print(f"  O período Jan/25-Abr/26 foi de queda de BTC ({bh_btc_ret:.1f}%).")
        print(f"  Estratégias long-only SEMPRE vão perder em período de queda do ativo.")
        print(f"  A solução real é: ou aceitar isso, ou incluir operações SHORT.")
    print(sep)


if __name__ == "__main__":
    main()
