"""
regime_backtest.py — Calibração de pesos por regime via simulação histórica.

Roda BacktestEngine para cada estratégia contra 6+ meses de candles históricos,
segmenta os trades por regime e calcula pesos ótimos por (strategy, regime).

Salva em data/models/regime_weights.json — lido pelo WeightEngine em produção.

Uso:
  python scripts/regime_backtest.py
  python scripts/regime_backtest.py --symbols BTC-USDT ETH-USDT SOL-USDT --months 6

Proteção anti-overfitting (WFO):
  IS = primeiros 70% dos candles (calibração)
  OOS = últimos 30% (validação)
  Pesos só são aplicados se IS Sharpe > 0.3 E degradação IS→OOS < 50%
"""

import asyncio
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Garante que src/ está no path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.models import Candle  # noqa: E402
from src.replay.backtest_engine import BacktestEngine, _detect_regime  # noqa: E402
from src.strategies.momentum.momentum_strategy import MomentumStrategy  # noqa: E402
from src.strategies.reversal.reversal_strategy import ReversalStrategy1H as ReversalStrategy  # noqa: E402
from src.strategies.weight_engine import WeightEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regime_backtest")

MODELS_DIR   = ROOT / "data" / "models"
WEIGHTS_PATH = MODELS_DIR / "regime_weights.json"

# Configurações
SYMBOLS          = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
MONTHS_HISTORY   = 6
INITIAL_CAPITAL  = 10_000.0
IS_SPLIT         = 0.70    # 70% calibração / 30% validação
MIN_TRADES_VALID = 30      # mínimo de trades por regime para ser válido (significância estatística)
MIN_IS_SHARPE    = 0.20    # mínimo para considerar o regime calibrável
MIN_OOS_TRADES   = 10      # mínimo de trades OOS para validação de degradação

# Estratégias a simular — apenas as que operam em granularidade 1H.
# trend_v1 (TrendStrategy) usa candles diários agregados + EMA50D e lógica
# fundamentalmente diferente (Long/Flat vs Long/Short). Não é calibrável via
# este backtest de regime. Os pesos de trend_v1 no WeightEngine são derivados
# de domain knowledge (SIM_WEIGHTS_DEFAULT) e atualizados via record_trade() online.
STRATEGIES_1H = {
    "momentum_v2":  lambda syms: MomentumStrategy(symbols=syms),
    "reversal_v1":  lambda syms: ReversalStrategy(symbols=syms),
}


async def fetch_candles_from_db(symbol: str, months: int = 6) -> list[Candle]:
    """
    Tenta buscar candles históricos do PostgreSQL local.
    Se indisponível, usa OKX REST API como fallback.
    """
    try:
        from src.core.config import settings
        from src.persistence.postgres import Database
        db = Database(settings.database_url)
        await db.connect()
        cutoff = datetime.now(UTC) - timedelta(days=months * 30)
        rows = await db.fetch(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles_1h
            WHERE symbol = $1 AND timestamp >= $2
            ORDER BY timestamp ASC
            """,
            symbol, cutoff,
        )
        await db.disconnect()
        if rows and len(rows) >= 100:
            candles = [
                Candle(
                    symbol=symbol,
                    granularity="1H",
                    timestamp=r["timestamp"],
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                    confirmed=True,
                )
                for r in rows
            ]
            logger.info("%s: %d candles do PostgreSQL", symbol, len(candles))
            return candles
    except Exception as exc:
        logger.debug("DB indisponível (%s), usando OKX REST", exc)

    return await fetch_candles_from_okx(symbol, months)


async def fetch_candles_from_okx(symbol: str, months: int = 6) -> list[Candle]:
    """Busca candles históricos da OKX REST API (paginado)."""
    import httpx

    inst = symbol.replace("-", "-")
    url  = "https://www.okx.com/api/v5/market/history-candles"
    all_candles: list[Candle] = []
    after = ""
    cutoff = datetime.now(UTC) - timedelta(days=months * 30)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            params: dict = {"instId": inst, "bar": "1H", "limit": "100"}
            if after:
                params["after"] = after

            resp = await client.get(url, params=params)
            data = resp.json()
            rows = data.get("data", [])
            if not rows:
                break

            for r in rows:
                ts_ms = int(r[0])
                ts    = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                if ts < cutoff:
                    break
                all_candles.append(Candle(
                    symbol=symbol,
                    granularity="1H",
                    timestamp=ts,
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                    confirmed=True,
                ))
            else:
                # Se não quebrou no loop → continua paginando
                after = rows[-1][0]
                if datetime.fromtimestamp(int(after) / 1000, tz=UTC) < cutoff:
                    break
                continue
            break   # quebrou internamente → fim dos dados

    # OKX retorna newest-first → inverte para oldest-first
    all_candles.sort(key=lambda c: c.timestamp)
    logger.info("%s: %d candles da OKX REST (%.1f meses)", symbol, len(all_candles), months)
    return all_candles


def compute_sharpe(pnls: list[float]) -> float:
    """Sharpe anualizado simplificado dos retornos por trade."""
    if len(pnls) < 3:
        return 0.0
    avg = sum(pnls) / len(pnls)
    var = sum((p - avg) ** 2 for p in pnls) / len(pnls)
    std = var ** 0.5
    if std < 1e-9:
        return 0.0
    # Anualização: ~252 trades/ano é referência; usamos sqrt(len) como proxy
    return (avg / std) * (len(pnls) ** 0.5)


def wilson_ci95_lower(wins: int, n: int) -> float:
    """
    Limite inferior do intervalo de confiança de Wilson a 95% para proporções.
    Retorna a estimativa conservadora do win rate real.
    Se o CI inferior cruzar 0.5 (break-even para RR=1), o regime é duvidoso.
    """
    if n == 0:
        return 0.0
    import math
    z = 1.96  # 95% CI
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def pnl_to_weight(expectancy: float, win_rate: float, n: int, wins: int | None = None) -> float:
    """
    Converte métricas de performance em peso [0.05, 1.20].
    Estratégias com expectancy positiva e WR > 40% recebem peso maior.
    Usa CI95% de Wilson para penalizar estimativas com alta incerteza.
    """
    wr_score  = min(max(win_rate * 2, 0.0), 1.2)           # 0.5 WR → 1.0
    exp_score = min(max(expectancy * 20 + 0.5, 0.0), 1.2)  # exp > 0 → > 0.5
    base = (wr_score + exp_score) / 2

    # Penaliza incerteza: usa CI95% de Wilson como estimativa conservadora
    if wins is not None and n >= 5:
        ci_lower = wilson_ci95_lower(wins, n)
        # Se CI inferior < 0.35 (abaixo do WR calibrado mínimo esperado), reduz peso
        if ci_lower < 0.35:
            base *= 0.5
    else:
        # Sem CI (n < 5): penaliza pela falta de dados
        confidence = min(n / MIN_TRADES_VALID, 1.0)
        base *= confidence

    return max(round(base, 4), 0.05)


async def run_simulation(
    strategy_id: str,
    strategy_factory,
    symbol: str,
    candles: list[Candle],
) -> dict[str, dict]:
    """
    Roda o backtest e retorna resultados por regime.
    Aplica IS/OOS split para validação anti-overfitting.
    """
    if len(candles) < 50:
        return {}

    split = int(len(candles) * IS_SPLIT)
    is_candles  = candles[:split]
    oos_candles = candles[split:]

    results: dict[str, dict] = {}

    for label, cands in [("IS", is_candles), ("OOS", oos_candles)]:
        if len(cands) < 30:
            continue

        strategy = strategy_factory([symbol])
        engine   = BacktestEngine(
            strategy=strategy,
            symbol=symbol,
            initial_capital=INITIAL_CAPITAL,
            seed=42,
        )
        try:
            result = await engine.run(cands, warmup=21)
        except Exception as exc:
            logger.warning("%s %s %s: backtest falhou — %s", strategy_id, symbol, label, exc)
            continue

        # Segmenta trades por regime
        by_regime: dict[str, list[float]] = defaultdict(list)
        for trade in result.trades:
            # Detecta regime no momento do trade (candles antes da entrada)
            idx = next(
                (i for i, c in enumerate(cands) if c.timestamp >= trade.entry_time),
                len(cands) - 1,
            )
            history_slice = list(reversed(cands[:idx + 1]))
            regime = _detect_regime(history_slice) if len(history_slice) >= 20 else "UNKNOWN"
            by_regime[regime].append(trade.pnl)

        for regime, pnls in by_regime.items():
            if len(pnls) < 2:
                continue
            wr  = sum(1 for p in pnls if p > 0) / len(pnls)
            exp = sum(pnls) / len(pnls)
            shr = compute_sharpe(pnls)

            if regime not in results:
                results[regime] = {}
            results[regime][label] = {
                "n":          len(pnls),
                "win_rate":   round(wr, 4),
                "expectancy": round(exp, 4),
                "sharpe":     round(shr, 4),
                "pnl_total":  round(sum(pnls), 2),
            }

    return results


async def main() -> None:
    logger.info("=" * 60)
    logger.info("REGIME BACKTEST — calibração de pesos por regime")
    logger.info("Estratégias: %s", list(STRATEGIES_1H.keys()))
    logger.info("Símbolos: %s | Histórico: %d meses", SYMBOLS, MONTHS_HISTORY)
    logger.info("IS split: %.0f%% | OOS: %.0f%%", IS_SPLIT * 100, (1 - IS_SPLIT) * 100)
    logger.info("=" * 60)

    weight_engine = WeightEngine(WEIGHTS_PATH)
    all_results: dict[str, dict[str, dict[str, dict]]] = {}
    # all_results[strategy_id][symbol][regime] = {IS: {...}, OOS: {...}}

    # Coleta candles históricos
    candles_by_symbol: dict[str, list[Candle]] = {}
    for symbol in SYMBOLS:
        logger.info("Buscando candles de %s...", symbol)
        candles_by_symbol[symbol] = await fetch_candles_from_db(symbol, MONTHS_HISTORY)

    # Roda simulação por estratégia × símbolo
    for strategy_id, factory in STRATEGIES_1H.items():
        all_results[strategy_id] = {}
        logger.info("\n── Simulando %s ─────────────────────", strategy_id)

        for symbol in SYMBOLS:
            candles = candles_by_symbol.get(symbol, [])
            if not candles:
                logger.warning("  %s: sem candles disponíveis", symbol)
                continue

            logger.info("  %s × %s (%d candles)...", strategy_id, symbol, len(candles))
            sim_results = await run_simulation(strategy_id, factory, symbol, candles)
            all_results[strategy_id][symbol] = sim_results

    # Agrega resultados por (strategy, regime) através de todos os símbolos
    # e calcula pesos finais
    logger.info("\n── Calculando pesos ─────────────────────")

    strategy_regime_agg: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for strat_id, sym_results in all_results.items():
        for symbol, regime_results in sym_results.items():
            for regime, splits in regime_results.items():
                agg = strategy_regime_agg[strat_id][regime]
                for label, metrics in splits.items():
                    if label not in agg:
                        agg[label] = {"n": 0, "pnl_total": 0.0, "wins": 0}
                    agg[label]["n"]         += metrics["n"]
                    agg[label]["pnl_total"] += metrics["pnl_total"]
                    agg[label]["wins"]      += int(metrics["win_rate"] * metrics["n"])

    # Aplica pesos no WeightEngine
    applied = 0
    skipped = 0

    for strat_id, regime_agg in strategy_regime_agg.items():
        for regime, splits in regime_agg.items():
            is_data = splits.get("IS", {})
            oos_data = splits.get("OOS", {})

            n_is = is_data.get("n", 0)
            if n_is < MIN_TRADES_VALID:
                logger.warning("  SKIP %s/%s: apenas %d trades IS (mínimo %d)",
                               strat_id, regime, n_is, MIN_TRADES_VALID)
                skipped += 1
                continue

            wr_is  = is_data.get("wins", 0) / max(n_is, 1)
            exp_is = is_data.get("pnl_total", 0) / max(n_is, 1)
            shr_is = compute_sharpe([is_data.get("pnl_total", 0) / max(n_is, 1)] * n_is)

            wins_is = is_data.get("wins", 0)
            ci_lower = wilson_ci95_lower(wins_is, n_is)

            if shr_is < MIN_IS_SHARPE:
                # Sem edge suficiente no IS → peso baixo (não bloqueia, apenas reduz)
                weight = 0.10
                reason = f"IS Sharpe={shr_is:.2f} < {MIN_IS_SHARPE}"
            else:
                weight = pnl_to_weight(exp_is, wr_is, n_is, wins=wins_is)
                reason = (f"IS Sharpe={shr_is:.2f} wr={wr_is:.1%} "
                          f"exp={exp_is:.2f} CI95_lower={ci_lower:.1%}")

            # Validação OOS
            oos_verdict = "N/A"
            n_oos = oos_data.get("n", 0)
            if n_oos >= MIN_OOS_TRADES:
                exp_oos = oos_data.get("pnl_total", 0) / max(n_oos, 1)
                degradation = (exp_is - exp_oos) / abs(exp_is) if exp_is != 0 else 0
                if degradation > 0.70:
                    weight = max(weight * 0.5, 0.10)
                    oos_verdict = f"DEGRADED {degradation:.0%}"
                else:
                    oos_verdict = f"OK degradation={degradation:.0%}"

            weight_engine.update_from_simulation(
                regime=regime,
                strategy_id=strat_id,
                sim_weight=weight,
                sim_win_rate=wr_is,
                sim_expectancy=exp_is,
                n_sim_trades=n_is,
            )

            logger.info(
                "  ✓ %s/%s: weight=%.2f | %s | OOS=%s | n_is=%d n_oos=%d",
                strat_id, regime, weight, reason, oos_verdict, n_is, n_oos,
            )
            applied += 1

    logger.info("\n%s", "=" * 60)
    logger.info("Pesos aplicados: %d | Skipped: %d", applied, skipped)
    logger.info("Salvo em: %s", WEIGHTS_PATH)
    logger.info("=" * 60)

    # Exibe resumo final
    logger.info("\nRESUMO DOS PESOS POR REGIME:")
    summary = weight_engine.summary()
    for regime, strats in sorted(summary.items()):
        logger.info("  %s:", regime)
        for sid, info in sorted(strats.items()):
            logger.info(
                "    %-20s blended=%.2f  sim=%.2f  confidence=%s  n_real=%d",
                sid, info["blended"], info["sim"], info["confidence"], info["n_real"],
            )


if __name__ == "__main__":
    asyncio.run(main())
