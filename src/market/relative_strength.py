"""
RelativeStrengthCollector — calcula força relativa de cada símbolo vs BTC.

Conceito:
  RS(ETH, 5h) = retorno_ETH_5h / retorno_BTC_5h

  RS > 1  → ETH superando BTC no período → momentum relativo bullish
  RS < 1  → ETH ficando para trás → fraqueza relativa
  RS = 1  → paridade com BTC

Fator M7 compõe:
  1. RS ponderado multi-horizonte (1h, 5h, 24h)       50%
  2. BTC leadership score (BTC subindo = maré alta)   30%
  3. Trend de RS (RS melhorando nos últimos períodos)  20%

Para BTC:
  - RS = 1 por definição (benchmark de si mesmo)
  - Leadership = BTC vs média das alts (proxy de dominância)

Ciclo: a cada 15 minutos (acoplado ao FuturesFlowCollector).
Cache Redis: `relative_strength:{symbol}` TTL=1800s.
"""

import asyncio
import json
import logging
import math
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

POLL_INTERVAL = 900    # 15 minutos
REDIS_TTL     = 1800   # 30 minutos
BTC_SYMBOL    = "BTC-USDT"
SYMBOLS       = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

# Horizontes para cálculo de RS (em candles 1H)
RS_HORIZONS = [1, 5, 24]

# Pesos dos horizontes no RS ponderado
RS_WEIGHTS = {1: 0.40, 5: 0.35, 24: 0.25}


def _safe_return(closes: list[float], horizon: int) -> float | None:
    """Retorno simples entre closes[0] e closes[horizon]."""
    if len(closes) <= horizon or closes[horizon] <= 0:
        return None
    return (closes[0] - closes[horizon]) / closes[horizon]


def _rs_ratio(sym_ret: float | None, btc_ret: float | None) -> float | None:
    """
    Relative Strength ratio = sym_ret / btc_ret.
    Retorna None se qualquer entrada for None ou BTC_ret == 0.
    """
    if sym_ret is None or btc_ret is None:
        return None
    if abs(btc_ret) < 1e-8:
        return 1.0   # BTC parado → sem sinal direcional
    return sym_ret / btc_ret if btc_ret != 0 else None


def _score_rs(rs: float | None) -> float:
    """
    Converte RS ratio em score [0, 1].

    Lógica econômica:
      rs >> 1 → símbolo liderando BTC → score alto (momentum relativo forte)
      rs == 1 → neutro → 0.5
      rs << 1 → símbolo atrasando BTC → score baixo (fraqueza relativa)

    Usa sigmoid suavizada: score = 1 / (1 + exp(-k*(rs-1)))
    com k=8 para RS em [-0.5, +0.5] relativo a 1.0.
    """
    if rs is None:
        return 0.5
    # Centraliza em 1.0, aplica sigmoid
    x = (rs - 1.0) * 8.0
    return round(1.0 / (1.0 + math.exp(-x)), 4)


def _score_btc_leadership(btc_closes: list[float], alt_closes_list: list[list[float]]) -> float:
    """
    BTC leadership score [0, 1]:

    Mede se o BTC está subindo e liderando as alts.
    - BTC 1H positivo E alts 1H positivas → score alto (maré sobe todos)
    - BTC 1H positivo E alts negativas → BTC isolado, incerto
    - BTC 1H negativo → risk-off → score baixo

    Para BTC: verifica se BTC está outperformando as alts (dominância subindo).
    """
    if len(btc_closes) < 2:
        return 0.5

    btc_ret_1h = _safe_return(btc_closes, 1) or 0.0
    btc_ret_5h = _safe_return(btc_closes, 5) or 0.0

    # Base: BTC em alta?
    btc_score = 0.5 + btc_ret_1h * 20 + btc_ret_5h * 5
    btc_score = min(max(btc_score, 0.0), 1.0)

    # Confirmação: alts também em alta? (maré alta levanta todos)
    if alt_closes_list:
        alt_rets = []
        for alt_c in alt_closes_list:
            r = _safe_return(alt_c, 1)
            if r is not None:
                alt_rets.append(r)
        if alt_rets:
            avg_alt_ret = sum(alt_rets) / len(alt_rets)
            # Se BTC e alts subindo juntos → mercado saudável → bonus
            if btc_ret_1h > 0 and avg_alt_ret > 0:
                btc_score = min(btc_score + 0.1, 1.0)
            elif btc_ret_1h < 0:
                btc_score = max(btc_score - 0.1, 0.0)

    return round(btc_score, 4)


def _score_rs_trend(rs_history: list[float | None]) -> float:
    """
    Trend de RS: está o RS melhorando ou piorando nos últimos pontos?
    rs_history = [RS mais recente, RS anterior, ...]

    Retorna score [0.2, 0.8] baseado na direção do RS.
    """
    valid = [x for x in rs_history if x is not None]
    if len(valid) < 2:
        return 0.5
    # Diferença entre primeiro e último ponto (mais recente vs mais antigo)
    delta = valid[0] - valid[-1]
    # Mapeia delta de RS para score: +0.1 = improving, -0.1 = degrading
    score = 0.5 + delta * 3.0
    return round(min(max(score, 0.2), 0.8), 4)


class RelativeStrengthCollector:
    """
    Calcula e cacheia força relativa para alimentar M7 do MomentumStrategy.
    Acoplado ao MarketEngine — lê candles 1H já em memória.
    """

    def __init__(self, market, cache, symbols: list[str] = SYMBOLS) -> None:
        self._market  = market
        self._cache   = cache
        self._symbols = symbols
        self._task: asyncio.Task | None = None
        self._running = False

        # Histórico de RS por símbolo para calcular trend
        # {symbol: [rs_mais_recente, rs_anterior, ...]}  (max 3 pontos)
        self._rs_history: dict[str, list[float | None]] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._compute_all()
        self._task = asyncio.create_task(self._loop(), name="rs_collector")
        logger.info("RelativeStrengthCollector started for %d symbols", len(self._symbols))

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._compute_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("RelativeStrengthCollector error: %s", exc)

    async def _compute_all(self) -> None:
        # Busca candles BTC (benchmark)
        btc_candles = self._market.get_candles(BTC_SYMBOL, "1H")
        if len(btc_candles) < 25:
            logger.debug("RS: candles BTC insuficientes (%d)", len(btc_candles))
            return

        btc_closes = [c.close for c in btc_candles[:25]]

        # Alts para BTC leadership (todos exceto BTC)
        alt_closes_list = []
        for sym in self._symbols:
            if sym == BTC_SYMBOL:
                continue
            alt_c = self._market.get_candles(sym, "1H")
            if len(alt_c) >= 2:
                alt_closes_list.append([c.close for c in alt_c[:25]])

        # BTC leadership (igual para todos os símbolos no ciclo)
        leadership_score = _score_btc_leadership(btc_closes, alt_closes_list)

        for symbol in self._symbols:
            try:
                await self._compute_one(symbol, btc_closes, leadership_score)
            except Exception as exc:
                logger.debug("RS %s: %s", symbol, exc)

    async def _compute_one(
        self,
        symbol: str,
        btc_closes: list[float],
        leadership_score: float,
    ) -> None:
        sym_candles = self._market.get_candles(symbol, "1H")
        if len(sym_candles) < 25:
            return

        sym_closes = [c.close for c in sym_candles[:25]]

        # ── RS multi-horizonte ────────────────────────────────────────────────
        rs_by_horizon: dict[int, float | None] = {}
        scores_by_horizon: dict[int, float] = {}

        for h in RS_HORIZONS:
            sym_ret = _safe_return(sym_closes, h)
            btc_ret = _safe_return(btc_closes, h)

            if symbol == BTC_SYMBOL:
                # BTC vs média das alts — detecta dominância
                rs = 1.0   # neutro por definição
            else:
                rs = _rs_ratio(sym_ret, btc_ret)

            rs_by_horizon[h] = rs
            scores_by_horizon[h] = _score_rs(rs)

        # RS ponderado
        rs_weighted_score = sum(
            scores_by_horizon[h] * RS_WEIGHTS[h]
            for h in RS_HORIZONS
        )

        # ── RS Trend ─────────────────────────────────────────────────────────
        # Atualiza histórico (max 3 pontos para trend)
        current_rs_1h = rs_by_horizon.get(1)
        hist = self._rs_history.get(symbol, [])
        hist = [current_rs_1h] + hist[:2]   # insere na frente, mantém max 3
        self._rs_history[symbol] = hist
        rs_trend_score = _score_rs_trend(hist)

        # ── M7 Score Final ────────────────────────────────────────────────────
        m7_score = (
            rs_weighted_score * 0.50 +
            leadership_score  * 0.30 +
            rs_trend_score    * 0.20
        )

        # ── Retornos brutos para informação ──────────────────────────────────
        sym_ret_1h  = _safe_return(sym_closes, 1)
        sym_ret_24h = _safe_return(sym_closes, 24)
        btc_ret_1h  = _safe_return(btc_closes, 1)
        btc_ret_24h = _safe_return(btc_closes, 24)

        payload = {
            "symbol":         symbol,
            "btc_symbol":     BTC_SYMBOL,
            "rs_1h":          round(rs_by_horizon.get(1) or 1.0, 4),
            "rs_5h":          round(rs_by_horizon.get(5) or 1.0, 4),
            "rs_24h":         round(rs_by_horizon.get(24) or 1.0, 4),
            "sym_ret_1h_pct": round((sym_ret_1h or 0) * 100, 4),
            "sym_ret_24h_pct":round((sym_ret_24h or 0) * 100, 3),
            "btc_ret_1h_pct": round((btc_ret_1h or 0) * 100, 4),
            "btc_ret_24h_pct":round((btc_ret_24h or 0) * 100, 3),
            "scores": {
                "rs_1h":       round(scores_by_horizon.get(1, 0.5), 4),
                "rs_5h":       round(scores_by_horizon.get(5, 0.5), 4),
                "rs_24h":      round(scores_by_horizon.get(24, 0.5), 4),
                "rs_weighted": round(rs_weighted_score, 4),
                "leadership":  round(leadership_score, 4),
                "rs_trend":    round(rs_trend_score, 4),
                "m7":          round(m7_score, 4),
            },
            "updated_at": datetime.now(UTC).isoformat(),
        }

        await self._cache.set(
            f"relative_strength:{symbol}",
            json.dumps(payload),
            ttl=REDIS_TTL,
        )
        logger.debug(
            "RS %s — rs_1h=%.3f rs_24h=%.3f leadership=%.3f M7=%.3f",
            symbol, rs_by_horizon.get(1) or 1.0,
            rs_by_horizon.get(24) or 1.0,
            leadership_score, m7_score,
        )

    async def get_rs(self, symbol: str) -> dict | None:
        """Lê do Redis o dado de RS mais recente para um símbolo."""
        raw = await self._cache.get(f"relative_strength:{symbol}")
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return None
