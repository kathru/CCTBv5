"""
LongAndChill — Módulo v5.20 Phase C

Modo de hold estendido para sinais de alta convicção em TREND_EXPANSION.

Problema que resolve:
  O PositionMonitor já converte TP em Running Mode (trailing 2.5× ATR),
  mas o trailing ATR é calculado sobre a granularidade 1H. Em tendências
  fortes de vários dias, o ruído horário encosta no trailing e fecha a
  posição muito cedo, perdendo os movimentos de +6R, +8R, +12R que pagam
  dezenas de losses.

Solução — trailing baseado na SMA 6H:
  1. Quando um ExitPlan atinge running_mode E o sinal original tinha
     score ≥ CHILL_SCORE_THRESHOLD (0.72) em TREND_EXPANSION:
     → Ativa chill_mode no ExitPlan (flag novo no dataclass)
  2. A cada ciclo de 15 min (alinhado com o candle 6H do MarketEngine):
     → Calcula SMA(20) dos closes dos últimos 20 candles 6H
     → Trailing = max(trailing_atual, SMA_6H - 1.0 × ATR_6H)
     → Nunca abaixa o trailing (apenas sobe — ratchet)
  3. Enquanto o preço está bem acima da SMA, a posição continua aberta
     deixando tendências longas correrem dias ou semanas.

Regras de ativação:
  - entry_regime == "TREND_EXPANSION"
  - score do sinal original >= CHILL_SCORE_THRESHOLD (0.72)
  - Posição já em running_mode (TP convertido pelo PositionMonitor)
  - Não aplicado em outros regimes (CHOP/BEAR não têm tendências longas)

Trailing SMA 6H:
  - SMA_PERIOD = 20 candles 6H = ~5 dias de lookback
  - trail = SMA_20_6H - ATR_6H × ATR_CHILL_MULT (1.0)
  - Comparado com trailing ATR 1H atual — usa o MAIOR (mais proteção)
  - Se candles 6H insuficientes (< MIN_CANDLES = 10): mantém ATR 1H trail

Coexistência com PositionMonitor:
  - LongAndChill só ATUALIZA plan.trailing_stop e plan.chill_mode
  - Nunca abre nem fecha posições diretamente
  - A decisão de saída por trailing continua no PositionMonitor
  - SL absoluto do PositionMonitor continua intocável

Poll: a cada 15 minutos (alinhado com granularidade 6H do MarketEngine).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

# Score mínimo do sinal original para ativar chill mode
CHILL_SCORE_THRESHOLD = 0.72

# SMA 6H: 20 candles × 6h = 5 dias de lookback
SMA_PERIOD  = 20
MIN_CANDLES = 10   # mínimo para calcular SMA confiável

# Trailing = SMA_6H - ATR_6H × este multiplicador
# 1.0 = trail bem abaixo da SMA (posição tem espaço para respirar)
ATR_CHILL_MULT = 1.0

# Regime alvo (único — LongAndChill só faz sentido em tendência forte)
TARGET_REGIME = "TREND_EXPANSION"

# Poll interval (alinhado com M6/M7 e MarketEngine)
POLL_INTERVAL = 900   # 15 min

# Granularidade dos candles usada para SMA trailing
CANDLE_GRAN = "6H"


# ── Motor principal ───────────────────────────────────────────────────────────

class LongAndChill:
    """
    Atualiza o trailing stop de posições de alta convicção para seguir
    a SMA 6H, permitindo que winners corram mais tempo.

    Uso em TradingLoop:
        self._long_and_chill = LongAndChill(
            market=self._market,
            position_monitor=self._position_monitor,
        )
        await self._long_and_chill.start()
    """

    def __init__(self, market, position_monitor) -> None:
        self._market  = market
        self._pm      = position_monitor
        self._running = False
        self._task: asyncio.Task | None = None

    # ── API pública ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="long_and_chill")
        logger.info(
            "LongAndChill: started (score≥%.2f SMA%d-6H trail)",
            CHILL_SCORE_THRESHOLD, SMA_PERIOD,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Loop principal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        await asyncio.sleep(120)   # aguarda posições existentes serem restauradas
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.warning("LongAndChill._tick error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        """Avalia cada ExitPlan ativo e atualiza trailing se condições atendidas."""
        plans = self._pm._plans   # acesso direto ao dict — mesmo processo, sem lock

        for symbol, plan in list(plans.items()):

            # ── 1. Ativação de chill_mode ─────────────────────────────────────
            # Só ativa quando: running_mode já ligado + regime certo + score alto
            if not plan.running_mode:
                continue

            if plan.entry_regime != TARGET_REGIME:
                continue

            entry_score = float(
                (plan.signal_factors or {}).get("score", 0.0)
                or (plan.signal_factors or {}).get("calibrated_score", 0.0)
                or 0.0
            )

            if entry_score < CHILL_SCORE_THRESHOLD:
                continue

            # ── 2. Calcula SMA 6H e ATR 6H ───────────────────────────────────
            sma_trail = await self._compute_sma_trail(symbol)
            if sma_trail is None:
                # Candles insuficientes — mantém ATR 1H trail do PositionMonitor
                continue

            current_trail = plan.trailing_stop or 0.0

            if sma_trail <= current_trail:
                # SMA trail já está abaixo do trailing atual — sem atualização
                logger.debug(
                    "LongAndChill: %s SMA_trail=%.4f <= current=%.4f — mantém",
                    symbol, sma_trail, current_trail,
                )
                continue

            # ── 3. Atualiza trailing (ratchet: só sobe) ───────────────────────
            plan.trailing_stop      = sma_trail
            plan.trailing_activated = True
            if not plan.chill_mode:
                plan.chill_mode = True
                logger.info(
                    "LongAndChill: %s CHILL MODE ativado — "
                    "trailing %.4f → SMA6H_trail=%.4f (score=%.3f)",
                    symbol, current_trail, sma_trail, entry_score,
                )
            else:
                logger.debug(
                    "LongAndChill: %s trail atualizado %.4f → %.4f (SMA6H)",
                    symbol, current_trail, sma_trail,
                )

    # ── Cálculo SMA 6H ───────────────────────────────────────────────────────

    async def _compute_sma_trail(self, symbol: str) -> float | None:
        """
        Calcula o trailing stop baseado na SMA 6H.

        Returns:
            trail = SMA(close, 20, 6H) - ATR(14, 6H) × ATR_CHILL_MULT
            None se candles insuficientes.
        """
        candles = self._market.get_candles(symbol, CANDLE_GRAN, limit=SMA_PERIOD + 14)
        if len(candles) < MIN_CANDLES:
            return None

        closes = [c.close for c in candles]
        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]

        # SMA dos últimos SMA_PERIOD closes
        sma_closes = closes[-SMA_PERIOD:]
        if len(sma_closes) < MIN_CANDLES:
            return None
        sma = sum(sma_closes) / len(sma_closes)

        # ATR 6H (True Range médio)
        atr_6h = self._calc_atr(closes, highs, lows, period=14)
        if atr_6h <= 0:
            return None

        trail = sma - atr_6h * ATR_CHILL_MULT
        return round(trail, 4)

    @staticmethod
    def _calc_atr(
        closes: list[float],
        highs:  list[float],
        lows:   list[float],
        period: int = 14,
    ) -> float:
        """Calcula ATR como média dos True Ranges dos últimos `period` candles."""
        if len(closes) < 2:
            return 0.0
        trs: list[float] = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i]  - lows[i],
                abs(highs[i]  - closes[i - 1]),
                abs(lows[i]   - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return 0.0
        recent = trs[-period:]
        return sum(recent) / len(recent)
