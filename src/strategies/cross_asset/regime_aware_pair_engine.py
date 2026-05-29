"""
RegimeAwarePairEngine — Módulo v5.20 Phase B

Extensão do CrossAssetEngine com lógica de pares adaptada ao regime macro.

Problema do CrossAssetEngine original:
  - Bloqueia em BEAR_TREND/PANIC/HIGH_CORRELATION (3 dos 7 regimes)
  - Parâmetros fixos: mesmo SPREAD_THRESHOLD e MAX_HOLD para todos os regimes
  - Resultado: 43% do tempo o motor de pares fica completamente inativo

Solução — 3 modos distintos por regime:

  ─────────────────────────────────────────────────────────────────────
  MODO BULL   (TREND_EXPANSION, VOLATILITY_COMPRESSION)
  ─────────────────────────────────────────────────────────────────────
  Igual ao CrossAssetEngine mas com parâmetros adaptativos:
    - SPREAD_THRESHOLD reduzido em TREND_EXPANSION (mercado mais direcional)
    - MAX_HOLD aumentado (trends duram mais)
    - MAX_ALLOCATION ligeiramente maior (regime favorável = mais confiança)
  Coexiste com CrossAssetEngine: este faz o par "normal", RAPE faz pares
  com threshold adaptado que CrossAsset poderia perder.

  ─────────────────────────────────────────────────────────────────────
  MODO BEAR   (BEAR_TREND, PANIC_LIQUIDATION) ← LACUNA PRINCIPAL
  ─────────────────────────────────────────────────────────────────────
  Correlação arb em mercado em queda:
    - Em BEAR, TODOS os ativos caem mas em VELOCIDADES DIFERENTES
    - Long o mais resiliente (caindo menos = RS score mais alto em bear)
    - Short o mais fraco (caindo mais = RS score mais baixo em bear)
    - O alpha vem da DIVERGÊNCIA de velocidade de queda, não da direção
    - Spread mínimo maior (0.15) — bear = mais volátil, exige spread real
    - Hold máximo menor (24h) — bear = sem tempo para esperar reversão
    - Drawdown máximo menor (2%) — risco maior, SL mais apertado

  ─────────────────────────────────────────────────────────────────────
  MODO CHOP   (MEAN_REVERTING_CHOP, TREND_EXHAUSTION)
  ─────────────────────────────────────────────────────────────────────
  Complementa SectorPairDetector com timeframe mais longo (M7 RS):
    - Onde SectorPairDetector usa Z-score de preço (12h lookback),
      RAPE CHOP usa divergência de RS scores (M7) entre pares
    - Entrada quando RS diverge > 0.20 (um muito alto, outro muito baixo)
    - Direção INVERSA ao BULL: long laggard, short leader (reversão esperada)
    - Hold máximo: 12h (CHOP reverte rápido)
    - Threshold menor (0.10) — CHOP tem spreads menores por natureza

Coexistência com outros módulos:
  - CrossAssetEngine: faz pares em BULL com threshold fixo (complementar)
  - SectorPairDetector: Z-score de preço em CHOP (diferente do RS-based RAPE)
  - ShortSqueezeDetector: squeeze spots single-asset (diferente de pares)
  - RegimeAwarePairEngine: adiciona BEAR + adaptativos para os gaps

Execução:
  Mesma infraestrutura: spot long + swap short via OKX
  Máx 1 par ativo (RAPE é mais seletivo — regime-aware = alta convicção)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Símbolos e contratos ──────────────────────────────────────────────────────

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

SWAP_SYMBOLS = {
    "BTC-USDT": "BTC-USDT-SWAP",
    "ETH-USDT": "ETH-USDT-SWAP",
    "SOL-USDT": "SOL-USDT-SWAP",
}

SWAP_CONTRACT_SIZE = {
    "BTC-USDT-SWAP": 0.01,
    "ETH-USDT-SWAP": 0.1,
    "SOL-USDT-SWAP": 1.0,
}

# ── Modos de operação ─────────────────────────────────────────────────────────

MODE_BULL = "BULL"
MODE_BEAR = "BEAR"
MODE_CHOP = "CHOP"
MODE_IDLE = "IDLE"

# Mapeamento regime → modo
REGIME_TO_MODE: dict[str, str] = {
    "TREND_EXPANSION":        MODE_BULL,
    "VOLATILITY_COMPRESSION": MODE_BULL,
    "BEAR_TREND":             MODE_BEAR,
    "PANIC_LIQUIDATION":      MODE_BEAR,
    "MEAN_REVERTING_CHOP":    MODE_CHOP,
    "TREND_EXHAUSTION":       MODE_CHOP,
    "HIGH_CORRELATION_RISK":  MODE_IDLE,   # todos correlacionados = sem edge
}

# ── Parâmetros por modo ───────────────────────────────────────────────────────

MODE_PARAMS: dict[str, dict] = {
    MODE_BULL: {
        "spread_threshold":  0.09,    # < 0.12 do CrossAsset (mais oportunidades)
        "spread_exit":       0.03,
        "max_hold_hours":    96,      # > 72h (trends duram mais)
        "max_drawdown":      0.045,   # ligeiramente maior (regime favorável)
        "allocation_pct":   0.08,     # 8% (vs 10% CrossAsset — complementar)
        "reverse_direction": False,   # long líder, short laggard (tendência)
    },
    MODE_BEAR: {
        "spread_threshold":  0.15,    # maior — bear é mais volátil, exige spread real
        "spread_exit":       0.06,
        "max_hold_hours":    24,      # menor — bear não espera
        "max_drawdown":      0.02,    # apertado — risco maior em bear
        "allocation_pct":   0.05,     # 5% — posição menor em bear
        "reverse_direction": False,   # long mais resiliente, short mais fraco
    },
    MODE_CHOP: {
        "spread_threshold":  0.10,    # menor — CHOP tem spreads menores
        "spread_exit":       0.02,
        "max_hold_hours":    12,      # CHOP reverte rápido
        "max_drawdown":      0.025,
        "allocation_pct":   0.06,
        "reverse_direction": True,    # long laggard, short líder (reversão)
    },
}

# Poll
POLL_INTERVAL = 900   # 15 min


# ── Estado de posição ─────────────────────────────────────────────────────────

@dataclass
class RegimePairPosition:
    """Par ativo gerenciado pelo RegimeAwarePairEngine."""
    mode:              str         # MODE_BULL / MODE_BEAR / MODE_CHOP
    long_symbol:       str
    short_symbol:      str
    short_swap_sym:    str
    entry_long_price:  float
    entry_short_price: float
    long_qty:          float
    short_contracts:   int
    notional:          float
    entry_spread:      float
    long_eid:          str = ""
    short_eid:         str = ""
    opened_at:         datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def age_hours(self) -> float:
        return (datetime.now(UTC) - self.opened_at).total_seconds() / 3600

    def pnl_pct(self, long_px: float, short_px: float) -> float:
        long_ret  = (long_px  - self.entry_long_price)  / self.entry_long_price
        short_ret = (self.entry_short_price - short_px) / self.entry_short_price
        return (long_ret + short_ret) / 2


# ── Motor principal ───────────────────────────────────────────────────────────

class RegimeAwarePairEngine:
    """
    Motor de pares regime-adaptativo. Complementa CrossAssetEngine com:
      - Modo BEAR: correlação arb em queda (lacuna do CrossAsset)
      - Modo BULL: parâmetros adaptativos (threshold menor, hold maior)
      - Modo CHOP: RS-based reversion (diferente do Z-score do SectorPairDetector)

    Uso em TradingLoop:
        self._rape = RegimeAwarePairEngine(cache=self._cache, okx_client=self._okx)
        self._rape.set_portfolio_value(pv)
        await self._rape.start()
    """

    def __init__(self, cache, okx_client) -> None:
        self._cache           = cache
        self._okx             = okx_client
        self._portfolio_value = 0.0
        self._running         = False
        self._task: asyncio.Task | None = None
        self._position: RegimePairPosition | None = None
        self._current_mode    = MODE_IDLE

    # ── API pública ───────────────────────────────────────────────────────────

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._recover_position_from_cache()
        self._task = asyncio.create_task(self._loop(), name="regime_aware_pair_engine")
        logger.info("RegimeAwarePairEngine: started (BULL/BEAR/CHOP adaptive pairs)")

    async def _recover_position_from_cache(self) -> None:
        """
        Restaura posição aberta do Redis após restart.

        Se o processo reiniciar com par aberto no OKX, o estado em memória
        (_position) seria perdido. Lê o snapshot gravado em 'regime_pair:position'
        para reconstruir o estado e evitar posições órfãs.
        """
        try:
            raw = await self._cache.get("regime_pair:position")
            if not raw:
                return
            data = raw if isinstance(raw, dict) else json.loads(raw)
            # Reconstrói posição com dados mínimos para gestão de saída
            long_sym  = data.get("long")
            short_sym = data.get("short")
            mode      = data.get("mode", MODE_IDLE)
            notional  = float(data.get("notional", 0))
            opened_at_str = data.get("opened_at")
            if not long_sym or not short_sym or notional <= 0:
                return
            swap_sym = SWAP_SYMBOLS.get(short_sym)
            if not swap_sym:
                return
            opened_at = (
                datetime.fromisoformat(opened_at_str)
                if opened_at_str else datetime.now(UTC)
            )
            self._position = RegimePairPosition(
                mode=mode,
                long_symbol=long_sym,
                short_symbol=short_sym,
                short_swap_sym=swap_sym,
                entry_long_price=0.0,    # não conhecido — saída via spread/tempo
                entry_short_price=0.0,
                long_qty=notional / max(1.0, notional),   # approx: será recalculado
                short_contracts=1,
                notional=notional,
                entry_spread=float(data.get("spread", 0.0)),
                long_eid="recovered",
                short_eid="recovered",
            )
            # Corrige o opened_at para o valor original
            self._position.opened_at = opened_at
            self._current_mode = mode
            logger.info(
                "RegimeAwarePairEngine: posição recuperada do cache — "
                "LONG %s SHORT %s mode=%s notional=%.0f",
                long_sym, short_sym, mode, notional,
            )
        except Exception as exc:
            logger.warning("RegimeAwarePairEngine: falha ao recuperar posição do cache: %s", exc)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_status(self) -> dict:
        pos = self._position
        return {
            "mode":         self._current_mode,
            "active":       pos is not None,
            "long_symbol":  pos.long_symbol  if pos else None,
            "short_symbol": pos.short_symbol if pos else None,
            "mode_entry":   pos.mode         if pos else None,
            "notional":     pos.notional     if pos else None,
            "age_h":        round(pos.age_hours, 1) if pos else None,
        }

    # ── Loop principal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        await asyncio.sleep(90)   # aguarda M7 popular Redis
        while self._running:
            try:
                await self._evaluate()
            except Exception as exc:
                logger.warning("RegimeAwarePairEngine._evaluate error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    async def _evaluate(self) -> None:
        regime = await self._get_macro_regime()
        mode   = REGIME_TO_MODE.get(regime, MODE_IDLE)
        self._current_mode = mode

        if mode == MODE_IDLE:
            logger.debug("RegimeAwarePairEngine: regime=%s → IDLE", regime)
            # Fecha posição ativa se regime virou HIGH_CORRELATION_RISK
            if self._position:
                await self._close_position("regime_idle")
            return

        rs_scores = await self._get_rs_scores()
        if len(rs_scores) < 2:
            return

        prices = await self._get_prices()
        if len(prices) < 2:
            return

        params = MODE_PARAMS[mode]

        if self._position:
            await self._check_exit(rs_scores, prices, params)
        else:
            await self._check_entry(mode, regime, rs_scores, prices, params)

    # ── Entrada ───────────────────────────────────────────────────────────────

    async def _check_entry(
        self,
        mode: str,
        regime: str,
        rs_scores: dict[str, float],
        prices: dict[str, float],
        params: dict,
    ) -> None:
        ranked = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return

        # Determina long/short baseado no modo
        if params["reverse_direction"]:
            # CHOP: long laggard (menor RS), short líder (maior RS)
            long_sym,  long_score  = ranked[-1]   # menor RS = laggard = vai reverter
            short_sym, short_score = ranked[0]    # maior RS = líder = vai cair
        else:
            # BULL/BEAR: long mais forte, short mais fraco
            long_sym,  long_score  = ranked[0]    # maior RS = mais resiliente
            short_sym, short_score = ranked[-1]   # menor RS = mais fraco

        spread = abs(long_score - short_score)

        if spread < params["spread_threshold"]:
            logger.debug(
                "RegimeAwarePairEngine [%s]: spread=%.3f < %.3f — aguarda",
                mode, spread, params["spread_threshold"],
            )
            return

        long_px  = prices.get(long_sym)
        short_px = prices.get(short_sym)
        if not long_px or not short_px:
            return

        # Sizing
        if self._portfolio_value <= 0:
            return
        notional    = self._portfolio_value * params["allocation_pct"] / 2
        swap_sym    = SWAP_SYMBOLS.get(short_sym)
        cs          = SWAP_CONTRACT_SIZE.get(swap_sym, 1.0)
        if not swap_sym or cs <= 0:
            return
        long_qty        = notional / long_px
        short_contracts = max(1, int(notional / (short_px * cs)))

        logger.info(
            "RegimeAwarePairEngine [%s] ENTRADA: LONG %s(RS=%.3f) "
            "SHORT %s(RS=%.3f) spread=%.3f regime=%s notional=%.0f",
            mode, long_sym, long_score, short_sym, short_score,
            spread, regime, notional,
        )

        try:
            coid_l = f"rp_long_{uuid.uuid4().hex[:10]}"
            coid_s = f"rp_short_{uuid.uuid4().hex[:10]}"
            long_eid, short_eid = await asyncio.gather(
                self._okx.place_order(
                    symbol=long_sym, side="buy", order_type="market",
                    quantity=long_qty, price=None, client_order_id=coid_l,
                ),
                self._okx.place_swap_order(
                    symbol=swap_sym, side="sell", pos_side="short",
                    quantity=float(short_contracts), order_type="market",
                    client_order_id=coid_s,
                ),
            )
        except Exception as exc:
            logger.error("RegimeAwarePairEngine: falha ao abrir par: %s", exc)
            return

        self._position = RegimePairPosition(
            mode=mode,
            long_symbol=long_sym,
            short_symbol=short_sym,
            short_swap_sym=swap_sym,
            entry_long_price=long_px,
            entry_short_price=short_px,
            long_qty=long_qty,
            short_contracts=short_contracts,
            notional=notional,
            entry_spread=spread,
            long_eid=long_eid or coid_l,
            short_eid=short_eid or coid_s,
        )

        await self._cache.set(
            "regime_pair:position",
            json.dumps({
                "mode": mode, "regime": regime,
                "long": long_sym, "short": short_sym,
                "spread": spread, "notional": notional,
                "opened_at": self._position.opened_at.isoformat(),
            }),
            ttl=86400,
        )

    # ── Saída ─────────────────────────────────────────────────────────────────

    async def _check_exit(
        self,
        rs_scores: dict[str, float],
        prices: dict[str, float],
        params: dict,
    ) -> None:
        pos = self._position
        assert pos is not None

        long_px  = prices.get(pos.long_symbol)
        short_px = prices.get(pos.short_symbol)
        if not long_px or not short_px:
            return

        pnl_pct = pos.pnl_pct(long_px, short_px)

        # Spread atual: long_RS - short_RS (sempre positivo na entrada; pode inverter)
        current_spread = rs_scores.get(pos.long_symbol, 0.5) - rs_scores.get(pos.short_symbol, 0.5)

        # Em CHOP (reverse), spread esperado fica negativo quando reversão ocorre
        if pos.mode == MODE_CHOP:
            spread_closed = current_spread >= 0   # reverteu: laggard superou líder
        else:
            spread_closed = current_spread < params["spread_exit"]

        reasons: list[str] = []

        if spread_closed:
            reasons.append(f"spread_convergiu({current_spread:.3f})")

        if pos.age_hours > params["max_hold_hours"]:
            reasons.append(f"max_hold({pos.age_hours:.0f}h)")

        if pnl_pct < -params["max_drawdown"]:
            reasons.append(f"drawdown({pnl_pct:.1%})")

        # Mudança de modo (regime mudou)
        regime = await self._get_macro_regime()
        new_mode = REGIME_TO_MODE.get(regime, MODE_IDLE)
        if new_mode != pos.mode:
            reasons.append(f"modo_mudou({pos.mode}→{new_mode})")

        if not reasons:
            logger.debug(
                "RegimeAwarePairEngine [%s]: %s/%s ativo pnl=%.2f%% spread=%.3f age=%.1fh",
                pos.mode, pos.long_symbol, pos.short_symbol,
                pnl_pct * 100, current_spread, pos.age_hours,
            )
            return

        logger.info(
            "RegimeAwarePairEngine [%s] SAÍDA: %s/%s motivo=%s pnl=%.2f%%",
            pos.mode, pos.long_symbol, pos.short_symbol,
            ", ".join(reasons), pnl_pct * 100,
        )
        await self._close_position(", ".join(reasons))

    async def _close_position(self, reason: str = "") -> None:
        pos = self._position
        if not pos:
            return
        try:
            coid_cl = f"rp_cl_{uuid.uuid4().hex[:10]}"
            coid_cs = f"rp_cs_{uuid.uuid4().hex[:10]}"
            await asyncio.gather(
                self._okx.place_order(
                    symbol=pos.long_symbol, side="sell", order_type="market",
                    quantity=pos.long_qty, price=None, client_order_id=coid_cl,
                ),
                self._okx.place_swap_order(
                    symbol=pos.short_swap_sym, side="buy", pos_side="short",
                    quantity=float(pos.short_contracts), order_type="market",
                    client_order_id=coid_cs,
                ),
            )
        except Exception as exc:
            logger.error("RegimeAwarePairEngine: falha ao fechar par: %s", exc)
        finally:
            self._position = None
            await self._cache.delete("regime_pair:position")
            logger.info(
                "RegimeAwarePairEngine: par fechado — motivo='%s'", reason
            )

    # ── Dados de mercado ──────────────────────────────────────────────────────

    async def _get_rs_scores(self) -> dict[str, float]:
        """Busca scores M7 (relative strength) do Redis."""
        scores: dict[str, float] = {}
        for sym in SYMBOLS:
            try:
                raw = await self._cache.get(f"relative_strength:{sym}")
                if raw:
                    data  = raw if isinstance(raw, dict) else json.loads(raw)
                    score = data.get("m7_score") or data.get("score")
                    if score is not None:
                        scores[sym] = float(score)
            except Exception:
                pass
        return scores

    async def _get_prices(self) -> dict[str, float]:
        """Busca preços atuais do Redis."""
        prices: dict[str, float] = {}
        for sym in SYMBOLS:
            try:
                raw = await self._cache.get(f"ticker:{sym}")
                if raw:
                    data  = raw if isinstance(raw, dict) else json.loads(raw)
                    price = data.get("last") or data.get("mid") or data.get("close")
                    if price:
                        prices[sym] = float(price)
            except Exception:
                pass
        return {k: v for k, v in prices.items() if v > 0}

    async def _get_macro_regime(self) -> str:
        """Busca regime macro do BTC do Redis."""
        try:
            raw = await self._cache.get("signal:BTC-USDT")
            if raw:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                return data.get("regime", "")
        except Exception:
            pass
        return ""
