"""
ShortSqueezeDetector — Módulo v5.20 Phase B

Detecta condições de short squeeze iminente e abre LONG antes da aceleração.

O que é um short squeeze:
  1. Muitos traders estão SHORT num ativo (funding negativo / OI alto de shorts)
  2. O preço começa a subir contra a posição deles
  3. Shorts são forçados a comprar para cobrir → amplifica a alta
  4. O movimento pode ser violento e rápido (+5% a +20% em poucas horas)

Sinais que o detector combina (todos do Redis — sem chamadas extras à API):

  A. Funding reversão (M6):
     - Funding estava negativo (shorts dominantes) nos últimos ciclos
     - Agora está revertendo para neutro/positivo
     - Indica que shorts estão cobrindo / longs voltando

  B. OI queda com preço subindo (M6):
     - OI caindo = posições sendo fechadas = short covering
     - Se o preço sobe ao mesmo tempo → squeeze em andamento

  C. RS reversal (M7):
     - Ativo estava underperforming peers (laggard)
     - Agora começa a outperformar (líder reverso)
     - Indica rotação de capital voltando para o ativo

  D. Confirmação de preço:
     - Price change 1h > 0 (preço já começou a mover)
     - Evita entrar antes do gatilho real

Score de squeeze [0, 1]:
  squeeze_score = A×0.35 + B×0.30 + C×0.25 + D×0.10

  Entrada: squeeze_score ≥ SQUEEZE_THRESHOLD (0.65)

Execução:
  - Long spot no ativo com maior squeeze_score
  - Exit gerenciado pelo PositionMonitor (ExitPlan normal com ATR)
  - Sizing: SQUEEZE_ALLOC_PCT (4%) do portfolio
  - Máx 1 squeeze ativo por vez (eventos são raros e intensos)
  - Cooldown de 4h por símbolo após saída (evita reentrada prematura)

Regimes bloqueados:
  - BEAR_TREND, PANIC_LIQUIDATION (squeeze impossível em queda estrutural)

Diferença do MomentumStrategy:
  MomentumStrategy opera em TODOS os regimes com score geral.
  ShortSqueezeDetector é especializado: aguarda setup específico de squeeze
  e entra mais agressivamente (4% vs 2-3% Kelly normal).

Cache Redis lido:
  - futures_flow:{symbol}    → funding_rate, oi_change_pct, scores.funding, scores.oi_change
  - relative_strength:{symbol} → m7_score, momentum_1h
  - ticker:{symbol}          → last price
  - signal:{symbol}          → regime

Cache escrito:
  - squeeze:status           → estado atual para dashboard (TTL 1h)
  - squeeze:cooldown:{symbol} → cooldown após trade (TTL 4h)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ...persistence.strategy_trade_log import strategy_trade_log as _stl

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

# Score mínimo para disparar entrada
SQUEEZE_THRESHOLD    = 0.65

# Pesos dos componentes do squeeze_score
W_FUNDING_REVERSAL   = 0.35   # A: funding revertendo de negativo
W_OI_DROP            = 0.30   # B: OI caindo com preço subindo
W_RS_REVERSAL        = 0.25   # C: RS melhorando (laggard vira líder)
W_PRICE_CONFIRM      = 0.10   # D: preço já em alta

# Sizing e risco
SQUEEZE_ALLOC_PCT    = 0.04   # 4% do portfolio (squeeze = alta convicção)
MAX_ACTIVE_SQUEEZES  = 1      # só 1 por vez
SQUEEZE_ATR_MULT_SL  = 1.2    # SL = entry - 1.2× ATR
SQUEEZE_ATR_MULT_TP  = 3.0    # TP = entry + 3.0× ATR (squeeze pode ser violento)
SQUEEZE_ATR_FALLBACK = 0.02   # 2% do preço se ATR indisponível

# Cooldown após saída (evita reentrada em falso squeeze já esgotado)
COOLDOWN_HOURS       = 4
COOLDOWN_TTL         = COOLDOWN_HOURS * 3600

# Regimes que bloqueiam entrada
BLOCKED_REGIMES      = {"BEAR_TREND", "PANIC_LIQUIDATION"}

# Thresholds dos sub-scores
# A: funding reversão — funding_rate saindo de negativo
FUNDING_NEG_THRESHOLD  = -0.0001   # funding < -0.01% = estava negativo
FUNDING_REV_THRESHOLD  =  0.0000   # funding ≥ 0 = revertendo

# B: OI drop — queda de OI enquanto preço sobe
OI_DROP_THRESHOLD      = -1.0      # OI_change_pct < -1% = short covering

# C: RS reversal — m7 de baixo subindo
RS_LOW_THRESHOLD       = 0.40      # estava abaixo de 0.40 (underperformer)
RS_RISE_THRESHOLD      = 0.50      # agora acima de 0.50 (revertendo)

# D: confirmação de preço — m7.momentum_1h > 0
PRICE_CONFIRM_POSITIVE = 0.0

# Poll interval
POLL_INTERVAL          = 900   # 15 min (alinha com M6/M7 collectors)

# Cache
CACHE_KEY_STATUS    = "squeeze:status"
CACHE_KEY_COOLDOWN  = "squeeze:cooldown:{symbol}"
CACHE_TTL_STATUS    = 3600


# ── Posição ativa ─────────────────────────────────────────────────────────────

@dataclass
class SqueezePosition:
    """Estado de um trade de squeeze ativo."""
    symbol:       str
    entry_price:  float
    quantity:     float
    stop_loss:    float
    take_profit:  float
    squeeze_score: float
    exchange_oid:  str = ""
    opened_at:     datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def age_hours(self) -> float:
        return (datetime.now(UTC) - self.opened_at).total_seconds() / 3600

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price


# ── Motor principal ───────────────────────────────────────────────────────────

class ShortSqueezeDetector:
    """
    Detecta e opera short squeezes em BTC/ETH/SOL.
    Roda como background task independente.

    Uso em TradingLoop:
        self._squeeze = ShortSqueezeDetector(
            cache=self._cache,
            okx_client=self._okx,
        )
        self._squeeze.set_portfolio_value(pv)
        await self._squeeze.start()
    """

    def __init__(self, cache, okx_client) -> None:
        self._cache           = cache
        self._okx             = okx_client
        self._portfolio_value = 0.0
        self._running         = False
        self._task: asyncio.Task | None = None
        self._active: dict[str, SqueezePosition] = {}
        # Histórico de funding por símbolo para detectar reversão
        self._funding_history: dict[str, list[float]] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="short_squeeze_detector")
        logger.info(
            "ShortSqueezeDetector: started (threshold=%.2f alloc=%.0f%%)",
            SQUEEZE_THRESHOLD, SQUEEZE_ALLOC_PCT * 100,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_status(self) -> dict:
        active = {
            sym: {
                "entry": pos.entry_price,
                "sl": pos.stop_loss, "tp": pos.take_profit,
                "score": pos.squeeze_score,
                "age_h": round(pos.age_hours, 1),
            }
            for sym, pos in self._active.items()
        }
        return {"active_squeezes": len(self._active), "positions": active}

    # ── Loop principal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        await asyncio.sleep(150)   # aguarda M6/M7 coletores popularem Redis
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.warning("ShortSqueezeDetector._tick error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        # 1. Verifica saídas das posições ativas
        for symbol in list(self._active.keys()):
            await self._check_exit(symbol)

        # 2. Verifica entradas se slots disponíveis
        if len(self._active) >= MAX_ACTIVE_SQUEEZES:
            return

        regime = await self._get_macro_regime("BTC-USDT")
        if regime in BLOCKED_REGIMES:
            logger.debug("ShortSqueezeDetector: bloqueado regime=%s", regime)
            return

        # Avalia squeeze_score para cada símbolo e ordena
        candidates: list[tuple[str, float, dict]] = []
        for symbol in SYMBOLS:
            if symbol in self._active:
                continue
            if await self._is_in_cooldown(symbol):
                continue
            m6  = await self._get_m6(symbol)
            m7  = await self._get_m7(symbol)
            if not m6 and not m7:
                continue
            score, components = self._compute_squeeze_score(symbol, m6, m7)
            if score >= SQUEEZE_THRESHOLD:
                candidates.append((symbol, score, components))

        if not candidates:
            return

        # Escolhe o símbolo com maior squeeze_score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sym, best_score, best_comp = candidates[0]
        await self._enter_squeeze(best_sym, best_score, best_comp)

    # ── Score de squeeze ──────────────────────────────────────────────────────

    def _compute_squeeze_score(
        self,
        symbol: str,
        m6: dict,
        m7: dict,
    ) -> tuple[float, dict]:
        """
        Computa squeeze_score [0, 1] a partir de M6 + M7.

        Retorna (score, componentes para log/debug).
        """
        components: dict[str, float] = {}

        # ── A. Funding Reversal (M6) ──────────────────────────────────────────
        # Detecta: funding estava negativo, agora revertendo para neutro/positivo
        funding_rate    = m6.get("funding_rate") or 0.0
        funding_history = self._funding_history.get(symbol, [])

        # Atualiza histórico local de funding
        if funding_rate:
            funding_history.append(funding_rate)
            if len(funding_history) > 6:   # mantém 6 ciclos (~48h)
                funding_history.pop(0)
            self._funding_history[symbol] = funding_history

        # Score A: funding vinha negativo e agora virou neutro/positivo
        had_negative = any(f < FUNDING_NEG_THRESHOLD for f in funding_history[:-1]) if len(funding_history) > 1 else False
        now_reversing = funding_rate >= FUNDING_REV_THRESHOLD

        if had_negative and now_reversing:
            # Quanto mais negativo era o fundo, mais forte o squeeze potencial
            min_historical = min(funding_history[:-1]) if len(funding_history) > 1 else 0.0
            reversal_depth = abs(min_historical) / 0.001   # normaliza por 0.1%
            score_a = min(1.0, 0.5 + reversal_depth * 0.5)
        elif now_reversing and funding_rate > 0:
            score_a = 0.6   # funding positivo sem histórico negativo = leve squeeze
        else:
            score_a = 0.1   # sem sinal de reversão
        components["funding_reversal"] = round(score_a, 3)

        # ── B. OI Drop + preço subindo (M6) ──────────────────────────────────
        oi_change_pct = m6.get("oi_change_pct") or 0.0
        oi_score_raw  = m6.get("scores", {}).get("oi_change", 0.5)
        m7_momentum   = m7.get("momentum_1h", 0.0) or 0.0

        # OI caindo enquanto preço sobe = short covering forçado
        if oi_change_pct <= OI_DROP_THRESHOLD and m7_momentum > PRICE_CONFIRM_POSITIVE:
            # Quanto mais OI caiu + mais o preço subiu = mais forte o squeeze
            oi_intensity    = min(1.0, abs(oi_change_pct) / 3.0)   # normaliza em 3%
            price_intensity = min(1.0, m7_momentum / 2.0)          # normaliza em +2%
            score_b = 0.5 + oi_intensity * 0.25 + price_intensity * 0.25
        elif oi_change_pct <= 0 and m7_momentum >= 0:
            score_b = 0.45   # neutro
        else:
            score_b = max(0.1, 1.0 - oi_score_raw)  # OI crescendo = não é squeeze
        components["oi_drop"] = round(score_b, 3)

        # ── C. RS Reversal (M7) ──────────────────────────────────────────────
        m7_score     = m7.get("m7_score", 0.5) or 0.5
        rs_prev_list = m7.get("rs_history", []) or []
        # Fallback: usa m7_score como proxy se histórico não disponível
        rs_prev      = rs_prev_list[-2] if len(rs_prev_list) >= 2 else (m7_score - 0.05)

        was_laggard    = rs_prev < RS_LOW_THRESHOLD
        now_recovering = m7_score >= RS_RISE_THRESHOLD

        if was_laggard and now_recovering:
            recovery_magnitude = (m7_score - rs_prev) / max(rs_prev, 0.01)
            score_c = min(1.0, 0.6 + recovery_magnitude * 0.4)
        elif m7_score > RS_RISE_THRESHOLD:
            score_c = 0.55   # RS acima da média mas sem reversão clara
        else:
            score_c = max(0.0, m7_score)   # abaixo de 0.5 = não indica squeeze
        components["rs_reversal"] = round(score_c, 3)

        # ── D. Confirmação de preço (M7 momentum_1h) ─────────────────────────
        if m7_momentum > 0.5:    # subindo > 0.5%
            score_d = min(1.0, 0.5 + m7_momentum / 2.0)
        elif m7_momentum > 0:
            score_d = 0.55
        else:
            score_d = 0.2    # preço não confirmou ainda
        components["price_confirm"] = round(score_d, 3)

        # ── Composição final ──────────────────────────────────────────────────
        score = (
            score_a * W_FUNDING_REVERSAL +
            score_b * W_OI_DROP +
            score_c * W_RS_REVERSAL +
            score_d * W_PRICE_CONFIRM
        )
        score = round(min(max(score, 0.0), 1.0), 4)
        components["total"] = score

        return score, components

    # ── Entrada ───────────────────────────────────────────────────────────────

    async def _enter_squeeze(
        self, symbol: str, score: float, components: dict
    ) -> None:
        price = await self._get_price(symbol)
        if not price:
            return

        # Sizing
        notional = self._portfolio_value * SQUEEZE_ALLOC_PCT
        if notional < 10:
            logger.debug("ShortSqueezeDetector: portfolio insuficiente para %s", symbol)
            return
        qty = notional / price

        # SL / TP via ATR sintético
        atr = price * SQUEEZE_ATR_FALLBACK
        sl  = round(price - atr * SQUEEZE_ATR_MULT_SL, 4)
        tp  = round(price + atr * SQUEEZE_ATR_MULT_TP, 4)

        logger.info(
            "ShortSqueezeDetector ENTRADA: %s @ %.4f score=%.3f "
            "components=%s sl=%.4f tp=%.4f qty=%.6f",
            symbol, price, score, components, sl, tp, qty,
        )

        # Coloca ordem spot LONG
        coid = f"sq_{uuid.uuid4().hex[:12]}"
        try:
            eid = await self._okx.place_order(
                symbol=symbol,
                side="buy",
                order_type="market",
                quantity=qty,
                price=None,
                client_order_id=coid,
            )
        except Exception as exc:
            logger.error("ShortSqueezeDetector: falha ao abrir %s: %s", symbol, exc)
            return

        pos = SqueezePosition(
            symbol=symbol,
            entry_price=price,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            squeeze_score=score,
            exchange_oid=eid or coid,
        )
        self._active[symbol] = pos

        await self._persist_status()
        logger.info(
            "ShortSqueezeDetector: %s LONG aberto — eid=%s score=%.3f",
            symbol, eid, score,
        )

    # ── Saída ─────────────────────────────────────────────────────────────────

    async def _check_exit(self, symbol: str) -> None:
        pos = self._active.get(symbol)
        if not pos:
            return

        price = await self._get_price(symbol)
        if not price:
            return

        pnl_pct = pos.pnl_pct(price)
        reasons: list[str] = []

        # A. Take Profit atingido
        if price >= pos.take_profit:
            reasons.append(f"tp({price:.2f}>={pos.take_profit:.2f})")

        # B. Stop Loss atingido
        if price <= pos.stop_loss:
            reasons.append(f"sl({price:.2f}<={pos.stop_loss:.2f})")

        # C. Squeeze esgotado: score voltou abaixo do limiar
        m6  = await self._get_m6(symbol)
        m7  = await self._get_m7(symbol)
        if m6 or m7:
            current_score, _ = self._compute_squeeze_score(symbol, m6 or {}, m7 or {})
            if current_score < SQUEEZE_THRESHOLD * 0.7:   # 30% de margem
                reasons.append(f"squeeze_esgotado(score={current_score:.3f})")

        # D. Regime virou bearish
        regime = await self._get_macro_regime(symbol)
        if regime in BLOCKED_REGIMES:
            reasons.append(f"regime_{regime}")

        # E. Max hold: 12h (squeeze é curto por natureza)
        if pos.age_hours >= 12:
            reasons.append(f"max_hold({pos.age_hours:.1f}h)")

        if not reasons:
            logger.debug(
                "ShortSqueezeDetector: %s ativo pnl=%.2f%% age=%.1fh",
                symbol, pnl_pct * 100, pos.age_hours,
            )
            return

        logger.info(
            "ShortSqueezeDetector SAÍDA: %s motivo=%s pnl=%.2f%%",
            symbol, ", ".join(reasons), pnl_pct * 100,
        )
        await self._close_squeeze(symbol, pos, reason=", ".join(reasons))

    async def _close_squeeze(
        self, symbol: str, pos: SqueezePosition, reason: str = ""
    ) -> None:
        """Fecha o LONG de squeeze via venda spot."""
        coid = f"sq_close_{uuid.uuid4().hex[:12]}"
        exit_price: float | None = None
        try:
            await self._okx.place_order(
                symbol=symbol,
                side="sell",
                order_type="market",
                quantity=pos.quantity,
                price=None,
                client_order_id=coid,
            )
            exit_price = await self._get_price(symbol)
        except Exception as exc:
            logger.error("ShortSqueezeDetector: falha ao fechar %s: %s", symbol, exc)
            return

        # Persiste trade no strategy_trades
        pnl_pct_val  = pos.pnl_pct(exit_price) if exit_price else None
        pnl_usdt_val = (pnl_pct_val * pos.quantity * pos.entry_price) if pnl_pct_val is not None else None
        await _stl.log_trade(
            strategy_id="short_squeeze",
            symbol=symbol,
            side="long",
            entry_price=pos.entry_price,
            exit_price=exit_price,
            notional=pos.quantity * pos.entry_price,
            pnl_pct=pnl_pct_val,
            pnl_usdt=pnl_usdt_val,
            reason=reason or "manual_close",
            opened_at=pos.opened_at,
            closed_at=datetime.now(UTC),
            extra={"squeeze_score": pos.squeeze_score},
        )

        self._active.pop(symbol, None)

        # Registra cooldown para evitar reentrada prematura
        await self._cache.set(
            CACHE_KEY_COOLDOWN.format(symbol=symbol),
            "1",
            ttl=COOLDOWN_TTL,
        )

        await self._persist_status()
        logger.info(
            "ShortSqueezeDetector: %s fechado (%.1fh hold) pnl=%.2f%%",
            symbol, pos.age_hours, (pnl_pct_val or 0) * 100,
        )

    # ── Dados de mercado ──────────────────────────────────────────────────────

    async def _get_m6(self, symbol: str) -> dict:
        """Lê dados M6 (FuturesFlow) do Redis."""
        try:
            raw = await self._cache.get(f"futures_flow:{symbol}")
            if raw:
                return raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            pass
        return {}

    async def _get_m7(self, symbol: str) -> dict:
        """Lê dados M7 (RelativeStrength) do Redis."""
        try:
            raw = await self._cache.get(f"relative_strength:{symbol}")
            if raw:
                return raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            pass
        return {}

    async def _get_price(self, symbol: str) -> float | None:
        """Lê preço atual do Redis."""
        try:
            raw = await self._cache.get(f"ticker:{symbol}")
            if raw:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                p = data.get("last") or data.get("mid") or data.get("close")
                if p:
                    return float(p)
        except Exception:
            pass
        try:
            t = await self._okx.get_ticker(symbol)
            if t:
                return float(t.last or t.mid or 0) or None
        except Exception:
            pass
        return None

    async def _get_macro_regime(self, symbol: str) -> str:
        """Lê regime macro do Redis."""
        try:
            raw = await self._cache.get(f"signal:{symbol}")
            if raw:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                return data.get("regime", "")
        except Exception:
            pass
        return ""

    async def _is_in_cooldown(self, symbol: str) -> bool:
        """Verifica se o símbolo está no período de cooldown pós-saída."""
        try:
            val = await self._cache.get(CACHE_KEY_COOLDOWN.format(symbol=symbol))
            return bool(val)
        except Exception:
            return False

    async def _persist_status(self) -> None:
        """Persiste estado resumido no Redis para o dashboard."""
        try:
            payload = {
                "active_squeezes": len(self._active),
                "positions": [
                    {
                        "symbol":  sym,
                        "entry":   round(pos.entry_price, 2),
                        "score":   pos.squeeze_score,
                        "age_h":   round(pos.age_hours, 1),
                        "pnl_pct": 0.0,   # atualizado no check_exit
                    }
                    for sym, pos in self._active.items()
                ],
                "updated_at": datetime.now(UTC).isoformat(),
            }
            await self._cache.set(CACHE_KEY_STATUS, json.dumps(payload), ttl=CACHE_TTL_STATUS)
        except Exception as exc:
            logger.debug("ShortSqueezeDetector: persist_status error: %s", exc)
