"""
autonomous_review.py -- Weekly Autonomous Review Agent CCTBv5

Roda toda segunda-feira 08:00 UTC via GitHub Actions.
Incorpora a logica de analise do daily_agent: coleta dados completos
de todos os endpoints, analisa M1-M9, portfolio, regime, alpha signals,
execution quality e posta relatorio estruturado no Discord.

Variaveis de ambiente:
  ANTHROPIC_API_KEY     chave Anthropic (Claude API)
  DISCORD_WEBHOOK_URL   webhook Discord para o canal #review
  ORACLE_API_BASE       URL base da API Oracle (ex: http://137.131.220.216:8001)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# -- Configuracao --------------------------------------------------------------

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
ORACLE_API_BASE     = os.environ.get("ORACLE_API_BASE", "http://137.131.220.216:8001").rstrip("/")

CLAUDE_MODEL = "claude-opus-4-7"   # modelo mais capaz para analise quantitativa
MAX_TOKENS   = 8192

M_FACTORS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "mc1", "mc2"]


# -- Coleta de dados ----------------------------------------------------------

async def fetch_all(base: str) -> dict:
    """Coleta todos os endpoints relevantes do Oracle."""
    endpoints = {
        # Core
        "health":              "/health",
        "version":             "/version",
        # Metrics
        "system":              "/api/metrics/system",
        "performance":         "/api/metrics/performance",
        "prices":              "/api/metrics/prices",
        # Portfolio e posicoes
        "portfolio":           "/api/portfolio/summary",
        "positions_open":      "/api/positions/open",
        # Ordens
        "orders_filled":       "/api/orders/filled",
        "orders_recent":       "/api/orders/recent",
        # Sinais
        "signals_log":         "/api/signals/log",
        "signals_calibration": "/api/signals/calibration",
        "signals_funnel":      "/api/signals/funnel",
        # Analytics
        "distribution":        "/api/analytics/distribution",
        "meta_regime":         "/api/analytics/meta_regime",
        "volatility_state":    "/api/analytics/volatility_state",
        "futures_flow":        "/api/analytics/futures_flow",
        "relative_strength":   "/api/analytics/relative_strength",
        "reality_check":       "/api/analytics/reality_check",
        "model_health":        "/api/analytics/model_health",
        "alpha_orthogonality": "/api/analytics/alpha_orthogonality",
        "quantitative":        "/api/analytics/quantitative",
        # Governance (Feature drift / PSI)
        "governance_drift":    "/api/governance/drift",
        "governance_live":     "/api/governance/live-stats",
        "governance_importance": "/api/governance/importance",
        # Watchdog
        "watchdog":            "/api/watchdog/status",
        # CrossAsset market-neutral
        "cross_asset":         "/api/cross_asset/status",
        # v5.20 Strategy Performance (edge alpha, WR, PF por estratégia)
        "strategy_performance": "/api/analytics/strategy_performance",
    }

    results: dict = {}
    async with httpx.AsyncClient(timeout=20) as client:
        for key, path in endpoints.items():
            try:
                resp = await client.get(f"{base}{path}")
                if resp.status_code == 200:
                    results[key] = resp.json()
                    logger.info("  v %s", key)
                else:
                    logger.warning("  x %s -- HTTP %d", key, resp.status_code)
                    results[key] = {}
            except Exception as exc:
                logger.warning("  x %s -- %s", key, exc)
                results[key] = {}

    return results


# -- Formatacao para Claude ---------------------------------------------------

def _safe_json(obj, limit=1500) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)[:limit]
    except Exception:
        return str(obj)[:limit]


def build_prompt(data: dict, week_start: str, week_end: str) -> str:
    health      = data.get("health", {})
    version     = data.get("version", {})
    _system     = data.get("system", {})  # noqa: F841 – coletado mas não exibido diretamente
    perf        = data.get("performance", {})
    portfolio   = data.get("portfolio", {})
    _signals_log = data.get("signals_log", {})  # noqa: F841 – usado via week_trades
    funnel      = data.get("signals_funnel", {})
    cal         = data.get("signals_calibration", {})
    dist        = data.get("distribution", {})
    meta        = data.get("meta_regime", {})
    vol         = data.get("volatility_state", {})
    ff          = data.get("futures_flow", {})
    rs          = data.get("relative_strength", {})
    rc          = data.get("reality_check", {})
    mh          = data.get("model_health", {})
    alpha       = data.get("alpha_orthogonality", {})
    drift       = data.get("governance_drift", {})
    live_stats  = data.get("governance_live", {})
    importance  = data.get("governance_importance", {})
    quant       = data.get("quantitative", {})
    positions   = data.get("positions_open", [])
    cross_asset = data.get("cross_asset", {})
    wd          = data.get("watchdog", {})

    orders = data.get("orders_filled", {})
    if isinstance(orders, dict):
        orders = orders.get("orders", [])
    if not isinstance(orders, list):
        orders = []

    # Filtra trades da semana
    cutoff = datetime.now(UTC) - timedelta(days=7)
    week_trades = []
    for o in orders:
        try:
            ts = datetime.fromisoformat(str(o.get("filled_at", "")).replace("Z", "+00:00"))
            if ts.replace(tzinfo=UTC) >= cutoff:
                week_trades.append({
                    "symbol":   o.get("symbol"),
                    "side":     o.get("side"),
                    "pnl":      o.get("pnl_usdt"),
                    "strategy": o.get("strategy_id"),
                    "regime":   o.get("regime"),
                    "date":     str(o.get("filled_at", ""))[:10],
                })
        except Exception:
            pass

    return f"""PERIODO DA ANALISE: {week_start} -> {week_end}
VERSAO DO SISTEMA: {version.get("version", "?")} | Fase: {version.get("minor", "?")} -- Alpha Orthogonality

=== SAUDE GERAL ===
{_safe_json(health, 400)}

=== METRICAS DE PERFORMANCE ===
{_safe_json(perf, 1200)}

=== PORTFOLIO ===
{_safe_json(portfolio, 800)}

=== TRADES DA SEMANA ({len(week_trades)} trades) ===
{_safe_json(week_trades, 1500)}

=== FUNIL DE SINAIS ===
{_safe_json(funnel, 600)}

=== CALIBRACAO (WR, Platt) ===
{_safe_json(cal, 600)}

=== DISTRIBUICAO DE TRADES ===
{_safe_json(dist, 800)}

=== META REGIME (M1-M9 regime context) ===
{_safe_json(meta, 800)}

=== ESTADO DE VOLATILIDADE (M8) ===
{_safe_json(vol, 600)}

=== FUTURES FLOW (M6) ===
{_safe_json(ff, 600)}

=== RELATIVE STRENGTH (M7) ===
{_safe_json(rs, 600)}

=== REALITY CHECK ===
{_safe_json(rc, 600)}

=== MODEL HEALTH (PSI, drift por feature) ===
{_safe_json(mh, 800)}

=== ALPHA ORTHOGONALITY (FD/LV/OD/MRM - Fase 18) ===
{_safe_json(alpha, 800)}

=== GOVERNANCE DRIFT (PSI por feature) ===
{_safe_json(drift, 800)}

=== GOVERNANCE LIVE STATS ===
{_safe_json(live_stats, 600)}

=== FEATURE IMPORTANCE ===
{_safe_json(importance, 600)}

=== ANALYTICS QUANTITATIVOS ===
{_safe_json(quant, 800)}

=== POSICOES ABERTAS (LONG spot) ===
{_safe_json(positions, 600)}

=== CROSS-ASSET ENGINE (market-neutral long/short) ===
{_safe_json(cross_asset, 400)}

=== WATCHDOGS ===
{_safe_json(wd, 400)}
"""


SYSTEM_PROMPT = """Voce e o Analista Quantitativo Senior do CCTBv5, sistema de trading algoritmico de criptomoedas.

ARQUITETURA ATUAL (v3.1.0 -- Composites + CrossAsset + SHORT):
- Motor Probabilistico: Platt sigmoid, WR baseline=37.5%, modelo FROZEN ate Day 90
- Score v3.1.0: M3=37% M6=20% M7=17% M8=5% M9=13% mc1=4% mc2=4%
  mc1=M1xM2 (RSI x Consistencia), mc2=M4xM3 (RegimeStrength x Volume)
- Fee correto: round-trip 0.25% (0.10% maker + 0.15% taker)
- Estrategias:
  * momentum_v2 (LONG) — principal, spot OKX
  * CrossAssetEngine (market-neutral) — long strongest RS / short weakest via SWAP perp
  * Bear SHORT: BEAR_TREND gera sinal SHORT via OKX swap, kelly fixo=3%
- Regimes: TREND_EXPANSION, VOLATILITY_COMPRESSION, MEAN_REVERTING_CHOP, TREND_EXHAUSTION,
           HIGH_CORRELATION_RISK, BEAR_TREND (→SHORT), PANIC_LIQUIDATION, TRANSITION
- SizingEngine: kelly = base x regime x vol_state x score x exceptional
  (drift_mult e calibration_mult REMOVIDOS — ja tratados pelo EdgeConditioner)
- EdgeConditioner: 4 gates (PSI canonico 6-bin, health, liquidity, WR PnL-based)
- DrawdownEngine: slope linear regressao sobre janela 30d de equity diario
  (aceleracao = slope < -0.5%/dia, nao mais 10 amostras a 15s)
- PSI canonico: 6 bins por percentil p10/p25/p50/p75/p90 (nao mais z-score)
- WR live: calculado via PnL real de ordens fechadas FIFO (nao taxa de execucao)
- WeightEngine: online learning sim->real, convergencia em 30 trades
- EdgeConditioner: PSI mult, Health mult, Liq mult, WR mult, Thr mult
- Portfolio Intelligence (Fase 15): edge ranking cross-asset, kelly proporcional
- Meta-learning de Regimes (Fase 16): softmax probabilistico, EMA alpha=0.35, threshold blendado
- Execution Intelligence (Fase 17): SmartOrderRouter maker/taker, SlippageTracker 20-trade rolling
- Alpha Orthogonality (Fase 18): FD (Funding Dislocation), LV (Liquidity Vacuum),
  OD (Overnight Drift), MRM (Mean Reversion Micro) -- kelly x[0.65-1.30], thr adj +-3%

FATORES M (v3.1.0):
- M1: RSI(14) — entra via mc1=M1xM2 (peso individual 0%)
- M2: Trend Consistency — entra via mc1=M1xM2 (peso individual 0%)
- M3: Directional Volume (buy vs sell pressure, 12 candles) — 37% do score
- M4: Regime Strength (SMA5 vs SMA20 + regime enum) — entra via mc2=M4xM3 (peso individual 0%)
- M5: Candle Structure — desativado (perm_imp=-0.001, peso=0%)
- M6: Futures Flow (funding rate + OI) — 20%
- M7: Relative Strength vs BTC/alts — 17%
- M8: Vol State (EXPANDING/TREND/COMPRESSED/MEAN_REVERTING/CHAOTIC) — 5%
- M9: News Sentiment (Fear&Greed + CoinGecko) — 13%
- mc1: M1 x M2 (RSI x Consistencia) — 4% (interacao RSI em tendencia firme)
- mc2: M4 x M3 (RegimeStrength x Volume) — 4% (volume confirma forca do regime)

REGRAS DO MODELO FROZEN:
- NAO sugerir mudancas nos pesos de features M1-M9/mc1/mc2 (frozen ate Day 90)
- Mudancas auto-aplicaveis: regime_weights (delta < 20%), feature_baseline
- Mudancas que requerem aprovacao humana: Platt, thresholds criticos, nova logica

ACTION_SPEC para cada pending_change (obrigatorio):
- Para mudanca em constante Python: {"type":"python_replace","file":"src/...","old":"linha exata atual","new":"linha exata nova"}
- Para mudanca em arquivo JSON: {"type":"json_patch","file":"data/...","patches":[{"op":"replace","path":"/chave","value":novo_valor}]}
- Use o campo "old" com a string EXATA como aparece no arquivo (incluindo espacos e comentarios)
- Se nao souber o arquivo exato, omita action_spec (o campo ficara ausente)

Responda em JSON estruturado conforme solicitado. Seja quantitativo e acionavel."""


def build_claude_message(data: dict, week_start: str, week_end: str) -> str:
    system_data_text = build_prompt(data, week_start, week_end)

    return f"""{system_data_text}

=== TAREFA: RELATORIO SEMANAL COMPLETO ===

Analise todos os dados acima e responda em JSON com EXATAMENTE esta estrutura:

{{
  "period": "{week_start} -> {week_end}",
  "overall_health": "SAUDAVEL|ATENCAO|CRITICO",
  "executive_summary": "Resumo executivo em 3-4 frases, quantitativo",

  "performance": {{
    "pnl_week_usdt": null,
    "win_rate_week": null,
    "trades_count": 0,
    "best_trade": "descricao",
    "worst_trade": "descricao",
    "sharpe_estimate": null,
    "assessment": "avaliacao qualitativa"
  }},

  "m_factors": {{
    "M3":  {{"status": "OK|ATENCAO|DEGRADADO", "weight": "37%", "observations": "...", "psi": null, "recommendation": "..."}},
    "M6":  {{"status": "...", "weight": "20%", "observations": "...", "psi": null, "recommendation": "..."}},
    "M7":  {{"status": "...", "weight": "17%", "observations": "...", "psi": null, "recommendation": "..."}},
    "M9":  {{"status": "...", "weight": "13%", "observations": "...", "psi": null, "recommendation": "..."}},
    "M8":  {{"status": "...", "weight": "5%",  "observations": "...", "psi": null, "recommendation": "..."}},
    "mc1": {{"status": "...", "weight": "4%",  "observations": "M1xM2 RSI x Consistency", "recommendation": "..."}},
    "mc2": {{"status": "...", "weight": "4%",  "observations": "M4xM3 RegimeStrength x Volume", "recommendation": "..."}},
    "M1":  {{"status": "...", "weight": "0% individual", "observations": "contribui via mc1", "recommendation": "..."}},
    "M2":  {{"status": "...", "weight": "0% individual", "observations": "contribui via mc1", "recommendation": "..."}},
    "M4":  {{"status": "...", "weight": "0% individual", "observations": "contribui via mc2", "recommendation": "..."}},
    "M5":  {{"status": "DESATIVADO", "weight": "0%", "observations": "perm_imp=-0.001", "recommendation": "manter desativado"}}
  }},

  "regime_analysis": {{
    "dominant_regime": "nome do regime dominante da semana",
    "regime_distribution": {{}},
    "threshold_mult_blended": null,
    "meta_confidence": null,
    "assessment": "impacto do regime na performance"
  }},

  "alpha_orthogonality": {{
    "active_signals": [],
    "boost_impact": "impacto estimado dos sinais ortogonais",
    "assessment": "eficacia dos sinais FD/LV/OD/MRM esta semana"
  }},

  "cross_asset": {{
    "engine_active": false,
    "current_position": null,
    "long_symbol": null,
    "short_symbol": null,
    "notional_usdt": null,
    "age_hours": null,
    "pnl_assessment": "avaliacao do PnL do par market-neutral",
    "spread_status": "status do spread RS (entry/exit threshold)"
  }},

  "execution_quality": {{
    "slippage_assessment": "avaliacao do slippage (SmartOrderRouter)",
    "maker_taker_ratio": "estimativa",
    "funnel_blockage": "analise do funil (regime_blocked vs liquidity_gate vs score)"
  }},

  "risk_assessment": {{
    "model_drift_psi": "resumo do PSI por feature",
    "calibration_health": "WR calibration status",
    "drawdown_current": null,
    "beta_portfolio": null,
    "watchdog_status": "OK|ATENCAO|CRITICO"
  }},

  "pending_changes": [
    {{
      "change_type": "threshold|param|logic|scoring_weight",
      "target": "arquivo ou componente",
      "description": "descricao da mudanca",
      "requires_human_approval": true,
      "priority": "HIGH|MEDIUM|LOW",
      "justification": "dados que justificam",
      "action_spec": {{
        "type": "python_replace",
        "file": "caminho/relativo/ao/projeto.py",
        "old": "linha exata atual no codigo",
        "new": "linha exata com a mudanca aplicada"
      }}
    }}
  ],

  "alerts": [
    {{
      "severity": "CRITICAL|WARNING|INFO",
      "message": "descricao do alerta",
      "action": "acao recomendada"
    }}
  ],

  "outlook_next_week": "perspectiva para a proxima semana baseada em dados (3-4 frases)",

  "discord_summary": "Resumo para Discord em markdown com emojis (max 1800 chars)"
}}

Retorne APENAS o JSON, sem texto antes ou depois."""


# -- Claude API ---------------------------------------------------------------

async def call_claude(data: dict, week_start: str, week_end: str) -> dict:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    user_message = build_claude_message(data, week_start, week_end)

    logger.info("Chamando Claude %s (max_tokens=%d)...", CLAUDE_MODEL, MAX_TOKENS)

    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
    )

    # Extrai o bloco de texto (ignora thinking blocks)
    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text.strip()
            break

    logger.info("Tokens: input=%d output=%d", response.usage.input_tokens, response.usage.output_tokens)

    # Remove code fences se presentes
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("JSON invalido do Claude: %s", exc)
        return {"error": str(exc), "raw": raw_text[:500]}


# -- Discord ------------------------------------------------------------------

async def post_discord(analysis: dict, data: dict) -> None:
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL nao configurado -- pulando")
        return

    health      = analysis.get("overall_health", "?")
    health_emoji = {"SAUDAVEL": "v", "ATENCAO": "o", "CRITICO": "x"}.get(health, "?")
    now_str     = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")

    perf  = analysis.get("performance", {})
    pnl   = perf.get("pnl_week_usdt")
    wr    = perf.get("win_rate_week")
    trades = perf.get("trades_count", 0)

    pnl_str = f"${pnl:+,.2f}" if pnl is not None else "N/A"
    wr_str  = f"{wr*100:.1f}%" if wr is not None else "N/A"

    # M-factor status resumido
    mf = analysis.get("m_factors", {})
    mf_status = " | ".join(
        f"**{m}**: {mf.get(m, {}).get('status', '?')}"
        for m in M_FACTORS
    )

    # Alertas criticos
    alerts = analysis.get("alerts", [])
    critical_alerts = [a for a in alerts if a.get("severity") == "CRITICAL"]
    alert_text = ""
    if critical_alerts:
        alert_text = "\n".join(f"x **{a['message']}**" for a in critical_alerts[:3])
        alert_text = f"\n\n**ALERTAS CRITICOS:**\n{alert_text}"

    # Alpha signals
    alpha = analysis.get("alpha_orthogonality", {})
    active = alpha.get("active_signals", [])
    alpha_text = f"\n\n**Alpha Signals ativos:** {', '.join(active) if active else 'nenhum'}" if active else ""

    discord_summary = analysis.get("discord_summary", analysis.get("executive_summary", ""))

    header = (
        f"**CCTBv5 -- Relatorio Semanal** | {now_str}\n"
        f"{health_emoji} **{health}** | P&L: **{pnl_str}** | WR: **{wr_str}** | Trades: **{trades}**\n"
        f"{mf_status}"
        f"{alert_text}"
        f"{alpha_text}\n\n"
    )

    full_msg = header + discord_summary
    if len(full_msg) > 1990:
        full_msg = full_msg[:1987] + "..."

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(DISCORD_WEBHOOK_URL, json={"content": full_msg, "username": "CCTBv5 Weekly Review"})
        if resp.status_code in (200, 204):
            logger.info("Discord: relatorio postado com sucesso")
        else:
            logger.warning("Discord: status %d -- %s", resp.status_code, resp.text[:200])


# -- Salvar pending_changes ---------------------------------------------------

def save_pending(analysis: dict) -> dict:
    from pathlib import Path
    pending_path = Path(__file__).parent.parent / "data" / "agent" / "pending_changes.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    # Adiciona ID sequencial a cada mudanca pendente
    raw_changes = [
        c for c in analysis.get("pending_changes", [])
        if c.get("requires_human_approval", True)
    ]
    for idx, c in enumerate(raw_changes):
        c["id"] = f"change_{idx}"

    pending = {
        "generated_at":       datetime.now(UTC).isoformat(),
        "source":             "weekly_review",
        "overall_health":     analysis.get("overall_health"),
        "executive_summary":  analysis.get("executive_summary"),
        "m_factors_status":   {m: analysis.get("m_factors", {}).get(m, {}).get("status", "?") for m in M_FACTORS},
        "pending_for_approval": raw_changes,
        "alerts": analysis.get("alerts", []),
        "outlook": analysis.get("outlook_next_week", ""),
    }

    pending_path.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("pending_changes.json salvo em %s", pending_path)
    return pending


async def post_discord_approval(pending: dict) -> None:
    """Posta mensagem separada de aprovacao com link para o workflow do GitHub."""
    if not DISCORD_WEBHOOK_URL:
        return

    changes = pending.get("pending_for_approval", [])
    if not changes:
        return

    # Link para o workflow de aprovacao no GitHub
    # O usuario precisa configurar GITHUB_REPO como variavel de ambiente ou hardcodar
    github_repo   = os.environ.get("GITHUB_REPOSITORY", "seu-usuario/CCTBv5")
    workflow_link = f"https://github.com/{github_repo}/actions/workflows/approve_changes.yml"

    health      = pending.get("overall_health", "?")
    health_emoji = {"SAUDAVEL": ":white_check_mark:", "ATENCAO": ":warning:", "CRITICO": ":red_circle:"}.get(health, ":grey_question:")

    priority_emoji = {"HIGH": ":red_circle:", "MEDIUM": ":orange_circle:", "LOW": ":yellow_circle:"}

    lines = [
        f"{health_emoji} **CCTBv5 -- Aprovacoes Pendentes** | {len(changes)} mudanca(s)",
        "",
    ]

    all_ids = ",".join(c.get("id", "?") for c in changes)

    for c in changes:
        cid      = c.get("id", "?")
        priority = c.get("priority", "?")
        target   = c.get("target", "?")
        desc     = c.get("description", "")[:120]
        p_emoji  = priority_emoji.get(priority, ":white_circle:")

        lines.append(f"{p_emoji} **`{cid}`** | {priority} | {target}")
        lines.append(f"   {desc}")

        spec = c.get("action_spec")
        if spec:
            lines.append(f"   > `{spec.get('type')}` em `{spec.get('file')}`")

        lines.append("")

    lines += [
        "---",
        "**Para aprovar, acesse o link abaixo e clique em \"Run workflow\":**",
        f":link: {workflow_link}",
        "",
        f"**Aprovados IDs:** `{all_ids}` (todos) ou IDs especificos separados por virgula",
        "**Dry-run disponivel** para testar sem modificar o sistema.",
    ]

    msg = "\n".join(lines)
    if len(msg) > 1990:
        msg = msg[:1987] + "..."

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            DISCORD_WEBHOOK_URL,
            json={"content": msg, "username": "CCTBv5 Aprovacoes"},
        )
        if resp.status_code in (200, 204):
            logger.info("Discord: mensagem de aprovacao postada")
        else:
            logger.warning("Discord aprovacao: status %d -- %s", resp.status_code, resp.text[:200])


# -- Entry point --------------------------------------------------------------

async def main() -> None:

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY nao configurada -- abortando")
        sys.exit(1)

    now       = datetime.now(UTC)
    week_end  = now.strftime("%Y-%m-%d")
    week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info("=== CCTBv5 Weekly Review Agent ===")
    logger.info("Periodo: %s -> %s", week_start, week_end)
    logger.info("Oracle: %s", ORACLE_API_BASE)
    logger.info("Modelo: %s", CLAUDE_MODEL)

    # 1. Coleta dados
    logger.info("Coletando dados do Oracle (%d endpoints)...", 27)
    data = await fetch_all(ORACLE_API_BASE)
    ok = sum(1 for v in data.values() if v)
    logger.info("  %d/%d endpoints responderam", ok, len(data))

    # 2. Chama Claude
    analysis = await call_claude(data, week_start, week_end)

    if "error" in analysis:
        logger.error("Analise falhou: %s", analysis["error"])
        sys.exit(1)

    logger.info(
        "Analise concluida: health=%s alerts=%d pending=%d",
        analysis.get("overall_health", "?"),
        len(analysis.get("alerts", [])),
        len(analysis.get("pending_changes", [])),
    )

    # 3. Salva pending_changes
    pending = save_pending(analysis)

    # 4. Posta relatorio semanal no Discord
    await post_discord(analysis, data)

    # 5. Posta mensagem de aprovacao (se houver mudancas pendentes)
    if pending.get("pending_for_approval"):
        await post_discord_approval(pending)

    logger.info("=== Weekly Review concluido ===")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
