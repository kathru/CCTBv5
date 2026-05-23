"""
Autonomous Review Agent — Phase 3 CCTBv5

Roda semanalmente via GitHub Actions.
Coleta dados de performance do Oracle, usa Claude para analisar e
posta um relatório estruturado no Discord.

Variáveis de ambiente necessárias:
  ANTHROPIC_API_KEY     — chave Anthropic (Claude API)
  DISCORD_WEBHOOK_URL   — webhook Discord para o canal #review
  ORACLE_API_BASE       — URL base da API Oracle (ex: http://137.131.220.216:8001)
  ORACLE_API_TOKEN      — token Bearer opcional
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

# ── Configuração ──────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
ORACLE_API_BASE     = os.environ.get("ORACLE_API_BASE", "http://137.131.220.216:8001").rstrip("/")
ORACLE_API_TOKEN    = os.environ.get("ORACLE_API_TOKEN", "")

CLAUDE_MODEL = "claude-opus-4-5"   # melhor para análise e geração de texto
MAX_TOKENS   = 1500

# ── Coleta de dados da API Oracle ─────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if ORACLE_API_TOKEN:
        h["Authorization"] = f"Bearer {ORACLE_API_TOKEN}"
    return h


async def fetch_data() -> dict:
    """Coleta todos os dados de performance da API Oracle."""
    results: dict = {}
    async with httpx.AsyncClient(timeout=30, verify=False) as client:
        endpoints = {
            "portfolio":   f"{ORACLE_API_BASE}/api/portfolio/summary",
            "performance": f"{ORACLE_API_BASE}/api/metrics/performance",
            "analytics":   f"{ORACLE_API_BASE}/api/analytics/summary",
            "system":      f"{ORACLE_API_BASE}/api/metrics/system",
        }
        for key, url in endpoints.items():
            try:
                resp = await client.get(url, headers=_headers())
                resp.raise_for_status()
                results[key] = resp.json()
                logger.info("  ✓ %s — %d bytes", key, len(resp.content))
            except Exception as exc:
                logger.warning("  ✗ %s — %s", key, exc)
                results[key] = {}

    # Últimas ordens filled (para análise de trades recentes)
    try:
        async with httpx.AsyncClient(timeout=30, verify=False) as client:
            resp = await client.get(
                f"{ORACLE_API_BASE}/api/orders/filled",
                params={"limit": 30},
                headers=_headers(),
            )
            resp.raise_for_status()
            results["recent_orders"] = resp.json()
    except Exception as exc:
        logger.warning("  ✗ recent_orders — %s", exc)
        results["recent_orders"] = {}

    return results


def _fmt_data_for_claude(data: dict) -> str:
    """Formata os dados coletados como texto estruturado para o prompt."""
    p   = data.get("portfolio", {})
    perf = data.get("performance", {})
    ana  = data.get("analytics", {})
    sys_ = data.get("system", {})
    orders = data.get("recent_orders", {}).get("orders", [])

    now = datetime.now(UTC)
    week_start = (now - timedelta(days=7)).strftime("%d/%m/%Y")

    lines = [
        f"# CCTBv5 — Relatório Semanal ({week_start} → {now.strftime('%d/%m/%Y')})",
        "",
        "## Portfolio",
        f"- Valor total: ${p.get('total_value', 0):,.2f}",
        f"- Capital inicial: ${p.get('initial_capital', 0):,.2f}",
        f"- Retorno acumulado: {p.get('total_return_pct', 0)*100:+.2f}%",
        f"- Drawdown atual: {p.get('drawdown_pct', 0)*100:.2f}%",
        f"- P&L realizado: ${p.get('realized_pnl', 0):,.2f}",
        f"- P&L não realizado: ${p.get('unrealized_pnl', 0):,.2f}",
        f"- Posições abertas: {p.get('open_position_count', 0)}",
        f"- Exposição: {p.get('total_exposure_pct', 0)*100:.1f}%",
        "",
        "## Performance Operacional",
        f"- Total de trades (ordens filled): {perf.get('total_trades', 0)}",
        f"- Compras: {perf.get('total_buys', 0)} | Vendas: {perf.get('total_sells', 0)}",
        f"- Volume total negociado: ${perf.get('total_volume', 0):,.2f}",
        f"- Taxas pagas: ${perf.get('total_fees', 0):,.2f}",
        f"- P&L realizado (ordens): ${perf.get('total_pnl', 0):,.2f}",
        f"- P&L por símbolo: {json.dumps(perf.get('pnl_by_symbol', {}), indent=2)}",
        "",
        "## Analytics",
        f"- Sharpe Ratio: {ana.get('sharpe_ratio', 'N/A')}",
        f"- Calmar Ratio: {ana.get('calmar_ratio', 'N/A')}",
        f"- Max Drawdown: {ana.get('max_drawdown_pct', ana.get('max_drawdown', 'N/A'))}",
        f"- Win Rate Rolling: {ana.get('win_rate', ana.get('rolling_win_rate', 'N/A'))}",
        f"- Expectância média: {ana.get('avg_expectancy', 'N/A')}",
        "",
        "## Sistema",
        f"- Modo: {sys_.get('mode', 'unknown')} | Monitor only: {sys_.get('monitor_only', False)}",
        f"- Versão: {sys_.get('version', 'N/A')}",
        f"- Status: {sys_.get('system_status', 'N/A')}",
        "",
    ]

    if orders:
        lines.append("## Últimas 30 Ordens Filled (bot)")
        for o in orders[:30]:
            side = str(o.get("side", "")).upper()
            sym  = o.get("symbol", "?").replace("-USDT", "")
            qty  = o.get("filled_quantity", 0)
            px   = o.get("avg_fill_price", 0)
            fee  = o.get("fees_paid", 0)
            ts   = o.get("filled_at", "")[:10]
            lines.append(f"  {ts} {side:4s} {sym:3s} qty={qty:.4f} px=${px:,.2f} fee=${fee:.2f}")

    return "\n".join(lines)


# ── Claude API ────────────────────────────────────────────────────────────────

async def call_claude(data_text: str) -> str:
    """Envia dados para Claude e obtém análise estruturada."""
    import anthropic   # importação tardia — só disponível em Actions

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    system_prompt = (
        "Você é um analista quantitativo especializado em sistemas de trading algorítmico."
        " Analise os dados de performance do bot CCTBv5 e produza um relatório semanal"
        " objetivo e acionável.\n\n"
        "Formato obrigatório (use exatamente estas seções com os emojis):\n"
        "📊 **RESUMO EXECUTIVO** — 2-3 frases sobre o resultado da semana\n"
        "📈 **PERFORMANCE** — métricas-chave com interpretação (Sharpe, Calmar, WR, P&L)\n"
        "🎯 **SINAIS & REGIME** — qualidade dos sinais, distribuição, taxa de disparo\n"
        "⚠️ **ALERTAS** — drawdown, exposição excessiva, degradação de WR, anomalias\n"
        "🔧 **RECOMENDAÇÕES** — 3-5 ações concretas e priorizadas (curto/médio prazo)\n"
        "🔮 **PERSPECTIVA** — outlook para a próxima semana baseado nos dados\n\n"
        "Seja direto, quantitativo e acionável. Máximo 1200 palavras."
    )

    user_message = f"""Dados coletados agora do sistema CCTBv5:

{data_text}

Parâmetros do motor (referência):
- Motor Probabilístico v2.4.0 (Platt calibrado, WR calibrado: 37.5%, n=38788)
- Pesos: M1=2%, M2=12%, M3=35%, M5=6%, M6=13%, M7=12%, M8=20%
- Thresholds: TREND_EXPANSION=0.56, VOL_COMP=0.58, CHOP=0.68
- Regimes bloqueados: TREND_EXHAUSTION, BEAR_TREND, HIGH_CORRELATION_RISK, PANIC_LIQUIDATION

Produza o relatório semanal."""

    message = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


# ── Discord ───────────────────────────────────────────────────────────────────

async def post_to_discord(report: str, data: dict) -> None:
    """Posta o relatório no Discord via webhook com embed."""
    p    = data.get("portfolio", {})
    perf = data.get("performance", {})
    total = p.get("total_value", 0)
    ret   = p.get("total_return_pct", 0)
    dd    = p.get("drawdown_pct", 0)
    pnl   = perf.get("total_pnl", 0)
    trades = perf.get("total_trades", 0)

    # Cor do embed baseada em performance
    color = 0x00C853 if pnl >= 0 else 0xFF1744  # verde ou vermelho

    # Divide o relatório em chunks (Discord tem limite de 4096 chars por field)
    chunks = [report[i:i+1000] for i in range(0, len(report), 1000)]

    now_str = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")

    embed: dict = {
        "title": f"🤖 Relatório Semanal CCTBv5 — {now_str}",
        "color": color,
        "description": chunks[0] if chunks else "Sem dados",
        "fields": [
            {
                "name": "📊 Resumo Rápido",
                "value": (
                    f"Portfolio: **${total:,.2f}** | "
                    f"Retorno: **{ret*100:+.2f}%** | "
                    f"P&L: **${pnl:,.2f}** | "
                    f"Trades: **{trades}** | "
                    f"DD: **{dd*100:.2f}%**"
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "CCTBv5 Autonomous Review Agent • Phase 3"},
    }

    # Adiciona chunks adicionais como fields
    for i, chunk in enumerate(chunks[1:], 1):
        embed["fields"].append({
            "name": f"(continuação {i})",
            "value": chunk,
            "inline": False,
        })

    payload = {"embeds": [embed]}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(DISCORD_WEBHOOK_URL, json=payload)
        resp.raise_for_status()
        logger.info("Discord: relatório postado (status=%d)", resp.status_code)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY não configurada — abortando")
        sys.exit(1)
    if not DISCORD_WEBHOOK_URL:
        logger.error("DISCORD_WEBHOOK_URL não configurada — abortando")
        sys.exit(1)

    logger.info("=== CCTBv5 Autonomous Review Agent ===")
    logger.info("Oracle API: %s", ORACLE_API_BASE)

    # 1. Coleta dados
    logger.info("Coletando dados da API Oracle...")
    data = await fetch_data()

    # 2. Formata para Claude
    data_text = _fmt_data_for_claude(data)
    logger.info("Texto preparado: %d chars", len(data_text))

    # 3. Chama Claude
    logger.info("Chamando Claude %s...", CLAUDE_MODEL)
    report = await call_claude(data_text)
    logger.info("Relatório gerado: %d chars", len(report))

    # 4. Posta no Discord
    logger.info("Postando no Discord...")
    await post_to_discord(report, data)

    logger.info("✓ Autonomous Review concluído")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
