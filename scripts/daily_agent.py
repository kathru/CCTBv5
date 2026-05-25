#!/usr/bin/env python3
"""
daily_agent.py — Agente LLM de Análise Diária do CCTBv5.

Responsabilidades:
  1. Coleta dados do sistema (APIs Oracle, arquivos locais, Redis)
  2. Roda scripts de recalibração (feature_analysis, recalibrate, regime_backtest)
  3. Chama Claude (claude-sonnet-4-6) para análise inteligente com prompt caching
  4. Gera relatório estruturado por fator M1-M9
  5. Aplica ajustes seguros automaticamente (regime_weights, feature_baseline)
  6. Salva mudanças pendentes (require aprovação humana) em data/agent/pending_changes.json
  7. Envia relatório formatado ao Discord

Uso:
  python scripts/daily_agent.py
  python scripts/daily_agent.py --dry-run           # não aplica nenhuma mudança
  python scripts/daily_agent.py --no-discord        # pula envio ao Discord
  python scripts/daily_agent.py --no-recalibrate    # pula scripts de recalibração
  python scripts/daily_agent.py --oracle-url http://137.131.220.216:8001

Ambientes:
  Oracle  → http://137.131.220.216:8001 (processa tudo, monitor_only=False)
  localhost → http://localhost:8001 (monitor_only=True, consome do Oracle)
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily_agent")

# ── Configurações ──────────────────────────────────────────────────────────────

ORACLE_URL      = "http://137.131.220.216:8001"
AGENT_DIR       = ROOT / "data" / "agent"
PENDING_PATH    = AGENT_DIR / "pending_changes.json"
REPORT_PATH     = AGENT_DIR / "daily_report.json"
MODELS_DIR      = ROOT / "data" / "models"

# Limites para ajustes automáticos seguros
AUTO_SAFE_WEIGHT_CHANGE = 0.20    # diff máximo em regime_weight para auto-aplicar
AUTO_SAFE_BASELINE_ONLY = True    # só atualiza feature_baseline automaticamente (não scoring weights)

# Modelo frozen até Day 90 — nunca alterar scoring weights automaticamente
MODEL_FROZEN = True

# Fator M a ser monitorado
M_FACTORS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


# ── Coleta de Dados ────────────────────────────────────────────────────────────

async def fetch_api(client: httpx.AsyncClient, url: str, path: str) -> dict | list | None:
    """Busca dados de uma API com timeout e fallback seguro."""
    try:
        resp = await client.get(f"{url}{path}", timeout=15.0)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("API %s retornou %d", path, resp.status_code)
    except Exception as exc:
        logger.warning("Falha ao buscar %s: %s", path, exc)
    return None


async def collect_system_data(oracle_url: str) -> dict:
    """Coleta todos os dados relevantes do sistema."""
    data: dict = {
        "collected_at": datetime.now(UTC).isoformat(),
        "oracle_url": oracle_url,
    }

    async with httpx.AsyncClient() as client:
        # Status geral
        data["health"]           = await fetch_api(client, oracle_url, "/health")
        data["model_health"]     = await fetch_api(client, oracle_url, "/api/analytics/model_health")
        data["signals_log"]      = await fetch_api(client, oracle_url, "/api/signals/log")
        data["signals_calibration"] = await fetch_api(client, oracle_url, "/api/signals/calibration")
        data["orders_filled"]    = await fetch_api(client, oracle_url, "/api/orders/filled")
        data["orders_recent"]    = await fetch_api(client, oracle_url, "/api/orders/recent")
        data["trade_distribution"] = await fetch_api(client, oracle_url, "/api/analytics/distribution")
        data["meta_regime"]      = await fetch_api(client, oracle_url, "/api/analytics/meta_regime")
        data["volatility_state"] = await fetch_api(client, oracle_url, "/api/analytics/volatility_state")
        data["futures_flow"]     = await fetch_api(client, oracle_url, "/api/analytics/futures_flow")
        data["relative_strength"] = await fetch_api(client, oracle_url, "/api/analytics/relative_strength")
        data["reality_check"]    = await fetch_api(client, oracle_url, "/api/analytics/reality_check")
        data["positions_open"]   = await fetch_api(client, oracle_url, "/api/positions/open")
        data["portfolio_summary"] = await fetch_api(client, oracle_url, "/api/portfolio/summary")
        data["metrics_system"]   = await fetch_api(client, oracle_url, "/api/metrics/system")
        data["metrics_performance"] = await fetch_api(client, oracle_url, "/api/metrics/performance")
        data["governance_drift"] = await fetch_api(client, oracle_url, "/api/governance/drift")
        data["governance_live"]  = await fetch_api(client, oracle_url, "/api/governance/live-stats")
        data["governance_importance"] = await fetch_api(client, oracle_url, "/api/governance/importance")
        data["quantitative"]     = await fetch_api(client, oracle_url, "/api/analytics/quantitative")

    # Arquivos locais
    data["regime_weights"] = _load_json(MODELS_DIR / "regime_weights.json")
    data["feature_baseline"] = _load_json(MODELS_DIR / "feature_baseline.json")
    data["platt_model"] = _load_json(MODELS_DIR / "platt_model.json")
    data["pending_changes_existing"] = _load_json(PENDING_PATH)

    return data


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Falha ao ler %s: %s", path, exc)
    return None


# ── Scripts de Recalibração ────────────────────────────────────────────────────

def run_recalibration_scripts(dry_run: bool = False) -> dict:
    """Roda feature_analysis, recalibrate e regime_backtest. Retorna sumário dos resultados."""
    results = {}

    scripts = [
        ("feature_analysis",  ["python", str(ROOT / "scripts" / "feature_analysis.py")]),
        ("recalibrate",       ["python", str(ROOT / "scripts" / "recalibrate.py"), "--incremental"]),
        ("regime_backtest",   ["python", str(ROOT / "scripts" / "regime_backtest.py")]),
    ]

    for name, cmd in scripts:
        if dry_run:
            logger.info("[DRY-RUN] Pulando %s", name)
            results[name] = {"status": "skipped_dry_run"}
            continue

        logger.info("Rodando %s...", name)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(ROOT),
            )
            results[name] = {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
            }
            if proc.returncode != 0:
                logger.warning("%s falhou (returncode=%d): %s",
                               name, proc.returncode, proc.stderr[-300:])
            else:
                logger.info("%s OK", name)
        except subprocess.TimeoutExpired:
            results[name] = {"status": "timeout"}
            logger.warning("%s: timeout após 300s", name)
        except Exception as exc:
            results[name] = {"status": "exception", "error": str(exc)}
            logger.warning("%s: exceção — %s", name, exc)

    return results


# ── Prompt para o Claude ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """Você é o Analista Quantitativo do CCTBv5, um sistema de trading algorítmico de criptomoedas.

Sua responsabilidade é analisar os dados diários do sistema e identificar:
1. Desvios de comportamento em cada fator M1-M9
2. Degradação do modelo (PSI drift, WR calibration)
3. Eficácia das estratégias por regime (momentum_v2, reversal_v1, trend_v1)
4. Oportunidades de melhoria nos pesos do regime (WeightEngine)
5. Anomalias nos padrões de sinal e trade

ARQUITETURA DO SISTEMA:
- Score calibrado: Platt v12, WR baseline=37.5%, modelo FROZEN até Day 90
- Estratégias: momentum_v2 (principal), reversal_v1 (reversão), trend_v1 (tendência)
- Regimes: TREND_EXPANSION, VOLATILITY_COMPRESSION, MEAN_REVERTING_CHOP, TREND_EXHAUSTION, HIGH_CORRELATION_RISK, BEAR_TREND, PANIC_LIQUIDATION
- Hold Engine: Conviction Score 0-100 (HOLD≥70, WATCH 50-69, ALERT 30-49, EXIT<30)
- SizingEngine: kelly = base × regime × drift × vol_state × calibration × score × exceptional_mult
- WeightEngine: online learning sim→real, convergência em 30 trades

FATORES M:
- M1: Momentum (RSI, MACD, EMA cross)
- M2: Estrutura de mercado (suporte/resistência, padrões de candle)
- M3: Volume/OBV (confirmação de movimento)
- M4: Sentimento (neutro por decisão, weight=0%)
- M5: Score composto Platt
- M6: Fluxo de futuros (open interest, funding rate)
- M7: Força relativa vs BTC/mercado
- M8: Estado de volatilidade (EXPANDING/TREND/COMPRESSED/MEAN_REVERTING/CHAOTIC)
- M9: Sentimento de notícias (FinNLP)

REGRAS DE SAÍDA (Hold Engine):
- Conviction score composto de: trend_persistence, momentum_align, vol_regime, volume_confirm, price_structure, time_decay
- Trail ATR: HOLD(≥85)→2.5×, HOLD(70-84)→2.0×, WATCH→1.0×, ALERT→0.5×
- TP Conversion em TREND_EXPANSION: parcial + modo RUNNING (trailing 2.5×ATR, sem teto)
- Anti-bag-holding: P1(30d), P2(45d/DD>15%), P3(60d)

IMPORTANTE:
- O modelo de scoring está FROZEN (não sugerir mudanças nos pesos das features M1-M9)
- Mudanças automáticas permitidas: regime_weights (se delta < 20%), feature_baseline
- Mudanças que precisam de aprovação humana: qualquer coisa no Platt, thresholds críticos, nova lógica
- Sempre justifique com dados quantitativos (não achismos)

Responda SEMPRE em JSON estruturado conforme solicitado."""


def build_user_prompt(system_data: dict, recal_results: dict) -> str:
    """Monta o prompt do usuário com dados coletados."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Extrai métricas-chave para o prompt
    model_health = system_data.get("model_health") or {}
    signals_log  = system_data.get("signals_log") or {}
    trade_dist   = system_data.get("trade_distribution") or {}
    regime_sum   = system_data.get("regime_summary") or {}
    pnl          = system_data.get("pnl_summary") or {}
    weight_sum   = system_data.get("weight_summary") or {}
    orders       = system_data.get("orders_filled") or []

    # Resumo compacto de orders (últimas 24h)
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    recent_orders = []
    if isinstance(orders, list):
        for o in orders:
            try:
                ts = datetime.fromisoformat(o.get("filled_at", ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    recent_orders.append({
                        "symbol": o.get("symbol"),
                        "side": o.get("side"),
                        "pnl": o.get("pnl_usdt"),
                        "strategy": o.get("strategy_id"),
                        "regime": o.get("regime"),
                    })
            except Exception:
                pass

    prompt = f"""DATA/HORA DA ANÁLISE: {now}

=== DADOS DO SISTEMA ===

MODEL HEALTH:
{json.dumps(model_health, indent=2, ensure_ascii=False)[:2000]}

SINAIS (log resumido):
{json.dumps(signals_log, indent=2, ensure_ascii=False)[:1500]}

TRADES FECHADOS (últimas 24h, {len(recent_orders)} trades):
{json.dumps(recent_orders, indent=2, ensure_ascii=False)[:1500]}

DISTRIBUIÇÃO DE TRADES:
{json.dumps(trade_dist, indent=2, ensure_ascii=False)[:1500]}

META REGIME / VOLATILITY STATE:
{json.dumps(regime_sum, indent=2, ensure_ascii=False)[:800]}

PNL / PERFORMANCE:
{json.dumps(pnl, indent=2, ensure_ascii=False)[:800]}

WEIGHT ENGINE (pesos atuais por estratégia/regime):
{json.dumps(weight_sum, indent=2, ensure_ascii=False)[:1500]}

GOVERNANCE DRIFT (PSI por feature):
{json.dumps(system_data.get("governance_drift"), indent=2, ensure_ascii=False)[:1000]}

GOVERNANCE LIVE-STATS (WR live):
{json.dumps(system_data.get("governance_live"), indent=2, ensure_ascii=False)[:800]}

RESULTADOS DAS RECALIBRAÇÕES:
{json.dumps(recal_results, indent=2, ensure_ascii=False)[:1000]}

=== TAREFA ===

Analise todos os dados acima e responda em JSON com EXATAMENTE esta estrutura:

{{
  "analysis_date": "{now}",
  "overall_health": "SAUDÁVEL|ATENÇÃO|CRÍTICO",
  "overall_summary": "Resumo executivo em 2-3 frases",

  "m_factors": {{
    "M1": {{
      "status": "OK|ATENÇÃO|DEGRADADO",
      "observations": "O que você encontrou de relevante",
      "deviation": "Desvio quantitativo se houver (ex: WR 32% vs baseline 37.5%)",
      "recommendation": "O que fazer (se nada, 'Manter monitoramento')"
    }},
    "M2": {{ ... }},
    "M3": {{ ... }},
    "M4": {{ ... }},
    "M5": {{ ... }},
    "M6": {{ ... }},
    "M7": {{ ... }},
    "M8": {{ ... }},
    "M9": {{ ... }}
  }},

  "strategy_performance": {{
    "momentum_v2": {{
      "win_rate_24h": null,
      "trades_24h": 0,
      "best_regime": "regime onde mais performa",
      "worst_regime": "regime onde menos performa",
      "assessment": "avaliação qualitativa"
    }},
    "reversal_v1": {{ ... }},
    "trend_v1": {{ ... }}
  }},

  "regime_weight_proposals": [
    {{
      "strategy_id": "momentum_v2",
      "regime": "TREND_EXPANSION",
      "current_weight": 1.0,
      "proposed_weight": 1.0,
      "justification": "motivo da mudança (ou 'sem alteração')",
      "auto_apply": true
    }}
  ],

  "pending_changes": [
    {{
      "change_type": "threshold|param|logic|scoring_weight",
      "target": "arquivo ou componente",
      "description": "descrição da mudança necessária",
      "requires_human_approval": true,
      "priority": "HIGH|MEDIUM|LOW",
      "justification": "dados que justificam a mudança"
    }}
  ],

  "alerts": [
    {{
      "severity": "CRITICAL|WARNING|INFO",
      "message": "descrição do alerta",
      "action": "ação recomendada imediata"
    }}
  ],

  "discord_summary": "Resumo formatado para Discord em markdown (máx 1800 chars)"
}}

IMPORTANTE: Retorne APENAS o JSON, sem texto antes ou depois. O campo discord_summary deve usar emojis e markdown compatível com Discord."""

    return prompt


# ── Chamada ao Claude ──────────────────────────────────────────────────────────

async def call_claude(system_data: dict, recal_results: dict) -> dict:
    """Chama Claude para análise. Usa prompt caching no system prompt."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic não instalado. Rode: pip install anthropic")
        return {"error": "anthropic not installed"}

    # Resolve API key: .env direto → settings → variável de ambiente
    import os
    from dotenv import dotenv_values

    api_key = ""

    # Tenta ler direto do .env (mais confiável que pydantic-settings em scripts)
    env_file = ROOT / ".env"
    if env_file.exists():
        env_vals = dotenv_values(env_file)
        api_key = env_vals.get("ANTHROPIC_API_KEY", "")

    # Fallback: settings pydantic
    if not api_key:
        try:
            from src.core.config import settings
            api_key = settings.anthropic_api_key
        except Exception:
            pass

    # Fallback: variável de ambiente do shell
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return {"error": "ANTHROPIC_API_KEY não configurada em .env ou variável de ambiente"}

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = build_user_prompt(system_data, recal_results)

    logger.info("Chamando Claude (claude-opus-4-7) para análise...")

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        raw_text = response.content[0].text.strip()

        # Extrai JSON da resposta
        if raw_text.startswith("```"):
            # Remove code block se presente
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        analysis = json.loads(raw_text)

        logger.info(
            "Claude respondeu: health=%s alerts=%d proposals=%d pending=%d",
            analysis.get("overall_health", "?"),
            len(analysis.get("alerts", [])),
            len(analysis.get("regime_weight_proposals", [])),
            len(analysis.get("pending_changes", [])),
        )

        # Log de uso de tokens (para monitorar custo)
        usage = response.usage
        logger.info(
            "Tokens: input=%d output=%d cache_creation=%d cache_read=%d",
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
        )

        return analysis

    except json.JSONDecodeError as exc:
        logger.error("Claude retornou JSON inválido: %s", exc)
        logger.debug("Resposta raw: %s", raw_text[:500])
        return {"error": f"json_decode: {exc}", "raw": raw_text[:1000]}
    except Exception as exc:
        logger.error("Erro ao chamar Claude: %s", exc, exc_info=True)
        return {"error": str(exc)}


# ── Aplicar Mudanças Seguras ───────────────────────────────────────────────────

def apply_safe_changes(analysis: dict, dry_run: bool = False) -> list[str]:
    """
    Aplica automaticamente mudanças consideradas seguras:
    - regime_weights com delta < AUTO_SAFE_WEIGHT_CHANGE
    - NÃO altera scoring weights (modelo frozen)
    Retorna lista de mudanças aplicadas.
    """
    applied = []

    if MODEL_FROZEN:
        # Verifica se análise quer mudar scoring weights → move para pending
        for change in analysis.get("pending_changes", []):
            if change.get("change_type") == "scoring_weight":
                change["requires_human_approval"] = True
                change["frozen_override"] = "MODEL_FROZEN: não aplicado automaticamente"

    # Aplica regime_weight_proposals com auto_apply=True e delta seguro
    proposals = analysis.get("regime_weight_proposals", [])
    regime_weights = _load_json(MODELS_DIR / "regime_weights.json") or {}

    changed_weights = False
    for prop in proposals:
        if not prop.get("auto_apply", False):
            continue

        strategy_id  = prop.get("strategy_id", "")
        regime       = prop.get("regime", "")
        current_w    = prop.get("current_weight", 1.0)
        proposed_w   = prop.get("proposed_weight", 1.0)
        delta        = abs(proposed_w - current_w)

        if delta < 0.001:
            continue  # sem mudança real

        if delta > AUTO_SAFE_WEIGHT_CHANGE:
            logger.info(
                "SKIP auto-apply %s/%s: delta=%.3f > limite %.2f (requer aprovação)",
                strategy_id, regime, delta, AUTO_SAFE_WEIGHT_CHANGE
            )
            prop["auto_apply"] = False
            prop["skip_reason"] = f"delta {delta:.3f} excede limite seguro {AUTO_SAFE_WEIGHT_CHANGE}"
            continue

        if dry_run:
            logger.info("[DRY-RUN] Aplicaria %s/%s: %.3f → %.3f", strategy_id, regime, current_w, proposed_w)
            applied.append(f"[DRY-RUN] {strategy_id}/{regime}: {current_w:.3f} → {proposed_w:.3f}")
            continue

        # Aplica no arquivo de regime_weights
        weights = regime_weights.setdefault("weights", {})
        regime_data = weights.setdefault(regime, {})
        if strategy_id not in regime_data:
            regime_data[strategy_id] = {}
        entry = regime_data[strategy_id]
        old_sim = entry.get("sim_weight", current_w)
        entry["sim_weight"] = round(proposed_w, 4)
        entry["blended_weight"] = round(proposed_w, 4)
        entry["agent_updated_at"] = datetime.now(UTC).isoformat()
        changed_weights = True

        msg = f"{strategy_id}/{regime}: {old_sim:.3f} → {proposed_w:.3f} (delta={delta:.3f})"
        applied.append(msg)
        logger.info("AUTO-APPLIED regime_weight %s", msg)

    if changed_weights and not dry_run:
        regime_weights["updated_at"] = datetime.now(UTC).isoformat()
        try:
            (MODELS_DIR / "regime_weights.json").write_text(
                json.dumps(regime_weights, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("regime_weights.json salvo com %d atualizações", len(applied))
        except Exception as exc:
            logger.error("Falha ao salvar regime_weights.json: %s", exc)

    return applied


# ── Salvar Pending Changes ─────────────────────────────────────────────────────

def save_pending_changes(analysis: dict, applied: list[str]) -> None:
    """Salva mudanças pendentes que requerem aprovação humana."""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)

    pending = {
        "generated_at": datetime.now(UTC).isoformat(),
        "overall_health": analysis.get("overall_health"),
        "overall_summary": analysis.get("overall_summary"),
        "auto_applied": applied,
        "pending_for_approval": [
            c for c in analysis.get("pending_changes", [])
            if c.get("requires_human_approval", True)
        ],
        "alerts": analysis.get("alerts", []),
        "m_factors_status": {
            m: {
                "status": analysis.get("m_factors", {}).get(m, {}).get("status", "N/A"),
                "recommendation": analysis.get("m_factors", {}).get(m, {}).get("recommendation", ""),
            }
            for m in M_FACTORS
        },
    }

    try:
        PENDING_PATH.write_text(
            json.dumps(pending, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Pending changes salvo em %s", PENDING_PATH)
    except Exception as exc:
        logger.error("Falha ao salvar pending_changes.json: %s", exc)


# ── Discord ────────────────────────────────────────────────────────────────────

async def send_discord_report(
    analysis: dict,
    applied: list[str],
    recal_results: dict,
    webhook_url: str,
) -> None:
    """Envia relatório diário ao Discord."""
    if not webhook_url:
        logger.warning("discord_webhook_url não configurado — pulando envio")
        return

    # Usa o campo discord_summary gerado pelo Claude
    discord_summary = analysis.get("discord_summary", "")

    # Cabeçalho padronizado
    health = analysis.get("overall_health", "?")
    health_emoji = {"SAUDÁVEL": "✅", "ATENÇÃO": "⚠️", "CRÍTICO": "🚨"}.get(health, "❓")

    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    header = (
        f"**📊 CCTBv5 — Relatório Diário do Agente LLM**\n"
        f"{health_emoji} **{health}** | {now_str}\n\n"
    )

    # Resumo das recalibrações
    recal_status = []
    for name, result in recal_results.items():
        status = result.get("status", "?")
        icon = "✅" if status == "ok" else ("⏩" if "skip" in status else "❌")
        recal_status.append(f"{icon} {name}")
    recal_line = f"**Recalibrações:** {' | '.join(recal_status)}\n\n" if recal_status else ""

    # Auto-aplicações
    applied_line = ""
    if applied:
        applied_strs = "\n".join(f"  • {a}" for a in applied[:5])
        applied_line = f"**Auto-aplicados ({len(applied)}):**\n{applied_strs}\n\n"

    # Alertas críticos
    alerts = analysis.get("alerts", [])
    critical = [a for a in alerts if a.get("severity") == "CRITICAL"]
    alert_line = ""
    if critical:
        alert_strs = "\n".join(f"  🚨 {a.get('message', '')}" for a in critical[:3])
        alert_line = f"**ALERTAS CRÍTICOS:**\n{alert_strs}\n\n"

    full_message = header + alert_line + recal_line + applied_line + discord_summary

    # Discord tem limite de 2000 chars por mensagem
    if len(full_message) > 1990:
        full_message = full_message[:1987] + "..."

    payload = {"content": full_message, "username": "CCTBv5 Agent"}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(webhook_url, json=payload, timeout=15.0)
            if resp.status_code in (200, 204):
                logger.info("Discord: relatório enviado com sucesso")
            else:
                logger.warning("Discord: status %d — %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.error("Discord: falha ao enviar — %s", exc)


# ── Salvar Relatório Completo ──────────────────────────────────────────────────

def save_full_report(analysis: dict, system_data: dict, recal_results: dict, applied: list[str]) -> None:
    """Salva relatório completo em data/agent/daily_report.json."""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "analysis": analysis,
        "recalibration_results": recal_results,
        "auto_applied": applied,
        "data_collected_at": system_data.get("collected_at"),
    }

    try:
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Relatório completo salvo em %s", REPORT_PATH)
    except Exception as exc:
        logger.error("Falha ao salvar daily_report.json: %s", exc)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("CCTBv5 Daily Agent — %s", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    if args.dry_run:
        logger.info("MODO DRY-RUN: nenhuma mudança será aplicada")
    logger.info("=" * 60)

    # 1. Coleta dados do sistema
    logger.info("Coletando dados do sistema (Oracle: %s)...", args.oracle_url)
    system_data = await collect_system_data(args.oracle_url)
    logger.info("Dados coletados: %d endpoints responderam",
                sum(1 for v in system_data.values() if v is not None))

    # 2. Roda scripts de recalibração
    recal_results: dict = {}
    if not args.no_recalibrate:
        logger.info("Rodando scripts de recalibração...")
        recal_results = run_recalibration_scripts(dry_run=args.dry_run)
    else:
        logger.info("Pulando recalibração (--no-recalibrate)")

    # 3. Recarrega dados após recalibração (baseline pode ter mudado)
    if not args.no_recalibrate and not args.dry_run:
        system_data["regime_weights"]   = _load_json(MODELS_DIR / "regime_weights.json")
        system_data["feature_baseline"] = _load_json(MODELS_DIR / "feature_baseline.json")

    # 4. Chama Claude para análise
    analysis = await call_claude(system_data, recal_results)

    if "error" in analysis:
        logger.error("Análise do Claude falhou: %s", analysis["error"])
        # Continua com análise vazia para não travar o fluxo
        analysis = {
            "overall_health": "ATENÇÃO",
            "overall_summary": f"Análise LLM falhou: {analysis.get('error')}",
            "m_factors": {},
            "strategy_performance": {},
            "regime_weight_proposals": [],
            "pending_changes": [],
            "alerts": [{"severity": "WARNING", "message": "Análise LLM falhou", "action": "Verificar logs"}],
            "discord_summary": f"⚠️ Análise LLM falhou: {analysis.get('error', '?')}",
        }

    # 5. Aplica mudanças seguras automaticamente
    applied = apply_safe_changes(analysis, dry_run=args.dry_run)
    if applied:
        logger.info("Mudanças auto-aplicadas: %d", len(applied))

    # 6. Salva pending changes para aprovação humana
    save_pending_changes(analysis, applied)

    # 7. Salva relatório completo
    save_full_report(analysis, system_data, recal_results, applied)

    # 8. Envia ao Discord
    if not args.no_discord:
        try:
            from src.core.config import settings
            webhook_url = settings.discord_webhook_url
        except Exception:
            webhook_url = ""

        await send_discord_report(analysis, applied, recal_results, webhook_url)
    else:
        logger.info("Pulando Discord (--no-discord)")

    # Resumo final
    logger.info("=" * 60)
    logger.info("ANÁLISE CONCLUÍDA")
    logger.info("  Saúde geral: %s", analysis.get("overall_health", "?"))
    logger.info("  Mudanças auto-aplicadas: %d", len(applied))
    logger.info("  Pending (requer aprovação): %d",
                len([c for c in analysis.get("pending_changes", []) if c.get("requires_human_approval")]))
    logger.info("  Alertas: %d (críticos: %d)",
                len(analysis.get("alerts", [])),
                len([a for a in analysis.get("alerts", []) if a.get("severity") == "CRITICAL"]))
    logger.info("  Relatório: %s", REPORT_PATH)
    logger.info("  Pending: %s", PENDING_PATH)
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCTBv5 Daily LLM Agent — análise e recalibração diária"
    )
    parser.add_argument(
        "--oracle-url",
        default=ORACLE_URL,
        help=f"URL do Oracle (default: {ORACLE_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula sem aplicar nenhuma mudança",
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Pula envio do relatório ao Discord",
    )
    parser.add_argument(
        "--no-recalibrate",
        action="store_true",
        help="Pula execução dos scripts de recalibração",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
