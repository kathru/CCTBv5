"""
apply_changes.py -- Aplica mudancas aprovadas do pending_changes.json

Uso:
    python scripts/apply_changes.py --approved-ids change_0,change_1
    python scripts/apply_changes.py --all          # aprova tudo (cuidado)
    python scripts/apply_changes.py --dry-run --approved-ids change_0

Variaveis de ambiente:
    DISCORD_WEBHOOK_URL   webhook para confirmacao pos-aplicacao (opcional)
    PROJECT_ROOT          raiz do projeto (default: diretorio pai deste script)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

PROJECT_ROOT        = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
PENDING_FILE        = PROJECT_ROOT / "data" / "agent" / "pending_changes.json"
APPLY_LOG_FILE      = PROJECT_ROOT / "data" / "agent" / "apply_log.json"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# Executores de action_spec
# ---------------------------------------------------------------------------

def apply_python_replace(spec: dict, file_root: Path, dry_run: bool) -> str:
    """Substitui uma string exata num arquivo Python."""
    file_path = file_root / spec["file"]
    old_str   = spec["old"]
    new_str   = spec["new"]

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    content = file_path.read_text(encoding="utf-8")

    if old_str not in content:
        raise ValueError(
            f"String de busca nao encontrada em {spec['file']}:\n  {old_str!r}"
        )

    count = content.count(old_str)
    if count > 1:
        raise ValueError(
            f"String ambigua — encontrada {count}x em {spec['file']}. "
            "Use string mais especifica."
        )

    new_content = content.replace(old_str, new_str, 1)

    if not dry_run:
        file_path.write_text(new_content, encoding="utf-8")
        logger.info("    Arquivo atualizado: %s", spec["file"])
    else:
        logger.info("    [DRY-RUN] Substituicao em %s: %r -> %r", spec["file"], old_str, new_str)

    return f"python_replace OK: {spec['file']} | {old_str!r} -> {new_str!r}"


def apply_json_patch(spec: dict, file_root: Path, dry_run: bool) -> str:
    """Aplica patches RFC 6902 em arquivo JSON."""
    try:
        import jsonpatch  # type: ignore
    except ImportError:
        raise ImportError("jsonpatch nao instalado. Use: pip install jsonpatch")

    file_path = file_root / spec["file"]
    patches   = spec["patches"]

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    patched = jsonpatch.apply_patch(data, patches)

    if not dry_run:
        file_path.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("    JSON atualizado: %s", spec["file"])
    else:
        logger.info("    [DRY-RUN] JSON patch em %s: %s", spec["file"], patches)

    return f"json_patch OK: {spec['file']} | {len(patches)} patch(es)"


def apply_action_spec(spec: dict, file_root: Path, dry_run: bool) -> str:
    """Despacha para o executor correto conforme spec['type']."""
    spec_type = spec.get("type")
    if spec_type == "python_replace":
        return apply_python_replace(spec, file_root, dry_run)
    elif spec_type == "json_patch":
        return apply_json_patch(spec, file_root, dry_run)
    else:
        raise ValueError(f"Tipo de action_spec desconhecido: {spec_type!r}")


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def post_discord(results: list[dict], dry_run: bool) -> None:
    if not DISCORD_WEBHOOK_URL:
        return

    import urllib.request

    status_emoji = {"ok": "v", "erro": "x", "skipped": "-"}
    lines = ["**CCTBv5 -- Mudancas Aplicadas**" + (" [DRY-RUN]" if dry_run else "")]
    for r in results:
        emoji = status_emoji.get(r["status"], "?")
        lines.append(f"{emoji} `{r['id']}` | {r['description'][:80]}")
        if r["status"] == "erro":
            lines.append(f"   ERRO: {r['error'][:120]}")

    now = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")
    lines.append(f"\n_{now}_")

    body = json.dumps({"content": "\n".join(lines), "username": "CCTBv5 Aprovacoes"}).encode()
    req  = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            logger.info("Discord: confirmacao postada")
    except Exception as exc:
        logger.warning("Discord: falha ao postar -- %s", exc)


# ---------------------------------------------------------------------------
# Log de aplicacao
# ---------------------------------------------------------------------------

def save_log(results: list[dict], dry_run: bool) -> None:
    APPLY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "applied_at": datetime.now(UTC).isoformat(),
        "dry_run":    dry_run,
        "results":    results,
    }

    history: list = []
    if APPLY_LOG_FILE.exists():
        try:
            history = json.loads(APPLY_LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            history = []

    history.append(log_entry)
    # Manter apenas os ultimos 50 registros
    history = history[-50:]

    APPLY_LOG_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Log salvo em %s", APPLY_LOG_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica mudancas aprovadas do pending_changes.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--approved-ids", type=str, help="IDs separados por virgula (ex: change_0,change_1)")
    group.add_argument("--all", action="store_true", help="Aprova todas as mudancas pendentes")
    parser.add_argument("--dry-run", action="store_true", help="Simula sem modificar arquivos")
    args = parser.parse_args()

    if not PENDING_FILE.exists():
        logger.error("pending_changes.json nao encontrado em %s", PENDING_FILE)
        sys.exit(1)

    pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    changes = pending.get("pending_for_approval", [])

    if not changes:
        logger.info("Nenhuma mudanca pendente em pending_changes.json.")
        sys.exit(0)

    # Determina quais IDs aprovar
    if args.all:
        approved_ids = {c["id"] for c in changes if "id" in c}
    else:
        approved_ids = {s.strip() for s in args.approved_ids.split(",") if s.strip()}

    logger.info("IDs aprovados: %s", sorted(approved_ids))
    logger.info("Modo: %s", "DRY-RUN" if args.dry_run else "REAL")

    results: list[dict] = []
    any_error = False

    for change in changes:
        cid = change.get("id", "?")
        desc = change.get("description", "")[:100]

        if cid not in approved_ids:
            logger.info("  Pulando %s (nao aprovado)", cid)
            results.append({"id": cid, "status": "skipped", "description": desc})
            continue

        spec = change.get("action_spec")
        if not spec:
            logger.warning("  %s: sem action_spec definido — pulando", cid)
            results.append({"id": cid, "status": "skipped", "description": desc,
                            "error": "action_spec ausente"})
            continue

        logger.info("  Aplicando %s: %s", cid, change.get("target", ""))
        try:
            result_msg = apply_action_spec(spec, PROJECT_ROOT, args.dry_run)
            results.append({"id": cid, "status": "ok", "description": desc, "result": result_msg})
            logger.info("    OK %s", result_msg)
        except Exception as exc:
            any_error = True
            logger.error("    ERRO %s: %s", cid, exc)
            results.append({"id": cid, "status": "erro", "description": desc, "error": str(exc)})

    # Salva log e notifica Discord
    save_log(results, args.dry_run)
    post_discord(results, args.dry_run)

    # Sumario
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "erro")
    logger.info("Concluido: %d ok | %d erros | %d skipped", ok_count, err_count,
                sum(1 for r in results if r["status"] == "skipped"))

    if any_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
