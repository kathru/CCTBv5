"""
Version module -- le a versao de _version.txt na raiz do projeto.

REGRA DE VERSIONAMENTO: vX.Y.Z
  X = 5        Versao geral do projeto (fixo: CCTBv5)
  Y = fase     Fase atual -- lido de phases.json -> current
               (Claude atualiza phases.json ao concluir cada nova fase)
  Z = commits  Total de commits no branch (git rev-list --count HEAD)

Fontes de verdade (raiz do container em /app/):
  _version.txt  "X.Y.Z" gerado pelo deploy.ps1, copiado pelo deploy_oracle.sh
  phases.json   registro de fases, mantido pelo Claude, copiado a cada deploy

Cadeia de fallback (se _version.txt ausente):
  1. phases.json + _git_patch (baked no Dockerfile)
  2. "5.0.0"
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAJOR = 5

_ROOT         = Path(__file__).parent.parent.parent  # /app
_VERSION_FILE = _ROOT / "_version.txt"
_PHASES_FILE  = _ROOT / "phases.json"
_PATCH_FILE   = _ROOT / "_git_patch"


def _read_phases_minor() -> int:
    """Le o numero da fase atual de phases.json."""
    try:
        if _PHASES_FILE.exists():
            data = json.loads(_PHASES_FILE.read_text())
            return int(data.get("current", 0))
    except Exception:
        pass
    return 0


def get_version() -> str:
    """Retorna string de versao X.Y.Z lida de _version.txt."""
    if _VERSION_FILE.exists():
        v = _VERSION_FILE.read_text().strip()
        if v:
            return v

    # Fallback: monta a partir de phases.json + _git_patch
    minor = _read_phases_minor()
    patch = _PATCH_FILE.read_text().strip() if _PATCH_FILE.exists() else "0"
    return f"{MAJOR}.{minor}.{patch}"


def get_version_info() -> dict:
    """Retorna metadados completos de versao."""
    version = get_version()
    parts   = version.split(".")
    major   = int(parts[0]) if len(parts) > 0 else MAJOR
    minor   = int(parts[1]) if len(parts) > 1 else 0
    patch   = int(parts[2]) if len(parts) > 2 else 0

    # Nome da fase atual para o titulo
    phase_name = ""
    try:
        if _PHASES_FILE.exists():
            data = json.loads(_PHASES_FILE.read_text())
            for p in data.get("phases", []):
                if p.get("n") == minor:
                    phase_name = p.get("name", "")
                    break
    except Exception:
        pass

    return {
        "version":       version,
        "major":         major,
        "minor":         minor,
        "patch":         patch,
        "phase_name":    phase_name,
        "title_desktop": f"Claude Code Trading Bot v{version}",
        "title_mobile":  f"CCTB v.{version}",
    }
