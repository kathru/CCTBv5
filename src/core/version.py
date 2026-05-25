"""
Version module -- le a versao de _version.txt na raiz do projeto.

REGRA DE VERSIONAMENTO: vX.Y.Z
  X = 5        Versao geral do projeto (fixo enquanto estivermos no CCTBv5)
  Y = fases    Total de fases implementadas -- lido de _phase.txt
               (atualizar _phase.txt manualmente ao concluir cada nova fase)
  Z = commits  Total de commits no branch (git rev-list --count HEAD)

Fontes de verdade (raiz do projeto / raiz do container em /app/):
  _version.txt  -- "X.Y.Z" gerado pelo deploy.ps1 e copiado ao container
  _phase.txt    -- numero da fase atual, atualizado manualmente por sessao

Cadeia de fallback (apenas se _version.txt nao existir):
  1. _version.txt   preferido, atualizado a cada deploy
  2. _phase.txt + _git_patch baked no Dockerfile
  3. "5.0.0"        ultimo recurso
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAJOR = 5   # Versao geral do projeto (fixo)

_ROOT         = Path(__file__).parent.parent.parent  # /app
_VERSION_FILE = _ROOT / "_version.txt"
_PHASE_FILE   = _ROOT / "_phase.txt"
_PATCH_FILE   = _ROOT / "_git_patch"
_MINOR_FILE   = _ROOT / "_git_minor"


def get_version() -> str:
    """Retorna string de versao X.Y.Z lida de _version.txt."""
    if _VERSION_FILE.exists():
        v = _VERSION_FILE.read_text().strip()
        if v:
            return v

    # Fallback: monta a partir dos arquivos individuais
    minor = _PHASE_FILE.read_text().strip() if _PHASE_FILE.exists() else "0"
    patch = _PATCH_FILE.read_text().strip() if _PATCH_FILE.exists() else "0"
    return f"{MAJOR}.{minor}.{patch}"


def get_version_info() -> dict:
    """Retorna metadados completos de versao."""
    version = get_version()
    parts   = version.split(".")
    major   = int(parts[0]) if len(parts) > 0 else MAJOR
    minor   = int(parts[1]) if len(parts) > 1 else 0
    patch   = int(parts[2]) if len(parts) > 2 else 0
    return {
        "version":       version,
        "major":         major,
        "minor":         minor,
        "patch":         patch,
        "title_desktop": f"Claude Code Trading Bot v{version}",
        "title_mobile":  f"CCTB v.{version}",
    }
