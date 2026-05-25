"""
Version module -- le a versao de _version.txt na raiz do projeto.

REGRA DE VERSIONAMENTO: vX.Y.Z
  X = 5        Versao geral do projeto (fixo enquanto estivermos no CCTBv5)
  Y = fases    Numero total de fases implementadas (contagem de tags fase/* no git)
  Z = commits  Numero total de commits no branch atual (git rev-list --count HEAD)

Fonte de verdade: _version.txt (raiz do projeto / raiz do container em /app/)
  - Gerado automaticamente pelo deploy.ps1 antes de cada push
  - Copiado ao container via docker cp _version.txt /app/_version.txt
  - Lido aqui em runtime e exposto via /version e /health

Cadeia de fallback (apenas se _version.txt nao existir):
  1. _version.txt   -- preferido, atualizado a cada deploy
  2. _git_patch     -- baked no Dockerfile em build-time (menos preciso)
  3. "5.0.0"        -- ultimo recurso
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAJOR = 5   # Versao geral do projeto (fixo)

_BUILD_DIR    = Path(__file__).parent.parent.parent  # /app
_VERSION_FILE = _BUILD_DIR / "_version.txt"
_PATCH_FILE   = _BUILD_DIR / "_git_patch"


def get_version() -> str:
    """Retorna string de versao X.Y.Z lida de _version.txt."""
    if _VERSION_FILE.exists():
        v = _VERSION_FILE.read_text().strip()
        if v:
            return v

    # Fallback: _git_patch baked no Dockerfile
    if _PATCH_FILE.exists():
        patch = _PATCH_FILE.read_text().strip() or "0"
        minor = (Path(__file__).parent.parent.parent / "_git_minor").read_text().strip() \
            if (Path(__file__).parent.parent.parent / "_git_minor").exists() else "0"
        return f"{MAJOR}.{minor}.{patch}"

    return f"{MAJOR}.0.0"


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
