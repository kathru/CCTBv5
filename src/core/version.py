"""
Version module — reads version from _version.txt at project root.

  X.Y.Z
  ├── X  Major version (architecture generation)
  ├── Y  Minor version (current phase series — bump manually on milestone)
  └── Z  Patch (increments per phase commit — updated manually in _version.txt)

_version.txt is the single source of truth.
  • Updated manually on each phase commit (e.g. "5.6.270")
  • Deployed to container via docker cp alongside code changes
  • No git dependency — works regardless of container git history

Fallback chain:
  1. _version.txt  (preferred — explicitly controlled)
  2. _git_patch + hardcoded MINOR  (Docker build-time bake)
  3. "5.6.0"  (last resort)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAJOR = 5
MINOR = 6  # Current phase series — bump manually for next milestone

_BUILD_DIR   = Path(__file__).parent.parent.parent  # /app
_VERSION_FILE = _BUILD_DIR / "_version.txt"
_PATCH_FILE   = _BUILD_DIR / "_git_patch"


def get_version() -> str:
    """Return X.Y.Z version string from _version.txt."""
    # Priority 1: explicit _version.txt (updated per deploy)
    if _VERSION_FILE.exists():
        v = _VERSION_FILE.read_text().strip()
        if v:
            return v

    # Priority 2: _git_patch baked at Docker build time
    if _PATCH_FILE.exists():
        patch = _PATCH_FILE.read_text().strip() or "0"
        return f"{MAJOR}.{MINOR}.{patch}"

    return f"{MAJOR}.{MINOR}.0"


def get_version_info() -> dict:
    """Return full version metadata."""
    version = get_version()
    parts = version.split(".")
    return {
        "version": version,
        "major": int(parts[0]),
        "minor": int(parts[1]),
        "patch": int(parts[2]),
        "title_desktop": f"Claude Code Trading Bot v{version}",
        "title_mobile": f"CCTB v.{version}",
    }
