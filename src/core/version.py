"""
Version module — derives semantic version from git history.

  X.Y.Z
  ├── X  Major version (hardcoded — architecture generation)
  ├── Y  Minor version (hardcoded — current evolution milestone/phase series)
  └── Z  Patch (commit count — auto-increments with every commit)

Scheme: v5.6.x
  5 = CCTBv5 architecture
  6 = Phase series 6.x (Equity Analytics, Distribution, Reality Check, ...)
  x = commit count — each phase commit produces v5.6.1, v5.6.2, ...
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MAJOR = 5
MINOR = 6  # Hardcoded: current phase series (bump manually for next major milestone)

# Files written by Dockerfile build stage from git metadata
_BUILD_DIR = Path(__file__).parent.parent.parent  # /app
_PATCH_FILE  = _BUILD_DIR / "_git_patch"


def _read_file(path: Path, default: str = "0") -> str:
    try:
        return path.read_text().strip() or default
    except Exception:
        return default


def _run(cmd: list[str], default: str = "0") -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or default
    except Exception:
        return default


def get_version() -> str:
    """Return X.Y.Z version string.

    Z (patch) is read from a file generated at Docker build time (from git).
    Falls back to live git commit count when running outside Docker (dev mode).
    """
    if _PATCH_FILE.exists():
        patch = _read_file(_PATCH_FILE)
    else:
        patch = _run(["git", "rev-list", "--count", "HEAD"], default="0")

    return f"{MAJOR}.{MINOR}.{patch}"


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
