"""
Version module — derives semantic version from git history.

  X.Y.Z
  ├── X  Major version (hardcoded — architecture generation)
  ├── Y  Structural changes (number of git tags = milestone releases)
  └── Z  Commit count (total commits on current branch)

Examples:
  5.0.27  → v5, no tags yet, 27 commits
  5.2.41  → v5, 2 tagged releases, 41 commits
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MAJOR = 5

# Files written by Dockerfile build stage from git metadata
_BUILD_DIR = Path(__file__).parent.parent.parent  # /app
_MINOR_FILE = _BUILD_DIR / "_git_minor"
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

    Y and Z are read from files generated at Docker build time (from git).
    Falls back to live git commands when running outside Docker (dev mode).
    """
    if _MINOR_FILE.exists() and _PATCH_FILE.exists():
        # Docker: read pre-computed values from build stage
        minor = _read_file(_MINOR_FILE)
        patch = _read_file(_PATCH_FILE)
    else:
        # Dev: compute live from git
        tags_output = _run(["git", "tag"])
        minor = str(len([t for t in tags_output.splitlines() if t.strip()]))
        patch = _run(["git", "rev-list", "--count", "HEAD"], default="0")

    return f"{MAJOR}.{minor}.{patch}"


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
