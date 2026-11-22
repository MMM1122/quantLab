"""Application context shared by CLI, API and dashboard (composition root)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from quantlab.core.config import REPO_ROOT, Settings, load_settings
from quantlab.data.snapshots import SnapshotStore
from quantlab.storage.db import Database


@dataclass
class AppContext:
    settings: Settings
    db: Database
    snapshots: SnapshotStore


def build_context(data_dir: str | Path | None = None) -> AppContext:
    settings = load_settings(data_dir)
    db = Database(settings.db_path)
    return AppContext(settings=settings, db=db, snapshots=SnapshotStore(settings.snapshots_dir, db))


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Code version for provenance: git HEAD (+ -dirty), or QUANTLAB_GIT_COMMIT baked into images."""
    baked = os.environ.get("QUANTLAB_GIT_COMMIT", "").strip()
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return baked or None
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return baked or None
