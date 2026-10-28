# src/ivan/core/gitinfo.py
"""The commit the code is at, for run manifests: a replayed slate records which build produced it."""
from __future__ import annotations

import subprocess

from ivan.core.utils import PROJECT_ROOT


def git_state() -> dict:
    """{"commit": short sha, "dirty": uncommitted tracked changes, "subject": the commit's subject}; empty strings off a checkout."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    try:
        commit = run("rev-parse", "--short", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        subject = run("log", "-1", "--format=%s")
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "dirty": False, "subject": ""}
    return {"commit": commit, "dirty": dirty, "subject": subject}
