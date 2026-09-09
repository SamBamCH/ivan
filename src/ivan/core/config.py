# src/ivan/core/config.py
"""
Single entry point for per-sport JSON config plus environment overrides.

Resolution order for anything path-like:
  1. explicit CLI argument (handled by the calling script)
  2. environment variable  (IVAN_<SPORT>_DB_URL, IVAN_ENTRIES_CSV)
  3. config/<sport>.json, with relative paths resolved against the project root
  4. the on-disk convention in ivan.sports.<sport>.paths
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ivan.core.utils import PROJECT_ROOT

# .env at the project root holds secrets (ODDS_API_KEY) and per-machine
# overrides. A missing file is fine.
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"


def config_path(sport: str) -> Path:
    return CONFIG_DIR / f"{sport}.json"


def resolve_path(p: str | os.PathLike) -> Path:
    """Absolute path; relative inputs are taken from the project root, never the cwd."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(sport: str = "nfl", path: str | os.PathLike | None = None) -> dict:
    """Load config/<sport>.json (or an explicit path) and resolve its `paths` block."""
    cfg_file = resolve_path(path) if path else config_path(sport)
    with open(cfg_file, encoding="utf-8") as f:
        cfg = json.load(f)

    paths = cfg.setdefault("paths", {})
    for key, val in list(paths.items()):
        if isinstance(val, str) and val:
            paths[key] = str(resolve_path(val))

    env_entries = os.environ.get("IVAN_ENTRIES_CSV")
    if env_entries:
        paths["entries_csv"] = str(resolve_path(env_entries))

    env_db = os.environ.get(f"IVAN_{sport.upper()}_DB_URL")
    if env_db:
        cfg["database_url"] = env_db

    return cfg


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} is not set. Add it to {PROJECT_ROOT / '.env'} or export it.")
    return val
