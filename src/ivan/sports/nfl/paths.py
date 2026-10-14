# src/ivan/sports/nfl/paths.py
"""
Canonical on-disk layout for NFL data. Everything is anchored to the project
root so library defaults and scripts behave the same regardless of cwd.
"""
from ivan.core.utils import PROJECT_ROOT

NFL_DATA_DIR = PROJECT_ROOT / "data" / "nfl"
INPUT_DIR    = NFL_DATA_DIR / "input"
OUTPUT_DIR   = NFL_DATA_DIR / "output"
MODELS_DIR   = NFL_DATA_DIR / "models"
DIAG_DIR     = NFL_DATA_DIR / "diagnostics"
LOGS_DIR     = NFL_DATA_DIR / "logs"

DEFAULT_ENTRIES_CSV = INPUT_DIR / "entries.csv"

def models_dir():
    """
    The model directory for the current clock: MODELS_DIR live, or the newest
    as-of version (MODELS_DIR/asof/<cutoff>) on or before a replay's lock
    (ivan.core.model_store). Loaders call this instead of using MODELS_DIR.
    """
    from ivan.core import clock, model_store
    return model_store.resolve(MODELS_DIR, clock.as_of()).path


def models_info() -> dict:
    """Which model version the current clock resolves to, for manifests and reports."""
    from ivan.core import clock, model_store
    return model_store.resolve(MODELS_DIR, clock.as_of()).describe()


# Logged-in DK Chrome profile used only by the historical contest scraper.
CHROME_PROFILE_DIR = PROJECT_ROOT / "chrome-selenium" / "dk-profile"
