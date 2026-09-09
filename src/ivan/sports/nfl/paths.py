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

# Logged-in DK Chrome profile used only by the historical contest scraper.
CHROME_PROFILE_DIR = PROJECT_ROOT / "chrome-selenium" / "dk-profile"
