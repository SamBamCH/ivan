# src/ivan/sports/nba/paths.py
"""
Canonical on-disk layout for NBA data. Everything is anchored to the project
root so library defaults and scripts behave the same regardless of cwd.
"""
from ivan.core.utils import PROJECT_ROOT

NBA_DATA_DIR = PROJECT_ROOT / "data" / "nba"
INPUT_DIR    = NBA_DATA_DIR / "input"
OUTPUT_DIR   = NBA_DATA_DIR / "output"
MODELS_DIR   = NBA_DATA_DIR / "models"
DIAG_DIR     = NBA_DATA_DIR / "diagnostics"
LOGS_DIR     = NBA_DATA_DIR / "logs"

DEFAULT_ENTRIES_CSV   = INPUT_DIR / "entries.csv"
PROJECTIONS_CSV       = INPUT_DIR / "stok.csv"
MINUTES_OVERRIDES_CSV = INPUT_DIR / "minutes_overrides.csv"
CHOSEN_LINEUPS_CSV    = OUTPUT_DIR / "chosen_lineups.csv"
LATESWAP_BEST_CSV     = OUTPUT_DIR / "lateswap_best.csv"
FIELD_MODEL_PKL       = MODELS_DIR / "field_distribution_model.pkl"

# Logged-in DK Chrome profile used only by the historical contest scraper.
CHROME_PROFILE_DIR = PROJECT_ROOT / "chrome-selenium" / "dk-profile"
