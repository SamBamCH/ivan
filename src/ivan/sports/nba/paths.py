# src/ivan/sports/nba/paths.py
"""
Canonical on-disk layout for NBA data. Everything is anchored to the project
root so library defaults and scripts behave the same regardless of cwd.
"""
from ivan.core.utils import PROJECT_ROOT

NBA_DATA_DIR = PROJECT_ROOT / "data" / "nba"
INPUT_DIR    = NBA_DATA_DIR / "input"
OUTPUT_DIR   = NBA_DATA_DIR / "output"      # cross-slate outputs only (scorecards, backtests); a slate's files are under slates/<dg>/output
MODELS_DIR   = NBA_DATA_DIR / "models"
DIAG_DIR     = NBA_DATA_DIR / "diagnostics"
LOGS_DIR     = NBA_DATA_DIR / "logs"

# Everything that belongs to one slate (its pool database, edits, grids, projection store, entries, outputs) lives under
# data/nba/slates/<draft group>/ and is reached through ivan.sports.nba.slates (paths of the current slate), not here.

# Logged-in DK Chrome profile used only by the historical contest scraper.
CHROME_PROFILE_DIR = PROJECT_ROOT / "chrome-selenium" / "dk-profile"
