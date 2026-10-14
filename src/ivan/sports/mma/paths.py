# src/ivan/sports/mma/paths.py
"""
Canonical on-disk layout for MMA data. Everything is anchored to the project
root so library defaults and scripts behave the same regardless of cwd.
"""
from ivan.core.utils import PROJECT_ROOT

MMA_DATA_DIR = PROJECT_ROOT / "data" / "mma"
INPUT_DIR    = MMA_DATA_DIR / "input"
OUTPUT_DIR   = MMA_DATA_DIR / "output"
DIAG_DIR     = MMA_DATA_DIR / "diagnostics"

# The four hand-maintained inputs of a slate (see config/mma.json -> paths).
PLAYERS_CSV     = INPUT_DIR / "player_ids.csv"    # DraftKings "export to CSV" of the player pool
FIGHT_ODDS_CSV  = INPUT_DIR / "fight_odds.csv"    # per fight: u60 sec, 1st round, itd (American odds)
PLAYER_ODDS_CSV = INPUT_DIR / "player_odds.csv"   # per fighter: 1st round win, ITD, Win, sig strikes, takedowns
PAYOUTS_CSV     = INPUT_DIR / "payouts.csv"       # rank range -> prize, as copied from the DK contest page
