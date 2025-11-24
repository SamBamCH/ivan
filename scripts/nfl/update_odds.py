from pathlib import Path
import json
import logging

from sqlalchemy import select

from ivan.sports.common.odds import get_odds, parse_odds 
from ivan.sports.common.odds import get_game_odds, parse_game_odds
from ivan.sports.nfl.models import NFLOdds, NFLPlayer
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.ownership import project_ownership  # NEW
from ivan.sports.nfl.fpts_distribution_calculations import generate_and_fit_all

from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import select, distinct

from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLPlayer, NFLOdds, NFLGameOdds

logging.basicConfig(level=logging.INFO, format="%(message)s %(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")   

DIAG_DIR = Path("data/nfl/diagnostics")

def create_projection_snapshot(session):
    """
    Snapshot current NFLPlayer projections AND ownership.
    Run this AFTER generate_and_fit_all() and project_ownership() update projections.
    """
    from ivan.core.utils import normalize_name
    from ivan.sports.nfl.models import NFLPlayer, ProjectionSnapshot    
    # Get all current players
    players = session.query(NFLPlayer).all()
    gametimes = set()
    for player in players:
        gametimes.add(player.gametime)

    snapshot_date = min(gametimes)
    
    snapshot_records = []
    players_to_update = set()
    
    for player in players:
        norm_name = normalize_name(player.name)
        players_to_update.add(norm_name)
        
        snapshot_records.append({
            'player_id': player.id,  # DraftKings player ID
            'snapshot_date': snapshot_date,
            'player_name': player.name,
            'normalized_name': norm_name,
            'team': player.team,
            'opponent': player.opponent,
            'position': player.position,
            'salary': player.salary,
            'roster_slot_id': player.roster_slot_id,
            'fpts_mean': player.fpts_mean or 0.0,
            'fpts_median': player.fpts_p50 or 0.0,
            'fpts_std': player.fpts_std or 0.0,
            'ownership_proj': player.ownership_proj or 0.0,
        })
    
    if snapshot_records:
        # Delete only the specific players being updated for this date
        session.query(ProjectionSnapshot).filter(
            ProjectionSnapshot.snapshot_date == snapshot_date,
            ProjectionSnapshot.normalized_name.in_(players_to_update)
        ).delete(synchronize_session=False)
        
        session.bulk_insert_mappings(ProjectionSnapshot, snapshot_records)
        session.commit()
        logging.info(f"[snapshot] Created/updated projection snapshot for {snapshot_date.date()} "
                    f"with {len(snapshot_records)} players")


def write_odds_pool_mismatch_report(session, df_odds: pd.DataFrame, out_dir: Path = DIAG_DIR) -> None:
    """
    Writes three diagnostics files:
      1) odds_not_in_pool_YYYYMMDD_HHMMSS.csv
      2) dk_players_missing_odds_YYYYMMDD_HHMMSS.csv
      3) name_suggestions_YYYYMMDD_HHMMSS.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Distinct normalized names present in odds
    odds_names = set(df_odds["player"].astype(str).map(normalize_name))

    # DK pool
    dk_rows = (
        session.query(NFLPlayer.id, NFLPlayer.name, NFLPlayer.team, NFLPlayer.position)
        .all()
    )
    dk_df = pd.DataFrame(dk_rows, columns=["id", "name", "team", "position"])
    dk_df["normalized"] = dk_df["name"].astype(str).map(normalize_name)
    dk_normalized = set(dk_df["normalized"])

    # 1) Players with odds but not in DK pool
    odds_not_in_pool = sorted(odds_names - dk_normalized)
    if odds_not_in_pool:
        sample_markets = (
            df_odds.groupby("player")["market"]
                  .apply(lambda s: ",".join(sorted(set(s))[:6]))
                  .to_dict()
        )
        df1 = pd.DataFrame({
            "normalized_name": odds_not_in_pool,
            "example_markets": [sample_markets.get(nm, "") for nm in odds_not_in_pool],
        })
        df1.to_csv(out_dir / f"odds_not_in_pool_{ts}.csv", index=False)

    # 2) DK players that have NO odds match
    at_risk = dk_df[~dk_df["normalized"].isin(odds_names)].copy()
    at_risk = at_risk[["id", "name", "normalized", "team", "position"]].sort_values(["team","position","name"])
    if not at_risk.empty:
        at_risk.to_csv(out_dir / f"dk_players_missing_odds_{ts}.csv", index=False)

    # 3) Conservative rename suggestions
    suffixes = (" jr", " sr", " ii", " iii", " iv", " v")
    suggestions = {}
    for _, row in at_risk.iterrows():
        nm = row["normalized"]
        cand = nm
        for sfx in suffixes:
            if cand.endswith(sfx):
                cand = cand[: -len(sfx)].strip()
        if cand and cand in odds_names and cand != nm:
            suggestions[row["name"]] = cand

    if suggestions:
        with open(out_dir / f"name_suggestions_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)

    logging.info(f"[update_odds]    Wrote diagnostics to {out_dir}:\n"
                 f"                 1) odds_not_in_pool_{ts}.csv ({len(odds_not_in_pool)})\n"
                 f"                 2) dk_players_missing_odds_{ts}.csv ({len(at_risk)})\n"
                 f"                 3) name_suggestions_{ts}.json ({len(suggestions)})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

engine = init_db("nfl")
session = get_session(engine)

logging.info("[update_odds][1] ✓ Connected to database")

api_key = '58699e4fe47856c9182db84884772536' 
relevant_teams = set(session.scalars(select(NFLPlayer.team).distinct()))

logging.debug(f"Teams to fetch odds for: {sorted(relevant_teams)} ({len(relevant_teams) // 2} games)")

# Fetch player props odds
odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams))
logging.debug("raw odds from api: " + str(odds))
df_odds = parse_odds(odds)
logging.debug("parsed odds: " + str(df_odds.head()))
logging.info(f"[update_odds][2] ✓ Parsed {len(df_odds)} odds rows from API response")

# *** FIX: Clear existing player odds before inserting fresh ones ***
# This prevents stale odds from accumulating and allows clean re-runs
if not df_odds.empty:
    deleted = session.query(NFLOdds).delete(synchronize_session=False)
    session.commit()
    if deleted > 0:
        logging.info(f"[update_odds][2] Cleared {deleted} existing player odds rows")

records = df_odds.to_dict(orient='records') 
session.bulk_insert_mappings(NFLOdds, records)
session.commit()
logging.info(f"[update_odds][3] ✓ Inserted {len(records)} rows into nfl_odds table")

# Fetch game odds (spreads, totals, moneylines)
game_odds_json, headers = get_game_odds(
    sport="americanfootball_nfl",
    api_key=api_key,
    regions="us",
    markets="h2h,spreads,totals",
    oddsFormat="american",
    dateFormat="iso",
    bookmakers="draftkings"
)
logging.debug("raw game odds from api: " + str(game_odds_json))
logging.info(f"[update_odds][4] ✓ Fetched game odds from API")

if game_odds_json:
    df_game_odds = parse_game_odds(game_odds_json)
    logging.info(f"[update_odds][5] ✓ Parsed {len(df_game_odds)} game odds rows from API response")

    if not df_game_odds.empty:
        # Deduplicate within the API response itself
        dedup_cols = ["game_id","bookmaker","market","outcome_name","point","last_update"]
        df_game_odds = df_game_odds.drop_duplicates(subset=dedup_cols)
        
        # *** FIX: Delete existing odds for these games before inserting ***
        # This prevents UNIQUE constraint violations on re-runs
        game_ids = df_game_odds['game_id'].unique().tolist()
        if game_ids:
            deleted = (
                session.query(NFLGameOdds)
                .filter(NFLGameOdds.game_id.in_(game_ids))
                .delete(synchronize_session=False)
            )
            session.commit()
            if deleted > 0:
                logging.info(f"[update_odds][5] Cleared {deleted} existing odds rows for {len(game_ids)} game(s)")
        
        # Now insert fresh odds
        session.bulk_insert_mappings(NFLGameOdds, df_game_odds.to_dict(orient="records"))
        session.commit()

        rem = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        last = headers.get("x-requests-last")
        logging.info(f"[update_odds][6] ✓ Inserted {len(df_game_odds)} rows into nfl_game_odds table")
        logging.info(f"[update_odds]    the-odds-api quota used: {used} | remaining: {rem} | last request cost: {last}")
else:
    logging.error("[update_odds][5] No game odds returned from API")

# Diagnostics report
write_odds_pool_mismatch_report(session, df_odds)

# Generate FPTS distributions
model_dir = 'data/nfl/models'
logging.info(f"[update_odds][7] Generating and fitting player FPTS distributions")
generate_and_fit_all(session, Path(model_dir), n_samples=10000)

# *** NEW: Project ownership using field simulation ***
logging.info(f"[update_odds][8] Projecting player ownership")

# Load config for ownership projection settings
try:
    config_path = Path("config/nfl.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
except Exception as e:
    logging.warning(f"[update_odds] Could not load config: {e}, using defaults")
    config = {}

# Get ownership projection settings from config
ownership_config = config.get("ownership_projection", {})
num_simulations = ownership_config.get("num_simulations", 3)

try:
    ownership_results = project_ownership(
        session,
        config=config,
    )
    
    if ownership_results:
        logging.info(f"[update_odds][8] ✓ Projected ownership for {len(ownership_results)} players")
    else:
        logging.warning("[update_odds][8] ⚠️  No ownership projections generated (check if lineups exist)")
        logging.warning("[update_odds]     Run 'python scripts/nfl/generate_lineups.py' to create lineups first")
except Exception as e:
    logging.error(f"[update_odds][8] ✗ Ownership projection failed: {e}")
    logging.warning("[update_odds]     Continuing without ownership projections")

# Create projection snapshot (includes ownership now)
logging.info(f"[update_odds][9] Creating projection snapshot")
create_projection_snapshot(session)

logging.info(f"[update_odds][10] ✓ Complete")