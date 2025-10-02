from pathlib import Path

from sqlalchemy import select

from ivan.sports.common.odds import get_odds, parse_odds 
from ivan.sports.common.odds import get_game_odds, parse_game_odds
from ivan.sports.nfl.models import NFLOdds, NFLPlayer, ProjectionSnapshot
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.fpts_distribution_calculations import generate_and_fit_all

from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import select, distinct

from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLPlayer, NFLOdds, NFLGameOdds

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s %(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")   


DIAG_DIR = Path("data/nfl/diagnostics")

def create_projection_snapshot(session):
    """
    Snapshot current NFLPlayer projections.
    Run this AFTER generate_and_fit_all() updates projections.
    """
    from ivan.core.utils import normalize_name
    from ivan.sports.nfl.models import NFLPlayer, ProjectionSnapshot
    
    # Use today's date at midnight UTC for consistent matching
    snapshot_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Delete any existing snapshot for today (idempotent updates)
    session.query(ProjectionSnapshot).filter(
        ProjectionSnapshot.snapshot_date == snapshot_date
    ).delete()
    
    # Get all current players
    players = session.query(NFLPlayer).all()
    
    snapshot_records = []
    for player in players:
        snapshot_records.append({
            'snapshot_date': snapshot_date,
            'player_name': player.name,
            'normalized_name': normalize_name(player.name),
            'team': player.team,
            'opponent': player.opponent,
            'position': player.position,
            'salary': player.salary,
            'roster_slot_id': player.roster_slot_id,
            'fpts_mean': player.fpts_mean or 0.0,
            'fpts_median': player.fpts_p50 or 0.0,
            'fpts_std': player.fpts_std or 0.0,
        })
    
    if snapshot_records:
        session.bulk_insert_mappings(ProjectionSnapshot, snapshot_records)
        session.commit()
        logging.info(f"[snapshot] Created projection snapshot for {snapshot_date.date()} "
                    f"with {len(snapshot_records)} players")

def write_odds_pool_mismatch_report(session, df_odds: pd.DataFrame, out_dir: Path = DIAG_DIR) -> None:
    """
    Writes three diagnostics files:
      1) odds_not_in_pool_YYYYMMDD_HHMMSS.csv      ← players with odds but not in DK pool
      2) dk_players_missing_odds_YYYYMMDD_HHMMSS.csv ← DK players (by normalized name) with no odds match (at-risk deletions)
      3) name_suggestions_YYYYMMDD_HHMMSS.json     ← suggested renames (conservative, suffix/spacing fixes only)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Distinct normalized names present in odds ---
    odds_names = set(df_odds["player"].astype(str).map(normalize_name))

    # --- DK pool (as stored). Normalize here for safe matching ---
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
        # Include a sample of markets for each `odds_only` name for easier triage
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

    # 2) DK players that have NO odds match (normalized) → at-risk deletions
    at_risk = dk_df[~dk_df["normalized"].isin(odds_names)].copy()
    at_risk = at_risk[["id", "name", "normalized", "team", "position"]].sort_values(["team","position","name"])
    if not at_risk.empty:
        at_risk.to_csv(out_dir / f"dk_players_missing_odds_{ts}.csv", index=False)

    # 3) Conservative rename suggestions (suffix-only / punctuation-only)
    #    We DO NOT do fuzzy matching per your preference—just safe suffix trims.
    suffixes = (" jr", " sr", " ii", " iii", " iv", " v")
    suggestions = {}
    for _, row in at_risk.iterrows():
        nm = row["normalized"]
        cand = nm
        for sfx in suffixes:
            if cand.endswith(sfx):
                cand = cand[: -len(sfx)]
                cand = cand.strip()
        # If dropping a suffix fixes it, propose that suggestion
        if cand and cand in odds_names and cand != nm:
            # store suggestion under the exact DK display name for your rename dict
            suggestions[row["name"]] = cand

    if suggestions:
        import json
        with open(out_dir / f"name_suggestions_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)

    # Small console ping so the run log highlights the files
    logging.info(f"[update_odds]    Wrote diagnostics to {out_dir}:\n"
                 f"                 1) odds_not_in_pool_{ts}.csv ({len(odds_not_in_pool)})\n"
                 f"                 2) dk_players_missing_odds_{ts}.csv ({len(at_risk)})\n"
                 f"                 3) name_suggestions_{ts}.json ({len(suggestions)})")



engine = init_db("nfl", None)
session = get_session(engine)

logging.info("[update_odds][1] ✔ Connected to database")

api_key = '58699e4fe47856c9182db84884772536' 
relevant_teams = set(session.scalars(select(NFLPlayer.team).distinct()))

logging.debug(f"Teams to fetch odds for: {sorted(relevant_teams)} ({len(relevant_teams) // 2} games)")

odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams))

logging.debug("raw odds from api: " + str(odds))
df_odds = parse_odds(odds)
logging.debug("parsed odds: " + str(df_odds.head()))
logging.info(f"[update_odds][2] ✔ Parsed {len(df_odds)} odds rows from API response")

records = df_odds.to_dict(orient='records') 
session.bulk_insert_mappings(NFLOdds, records)
session.commit()
logging.info(f"[update_odds][3] ✔ Inserted {len(records)} rows into nfl_odds table")

game_odds_json, headers = get_game_odds(
    sport="americanfootball_nfl",
    api_key=api_key,
    regions="us",
    markets="h2h,spreads,totals",  # adjust to control quota/cost
    oddsFormat="american",
    dateFormat="iso",
    bookmakers="draftkings"
)
logging.debug("raw game odds from api: " + str(game_odds_json))
logging.info(f"[update_odds][4] ✔ Fetched game odds from API")

if game_odds_json:
    df_game_odds = parse_game_odds(game_odds_json)
    logging.info(f"[update_odds][5] ✔ Parsed {len(df_game_odds)} game odds rows from API response")

    # Optional: drop duplicates created by books mirroring same snapshot
    if not df_game_odds.empty:
        dedup_cols = ["game_id","bookmaker","market","outcome_name","point","last_update"]
        df_game_odds = df_game_odds.drop_duplicates(subset=dedup_cols)

        # bulk insert
        session.bulk_insert_mappings(NFLGameOdds, df_game_odds.to_dict(orient="records"))
        session.commit()

        # Quick visibility on quota usage
        rem = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        last = headers.get("x-requests-last")
        logging.info(f"[update_odds][6] ✔ Inserted {len(df_game_odds)} rows into nfl_game_odds table")
        logging.info(f"[update_odds]    the-odds-api quota used: {used} | remaining: {rem} | last request cost: {last}")
else:
    logging.error("[update_odds][5] No game odds returned from API")

write_odds_pool_mismatch_report(session, df_odds)

model_dir = 'data/nfl/models'

logging.info(f"[update_odds][7][1] Generating and fitting player FPTS distributions")
generate_and_fit_all(session, Path(model_dir), n_samples=10000)

logging.info(f"[update_odds][7][2] Creating projection snapshot")
create_projection_snapshot(session)

logging.info(f"[update_odds][8] ✓ Complete")