from pathlib import Path

from sqlalchemy import select

from ivan.sports.common.odds import get_odds, parse_odds 
from ivan.sports.common.odds import get_game_odds, parse_game_odds
from ivan.sports.nfl.models import NFLOdds, NFLPlayer
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.fpts_distribution_calculations import generate_and_fit_all

from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import select, distinct

from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLPlayer, NFLOdds, NFLGameOdds

DIAG_DIR = Path("data/nfl/diagnostics")

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
    print(f"[diagnostics] wrote mismatch reports to: {out_dir} (timestamp {ts})")



engine = init_db("nfl", None)
session = get_session(engine)


api_key = '58699e4fe47856c9182db84884772536' 
relevant_teams = set(session.scalars(select(NFLPlayer.team).distinct()))

odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams))
df_odds = parse_odds(odds)

records = df_odds.to_dict(orient='records') 
session.bulk_insert_mappings(NFLOdds, records)
session.commit()

game_odds_json, headers = get_game_odds(
    sport="americanfootball_nfl",
    api_key=api_key,
    regions="us",
    markets="h2h,spreads,totals",  # adjust to control quota/cost
    oddsFormat="american",
    dateFormat="iso",
    bookmakers="draftkings"
)

if game_odds_json:
    df_game_odds = parse_game_odds(game_odds_json)

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
        print(f"[odds-api] game_odds inserted: {len(df_game_odds)} rows | remaining={rem} used={used} last={last}")
else:
    print("[odds-api] no game odds returned; nothing inserted.")

write_odds_pool_mismatch_report(session, df_odds)

model_dir = 'data/nfl/models'
generate_and_fit_all(session, Path(model_dir), n_samples=10000, verbose=False, delete_players_without_odds=True, save_fpts_sample=True)#TODO: add a plot arg to plot the actual distribution of sampled values against the fitted model. 
