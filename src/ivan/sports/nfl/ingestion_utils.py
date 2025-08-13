import os
from datetime import datetime
import pandas as pd
from difflib import get_close_matches
from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLPlayer

def attach_player_ids_and_filter_players(session, df_odds: pd.DataFrame, strict: bool = False, report_dir: str = "data/nfl/logs") -> pd.DataFrame:
    """
    - Normalizes odds names (using your normalize_name + overrides)
    - Drops DST rows
    - Attaches DK player_id to odds rows (by name, optionally team)
    - Logs a CSV of unmatched odds names with suggestions
    - Filters NFLPlayer table down to the matched player_ids
    - Returns odds DataFrame with a 'player_id' column (matched only)
    """
    os.makedirs(report_dir, exist_ok=True)

    # Normalize odds names & ignore DST
    df = df_odds.copy()
    df["player_norm"] = df["player"].map(normalize_name)
    df = df[~df["player_norm"].str.contains(r"\bdst\b", na=False)].copy()

    # Snapshot DK players we just inserted
    q = session.query(
        NFLPlayer.id.label("dk_player_id"),
        NFLPlayer.name,
        NFLPlayer.team,
        NFLPlayer.position,
    )
    players_df = pd.read_sql(q.statement, session.bind)
    players_df["player_norm"] = players_df["name"].map(normalize_name)

    # Try a (player_norm, team) join if team exists in odds; else fall back to name-only
    if "team" in df.columns and "team" in players_df.columns:
        m = df.merge(
            players_df[["dk_player_id", "player_norm", "team"]],
            left_on=["player_norm", "team"],
            right_on=["player_norm", "team"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"dk_player_id": "player_id"})
    else:
        m = df.merge(
            players_df[["dk_player_id", "player_norm"]],
            on="player_norm",
            how="left",
            validate="many_to_one",
        ).rename(columns={"dk_player_id": "player_id"})

    # Handle unmatched odds names
    unmatched = sorted(m.loc[m["player_id"].isna(), "player_norm"].unique().tolist())
    if unmatched:
        dk_names = set(players_df["player_norm"])
        sugg_rows = [{"player_norm": nm, "suggestions": ", ".join(get_close_matches(nm, dk_names, n=3, cutoff=0.6))} for nm in unmatched]
        report_path = os.path.join(report_dir, f"unmatched_odds_{datetime.now():%Y%m%d_%H%M%S}.csv")
        pd.DataFrame(sugg_rows).to_csv(report_path, index=False)
        if strict:
            raise RuntimeError(
                f"{len(unmatched)} odds names did not match any DK player after normalize_name(). "
                f"See {report_path}. Add overrides to ivan.core.utils._NAME_OVERRIDES and rerun."
            )
        else:
            # Warn and drop
            import logging
            logging.warning(f"{len(unmatched)} odds names did not match; suggestions → {report_path}. Dropping those rows.")
            m = m[m["player_id"].notna()].copy()

    # Now filter NFLPlayer to only the matched ids
    keep_ids = m["player_id"].astype(int).unique().tolist()
    if keep_ids:  # safety guard so we don't wipe the table if odds were empty
        deleted = (
            session.query(NFLPlayer)
            .filter(~NFLPlayer.id.in_(keep_ids))
            .delete(synchronize_session=False)
        )
        session.commit()
        import logging
        logging.info(f"✔ Filtered NFLPlayer to {len(keep_ids)} players with odds (removed {deleted}).")
    else:
        import logging
        logging.warning("No matched player_ids from odds; NFLPlayer table left as-is.")

    return m
