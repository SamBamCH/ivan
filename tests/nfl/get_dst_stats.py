# scripts/nfl/write_schedule_points_allowed.py
from __future__ import annotations

import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from ivan.database.session import init_db, get_session


def load_schedules_safe(years: list[int], include_types=("REG", "POST")) -> pd.DataFrame:
    """
    Load schedules for the given years and filter to REG/POST (default).
    Skips years that aren't available yet.
    """
    try:
        df = nfl.import_schedules(years)
    except Exception as e:
        raise RuntimeError(f"import_schedules failed for {years}: {e}")

    if "game_type" in df.columns and include_types:
        df = df[df["game_type"].isin(include_types)].copy()

    # keep a tidy set of columns; you can add more if you like
    keep = [
        "game_id", "season", "week",
        "home_team", "home_score",
        "away_team", "away_score",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"Schedules missing required columns: {missing}")

    return df[keep].reset_index(drop=True)


def build_team_points_allowed(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    From one schedule row (home vs away), produce two team-rows:
      - home team: points_allowed = away_score
      - away team: points_allowed = home_score
    """
    rows = []
    it = schedule_df.itertuples(index=False)
    for r in it:
        # home team view
        rows.append({
            "game_id":       r.game_id,
            "season":        int(r.season),
            "week":          int(r.week),
            "team":          r.home_team,
            "opponent":      r.away_team,
            "points_allowed": int(r.away_score if r.away_score is not None else 0),
        })
        # away team view
        rows.append({
            "game_id":       r.game_id,
            "season":        int(r.season),
            "week":          int(r.week),
            "team":          r.away_team,
            "opponent":      r.home_team,
            "points_allowed": int(r.home_score if r.home_score is not None else 0),
        })

    out = pd.DataFrame(rows, columns=[
        "game_id", "season", "week", "team", "opponent", "points_allowed"
    ])
    return out


def write_tables(engine, schedule_df: pd.DataFrame, pa_df: pd.DataFrame,
                 schedule_table="nfl_schedule",
                 pa_table="nfl_team_points_allowed") -> None:
    """
    Persist both tables; create useful indexes.
    Uses REPLACE for simplicity—switch to UPSERT if you want incremental loads.
    """
    schedule_df.to_sql(schedule_table, engine, if_exists="replace", index=False)
    pa_df.to_sql(pa_table, engine, if_exists="replace", index=False)

    with engine.begin() as con:
        # Indexes help joins in downstream DST build
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{pa_table}_game      ON {pa_table}(game_id)"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{pa_table}_team_week ON {pa_table}(season, week, team)"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{schedule_table}_game ON {schedule_table}(game_id)"))


def main(years: list[int]) -> None:
    engine = init_db("nfl")                 # your helper; returns SQLAlchemy engine
    session = get_session(engine)           # not used here, but handy if you want tx control
    sched = load_schedules_safe(years)
    team_pa = build_team_points_allowed(sched)
    write_tables(engine, sched, team_pa)
    print(f"✅ wrote {len(sched)} schedule rows and {len(team_pa)} team-PA rows.")


if __name__ == "__main__":
    # choose completed seasons for reproducible training artifacts
    main([2023, 2024])  # add 2022, 2021, ... as needed
