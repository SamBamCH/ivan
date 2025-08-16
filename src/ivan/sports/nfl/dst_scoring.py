# src/ivan/sports/nfl/build_dst_from_pbp.py

from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from ivan.database.session import init_db, get_session

def _read_sql(session, stmt) -> pd.DataFrame:
    with session.bind.connect() as conn:
        return pd.read_sql(stmt, conn)

def _pa_bonus(pa: int) -> int:
    if pa == 0: return 10
    if 1 <= pa <= 6: return 7
    if 7 <= pa <= 13: return 4
    if 14 <= pa <= 20: return 1
    if 21 <= pa <= 27: return 0
    if 28 <= pa <= 34: return -1
    return -4

def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- NEW: build from OFFENSE events (negatives that give DST points to opponent)
def _load_offense_negatives_from_stats(session, years: list[int], table: str = "nfl_stats") -> pd.DataFrame:
    """
    Return per (season, week, OFFENSE team) the counts of offense-side negatives:
      sacks_taken, ints_thrown, fumbles_lost, blocked_kicks_allowed, safeties_taken, ret_tds_allowed (optional)
    We intentionally name the team column 'offense' here.
    """
    years_sql = ",".join(str(int(y)) for y in years)
    df = _read_sql(session, text(f"SELECT * FROM {table} WHERE season IN ({years_sql})"))
    if df.empty:
        return pd.DataFrame(columns=[
            "season","week","offense",
            "sacks_taken","ints_thrown","fumbles_lost"
        ])

    # Identify offense team/week/season columns
    offense_col = _first_present(df, ["recent_team", "team", "team_abbrev", "posteam"])
    week_col    = _first_present(df, ["week"])
    season_col  = _first_present(df, ["season"])
    if offense_col is None or week_col is None or season_col is None:
        raise RuntimeError("nfl_stats must include offense team + season + week (recent_team/team/team_abbrev, season, week).")

    # Map likely offense-side columns
    sacks_taken_col    = _first_present(df, ["sacks_taken", "sacked", "sacks"])  # often 'sacks' lives on QB/offense rows
    ints_thrown_col    = _first_present(df, ["interceptions_thrown", "interceptions", "int"])
    fumbles_lost_col   = _first_present(df, ["fumbles_lost", "rushing_fumbles_lost", "fum_lost"])

    for c in [sacks_taken_col, ints_thrown_col, fumbles_lost_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = (
        df.groupby([season_col, week_col, offense_col], dropna=False)
          .agg(
              sacks_taken            =(sacks_taken_col, "sum") if sacks_taken_col else ("season", "size"),
              ints_thrown            =(ints_thrown_col, "sum") if ints_thrown_col else ("season", "size"),
              fumbles_lost           =(fumbles_lost_col, "sum") if fumbles_lost_col else ("season", "size"),
          )
          .reset_index()
          .rename(columns={season_col: "season", week_col: "week", offense_col: "offense"})
    )

    # If a stat was missing, above produced a dummy count; force zeros instead.
    for c in ["sacks_taken","ints_thrown","fumbles_lost"]:
        if c not in agg.columns or (locals()[c.replace(" ", "_", -1) + "_col"] is None):
            if c not in agg.columns:
                agg[c] = 0
            else:
                # When we used dummy ("season","size") to satisfy agg signature, set to 0
                agg[c] = 0

    return agg[["season","week","offense","sacks_taken","ints_thrown","fumbles_lost"]]

def _load_pbp_keys_defense(session, years: list[int], pbp_table: str = "nfl_stats") -> pd.DataFrame:
    """
    Keys for rows known to exist in the PBP (defensive identity).
    We prefer a defensive team column; fall back to team/recent_team if that's all we have.
    """
    years_sql = ",".join(str(int(y)) for y in years)
    pbp = _read_sql(session, text(f"SELECT * FROM {pbp_table} WHERE season IN ({years_sql})"))
    if pbp.empty:
        return pd.DataFrame(columns=["season","week","team"])

    week_col   = _first_present(pbp, ["week"])
    season_col = _first_present(pbp, ["season"])
    def_col    = _first_present(pbp, ["defteam", "def_team", "defense_team", "recent_team", "team", "team_abbrev"])
    if week_col is None or season_col is None or def_col is None:
        return pd.DataFrame(columns=["season","week","team"])

    keys = (
        pbp[[season_col, week_col, def_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={season_col: "season", week_col: "week", def_col: "team"})
    )
    keys["season"] = keys["season"].astype(int)
    keys["week"]   = keys["week"].astype(int)
    return keys[["season","week","team"]]

def build_dst_game_table(db_url: str, years: list[int],
                         pa_table: str = "nfl_team_points_allowed",
                         pbp_table: str = "nfl_stats",
                         out_table: str = "nfl_dst_game") -> int:
    """
    Build per-(season, week, DEFENSE team) DST fantasy points *from offense-side negatives*.
    Output columns: season, week, team(DEF), opponent(OFF), dst_points
    """
    engine  = init_db("nfl", db_url)
    session = get_session(engine)

    # 1) OFFENSE negatives
    off_agg = _load_offense_negatives_from_stats(session, years, table="nfl_stats")

    if off_agg.empty:
        with engine.begin() as con:
            pd.DataFrame(columns=["season","week","team","opponent","dst_points"]).to_sql(
                out_table, con, if_exists="replace", index=False
            )
        return 0

    # 2) Points allowed table: identify the DEFENSE for each (season,week, offense)
    #    pa.team      = defense team
    #    pa.opponent  = offense team
    pa = _read_sql(session, text(f"SELECT season, week, team, opponent, points_allowed FROM {pa_table}"))
    if pa.empty:
        # We can still compute event-driven DST points, but no PA bonus without this table
        pa = pd.DataFrame(columns=["season","week","team","opponent","points_allowed"])

    # 3) Join offense stats to their matched defense via PA table (on offense == pa.opponent)
    df = off_agg.merge(
        pa,
        left_on=["season","week","offense"],
        right_on=["season","week","opponent"],
        how="inner"
    )
    if df.empty:
        with engine.begin() as con:
            pd.DataFrame(columns=["season","week","team","opponent","dst_points"]).to_sql(
                out_table, con, if_exists="replace", index=False
            )
        return 0

    # 4) Require presence in PBP for the DEFENSE (if desired)
    pbp_def_keys = _load_pbp_keys_defense(session, years, pbp_table=pbp_table)

    print(pbp_def_keys)
    if not pbp_def_keys.empty:
        df = df.merge(pbp_def_keys, on=["season","week","team"], how="inner")
        if df.empty:
            with engine.begin() as con:
                pd.DataFrame(columns=["season","week","team","opponent","dst_points"]).to_sql(
                    out_table, con, if_exists="replace", index=False
                )
            return 0

    # 5) Compute DST points FOR THE DEFENSE using offense negatives + PA bonus
    df["points_allowed"] = pd.to_numeric(df["points_allowed"], errors="coerce").fillna(0).astype(int)

    # Safe numeric coercion for all event columns
    for c in ["sacks_taken","ints_thrown","fumbles_lost","blocked_kicks_allowed","safeties_taken","ret_tds_allowed"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["dst_points"] = (
          df["sacks_taken"]           * 1
        + df["ints_thrown"]           * 2
        + df["fumbles_lost"]          * 2
        + df["ret_tds_allowed"]       * 6
        + df["safeties_taken"]        * 2
        + df["blocked_kicks_allowed"] * 2
        + df["points_allowed"].map(_pa_bonus)
    ).astype(float)

    out = (
        df[["season","week","team","opponent","dst_points"]]  # team = DEF, opponent = OFF
        .copy()
        .sort_values(["season","week","team"])
    )

    # 6) write
    with engine.begin() as con:
        out.to_sql(out_table, con, if_exists="replace", index=False)
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{out_table}_team_week ON {out_table}(season, week, team)"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{out_table}_opp_week  ON {out_table}(season, week, opponent)"))

    return len(out)
