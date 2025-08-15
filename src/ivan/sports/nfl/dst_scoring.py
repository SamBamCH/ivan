# src/ivan/sports/nfl/build_dst_from_pbp.py
from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from ivan.database.session import init_db, get_session

# ---- small helper so pandas always gets a Connection (SA 2.0 safe) ----------
def _read_sql(session, stmt) -> pd.DataFrame:
    with session.bind.connect() as conn:
        return pd.read_sql(stmt, conn)

# -----------------------------------------------------------------------------

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

def _load_defense_events_from_stats(session, years: list[int], table: str = "nfl_stats") -> pd.DataFrame:
    years_sql = ",".join(str(int(y)) for y in years)
    df = _read_sql(session, text(f"SELECT * FROM {table} WHERE season IN ({years_sql})"))
    if df.empty:
        return pd.DataFrame()

    team_col   = _first_present(df, ["recent_team", "team", "team_abbrev"])
    week_col   = _first_present(df, ["week"])
    season_col = _first_present(df, ["season"])
    if team_col is None or week_col is None or season_col is None:
        raise RuntimeError("nfl_stats must include team, season, week columns (recent_team/team/team_abbrev).")

    sacks_col    = _first_present(df, ["sacks", "sack"])
    ints_col     = _first_present(df, ["interceptions", "interception"])
    fumrec_col   = _first_present(df, ["rushing_fumbles_lost", "fumbles_recovered", "fumbles_rec"])
    blocked_col  = _first_present(df, ["blocked_kicks", "blocked_kick", "blocked"])
    safeties_col = _first_present(df, ["safeties", "safety"])
    dst_td_col   = _first_present(df, ["defense_td", "def_td", "def_st_tds", "return_touchdown"])

    for c in [sacks_col, ints_col, fumrec_col, blocked_col, safeties_col, dst_td_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    group_keys = [season_col, week_col, team_col]
    agg_dict = {}
    if sacks_col:    agg_dict["sacks"]         = (sacks_col, "sum")
    if ints_col:     agg_dict["interceptions"] = (ints_col, "sum")
    if fumrec_col:   agg_dict["fumbles_rec"]   = (fumrec_col, "sum")
    if blocked_col:  agg_dict["blocked_kicks"] = (blocked_col, "sum")
    if safeties_col: agg_dict["safeties"]      = (safeties_col, "sum")
    if dst_td_col:   agg_dict["def_st_tds"]    = (dst_td_col, "sum")

    if not agg_dict:
        return pd.DataFrame(columns=["season","week","team","sacks","interceptions","fumbles_rec","blocked_kicks","safeties","def_st_tds"])

    def_agg = (
        df.groupby(group_keys, dropna=False)
          .agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_dict.items()})
          .reset_index()
          .rename(columns={season_col: "season", week_col: "week", team_col: "team"})
    )

    for col in ["sacks","interceptions","fumbles_rec","blocked_kicks","safeties","def_st_tds"]:
        if col not in def_agg.columns:
            def_agg[col] = 0

    return def_agg[["season","week","team","sacks","interceptions","fumbles_rec","blocked_kicks","safeties","def_st_tds"]]

# If you want to require PBP presence (even if it’s in the same table), this loader stays SA 2.0 safe.
def _load_pbp_keys(session, years: list[int], pbp_table: str = "nfl_stats") -> pd.DataFrame:
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
    Build per-(season, week, team) DST fantasy points from your nfl_stats rows,
    optionally requiring presence in {pbp_table}.
    """
    engine  = init_db("nfl", db_url)
    session = get_session(engine)

    # 1) defensive events from weekly stats
    def_agg = _load_defense_events_from_stats(session, years, table="nfl_stats")
    if def_agg.empty:
        raise RuntimeError("No defensive events found in nfl_stats for the requested years.")

    # 2) points allowed (provides opponent + PA)
    pa = _read_sql(session, text(f"SELECT season, week, team, opponent, points_allowed FROM {pa_table}"))

    # 3) restrict to keys present in pbp_table (optional; default pbp_table='nfl_stats')
    pbp_keys = _load_pbp_keys(session, years, pbp_table=pbp_table)

    # 4) inner joins ensure rows are in both PA and PBP keys
    df = def_agg.merge(pa, on=["season","week","team"], how="inner")
    df = df.merge(pbp_keys, on=["season","week","team"], how="inner")
    if df.empty:
        with engine.begin() as con:
            pd.DataFrame(columns=["season","week","team","opponent","dst_points"]).to_sql(
                out_table, con, if_exists="replace", index=False
            )
        return 0

    # 5) compute points
    df["points_allowed"] = pd.to_numeric(df["points_allowed"], errors="coerce").fillna(0).astype(int)
    df["dst_points"] = (
        df["sacks"].fillna(0).astype(int)          * 1
      + df["interceptions"].fillna(0).astype(int) * 2
      + df["fumbles_rec"].fillna(0).astype(int)   * 2
      + df["def_st_tds"].fillna(0).astype(int)    * 6
      + df["safeties"].fillna(0).astype(int)      * 2
      + df["blocked_kicks"].fillna(0).astype(int) * 2
      + df["points_allowed"].map(_pa_bonus)
    ).astype(float)

    out = (
        df[["season","week","team","opponent","dst_points"]]
        .copy()
        .sort_values(["season","week","team"])
    )

    # 6) write
    with engine.begin() as con:
        out.to_sql(out_table, con, if_exists="replace", index=False)
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{out_table}_team_week ON {out_table}(season, week, team)"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{out_table}_opp_week  ON {out_table}(season, week, opponent)"))

    return len(out)
