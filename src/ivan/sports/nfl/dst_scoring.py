"""
DraftKings DST points per team-game, from nfl_stats and nfl_schedule.

The defense's points are what the *opposing offense* gave up (sacks taken,
interceptions thrown, fumbles lost, kicks blocked) plus its own touchdowns and
safeties and the points-allowed bracket. Points allowed come from the final
score in nfl_schedule, less touchdowns the opponent's defense scored, which
DraftKings does not charge to the DST.

Writes nfl_dst_game: one row per (season, week, team) with the components, so
the team correlation model, the positional table and the DST odds model all
read the same number.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from ivan.database.session import init_db, get_session, read_sql as _read_sql


def pa_bonus(pa: float) -> int:
    if pa <= 0: return 10
    if pa <= 6: return 7
    if pa <= 13: return 4
    if pa <= 20: return 1
    if pa <= 27: return 0
    if pa <= 34: return -1
    return -4


def dst_points_frame(session, years: list[int]) -> pd.DataFrame:
    years_sql = ",".join(str(int(y)) for y in years)
    st = _read_sql(session, text(f"""
        SELECT season, season_type, week, team, opponent_team,
               sacks_suffered, passing_interceptions,
               rushing_fumbles_lost, receiving_fumbles_lost, sack_fumbles_lost,
               fg_blocked, pat_blocked, def_tds, special_teams_tds, def_safeties
        FROM nfl_stats WHERE season IN ({years_sql})"""))
    if st.empty:
        return pd.DataFrame(columns=["season", "season_type", "week", "team", "opponent", "dst_points"])
    num_cols = ["sacks_suffered", "passing_interceptions", "rushing_fumbles_lost", "receiving_fumbles_lost",
                "sack_fumbles_lost", "fg_blocked", "pat_blocked", "def_tds", "special_teams_tds", "def_safeties"]
    for c in num_cols:
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)

    keys = ["season", "season_type", "week", "team"]
    g = st.groupby(keys + ["opponent_team"], as_index=False)[num_cols].sum()

    # What this team's offense surrendered -> credited to the opponent's DST.
    given = g.rename(columns={"team": "opponent", "opponent_team": "team"})
    given["sacks"] = given.sacks_suffered
    given["takeaways"] = given.passing_interceptions + given.rushing_fumbles_lost + given.receiving_fumbles_lost + given.sack_fumbles_lost
    given["blocks"] = given.fg_blocked + given.pat_blocked
    given["opp_def_tds"] = given.def_tds          # scored by the opponent's defense: not points allowed
    given = given[keys + ["opponent", "sacks", "takeaways", "blocks", "opp_def_tds"]]

    # What this team's defense and special teams scored themselves.
    own = g[keys].copy()
    own["def_tds"] = g.def_tds + g.special_teams_tds
    own["safeties"] = g.def_safeties

    df = given.merge(own, on=keys, how="left").fillna({"def_tds": 0.0, "safeties": 0.0})

    sched = _read_sql(session, text(f"""
        SELECT season, game_type, week, home_team, away_team, home_score, away_score
        FROM nfl_schedule WHERE season IN ({years_sql}) AND home_score IS NOT NULL"""))
    home = sched.rename(columns={"home_team": "team", "away_score": "opp_score"})[["season", "week", "team", "opp_score"]]
    away = sched.rename(columns={"away_team": "team", "home_score": "opp_score"})[["season", "week", "team", "opp_score"]]
    scores = pd.concat([home, away], ignore_index=True)
    df = df.merge(scores, on=["season", "week", "team"], how="inner")

    df["points_allowed"] = (df.opp_score - 7 * df.opp_def_tds).clip(lower=0)
    df["dst_points"] = (df.sacks + 2 * df.takeaways + 2 * df.blocks + 6 * df.def_tds + 2 * df.safeties
                        + df.points_allowed.map(pa_bonus)).astype(float)
    cols = ["season", "season_type", "week", "team", "opponent", "dst_points",
            "points_allowed", "sacks", "takeaways", "blocks", "def_tds", "safeties"]
    return df[cols].sort_values(["season", "week", "team"]).reset_index(drop=True)


def build_dst_game_table(db_url: str | None, years: list[int], out_table: str = "nfl_dst_game",
                         session=None) -> int:
    """Rebuild nfl_dst_game for the given seasons. Returns rows written."""
    if session is None:
        engine = init_db("nfl", db_url)
        session = get_session(engine)
    out = dst_points_frame(session, years)
    engine = session.get_bind()
    with engine.begin() as con:
        out.to_sql(out_table, con, if_exists="replace", index=False)
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{out_table}_team_week ON {out_table}(season, week, team)"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{out_table}_opp_week  ON {out_table}(season, week, opponent)"))
    return len(out)
