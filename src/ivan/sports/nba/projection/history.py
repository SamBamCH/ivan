"""
Trailing per-player state from the nba_stats table.

Everything here is "as of" a date: state_before(date) returns, for every
player, the exponentially weighted minutes and per-minute rates computed from
games strictly before that date. The whole table is processed once; the
backtest then asks for hundreds of dates cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

STATS = ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "fg3m"]

# Regular season (002) and playoffs (004). Preseason (001) and the all-star
# game (006) are noise for rotation purposes.
GAME_PREFIXES = ("002", "004")

MIN_HALFLIFE = 5      # games played; minutes move fast with role changes
RATE_HALFLIFE = 20    # games played; per-minute rates are stable (backtest 2026-10-06: 20 beat 5/10/40)
START_HALFLIFE = 3    # games played; starter share tracks the current lineup
PLAY_HALFLIFE = 5     # games dressed; how often a dressed player actually plays


def load_box_scores(session: Session) -> pd.DataFrame:
    """Box score rows for real games, one row per player-game, sorted by date."""
    q = f"""
        select game_id, game_date, player_name, team_abbreviation as team,
               position, minutes, dk_fpts, {", ".join(STATS)}
        from nba_stats
        where substr(game_id, 1, 3) in {GAME_PREFIXES}
        order by game_date, game_id
    """
    df = pd.read_sql(q, session.get_bind())
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["starter"] = df["position"].ne("BENCH").astype(float)
    df["played"] = df["minutes"] > 0
    return df


def _decay(halflife: int) -> float:
    return 0.5 ** (1.0 / halflife)


@dataclass
class HistoryState:
    """Per-player trailing state after each game they appeared in."""
    rows: pd.DataFrame          # one row per player-game with the state *after* that game
    team_rows: pd.DataFrame     # one row per team-game: trailing team fpts per game
    league_rates: dict[str, float]  # league per-minute rate per stat (prior for shrinkage)
    league_min_sd: float        # typical within-player minutes sd
    league_team_fpts: float     # league average team fantasy points per game

    def before(self, d: date) -> pd.DataFrame:
        """Latest state per player from games strictly before d, indexed by player."""
        sub = self.rows[self.rows["game_date"] < d]
        return sub.groupby("player_name").tail(1).set_index("player_name")

    def team_before(self, d: date) -> dict[str, float]:
        """Trailing team fantasy points per game, by team, from games before d."""
        sub = self.team_rows[self.team_rows["game_date"] < d]
        return sub.groupby("team").tail(1).set_index("team")["team_fpts_ewm"].to_dict()


def build_history(box: pd.DataFrame, min_halflife: int = MIN_HALFLIFE,
                  rate_halflife: int = RATE_HALFLIFE, team_halflife: int = 10) -> HistoryState:
    """
    Walk each player's games in order and record, after every game played:
      min_ewm      EWM of minutes (halflife MIN_HALFLIFE games)
      min_ewm_sd   EWM sd of minutes around min_ewm
      rate_<stat>  minute-weighted EWM of stat/minute (halflife RATE_HALFLIFE)
      eff_min      decayed minutes behind the rates (shrinkage weight)
      start_ewm    EWM of the starter flag (halflife START_HALFLIFE)
      play_ewm     EWM of "played when dressed" (halflife PLAY_HALFLIFE)
      games        games played so far
    A DNP game (dressed, 0 minutes) only moves play_ewm: it says nothing
    about minutes-when-playing or rates, but a lot about whether the coach
    will use him. Live, a DK pool lists every dressed player, so the baseline
    minutes are min_ewm * play_ewm.
    """
    played = box[box["played"]].copy()
    a_min, a_rate, a_st, a_pl = _decay(min_halflife), _decay(rate_halflife), _decay(START_HALFLIFE), _decay(PLAY_HALFLIFE)

    out = []
    for name, g in box.groupby("player_name", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        m_num = m_den = 0.0
        v_num = 0.0
        s_num = s_den = 0.0
        p_num = p_den = 0.0
        r_num = np.zeros(len(STATS)); r_den = 0.0
        n = 0
        mean = 0.0
        for row in g.itertuples(index=False):
            mins = float(row.minutes)
            p_num = a_pl * p_num + float(mins > 0); p_den = a_pl * p_den + 1.0
            if mins <= 0:
                if n == 0:
                    continue  # nothing to carry yet
                rec = dict(out[-1]); rec.update(game_date=row.game_date, game_id=row.game_id,
                                                team=row.team, play_ewm=p_num / p_den)
                out.append(rec)
                continue
            # minutes: EWM mean and variance
            m_num = a_min * m_num + mins; m_den = a_min * m_den + 1.0
            mean = m_num / m_den
            v_num = a_min * v_num + (mins - mean) ** 2
            # rates: minute-weighted
            r_num = a_rate * r_num + np.array([getattr(row, s) for s in STATS], dtype=float)
            r_den = a_rate * r_den + mins
            # starter share
            s_num = a_st * s_num + float(row.starter); s_den = a_st * s_den + 1.0
            n += 1
            rec = {
                "player_name": name, "team": row.team, "game_date": row.game_date,
                "game_id": row.game_id,
                "min_ewm": mean, "min_ewm_sd": float(np.sqrt(v_num / m_den)),
                "eff_min": r_den, "start_ewm": s_num / s_den, "play_ewm": p_num / p_den, "games": n,
            }
            for i, s in enumerate(STATS):
                rec[f"rate_{s}"] = r_num[i] / r_den
            out.append(rec)
    rows = pd.DataFrame(out)

    # team-level production: EWM of total fantasy points per team-game
    tg = played.groupby(["team", "game_date", "game_id"], sort=True)["dk_fpts"].sum().reset_index()
    a_t = _decay(team_halflife)
    trows = []
    for team, g in tg.groupby("team", sort=False):
        num = den = 0.0
        for row in g.sort_values(["game_date", "game_id"]).itertuples(index=False):
            num = a_t * num + float(row.dk_fpts); den = a_t * den + 1.0
            trows.append({"team": team, "game_date": row.game_date, "game_id": row.game_id,
                          "team_fpts_ewm": num / den})
    team_rows = pd.DataFrame(trows)

    league = {s: float(played[s].sum() / played["minutes"].sum()) for s in STATS}
    # within-player minutes sd for rotation players
    rot = rows[rows["games"] >= 10]
    sd = float(rot["min_ewm_sd"].median()) if len(rot) else 5.0
    return HistoryState(rows=rows, team_rows=team_rows, league_rates=league, league_min_sd=sd,
                        league_team_fpts=float(tg["dk_fpts"].mean()))
