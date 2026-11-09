# src/ivan/sports/nba/gamesim/usage.py
"""
Layer C: usage. The five on the floor share the team's possessions.

Every possession ends in one player's hands: a shot, a trip to the line or a turnover. A player's **share** of each is
his rate per possession on the floor, measured from the play-by-play (possessions.player_games: his counts and the
possessions, attempts and misses that happened around him while he was on):

    fga_r   field-goal attempts per own possession on the floor      ast_r   assists per teammate make on the floor
    fg3a_r  threes attempted per own possession                      oreb_r  offensive rebounds per teammate miss on the floor
    fta_r   free-throw attempts per own possession                   dreb_r  defensive rebounds per opponent miss on the floor
    tov_r   turnovers per own possession                             stl_r   steals per opponent possession
    fg3_r   threes attempted per attempt (the mix, layer D)          blk_r   blocks per opponent attempt
    pf_r    personal fouls per opponent possession (layer B's hazard)

Each rate is a trailing exponentially weighted estimate (half-life RATE_HALFLIFE games) shrunk toward his position
group's league rate with PRIOR_OPPS pseudo-opportunities, as of any date (UsageHistory.before, walk-forward).

The pie. Per possession the team takes about one ending (FGA + TOV + 0.44 FTA = poss + OREB), and the team's own
trailing mix of endings per possession (TeamProfile: fga, fta, tov per possession) is what the five on the floor split:
player i's expected attempts over N possessions with that five are

    N x team_fga_per_poss x fga_r_i / sum_j fga_r_j        (the five's shares renormalised to the team's pie)

so when a star sits, the four others and his replacement absorb his share in proportion to their own, and the team still
ends every possession. Rebounds, assists, steals and blocks work on their own opportunities (misses, makes, opponent
possessions) and renormalise the same way where the pie is a team quantity (rebounds: the team's share of available
boards is a team property; assists: of the team's makes, the share assisted).

check(events, box, groups, start, end): the layer alone. For every player-game in the window the real possession
sequence of that game is replayed with the real fives (possessions.events on_off), and the player's attempts, trips,
turnovers, rebounds, assists, steals and blocks are predicted from the shares as of that date; against the per-minute
rate x his true minutes (the current model's way, trailing per-minute rates from the box score) on the same player-games.
Per stat: MAE and bias, overall, for starters, for the bench, and on the nights a 25+ minute teammate sat.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import possessions as X

RATE_HALFLIFE = 20.0          # games
PRIOR_OPPS = 150.0            # pseudo-opportunities toward the group prior
TEAM_HALFLIFE = 10.0          # games, the team's mix of possession endings
ABSENT_MIN = 25.0             # a teammate with this many trailing minutes who sat makes it a 'teammate out' night
# rate -> (numerator column, denominator column) in player_games; fg3_r's denominator is the player's own attempts
RATES: Dict[str, Tuple[str, str]] = {
    "fga_r": ("fga", "off_poss"), "fg3a_r": ("fg3a", "off_poss"), "fta_r": ("fta", "off_poss"), "tov_r": ("tov", "off_poss"), "fg3_r": ("fg3a", "fga"),
    "ast_r": ("ast", "team_fgm_on"), "oreb_r": ("oreb", "team_miss_on"), "dreb_r": ("dreb", "opp_miss_on"),
    "stl_r": ("stl", "def_poss"), "blk_r": ("blk", "opp_fga_on"), "pf_r": ("pf", "def_poss"),
}
# the stats the check scores: stat -> (rate, which opportunity count of the real game feeds the prediction, the team pie it renormalises to)
PIE: Dict[str, Tuple[str, str, Optional[str]]] = {
    "fga": ("fga_r", "off_poss", "fga_pp"), "fg3a": ("fg3a_r", "off_poss", "fg3a_pp"), "fta": ("fta_r", "off_poss", "fta_pp"), "tov": ("tov_r", "off_poss", "tov_pp"),
    "ast": ("ast_r", "team_fgm_on", "ast_share"), "oreb": ("oreb_r", "team_miss_on", "oreb_share"), "dreb": ("dreb_r", "opp_miss_on", "dreb_share"),
    "stl": ("stl_r", "def_poss", None), "blk": ("blk_r", "opp_fga_on", None),
}
PER_MINUTE = ["fga", "fta", "tov", "ast", "oreb", "dreb", "stl", "blk", "fg3a"]


# ----------------------------------------------------------------------------- the player state
def _decay(halflife: float) -> float:
    return 0.5 ** (1.0 / halflife)


@dataclass
class UsageHistory:
    """Trailing numerators and denominators per player after each game; league priors per position group."""
    rows: pd.DataFrame                     # player, team, game_date, game_id, minutes, n_<stat>, d_<rate>, games
    priors: Dict[str, Dict[str, float]]    # group -> rate -> league rate
    league: Dict[str, float]               # rate -> league rate over everyone
    groups: Dict[str, str]
    pm_league: Dict[str, float]            # stat -> league per-minute rate (the baseline's prior)

    def before(self, d: date) -> pd.DataFrame:
        sub = self.rows[self.rows.game_date < d]
        return sub.groupby("player").tail(1).set_index("player")

    def rate(self, state: pd.DataFrame, player: str, r: str) -> float:
        """The shrunk rate as of the state; the group prior for a player without history."""
        prior = self.priors.get(self.groups.get(player, "wing"), self.league).get(r, self.league[r])
        if player not in state.index:
            return prior
        row = state.loc[player]
        num, den = float(row[f"n_{r}"]), float(row[f"d_{r}"])
        return (num + PRIOR_OPPS * prior) / (den + PRIOR_OPPS)

    def per_minute(self, state: pd.DataFrame, player: str, stat: str) -> float:
        """The baseline: the stat per minute, trailing, shrunk toward the league per-minute rate with 60 pseudo-minutes (rates.PRIOR_MINUTES)."""
        prior = self.pm_league[stat]
        if player not in state.index:
            return prior
        row = state.loc[player]
        return (float(row[f"pm_{stat}"]) + 60.0 * prior) / (float(row["pm_min"]) + 60.0)


def build_usage(pg: pd.DataFrame, minutes: pd.DataFrame, groups: Dict[str, str], halflife: float = RATE_HALFLIFE) -> UsageHistory:
    """
    pg: possessions.player_games; minutes: game_id, player, minutes from nba_stats (normalized names); groups: player -> big | guard | wing.
    Walks every player's games in order, keeping the decayed numerators and denominators of every rate after each game.
    """
    df = pg.merge(minutes, on=["game_id", "player"], how="left")
    df["minutes"] = df.minutes.fillna(0.0)
    df["game_date"] = pd.to_datetime(df.game_date).dt.date
    a = _decay(halflife)
    out = []
    for name, g in df.groupby("player", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        num = {r: 0.0 for r in RATES}; den = {r: 0.0 for r in RATES}
        pm = {s: 0.0 for s in PER_MINUTE}; pm_min = 0.0; n = 0
        for row in g.itertuples(index=False):
            for r, (nc, dc) in RATES.items():
                num[r] = a * num[r] + float(getattr(row, nc)); den[r] = a * den[r] + float(getattr(row, dc))
            for s in PER_MINUTE:
                pm[s] = a * pm[s] + float(getattr(row, s))
            pm_min = a * pm_min + float(row.minutes); n += 1
            rec = dict(player=name, team=row.team, game_date=row.game_date, game_id=row.game_id, minutes=float(row.minutes), games=n, pm_min=pm_min)
            rec.update({f"n_{r}": num[r] for r in RATES}); rec.update({f"d_{r}": den[r] for r in RATES}); rec.update({f"pm_{s}": pm[s] for s in PER_MINUTE})
            out.append(rec)
    rows = pd.DataFrame(out)
    # league rates: totals over everyone, and per group
    league = {r: float(df[nc].sum() / max(1.0, df[dc].sum())) for r, (nc, dc) in RATES.items()}
    priors: Dict[str, Dict[str, float]] = {}
    df["group"] = df.player.map(lambda p: groups.get(p, "wing"))
    for grp, d in df.groupby("group"):
        priors[grp] = {r: float(d[nc].sum() / max(1.0, d[dc].sum())) for r, (nc, dc) in RATES.items()}
    tot_min = float(df.minutes.sum())
    pm_league = {s: float(df[s].sum() / max(1.0, tot_min)) for s in PER_MINUTE}
    return UsageHistory(rows=rows, priors=priors, league=league, groups=groups, pm_league=pm_league)


# ----------------------------------------------------------------------------- the team pie
@dataclass
class TeamProfile:
    """Per team, as of a date: endings per possession and the team's shares of its own and the opponent's misses and makes."""
    rows: pd.DataFrame     # team, game_date, game_id, fga_pp, fta_pp, tov_pp, oreb_share, dreb_share, ast_share
    league: Dict[str, float]

    def before(self, team: str, d: date) -> Dict[str, float]:
        sub = self.rows[(self.rows.team == team) & (self.rows.game_date < d)]
        if sub.empty:
            return dict(self.league)
        return {k: float(sub.iloc[-1][k]) for k in self.league}


def build_team_profile(seg: pd.DataFrame, pg: pd.DataFrame, halflife: float = TEAM_HALFLIFE) -> TeamProfile:
    """
    From team_segments (possessions, attempts, makes per team-game) and player_games (the players' own turnovers, rebounds,
    assists: a team turnover or a team rebound is nobody's stat, so the pies the players split leave them out).
    """
    tg = seg.groupby(["game_id", "game_date", "team", "opp"], as_index=False)[["poss", "fga", "fg3a", "fta", "fgm"]].sum()
    own = pg.groupby(["game_id", "team"], as_index=False)[["tov", "oreb", "dreb", "ast"]].sum()
    tg = tg.merge(own, on=["game_id", "team"], how="left").fillna({"tov": 0.0, "oreb": 0.0, "dreb": 0.0, "ast": 0.0})
    opp = tg[["game_id", "team", "fga", "fgm"]].rename(columns={"team": "opp", "fga": "opp_fga", "fgm": "opp_fgm"})
    tg = tg.merge(opp, on=["game_id", "opp"], how="left")
    tg["miss"] = tg.fga - tg.fgm; tg["opp_miss"] = tg.opp_fga - tg.opp_fgm
    tg["game_date"] = pd.to_datetime(tg.game_date).dt.date
    a = _decay(halflife)
    keys = {"fga_pp": ("fga", "poss"), "fg3a_pp": ("fg3a", "poss"), "fta_pp": ("fta", "poss"), "tov_pp": ("tov", "poss"),
            "oreb_share": ("oreb", "miss"), "dreb_share": ("dreb", "opp_miss"), "ast_share": ("ast", "fgm")}
    out = []
    for team, g in tg.groupby("team", sort=False):
        num = {k: 0.0 for k in keys}; den = {k: 0.0 for k in keys}
        for row in g.sort_values(["game_date", "game_id"]).itertuples(index=False):
            rec = dict(team=team, game_date=row.game_date, game_id=row.game_id)
            for k, (nc, dc) in keys.items():
                num[k] = a * num[k] + float(getattr(row, nc)); den[k] = a * den[k] + float(getattr(row, dc))
                rec[k] = num[k] / den[k] if den[k] > 0 else np.nan
            out.append(rec)
    rows = pd.DataFrame(out)
    league = {k: float(tg[nc].sum() / max(1.0, tg[dc].sum())) for k, (nc, dc) in keys.items()}
    return TeamProfile(rows=rows, league=league)


def split_pie(rates: Dict[str, float], total: float) -> Dict[str, float]:
    """`total` expected endings over the five, split by their rates (renormalised): player -> expected count."""
    s = sum(max(0.0, v) for v in rates.values())
    if s <= 0:
        return {p: total / len(rates) for p in rates}
    return {p: total * max(0.0, v) / s for p, v in rates.items()}


# ----------------------------------------------------------------------------- the check
def _absent_nights(usage: UsageHistory, box_minutes: pd.DataFrame) -> set:
    """(game_id, team) where a player with ABSENT_MIN+ trailing minutes (his last state) did not play."""
    out = set()
    last = usage.rows.sort_values(["player", "game_date"]).groupby("player").tail(1)
    trailing = {}
    for team, g in usage.rows.groupby("team"):
        pass
    for r in last.itertuples(index=False):
        trailing[r.player] = float(r.minutes)
    played = box_minutes[box_minutes.minutes > 0].groupby(["game_id", "team"]).player.apply(set).to_dict()
    for (gid, team), names in box_minutes.groupby(["game_id", "team"]).player.apply(set).to_dict().items():
        present = played.get((gid, team), set())
        for nm in names - present:
            if trailing.get(nm, 0.0) >= ABSENT_MIN:
                out.add((gid, team)); break
    return out


def predict_game(events_game: pd.DataFrame, usage: UsageHistory, state: pd.DataFrame, profile: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    One game's real possession sequence replayed: per (team, player) the expected fga, fta, tov, fg3a, ast, oreb, dreb, stl, blk from
    the shares as of `state`, over the real fives. profile: team -> TeamProfile.before.
    Possessions by five: each possession-ending event's offence five gets the team's endings per possession split by shares; its
    defence five gets steals (per opponent possession). Attempts/misses/makes by five feed blocks, rebounds and assists.
    """
    acc: Dict[Tuple[str, str], Dict[str, float]] = {}

    def add(team, player, stat, v):
        acc.setdefault((team, player), {s: 0.0 for s in PIE})[stat] += v

    ends = events_game[events_game.poss_end & events_game.on_off.astype(bool) & events_game.on_def.astype(bool)]
    for (team, opp_team, on_off, on_def, poss_team), d in ends.groupby(["team", "opp", "on_off", "on_def", "poss_team"], sort=False):
        n = float(len(d))
        off_team = poss_team; def_team = opp_team if team == off_team else team
        off_five, def_five = (on_off, on_def) if team == off_team else (on_def, on_off)
        off = off_five.split("|"); de = def_five.split("|")
        prof = profile[off_team]
        for stat, pie_key in (("fga", "fga_pp"), ("fg3a", "fg3a_pp"), ("fta", "fta_pp"), ("tov", "tov_pp")):
            r = PIE[stat][0]
            rates = {p: usage.rate(state, p, r) for p in off}
            for p, v in split_pie(rates, n * prof[pie_key]).items():
                add(off_team, p, stat, v)
        for p in de:
            add(def_team, p, "stl", n * usage.rate(state, p, "stl_r"))
    shots = events_game[(events_game.kind == "fg") & events_game.on_off.astype(bool) & events_game.on_def.astype(bool)]
    for (team, opp_team, on_off, on_def), d in shots.groupby(["team", "opp", "on_off", "on_def"], sort=False):
        n_att = float(len(d)); n_made = float(d.made.sum()); n_miss = n_att - n_made
        off = on_off.split("|"); de = on_def.split("|")
        # blocks: the defenders' rates per opponent attempt (no team pie: a block is the shooter's loss)
        for p in de:
            add(opp_team, p, "blk", n_att * usage.rate(state, p, "blk_r"))
        # rebounds: the team's share of the misses, then the five's rates
        prof_o, prof_d = profile[team], profile[opp_team]
        for p, v in split_pie({p: usage.rate(state, p, "oreb_r") for p in off}, n_miss * prof_o["oreb_share"]).items():
            add(team, p, "oreb", v)
        for p, v in split_pie({p: usage.rate(state, p, "dreb_r") for p in de}, n_miss * prof_d["dreb_share"]).items():
            add(opp_team, p, "dreb", v)
        # assists: the team's assisted share of its makes, split by the five's rates (the shooter cannot assist himself; a
        # make by each of the five in turn is approximated by the five's shares over all makes)
        for p, v in split_pie({p: usage.rate(state, p, "ast_r") for p in off}, n_made * prof_o["ast_share"]).items():
            add(team, p, "ast", v)
    misses_ft = events_game[(events_game.kind == "ft") & (events_game.trip_i == events_game.trip_n) & ~events_game.made & events_game.on_off.astype(bool) & events_game.on_def.astype(bool)]
    for (team, opp_team, on_off, on_def), d in misses_ft.groupby(["team", "opp", "on_off", "on_def"], sort=False):
        n_miss = float(len(d)); off = on_off.split("|"); de = on_def.split("|")
        for p, v in split_pie({p: usage.rate(state, p, "oreb_r") for p in off}, n_miss * profile[team]["oreb_share"]).items():
            add(team, p, "oreb", v)
        for p, v in split_pie({p: usage.rate(state, p, "dreb_r") for p in de}, n_miss * profile[opp_team]["dreb_share"]).items():
            add(opp_team, p, "dreb", v)
    return pd.DataFrame([dict(team=team, player=p, **{f"pred_{s}": v[s] for s in PIE}) for (team, p), v in acc.items()])


def check(events: pd.DataFrame, box: pd.DataFrame, groups: Dict[str, str], start: date, end: date, log=print) -> dict:
    """
    Walk-forward over the games in [start, end]: the shares' predictions (real fives, real possessions) and the per-minute
    baseline (trailing stat per minute x true minutes) against the real counts, per stat, overall and by cut.
    Returns {"rows": player-game frame, "table": per stat and cut, "summary": dict}.
    """
    pg = X.player_games(events)
    seg = X.team_segments(events)
    b = box.copy()
    from ivan.sports.nba.names import normalize_name
    b["player"] = b.player_name.map(normalize_name); b = b.rename(columns={"team_abbreviation": "team"})
    minutes = b[["game_id", "player", "minutes"]]
    usage = build_usage(pg, minutes, groups)
    profile = build_team_profile(seg, pg)
    absent = _absent_nights(usage, b[["game_id", "team", "player", "minutes"]])
    starters = set(zip(b[b.position != "BENCH"].game_id, b[b.position != "BENCH"].player))
    ev = events.copy(); ev["game_date"] = pd.to_datetime(ev.game_date).dt.date
    games = ev[(ev.game_date >= start) & (ev.game_date <= end)][["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
    rows = []
    state_date = None; state = None
    for i, g in enumerate(games.itertuples(index=False), 1):
        if g.game_date != state_date:
            state_date = g.game_date; state = usage.before(g.game_date)
        eg = ev[ev.game_id == g.game_id]
        teams = eg.team.unique()
        prof = {t: profile.before(t, g.game_date) for t in teams}
        pred = predict_game(eg, usage, state, prof)
        actual = pg[pg.game_id == g.game_id][["team", "player"] + list(PIE)]
        m = actual.merge(pred, on=["team", "player"], how="left")
        mins = minutes[minutes.game_id == g.game_id].set_index("player").minutes
        for r in m.itertuples(index=False):
            mn = float(mins.get(r.player, 0.0))
            rec = dict(game_id=g.game_id, game_date=g.game_date, team=r.team, player=r.player, minutes=mn, starter=(g.game_id, r.player) in starters,
                       absent_night=(g.game_id, r.team) in absent)
            for s in PIE:
                rec[s] = float(getattr(r, s)); rec[f"pred_{s}"] = float(getattr(r, f"pred_{s}")) if pd.notna(getattr(r, f"pred_{s}")) else 0.0
                rec[f"pm_{s}"] = mn * usage.per_minute(state, r.player, s)
            rows.append(rec)
        if i % 50 == 0:
            log(f"  {i}/{len(games)} games")
    df = pd.DataFrame(rows)
    df = df[df.minutes > 0]
    table = []
    cuts = [("all", df), ("starters", df[df.starter]), ("bench", df[~df.starter]), ("teammate out", df[df.absent_night])]
    for cut, d in cuts:
        if d.empty:
            continue
        for s in PIE:
            table.append(dict(cut=cut, stat=s, n=int(len(d)), mean=float(d[s].mean()),
                              mae_shares=float((d[f"pred_{s}"] - d[s]).abs().mean()), bias_shares=float((d[f"pred_{s}"] - d[s]).mean()),
                              mae_per_minute=float((d[f"pm_{s}"] - d[s]).abs().mean()), bias_per_minute=float((d[f"pm_{s}"] - d[s]).mean())))
    table = pd.DataFrame(table)
    summary = dict(games=int(games.shape[0]), player_games=int(len(df)))
    if len(table):
        t = table[table.cut == "all"]
        summary["wins"] = int((t.mae_shares < t.mae_per_minute).sum()); summary["stats"] = int(len(t))
    return {"rows": df, "table": table, "summary": summary}
