# src/ivan/sports/nfl/refactor/usage.py
"""
Usage allocation: layer 2 of the game model.

Layer 1 (refactor.environment) says how much offense a team runs. The props
say what the priced players take of it. What is left is the volume of the
players nobody prices, and this module hands it out:

    team total of a stat               the QB's passing-yards line when it exists (the market's own view of the
                                       team's passing volume), otherwise the environment's dropbacks / rushes
                                       times the league's yards per dropback / completions per dropback / yards
                                       per carry
  - every mean already settled         priced players (the props are the marginals) and the QB's own rushing
                                       from the salary model, both passed in with add_fixed
  = the residual                       floored at RESIDUAL_FLOOR of the team total, since the priced lines can
                                       add up past it
  split among the unpriced RB/WR/TE    by depth-rank priors: the share of the team's stat that the k-th RB / WR /
                                       TE took in past seasons (share_priors), normalised over the unpriced
                                       players actually on the slate. Depth rank is the player's salary order
                                       within his position group on his team (ties: anytime-TD probability).

The salary model (refactor.volume) still predicts QB stats and touchdowns;
this module answers only receiving_yards, receptions and rushing_yards for
RB / WR / TE, and returns None for anything else so the caller falls back.

Everything here is a per-game expectation; the shape around it still comes
from the neighbour pool in refactor.stat. The joint draws of layer 1 are
not consumed yet (the sampler keeps its own team-points latents).

Measured 2022-2026 (share of team receiving yards by season rank): WR1 .28,
WR2 .19, WR3 .13, WR4 .09, WR5 .06, WR6+ .03; TE1 .17, TE2 .08, TE3+ .04;
RB1 .09, RB2 .05, RB3 .02, RB4+ .01. Rushing yards: RB1 .54, RB2 .25, RB3
.12, RB4+ .06; QB .13; WR/TE ~0. On a 2026 week 6 slate the priced
receiving-yards lines summed to 52-89% (mean 75%) of the QB's passing line,
so the residual is a quarter of the passing game, spread over 4-9 unpriced
players per team.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.database.session import read_sql

POSITIONS = ("RB", "WR", "TE")                 # the players the allocation covers
VOLUME_STATS = ("receiving_yards", "receptions", "rushing_yards")
RANK_CAP = {"RB": 4, "WR": 6, "TE": 3}        # depth ranks beyond the cap share the cap's prior
RANK_BY = {"receiving_yards": "targets", "receptions": "targets", "rushing_yards": "carries"}
RESIDUAL_FLOOR = 0.05                          # share of the team total always left for the unpriced players
PRIOR_SEASONS = 4                              # seasons of history behind the share priors and the efficiencies

# fallbacks when the history query returns nothing (a fresh database)
LEAGUE_EFFICIENCY = {"yards_per_dropback": 6.5, "completions_per_dropback": 0.60, "yards_per_completion": 11.0, "yards_per_carry": 4.3}
LEAGUE_TEAM = {"dropbacks": 37.0, "rushes": 26.0}
DEFAULT_PRIORS = {
    ("WR", 1): {"receiving_yards": .279, "receptions": .242, "rushing_yards": .027},
    ("WR", 2): {"receiving_yards": .192, "receptions": .166, "rushing_yards": .012},
    ("WR", 3): {"receiving_yards": .132, "receptions": .114, "rushing_yards": .005},
    ("WR", 4): {"receiving_yards": .088, "receptions": .078, "rushing_yards": .002},
    ("WR", 5): {"receiving_yards": .055, "receptions": .050, "rushing_yards": .001},
    ("WR", 6): {"receiving_yards": .030, "receptions": .029, "rushing_yards": .001},
    ("TE", 1): {"receiving_yards": .169, "receptions": .174, "rushing_yards": .008},
    ("TE", 2): {"receiving_yards": .075, "receptions": .081, "rushing_yards": .001},
    ("TE", 3): {"receiving_yards": .035, "receptions": .041, "rushing_yards": .001},
    ("RB", 1): {"receiving_yards": .092, "receptions": .134, "rushing_yards": .539},
    ("RB", 2): {"receiving_yards": .048, "receptions": .068, "rushing_yards": .253},
    ("RB", 3): {"receiving_yards": .021, "receptions": .034, "rushing_yards": .124},
    ("RB", 4): {"receiving_yards": .013, "receptions": .020, "rushing_yards": .061},
}


# --------------------------------------------------------------------------- history
def _season_window(before: Optional[Tuple[int, int]], session) -> Tuple[int, Optional[Tuple[int, int]]]:
    if before is None:
        last = read_sql(session, text("SELECT max(season) AS s FROM nfl_stats")).s.iloc[0]
        last = int(last) if last is not None and not pd.isna(last) else 2025
        return last - PRIOR_SEASONS + 1, None
    return int(before[0]) - PRIOR_SEASONS + 1, before


def _player_weeks(session, before: Optional[Tuple[int, int]]) -> pd.DataFrame:
    since, cut = _season_window(before, session)
    df = read_sql(session, text("""SELECT season, week, team, player_display_name AS name, position, targets, receptions, receiving_yards,
                                          carries, rushing_yards, attempts, completions, passing_yards, sacks_suffered
                                   FROM nfl_stats WHERE season >= :s AND season_type = 'REG'
                                     AND position IN ('QB', 'RB', 'WR', 'TE')""").bindparams(s=since))
    if cut is not None:
        s, w = cut
        df = df[(df.season < s) | ((df.season == s) & (df.week < w))]
    return df


def share_priors(session, before: Optional[Tuple[int, int]] = None) -> Dict[Tuple[str, int], Dict[str, float]]:
    """
    {(position, depth rank): {stat: mean share of the team's weekly total}} from the seasons before `before`
    (live: the last PRIOR_SEASONS). Depth rank is the player's rank within his team-season position group
    by season targets (receiving stats) or carries (rushing), capped at RANK_CAP. Falls back to DEFAULT_PRIORS.
    """
    df = _player_weeks(session, before)
    if df.empty:
        return {k: dict(v) for k, v in DEFAULT_PRIORS.items()}
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    skill = df[df.position.isin(POSITIONS)]
    for stat in VOLUME_STATS:
        tot = df.groupby(["season", "week", "team"])[stat].sum().rename("team_total")
        d = skill.merge(tot, on=["season", "week", "team"])
        by = RANK_BY[stat]
        sr = d.groupby(["season", "team", "position", "name"])[by].sum().reset_index()
        sr["rank"] = sr.groupby(["season", "team", "position"])[by].rank(ascending=False, method="first").astype(int)
        d = d.merge(sr[["season", "team", "position", "name", "rank"]], on=["season", "team", "position", "name"])
        d["rank"] = [min(int(r), RANK_CAP[p]) for r, p in zip(d["rank"], d.position)]
        d["share"] = d[stat] / d.team_total.replace(0, np.nan)
        for (pos, rank), g in d.groupby(["position", "rank"]):
            sh = g.share.dropna()
            if len(sh) >= 30:
                out.setdefault((str(pos), int(rank)), {})[stat] = float(sh.mean())
    for key, dflt in DEFAULT_PRIORS.items():              # anything history could not fill
        for stat, v in dflt.items():
            out.setdefault(key, {}).setdefault(stat, v)
    return out


def league_efficiency(session, before: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
    """Yards per dropback, completions per dropback, yards per completion and yards per carry, league-wide, from the same window."""
    df = _player_weeks(session, before)
    if df.empty:
        return dict(LEAGUE_EFFICIENCY)
    n = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).sum()
    dropbacks = n("attempts") + n("sacks_suffered")
    out = dict(LEAGUE_EFFICIENCY)
    if dropbacks > 0:
        out["yards_per_dropback"] = float(n("passing_yards") / dropbacks)
        out["completions_per_dropback"] = float(n("completions") / dropbacks)
    if n("completions") > 0:
        out["yards_per_completion"] = float(n("passing_yards") / n("completions"))
    if n("carries") > 0:
        out["yards_per_carry"] = float(n("rushing_yards") / n("carries"))
    return out


# --------------------------------------------------------------------------- the model
@dataclass
class _Candidate:
    pid: int
    name: str
    team: str
    position: str
    salary: float
    p_td: float
    rank: int = 0


@dataclass
class TeamUsage:
    team: str
    totals: Dict[str, float] = field(default_factory=dict)        # stat -> team expectation
    total_source: Dict[str, str] = field(default_factory=dict)    # stat -> "qb line" | "environment" | "league"
    fixed: Dict[str, float] = field(default_factory=dict)         # stat -> sum of settled means
    residual: Dict[str, float] = field(default_factory=dict)      # stat -> what the unpriced players share
    n_unpriced: Dict[str, int] = field(default_factory=dict)
    floored: Dict[str, bool] = field(default_factory=dict)


class UsageModel:
    """
    Collects the settled means (add_fixed) and every candidate (add_candidate)
    during the projection's first pass, then allocates the residual team volume
    to the unpriced RB / WR / TE on the first predict().

        usage = UsageModel(session, environment=slate_environment(session), before=cutoff)
        usage.add_candidate(player, p_td);  usage.add_fixed(player, "receiving_yards", 61.2)
        usage.predict(player, "receptions")   -> mean, or None when this module does not answer
    """

    def __init__(self, session=None, environment: Optional[pd.DataFrame] = None, before: Optional[Tuple[int, int]] = None,
                 priors: Optional[Dict] = None, efficiency: Optional[Dict[str, float]] = None, residual_floor: float = RESIDUAL_FLOOR):
        self.priors = priors if priors is not None else (share_priors(session, before) if session is not None else {k: dict(v) for k, v in DEFAULT_PRIORS.items()})
        self.eff = efficiency if efficiency is not None else (league_efficiency(session, before) if session is not None else dict(LEAGUE_EFFICIENCY))
        self.env: Dict[str, Dict[str, float]] = {}
        if environment is not None and len(environment):
            for r in environment.itertuples():
                plays, pr = float(r.plays_mean), float(r.pass_rate_mean)
                self.env[str(r.team)] = {"dropbacks": plays * pr, "rushes": plays * (1.0 - pr)}
        self.residual_floor = float(residual_floor)
        self.candidates: Dict[int, _Candidate] = {}
        self.fixed: Dict[Tuple[str, str], Dict[int, float]] = {}       # (team, stat) -> pid -> mean
        self.qb_pass: Dict[str, float] = {}                              # team -> priced passing yards (sum over its QBs)
        self.alloc: Dict[Tuple[int, str], float] = {}
        self.teams: Dict[str, TeamUsage] = {}
        self.fitted = False

    # ---- collection
    def add_candidate(self, player, p_td: Optional[float]) -> None:
        if self.fitted:
            raise RuntimeError("UsageModel is already fitted")
        if player.position not in POSITIONS or not player.team:
            return
        self.candidates[int(player.id)] = _Candidate(int(player.id), player.name, str(player.team), str(player.position),
                                                     float(player.salary or 0), float(p_td) if p_td is not None else 0.0)

    def add_fixed(self, player, stat: str, mean: float) -> None:
        """A stat mean settled before allocation: a priced stat of anyone, or the salary model's answer for a QB."""
        if self.fitted:
            raise RuntimeError("UsageModel is already fitted")
        if mean is None or not np.isfinite(mean) or not player.team:
            return
        team = str(player.team)
        if stat == "passing_yards":
            self.qb_pass[team] = self.qb_pass.get(team, 0.0) + float(mean)
        if stat in VOLUME_STATS:
            self.fixed.setdefault((team, stat), {})[int(player.id)] = float(mean)

    # ---- allocation
    def team_total(self, team: str, stat: str) -> Tuple[float, str]:
        e = self.env.get(team)
        if stat in ("receiving_yards", "receptions"):
            if team in self.qb_pass and self.qb_pass[team] > 0:
                yards = self.qb_pass[team]
                return (yards, "qb line") if stat == "receiving_yards" else (yards / self.eff["yards_per_completion"], "qb line")
            dropbacks, src = (e["dropbacks"], "environment") if e else (LEAGUE_TEAM["dropbacks"], "league")
            per = self.eff["yards_per_dropback"] if stat == "receiving_yards" else self.eff["completions_per_dropback"]
            return dropbacks * per, src
        rushes, src = (e["rushes"], "environment") if e else (LEAGUE_TEAM["rushes"], "league")
        return rushes * self.eff["yards_per_carry"], src

    def _prior(self, c: _Candidate, stat: str) -> float:
        p = self.priors.get((c.position, min(c.rank, RANK_CAP[c.position])))
        if not p:
            p = DEFAULT_PRIORS[(c.position, min(c.rank, RANK_CAP[c.position]))]
        return float(p.get(stat, 0.0))

    def fit(self) -> None:
        """Depth ranks, team totals, residuals and the allocation. Idempotent."""
        if self.fitted:
            return
        by_group: Dict[Tuple[str, str], List[_Candidate]] = {}
        for c in self.candidates.values():
            by_group.setdefault((c.team, c.position), []).append(c)
        for group in by_group.values():
            group.sort(key=lambda c: (-c.salary, -c.p_td, c.name))
            for i, c in enumerate(group, start=1):
                c.rank = i
        teams = sorted({c.team for c in self.candidates.values()})
        for team in teams:
            tu = TeamUsage(team)
            members = [c for c in self.candidates.values() if c.team == team]
            for stat in VOLUME_STATS:
                total, src = self.team_total(team, stat)
                fixed = self.fixed.get((team, stat), {})
                fixed_sum = float(sum(fixed.values()))
                floor = self.residual_floor * total
                residual = max(total - fixed_sum, floor)
                unpriced = [c for c in members if c.pid not in fixed]
                w = np.array([self._prior(c, stat) for c in unpriced], dtype=float)
                if len(unpriced) and w.sum() <= 0:
                    w = np.ones(len(unpriced))
                for c, wi in zip(unpriced, w):
                    self.alloc[(c.pid, stat)] = residual * float(wi) / float(w.sum()) if w.sum() > 0 else 0.0
                tu.totals[stat], tu.total_source[stat], tu.fixed[stat] = total, src, fixed_sum
                tu.residual[stat], tu.n_unpriced[stat], tu.floored[stat] = residual, len(unpriced), bool(total - fixed_sum < floor)
            self.teams[team] = tu
        self.fitted = True

    def predict(self, player, stat: str) -> Optional[float]:
        """The allocated mean of an unpriced volume stat for an RB / WR / TE on the slate; None when the salary model should answer."""
        if stat not in VOLUME_STATS or player.position not in POSITIONS:
            return None
        if not self.fitted:
            self.fit()
        v = self.alloc.get((int(player.id), stat))
        return None if v is None else float(max(v, 0.0))

    def rank_of(self, player) -> Optional[int]:
        c = self.candidates.get(int(player.id))
        return c.rank if c else None

    # ---- reporting
    def describe(self) -> str:
        if not self.fitted:
            self.fit()
        lines = [f"usage allocation (layer 2): {len(self.teams)} teams, environment for {len(self.env)}; "
                 f"league ypd {self.eff['yards_per_dropback']:.2f}, comp/db {self.eff['completions_per_dropback']:.3f}, ypc {self.eff['yards_per_carry']:.2f}"]
        for team, tu in sorted(self.teams.items()):
            parts = []
            for stat in VOLUME_STATS:
                short = {"receiving_yards": "rec yds", "receptions": "rec", "rushing_yards": "rush yds"}[stat]
                parts.append(f"{short} {tu.totals[stat]:.0f} ({tu.total_source[stat]}) - fixed {tu.fixed[stat]:.0f} = "
                             f"{tu.residual[stat]:.0f}{'*' if tu.floored[stat] else ''} over {tu.n_unpriced[stat]}")
            lines.append(f"  {team:4s} " + " | ".join(parts))
        if any(tu.floored[s] for tu in self.teams.values() for s in VOLUME_STATS):
            lines.append("  * residual floored: the settled lines add up past the team total")
        return "\n".join(lines)

    def team_rows(self) -> List[dict]:
        """One row per team and stat, for an artifact or a test."""
        if not self.fitted:
            self.fit()
        rows = []
        for team, tu in sorted(self.teams.items()):
            for stat in VOLUME_STATS:
                rows.append({"team": team, "stat": stat, "total": tu.totals[stat], "total_source": tu.total_source[stat],
                             "fixed": tu.fixed[stat], "residual": tu.residual[stat], "n_unpriced": tu.n_unpriced[stat],
                             "floored": tu.floored[stat]})
        return rows
