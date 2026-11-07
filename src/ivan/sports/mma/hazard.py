# src/ivan/sports/mma/hazard.py
"""
One risk of finish per fighter: the fight as a competing-risks survival
process.

Each fighter carries a finish hazard (stoppages per second) and two
production rates (significant strikes and takedowns per second). The fight
ends at the first stoppage; while it lasts the chance it is still going
decays as exp(-cumulative hazard), and the fighters produce at their rates.
At the final bell whatever is left is a decision.

Markets to parameters:
  * the fight's ITD price (or the two fighters' ITD prices summed) sets the
    total hazard: 1 - exp(-H(T)) = P(inside the distance);
  * each fighter's ITD price sets their share of the stoppages (the
    moneyline split when a side has no ITD price);
  * the moneyline minus the finish share sets who wins a decision;
  * a fighter's strikes / takedowns line is honoured as the mean of the
    simulated count, so the rate is the line over the expected weighted
    time fought.
The round-1 and under-60-seconds prices are not inputs. On the 2026-10-24
card a flat hazard from the ITD price reproduced the round-1 price within
5% (docs/mma/notes/hazard_2026-11-07.md); the first-minute price ran 40%
above flat, mostly vig on a 20-1 shot, so `first_minute_mult` is 1 by
default and exposed for anyone who wants the bump.

Production noise is a Gamma multiplier on the rate per simulation (between-
fight pace variation, `rate_cv`) and Poisson counts given the mean, so a
takedown is a whole takedown. The loser's production keeps the configured
share (`scoring.loser_production_share`), folded into the rate so the line
still holds in expectation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ivan.sports.mma.fight import DECISION, DEFAULT_SCORING, Fighter, ROUND_SECONDS
from ivan.sports.mma.odds import american_to_prob, win_split

FIRST_MINUTE = 60.0

DEFAULTS = {
    "model": "hazard",            # or "tree": the round-node model in fight.py
    "first_minute_mult": 1.0,     # hazard multiplier inside the first 60 seconds
    "default_itd_prob": 0.5,      # finish probability when no ITD price exists anywhere (UFC runs about half)
    "max_itd_prob": 0.97,         # cap on a finish probability built from one-sided prices
    "rate_cv": 0.25,              # sd of a fighter's per-second rate across simulations, as a share of its mean
}


@dataclass
class HazardFight:
    id: str
    f1: Fighter
    f2: Fighter
    rounds: int = 3
    itd_odds: Optional[float] = None
    u60_odds: Optional[float] = None       # accepted for interface parity with Fight; not used
    r1_odds: Optional[float] = None        # idem
    scoring: dict = field(default_factory=lambda: DEFAULT_SCORING)
    params: dict = field(default_factory=lambda: DEFAULTS)

    # ------------------------------------------------------------------ parameters from the markets
    @property
    def total_time(self) -> float:
        return ROUND_SECONDS * self.rounds

    def _param(self, key: str) -> float:
        return float(self.params.get(key, DEFAULTS[key]))

    def _fighter_itd(self, f: Fighter) -> Optional[float]:
        return None if f.itd_odds in (None, 0) else american_to_prob(f.itd_odds)

    def finish_prob(self) -> float:
        """P(the fight ends inside the distance)."""
        cap = self._param("max_itd_prob")
        if self.itd_odds not in (None, 0):
            return min(american_to_prob(self.itd_odds), cap)
        q1, q2 = self._fighter_itd(self.f1), self._fighter_itd(self.f2)
        if q1 is not None and q2 is not None:
            return min(q1 + q2, cap)
        return min(self._param("default_itd_prob"), cap)

    def finish_split(self) -> float:
        """Fighter 1's share of the stoppages: the ITD prices against each other, else the moneyline split."""
        q1, q2 = self._fighter_itd(self.f1), self._fighter_itd(self.f2)
        if q1 is not None and q2 is not None and q1 + q2 > 0:
            return q1 / (q1 + q2)
        return win_split(self.f1.win_odds, self.f2.win_odds)[0]

    def decision_split(self) -> float:
        """Fighter 1's chance of winning a decision, so that the overall win rate matches the moneyline."""
        p_win = win_split(self.f1.win_odds, self.f2.win_odds)[0]
        p, s1 = self.finish_prob(), self.finish_split()
        if p >= 1.0 - 1e-9:
            return p_win
        return float(np.clip((p_win - s1 * p) / (1.0 - p), 0.0, 1.0))

    def hazard(self) -> float:
        """The base hazard (per second) such that 1 - exp(-H(T)) = finish_prob, with the first-minute multiplier."""
        p = min(max(self.finish_prob(), 1e-9), 1.0 - 1e-9)
        m = self._param("first_minute_mult")
        T = self.total_time
        weighted_time = m * FIRST_MINUTE + (T - FIRST_MINUTE)
        return -np.log(1.0 - p) / weighted_time

    # ------------------------------------------------------------------ the survival curve
    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        lam, m = self.hazard(), self._param("first_minute_mult")
        t = np.asarray(t, dtype=float)
        return np.where(t <= FIRST_MINUTE, lam * m * t, lam * (m * FIRST_MINUTE + (t - FIRST_MINUTE)))

    def inverse_cumulative_hazard(self, h: np.ndarray) -> np.ndarray:
        lam, m = self.hazard(), self._param("first_minute_mult")
        h = np.asarray(h, dtype=float)
        h_first = lam * m * FIRST_MINUTE
        return np.where(h <= h_first, h / (lam * m), FIRST_MINUTE + (h - h_first) / lam)

    def survival(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.cumulative_hazard(t))

    def segments(self) -> list[tuple[str, float, float, float]]:
        """(bonus key, t0, t1, probability of a stoppage in [t0, t1)) for u60, the rest of round 1, each later round."""
        edges = [0.0, FIRST_MINUTE] + [ROUND_SECONDS * r for r in range(1, self.rounds + 1)]
        keys = ["u60", "r1"] + [f"r{r}" for r in range(2, self.rounds + 1)]
        S = self.survival(np.array(edges))
        return [(k, edges[i], edges[i + 1], float(S[i] - S[i + 1])) for i, k in enumerate(keys)]

    def decision_prob(self) -> float:
        return float(self.survival(np.array([self.total_time]))[0])

    def expected_time(self) -> tuple[float, float]:
        """(E[time fought], the part of it from fights that are stopped): the second excludes the decision mass."""
        lam, m, T = self.hazard(), self._param("first_minute_mult"), self.total_time
        first = (1.0 - np.exp(-lam * m * FIRST_MINUTE)) / (lam * m)
        rest = np.exp(-lam * m * FIRST_MINUTE) * (1.0 - np.exp(-lam * (T - FIRST_MINUTE))) / lam
        e_min = float(first + rest)                     # E[min(tau, T)] = integral of the survival curve
        return e_min, e_min - T * self.decision_prob()

    # ------------------------------------------------------------------ production
    def _weighted_time(self, f1_side: bool) -> float:
        """E[time fought x (1 if this fighter wins else loser share)], the divisor that turns a line into a rate."""
        h = float(self.scoring.get("loser_production_share", 0.5))
        s1, d1 = self.finish_split(), self.decision_split()
        s, d = (s1, d1) if f1_side else (1.0 - s1, 1.0 - d1)
        _, e_stop = self.expected_time()
        e_dec = self.total_time * self.decision_prob()
        return e_stop * (s + h * (1.0 - s)) + e_dec * (d + h * (1.0 - d))

    def rates(self, f1_side: bool) -> tuple[float, float]:
        """(significant strikes, takedowns) per second for fighter 1 or 2, honouring their lines as means."""
        f = self.f1 if f1_side else self.f2
        wt = max(self._weighted_time(f1_side), 1e-9)
        return float(f.sig_strikes or 0.0) / wt, float(f.takedowns or 0.0) / wt

    def _production(self, f1_side: bool, t: np.ndarray, wins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        r_sig, r_td = self.rates(f1_side)
        h = float(self.scoring.get("loser_production_share", 0.5))
        cv = self._param("rate_cv")
        weight = np.where(wins, 1.0, h) * t
        if cv > 0:
            shape = 1.0 / cv ** 2
            weight = weight * rng.gamma(shape, 1.0 / shape, size=t.shape[0])
        sig = rng.poisson(np.maximum(r_sig * weight, 0.0)) if r_sig > 0 else np.zeros(t.shape[0])
        td = rng.poisson(np.maximum(r_td * weight, 0.0)) if r_td > 0 else np.zeros(t.shape[0])
        return sig * float(self.scoring["sig_strike"]) + td * float(self.scoring["takedown"])

    def _bonus(self, t: np.ndarray, finished: np.ndarray) -> np.ndarray:
        b = self.scoring["bonus"]
        rnd = np.minimum(np.floor(t / ROUND_SECONDS).astype(int) + 1, self.rounds)
        by_round = np.array([b["u60"]] + [b[f"r{r}"] for r in range(1, self.rounds + 1)], dtype=float)
        idx = np.where(t < FIRST_MINUTE, 0, rnd)
        return np.where(finished, by_round[idx], float(b[DECISION]))

    # ------------------------------------------------------------------ simulate
    def simulate(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Fantasy points for (fighter1, fighter2) in each of n simulated fights."""
        T = self.total_time
        tau = self.inverse_cumulative_hazard(-np.log(rng.random(n)))
        finished = tau < T
        t = np.minimum(tau, T)
        s1, d1 = self.finish_split(), self.decision_split()
        u = rng.random(n)
        f1_wins = np.where(finished, u < s1, u < d1)
        bonus = self._bonus(t, finished)
        prod1 = self._production(True, t, f1_wins, rng)
        prod2 = self._production(False, t, ~f1_wins, rng)
        return prod1 + np.where(f1_wins, bonus, 0.0), prod2 + np.where(f1_wins, 0.0, bonus)

    # ------------------------------------------------------------------ expectations
    def expected_bonus(self, f1_side: bool) -> float:
        b = self.scoring["bonus"]
        s1, d1 = self.finish_split(), self.decision_split()
        s, d = (s1, d1) if f1_side else (1.0 - s1, 1.0 - d1)
        stop = sum(prob * float(b[key]) for key, _, _, prob in self.segments())
        return s * stop + d * self.decision_prob() * float(b[DECISION])

    def expected_points(self, f1_side: bool) -> float:
        """E[fantasy points]: the lines at their weights (held as means) plus the expected finish bonus."""
        f = self.f1 if f1_side else self.f2
        production = float(f.sig_strikes or 0.0) * float(self.scoring["sig_strike"]) + float(f.takedowns or 0.0) * float(self.scoring["takedown"])
        return production + self.expected_bonus(f1_side)

    def describe(self) -> dict:
        """The fitted parameters, for the ingest summary and the desk."""
        e_time, _ = self.expected_time()
        return {"fight": self.id, "rounds": self.rounds, "p_finish": self.finish_prob(), "hazard_per_min": float(self.hazard() * 60),
                "f1_finish_share": self.finish_split(), "f1_decision_share": self.decision_split(),
                "expected_minutes": e_time / 60, "p_decision": self.decision_prob(),
                "segments": {k: p for k, _, _, p in self.segments()}}


def build_fight(id: str, f1: Fighter, f2: Fighter, rounds: int, *, u60_odds=None, r1_odds=None, itd_odds=None,
                scoring: dict, fight_cfg: Optional[dict] = None):
    """A HazardFight or a tree Fight (fight.py) by `fight_cfg["model"]`; the one place that picks the model."""
    from ivan.sports.mma.fight import Fight
    params = {**DEFAULTS, **(fight_cfg or {})}
    if str(params.get("model", "hazard")).lower() == "tree":
        return Fight(id=id, f1=f1, f2=f2, rounds=rounds, u60_odds=u60_odds, r1_odds=r1_odds, itd_odds=itd_odds, scoring=scoring)
    return HazardFight(id=id, f1=f1, f2=f2, rounds=rounds, itd_odds=itd_odds, u60_odds=u60_odds, r1_odds=r1_odds,
                       scoring=scoring, params=params)
