# src/ivan/sports/mma/fight.py
"""
Monte Carlo model of one fight, driven by its betting markets.

How a fight ends is drawn from the fight-level markets (inside 60 seconds,
in round 1, inside the distance), who wins from the two fighters' markets
isolated to that kind of finish (the round-1 price for round 1, ITD minus
round 1 for a later stoppage, moneyline minus ITD for a decision; the
moneyline alone when a market is blank), and how long it lasted uniformly
inside the chosen round. Each
fighter's output (significant strikes, takedowns) is their expected full-
fight total scaled by the time fought with diminishing returns, plus noise.
The winner takes the finish bonus for that round; the loser keeps half of
their production.

Ported from mma_dfs/src/Fight.py, vectorised over simulations; the winner
split by finish kind is the one change from the original: simulate(n) returns n outcomes per fighter at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ivan.sports.mma.odds import american_to_prob, win_split

ROUND_SECONDS = 300
DECISION = "decision"

# DraftKings MMA scoring as the original model used it. The finish bonuses
# are DK's (90/70/45/40/40 by round, +25 for a finish inside 60 seconds,
# 30 for a decision); the per-strike weight is the model's own.
DEFAULT_SCORING = {
    "sig_strike": 0.4,
    "takedown": 5.0,
    "loser_production_share": 0.5,
    "bonus": {"u60": 115, "r1": 90, "r2": 70, "r3": 45, "r4": 40, "r5": 40, "decision": 30},
    "production_noise_cv": 0.2,      # sd of a fighter's per-second rate as a share of its mean
    "time_exponent": 1.5,            # production by time t of a T-second fight = full-fight line x (t/T)^time_exponent
                                     # (1.5 is the original's rate x sqrt scale; below 1 front-loads production as real fights do)
}


@dataclass
class Fighter:
    id: int
    name: str
    salary: int
    win_odds: Optional[float] = None
    itd_odds: Optional[float] = None
    r1_win_odds: Optional[float] = None
    sig_strikes: float = 0.0
    takedowns: float = 0.0
    win_prob: float = 0.5

    def odds_for(self, finish: str) -> Optional[float]:
        if finish == "r1":
            return self.r1_win_odds
        if finish == DECISION:
            return self.win_odds
        return self.itd_odds

    def finish_share(self, finish: str) -> Optional[float]:
        """
        This fighter's unnormalised probability of winning by `finish`, from
        the markets that isolate it: the round-1 price for round 1, ITD minus
        round 1 for a later stoppage, moneyline minus ITD for a decision.
        None when the markets needed are missing (the caller falls back to
        the moneyline), so a fighter never gets a share from a coin flip.
        """
        if finish == "r1":
            return None if self.r1_win_odds in (None, 0) else american_to_prob(self.r1_win_odds)
        if self.itd_odds in (None, 0):
            return None
        p_itd = american_to_prob(self.itd_odds)
        if finish == DECISION:
            if self.win_odds in (None, 0):
                return None
            return max(0.0, american_to_prob(self.win_odds) - p_itd)
        p_r1 = american_to_prob(self.r1_win_odds) if self.r1_win_odds not in (None, 0) else 0.0
        return max(0.0, p_itd - p_r1)


@dataclass
class Node:
    """One way the fight can end: which round, the time window, the winner's bonus."""
    key: str                 # u60, r1, r2, ..., decision
    round: object            # 1..5 or "decision"
    t0: float
    t1: float
    bonus: float
    prob: float = 0.0

    @property
    def finish_kind(self) -> str:
        if self.round == DECISION:
            return DECISION
        return "r1" if self.round == 1 else "itd"


@dataclass
class Fight:
    id: str
    f1: Fighter
    f2: Fighter
    rounds: int = 3
    u60_odds: Optional[float] = None
    r1_odds: Optional[float] = None
    itd_odds: Optional[float] = None
    scoring: dict = field(default_factory=lambda: DEFAULT_SCORING)

    @property
    def total_time(self) -> float:
        return ROUND_SECONDS * self.rounds

    def nodes(self) -> list[Node]:
        """Outcome nodes with normalised probabilities from the fight-level markets."""
        p_u60 = american_to_prob(self.u60_odds)
        p_r1 = american_to_prob(self.r1_odds)
        p_itd = american_to_prob(self.itd_odds)
        p_node1 = p_u60                          # finish inside 60 seconds
        p_node2 = max(0.0, p_r1 - p_u60)         # rest of round 1
        p_stop = max(0.0, p_itd - p_r1)          # stoppages after round 1, split evenly over the later rounds
        p_dec = max(0.0, 1.0 - (p_node1 + p_node2 + p_stop))
        b = self.scoring["bonus"]
        later = list(range(2, self.rounds + 1))
        out = [Node("u60", 1, 0, 60, b["u60"], p_node1),
               Node("r1", 1, 60, ROUND_SECONDS, b["r1"], p_node2)]
        for r in later:
            out.append(Node(f"r{r}", r, ROUND_SECONDS * (r - 1), ROUND_SECONDS * r, b[f"r{r}"], p_stop / len(later)))
        out.append(Node(DECISION, DECISION, self.total_time, self.total_time, b["decision"], p_dec))
        total = sum(n.prob for n in out)
        if total <= 0:
            for n in out:
                n.prob = 1.0 / len(out)
        else:
            for n in out:
                n.prob = n.prob / total
        return out

    def winner_share(self, finish: str) -> float:
        """
        Fighter 1's probability of being the winner given the fight ended by
        `finish`. Both fighters' isolated finish probabilities (Fighter.
        finish_share) are normalised against each other; when either side
        lacks the markets for it, or both come out at zero, the split falls
        back to the moneylines.
        """
        a, b = self.f1.finish_share(finish), self.f2.finish_share(finish)
        if a is None or b is None or a + b <= 0:
            return win_split(self.f1.win_odds, self.f2.win_odds)[0]
        return a / (a + b)

    def _production(self, fighter: Fighter, finish_time: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Significant strikes and takedowns landed by `finish_time` seconds, one value per simulation."""
        n = finish_time.shape[0]
        cv = float(self.scoring.get("production_noise_cv", 0.2))
        k = float(self.scoring.get("time_exponent", 1.5))
        share = np.power(np.clip(finish_time / self.total_time, 0.0, 1.0), k)   # share of the full-fight line landed by finish_time

        def draw(total: float) -> np.ndarray:
            base = float(total or 0.0)
            noisy = rng.normal(base, base * cv, size=n) if base > 0 else np.zeros(n)
            return np.maximum(noisy, 0.0) * share

        return draw(fighter.sig_strikes), draw(fighter.takedowns)

    def simulate(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Fantasy points for (fighter1, fighter2) in each of n simulated fights."""
        nodes = self.nodes()
        probs = np.array([nd.prob for nd in nodes], dtype=float)
        idx = rng.choice(len(nodes), size=n, p=probs)

        # winner: fighter 1's share of the way the fight ended (see winner_share)
        p1_by_node = np.array([self.winner_share(nd.finish_kind) for nd in nodes])
        f1_wins = rng.random(n) < p1_by_node[idx]

        t0 = np.array([nd.t0 for nd in nodes])[idx]
        t1 = np.array([nd.t1 for nd in nodes])[idx]
        finish_time = t0 + rng.random(n) * (t1 - t0)
        bonus = np.array([nd.bonus for nd in nodes])[idx]

        w_sig, w_td = float(self.scoring["sig_strike"]), float(self.scoring["takedown"])
        loser_share = float(self.scoring["loser_production_share"])
        sig1, td1 = self._production(self.f1, finish_time, rng)
        sig2, td2 = self._production(self.f2, finish_time, rng)
        prod1 = sig1 * w_sig + td1 * w_td
        prod2 = sig2 * w_sig + td2 * w_td

        fpts1 = np.where(f1_wins, prod1 + bonus, prod1 * loser_share)
        fpts2 = np.where(f1_wins, prod2 * loser_share, prod2 + bonus)
        return fpts1, fpts2


def project_fighter(fighter: Fighter, fight_itd_odds: Optional[float], scoring: dict = DEFAULT_SCORING) -> tuple[float, float]:
    """
    (win probability, projected fantasy points) for the pool and the field
    model. Production from the volume markets plus the expected finish
    bonus: win probability times a blend of a stoppage (round-2 bonus as
    the average) and a decision, weighted by how likely the fight ends early.
    """
    win_prob = american_to_prob(fighter.win_odds)
    production = (fighter.sig_strikes or 0.0) * float(scoring["sig_strike"]) + (fighter.takedowns or 0.0) * float(scoring["takedown"])
    itd_prob = american_to_prob(fight_itd_odds)
    b = scoring["bonus"]
    expected_bonus = win_prob * (itd_prob * b["r2"] + (1 - itd_prob) * b["decision"])
    return win_prob, production + expected_bonus
