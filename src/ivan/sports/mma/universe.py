# src/ivan/sports/mma/universe.py
"""
Every valid DraftKings MMA lineup for a slate: six fighters, at most one per
fight, under the cap. With 12-13 fights that is 50-110k lineups, small enough
to enumerate, so the field model can be fitted on all of them (including the
ones nobody played) and the field sampled from all of them, with no need for
an optimiser to guess which builds the field considers.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROSTER = 6


def enumerate_lineups(pool: Iterable[dict], win_prob: dict[int, float], salary_cap: int = 50000, min_salary: int = 0,
                      roster: int = ROSTER, main_event: Optional[set] = None) -> pd.DataFrame:
    """
    One row per valid lineup: key (sorted fighter ids), salary, salary_left,
    sum_win (summed win probability) and rank_win (1 = highest sum_win, ties
    broken by lower salary left then key). pool rows need id, salary, fight_id.
    """
    fights: dict = {}
    for r in pool:
        fights.setdefault(r["fight_id"], []).append((int(r["id"]), int(r["salary"]), float(win_prob.get(int(r["id"]), 0.5))))
    fl = list(fights.values())
    rows = []
    for combo in itertools.combinations(range(len(fl)), roster):
        for sides in itertools.product(*[fl[i] for i in combo]):
            s = sum(x[1] for x in sides)
            if min_salary <= s <= salary_cap:
                rows.append((tuple(sorted(x[0] for x in sides)), s, sum(x[2] for x in sides)))
    df = pd.DataFrame(rows, columns=["key", "salary", "sum_win"])
    df["salary_left"] = salary_cap - df.salary
    me = set(main_event or ())
    df["n_main"] = df.key.map(lambda k: sum(1 for i in k if i in me))
    df = df.sort_values(["sum_win", "salary_left", "key"], ascending=[False, True, True]).reset_index(drop=True)
    df["rank_win"] = np.arange(1, len(df) + 1)
    return df


def rank_within(lineups: list[list[int]], win_prob: dict[int, float], salary: dict[int, int], salary_cap: int = 50000) -> np.ndarray:
    """rank_win of each given lineup among the given lineups (1 = best), the same ordering enumerate_lineups uses."""
    sw = np.array([sum(win_prob.get(p, 0.5) for p in ids) for ids in lineups])
    left = np.array([salary_cap - sum(salary[p] for p in ids) for ids in lineups])
    order = np.lexsort((np.arange(len(lineups)), left, -sw))
    rank = np.empty(len(lineups), dtype=float)
    rank[order] = np.arange(1, len(lineups) + 1)
    return rank
