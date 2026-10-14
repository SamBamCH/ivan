#!/usr/bin/env python3
"""
scripts/nfl/backtest_correlation.py

How much should a team's correlation matrices lean on its own history versus
the league's?

For every game in the holdout weeks, build the matchup's correlation (within A,
within B, A x B cross) from history *before* that week under each candidate
league-shrinkage strength, then score the game's realised slot lines pair by
pair under the bivariate Gaussian copula each pair implies. Marginals are the
same for every candidate (each slot's empirical CDF over the team's prior
games), so the score differences come only from the dependence structure.
Higher log-likelihood is better; the table reports each candidate against the
raw team-specific matrices (shrink 0), per game, overall and by pair type
(same-team player pairs, pairs involving the team's points or takeaways,
cross-team player pairs).

    venv/bin/python scripts/nfl/backtest_correlation.py --since 2025 1
    venv/bin/python scripts/nfl/backtest_correlation.py --shrink 0 100 300 inf
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as st

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.refactor import team_correlation as tc

N = tc.N_SLOTS
TEAM_LEVEL = {tc.SLOT_IDX["DST"], tc.SLOT_IDX[tc.PTS_SLOT]}
CATEGORIES = ("within_players", "team_level", "cross_players")


def _spearman_to_pearson(R: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(np.clip(np.pi * R / 6.0, -np.pi / 2, np.pi / 2))


def _normal_scores(prior: np.ndarray, x: np.ndarray) -> np.ndarray:
    """z for each slot of x from the empirical CDF of the team's prior games; nan where unavailable."""
    z = np.full(len(x), np.nan)
    for j, v in enumerate(x):
        col = prior[:, j]
        col = col[~np.isnan(col)]
        if np.isnan(v) or len(col) < 8:
            continue
        u = ((col < v).sum() + 0.5 * (col == v).sum() + 0.5) / (len(col) + 1)
        z[j] = st.norm.ppf(np.clip(u, 0.01, 0.99))
    return z


def _pair_category(i: int, j: int) -> str:
    same_team = (i < N) == (j < N)
    if (i % N) in TEAM_LEVEL or (j % N) in TEAM_LEVEL:
        return "team_level"
    return "within_players" if same_team else "cross_players"


def _pairwise_loglik(z: np.ndarray, R: np.ndarray, max_abs_rho: float = 0.95) -> Dict[str, float]:
    """
    Bivariate Gaussian copula log density at the normal scores, summed over
    slot pairs by category (slots with nan dropped). Pairwise rather than the
    full multivariate density: the matrices are pairwise estimates and a full
    inverse would score their eigenvalue floor instead of their correlations.
    """
    out = {c: 0.0 for c in CATEGORIES}
    keep = np.where(~np.isnan(z))[0]
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            i, j = keep[a], keep[b]
            r = float(np.clip(R[i, j], -max_abs_rho, max_abs_rho))
            zi, zj = z[i], z[j]
            out[_pair_category(i, j)] += -0.5 * np.log(1 - r * r) - (r * r * (zi * zi + zj * zj) - 2 * r * zi * zj) / (2 * (1 - r * r))
    return out


def game_matrix(session, a: str, b: str, before, shrink: float) -> np.ndarray:
    """Pearson matrix over A's then B's slots under league shrinkage `shrink` (0 = raw team, inf = league only)."""
    Ra, _ = tc.build_team_correlation(session, a, league_shrink=shrink, before=before)
    Rb, _ = tc.build_team_correlation(session, b, league_shrink=shrink, before=before)
    Rx, _ = tc.build_cross_team_correlation(session, a, b, league_shrink=shrink, before=before)
    return _spearman_to_pearson(np.block([[Ra, Rx], [Rx.T, Rb]]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", nargs=2, type=int, default=[2025, 1], metavar=("SEASON", "WEEK"),
                    help="first holdout week (default 2025 1); every later played week is a holdout")
    ap.add_argument("--shrink", nargs="+", default=["0", "30", "100", "300", "1000", "inf"],
                    help="candidate league shrinkage in pseudo-games (0 = raw team matrices, inf = league only)")
    ap.add_argument("--db-url", default=None)
    args = ap.parse_args()
    shrinks = sorted({float(s) for s in args.shrink} | {0.0})
    label = lambda k: "team only" if k == 0 else ("league only" if not np.isfinite(k) else f"league k={k:g}")

    session = get_session(init_db("nfl", args.db_url))
    w = tc.league_wide_table(session)
    s0, w0 = args.since
    hold = w[(w.season > s0) | ((w.season == s0) & (w.week >= w0))]
    games = (hold.assign(a=np.minimum(hold.team, hold.opponent), b=np.maximum(hold.team, hold.opponent))
                 .drop_duplicates(["season", "week", "a", "b"])[["season", "week", "a", "b"]])
    print(f"{len(games)} holdout games from {s0} week {w0}; candidates: {[label(k) for k in shrinks]}", flush=True)

    rows = []
    t0 = time.time()
    for k_game, g in enumerate(games.itertuples(index=False)):
        before = (int(g.season), int(g.week))
        ga, gb = tc.team_games(session, g.a, before), tc.team_games(session, g.b, before)
        xa = w[(w.season == g.season) & (w.week == g.week) & (w.team == g.a)][tc.TEAM_SLOTS].to_numpy(float)
        xb = w[(w.season == g.season) & (w.week == g.week) & (w.team == g.b)][tc.TEAM_SLOTS].to_numpy(float)
        if len(ga) < tc.MIN_GAMES_WITHIN or len(gb) < tc.MIN_GAMES_WITHIN or len(xa) != 1 or len(xb) != 1:
            continue
        z = np.concatenate([_normal_scores(ga[tc.TEAM_SLOTS].to_numpy(float), xa[0]),
                            _normal_scores(gb[tc.TEAM_SLOTS].to_numpy(float), xb[0])])
        for k in shrinks:
            ll = _pairwise_loglik(z, game_matrix(session, g.a, g.b, before, k))
            rows.append({"season": g.season, "week": g.week, "game": f"{g.a}-{g.b}", "label": label(k),
                         "loglik": sum(ll.values()), **ll, "n_slots": int((~np.isnan(z)).sum())})
        if (k_game + 1) % 50 == 0:
            print(f"  {k_game + 1}/{len(games)} games, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no scorable games"); return 1
    base_label = label(0.0)
    key = ["season", "week", "game"]
    base = df[df.label == base_label].set_index(key)[["loglik"] + list(CATEGORIES)]
    df = df.join(base.add_suffix("_base"), on=key)
    for c in ["loglik"] + list(CATEGORIES):
        df[f"gain_{c}"] = df[c] - df[f"{c}_base"]
    ng = df.groupby("label").game.count().iloc[0]
    print(f"\nscored {ng} games, {df.n_slots.mean():.1f} slots per game on average")
    print(f"\nmean log-likelihood gain per game vs '{base_label}' (positive = better), paired-t z on the total, "
          "and the gain split by pair type:")
    summ = df[df.label != base_label].groupby("label").agg(
        gain=("gain_loglik", "mean"), sd=("gain_loglik", "std"), n=("gain_loglik", "count"),
        within_players=("gain_within_players", "mean"), team_level=("gain_team_level", "mean"),
        cross_players=("gain_cross_players", "mean"))
    summ["z"] = summ.gain / (summ.sd / np.sqrt(summ.n))
    print(summ.sort_values("gain", ascending=False)[["gain", "z", "within_players", "team_level", "cross_players", "n"]]
          .round(3).to_string())
    print("\nby season (total gain per game):")
    print(df[df.label != base_label].pivot_table(index="season", columns="label", values="gain_loglik", aggfunc="mean")
          .round(3).to_string())
    print(f"\ncurrent default: LEAGUE_SHRINK_GAMES = {tc.LEAGUE_SHRINK_GAMES:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
