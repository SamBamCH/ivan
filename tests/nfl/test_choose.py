"""The two lineup choosers (ivan.sports.nfl.choose): the portfolio method spreads the book across simulated worlds."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nfl import choose as C
from ivan.sports.nfl import lineup_export as LE
from ivan.sports.nfl.models import HistoricalContest, Lineup, LineupSimulationResult, NFLBase, NFLPlayer


def two_cluster_rois(n_per_cluster=40, worlds=400, seed=0):
    """
    Two families of lineups: family A wins big in worlds 0-9 (a shootout) and
    loses elsewhere; family B cashes a little in most worlds. On mean ROI every
    A lineup beats every B lineup, so a lineup-by-lineup chooser takes only A.
    """
    rng = np.random.default_rng(seed)
    R = -np.ones((2 * n_per_cluster, worlds), dtype=np.float32)
    for i in range(n_per_cluster):
        R[i, :10] = 60.0 + rng.normal(0, 3, 10)                       # A: 10 of 400 worlds pay 60x -> mean ROI +0.5
    for i in range(n_per_cluster, 2 * n_per_cluster):
        hit = rng.random(worlds) < 0.6
        R[i, hit] = 0.9                                                 # B: cash 60% of worlds at 1.9x -> mean ROI +0.14
    return R


def three_family_rois(n_per_family=30, worlds=300, seed=0):
    """
    Three families that pay in disjoint thirds of the worlds (a different game
    script each), with slightly different multipliers so that on mean ROI every
    A beats every B beats every C. A chooser that judges lineups one by one
    takes only A; one that judges the book spreads over all three scripts.
    """
    rng = np.random.default_rng(seed)
    R = -np.ones((3 * n_per_family, worlds), dtype=np.float32)
    third = worlds // 3
    for f, mult in enumerate((3.4, 3.2, 3.0)):
        for i in range(f * n_per_family, (f + 1) * n_per_family):
            w = f * third + rng.choice(third, size=int(0.9 * third), replace=False)
            R[i, w] = mult + rng.normal(0, 0.05, w.size)
    return R


def test_portfolio_spreads_across_worlds():
    R = three_family_rois()
    n = R.shape[0] // 3
    sel, trace = C.pick_portfolio(R, 15, bankroll=3.0)
    assert len(sel) == 15 and len(set(sel)) == 15 and [t["pick"] for t in trace] == list(range(1, 16))
    fam = [i // n for i in sel]
    assert set(fam) == {0, 1, 2}, "the book should cover all three game scripts"
    assert max(fam.count(f) for f in range(3)) <= 7
    # the quality (mean ROI) pick takes family A only
    top = np.argsort(-R.mean(axis=1))[:15]
    assert all(i < n for i in top)
    book = C.book_summary(R[sel]); top_book = C.book_summary(R[top])
    assert book["p_loss"] < top_book["p_loss"] and book["worst"] > top_book["worst"]
    assert trace[0]["score"] == pytest.approx(trace[0]["div"]) and all(t["div"] > 0 for t in trace)


def test_portfolio_prefers_steady_cashers_to_lottery_tickets():
    R = two_cluster_rois()
    n = R.shape[0] // 2
    sel, _ = C.pick_portfolio(R, 20, bankroll=3.0)
    assert sum(i >= n for i in sel) >= 15                              # mean ROI alone would take only family A
    assert all(i < n for i in np.argsort(-R.mean(axis=1))[:20])


def test_portfolio_is_deterministic_and_bounded():
    R = two_cluster_rois(seed=1)
    a, _ = C.pick_portfolio(R, 7, bankroll=2.0)
    b, _ = C.pick_portfolio(R, 7, bankroll=2.0)
    assert a == b
    sel, _ = C.pick_portfolio(R, 500, bankroll=2.0)                   # more picks than candidates
    assert len(sel) == R.shape[0] and len(set(sel)) == R.shape[0]
    assert C.pick_portfolio(np.zeros((0, 10), dtype=np.float32), 5) == ([], [])
    # a bankroll at or under 1x is lifted so the log stays finite when every entry loses
    sel, trace = C.pick_portfolio(-np.ones((5, 20), dtype=np.float32), 5, bankroll=0.5)
    assert len(sel) == 5 and all(np.isfinite(t["score"]) for t in trace)


def test_greedy_pick_matches_quality_then_diversity():
    q = np.array([0.1, 0.9, 0.8, 0.2])
    d = np.array([[0, 1, 0.1, 0.5], [1, 0, 0.5, 0.2], [0.1, 0.5, 0, 0.7], [0.5, 0.2, 0.7, 0]])
    sel, trace = C.pick_greedy(q, d, lam=0.0, num_lineups=2)
    assert sel == [1, 2]
    sel, trace = C.pick_greedy(q, d, lam=5.0, num_lineups=2)
    assert sel == [1, 0] and trace[1]["div"] == 1.0                  # with a big lambda the farthest lineup wins


def test_method_and_bankroll_parsing():
    assert C.method_of({}) == "greedy" and C.method_of({"method": "Portfolio"}) == "portfolio"
    with pytest.raises(ValueError):
        C.method_of({"method": "random"})
    assert C.bankroll_of({}) == C.DEFAULT_BANKROLL and C.bankroll_of({"bankroll": 0.2}) == C.MIN_BANKROLL
    w = LE.weights_resolver(flags={"method": "portfolio", "bankroll": None}, prompter=None)(1, "x", {})
    assert w["method"] == "portfolio" and w["bankroll"] == 3.0 and w["cash_pct"] == 0.5


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_choose_lineup_ids_both_methods(session):
    kick = datetime(2026, 10, 15, 19, 15)
    for pid, slot, pos in [(1, 511, "QB"), (2, 512, "QB"), (3, 512, "WR"), (4, 512, "RB"), (5, 512, "TE"), (6, 512, "K"), (7, 512, "DST"), (8, 512, "WR")]:
        session.add(NFLPlayer(id=pid, roster_slot_id=slot, name=f"p{pid}", team="A", opponent="B", position=pos, gametime=kick,
                              salary=5000, fpts_mean=10.0, fpts_p50=9.0, ownership_proj=20.0))
    session.add(HistoricalContest(contest_id=100, draft_group_id=1, name="Showdown A", date=kick, entry_fee=5.0,
                                  contest_type="Showdown Captain Mode", total_entries=1000, multi_entry=20))
    R = two_cluster_rois(n_per_cluster=6, worlds=100)
    rng = np.random.default_rng(0)
    flex = [2, 3, 4, 5, 6, 7, 8]
    for i in range(12):
        pids = [1] + sorted(rng.choice(flex, 5, replace=False).tolist())
        session.add(Lineup(id=i + 1, player_ids_list=pids))
        session.flush()
        session.add(LineupSimulationResult(lineup_id=i + 1, contest_id=100, contest_name="Showdown A", fpts=[float(x) for x in rng.normal(100, 10, 100)],
                                           payout=[float(x) for x in np.where(R[i] > -1, 5 * (1 + R[i]), 0.0)], roi=[float(x) for x in R[i]], rank=list(range(100))))
    session.commit()
    stats_g, stats_p = {}, {}
    greedy = C.choose_lineup_ids(session, {**LE.DEFAULT_WEIGHTS, "cash_pct": 0.0, "avg_roi_win": 1.0, "diversity_lambda": 0.0}, 4, "Showdown A", stats_g)
    port = C.choose_lineup_ids(session, {**LE.DEFAULT_WEIGHTS, "method": "portfolio"}, 4, "Showdown A", stats_p)
    assert len(greedy) == 4 and len(port) == 4
    assert stats_g["book"]["method"] == "greedy" and stats_p["book"]["method"] == "portfolio" and stats_p["book"]["bankroll"] == 3.0
    assert stats_p["viz"]["method"] == "portfolio" and stats_p["viz"]["book"]["chosen"]["entries"] == 4
    assert all(lid <= 6 for lid in greedy)                           # mean-ROI picks only the shootout family
    assert any(lid > 6 for lid in port)                               # the book holds some steady cashers too
    assert stats_p["book"]["chosen"]["p_loss"] < stats_g["book"]["chosen"]["p_loss"]

    # the full choose run logs the book line and carries the method into the outputs
    entries = pd.DataFrame({"Entry ID": ["11", "12"], "Contest Name": ["Showdown A"] * 2, "Contest ID": [100, 100], "Entry Fee": ["$5"] * 2})
    lines = []
    run = LE.choose_entries(session, entries, LE.weights_resolver(flags={"method": "portfolio"}), log=lines.append)
    assert run.contests[0]["weights"]["method"] == "portfolio" and any("book over 100 sim worlds" in m for m in lines)


def test_cross_fit_book_reports_selection_bias():
    """Choosing on half the worlds and scoring on the rest: picks ranked on sampled rare wins look worse out of sample."""
    rng = np.random.default_rng(3)
    N, S = 400, 400
    R = np.full((N, S), -1.0, dtype=np.float32)                   # every lineup: same true 1% chance of a 60x win, 25% cash at 1.5x
    R[rng.random((N, S)) < 0.25] = 0.5
    R[rng.random((N, S)) < 0.01] = 60.0
    cf = C.cross_fit_book("greedy", R, 10, 3.0, 0.0, np.zeros((N, N)), np.array([0.0, 1.0]))
    assert cf["worlds_fit"] == 200 and cf["worlds_score"] == 200
    # the chooser shrinks each row by its standard error before selecting, so a pure-noise pool no longer yields a
    # book whose in-sample claim beats its out-of-sample score: the selection bias is gone
    assert abs(cf["in_sample"]["mean"] - cf["out_of_sample"]["mean"]) < 0.3
    assert cf["selection_bias"] == pytest.approx(cf["in_sample"]["mean"] - cf["out_of_sample"]["mean"])
    pf = C.cross_fit_book("portfolio", R, 10, 3.0, 0.0, None, np.array([0.0, 1.0]))
    assert pf is not None and pf["in_sample"]["mean"] >= pf["out_of_sample"]["mean"] - 0.5
    assert C.cross_fit_book("portfolio", R[:, :10], 5, 3.0, 0.0, None, np.array([0.0, 1.0])) is None


def test_shrinkage_pulls_noisy_rows_toward_the_pool_and_keeps_precise_ones():
    rng = np.random.default_rng(0)
    S = 400
    true = np.array([0.0] * 50 + [1.0] * 10)                           # ten genuinely good lineups
    noise_sd = np.array([3.0] * 30 + [40.0] * 20 + [3.0] * 10)          # twenty noisy rows among the ordinary ones
    R = true[:, None] + rng.normal(0, 1, (60, S)) * noise_sd[:, None]
    out, info = C.shrink_rows(R)
    mu, mu_s = R.mean(axis=1), out.mean(axis=1)
    assert info["tau"] > 0 and 0 < info["lambda_median"] <= 1
    # the noisy rows' extreme estimates are pulled far more than the precise good ones
    pull = np.abs(mu - mu_s)
    assert pull[30:50].mean() > 5 * pull[50:].mean()
    # the good lineups still rank on top after shrinkage, and the top-20 claim came down
    assert set(np.argsort(-mu_s)[:10]) == set(range(50, 60))
    assert info["top_mean_after"] < info["top_mean_before"]
    # pure noise: the spread of true means is estimated near zero (it fluctuates, so not exactly), and the
    # top-20 claim collapses toward the pool mean
    out0, info0 = C.shrink_rows(rng.normal(0, 10, (40, S)))
    assert info0["lambda_median"] < 0.5 and abs(info0["top_mean_after"]) < 0.5 * abs(info0["top_mean_before"])
