"""The slate as one book (ivan.core.slate_book + portfolio.pick_book / improve_book), sport-agnostic."""
import numpy as np
import pytest

from ivan.core import slate_book as SB
from ivan.core.portfolio import book_utility, improve_book, pick_book


def _rows(rng, n, worlds, kind):
    """Lineups of one kind: 'steady' cash small and often, 'tail' win big and rarely, each a worlds-long ROI row."""
    R = np.full((n, worlds), -1.0, dtype=np.float32)
    for i in range(n):
        if kind == "steady":
            hit = rng.random(worlds) < 0.55
            R[i, hit] = 0.8
        else:
            hit = rng.random(worlds) < 0.04
            R[i, hit] = 25.0
    return R


def _contest(cid, name, fee, slots, R, rng):
    N, W = R.shape
    return SB.ContestCandidates(contest_id=cid, contest_name=name, fee=fee, slots=slots, lineup_ids=list(range(1, N + 1)), roi_raw=R,
                                fpts=rng.normal(100, 10, (N, W)), cash=(R > -1).mean(axis=1), counts=(N, N, N, N, 0))


def test_pick_book_fills_each_contests_slots_and_never_repeats_a_pair():
    rng = np.random.default_rng(0)
    net = rng.normal(0, 1, (12, 200)).astype(np.float32)
    group = np.array([0] * 7 + [1] * 5)
    sel, trace, diag = pick_book(net, 100.0, group, [3, 2])
    assert len(sel) == 5 and len(set(sel)) == 5
    assert sorted(group[sel].tolist()) == [0, 0, 0, 1, 1]
    assert [t["pick"] for t in trace] == [1, 2, 3, 4, 5] and all(t["group"] in (0, 1) for t in trace)
    assert diag["steps"] == 5


def test_improve_book_never_lowers_the_utility_and_finds_a_blocked_swap():
    """A greedy that starts from a deliberately bad pick: the exchange pass swaps it out and the utility rises."""
    rng = np.random.default_rng(1)
    W = 400
    good = rng.normal(0.5, 1.0, (6, W)).astype(np.float32)
    bad = np.full((2, W), -0.5, dtype=np.float32)
    net = np.vstack([good, bad])
    group = np.zeros(8, dtype=int)
    start = [6, 7, 0]                                        # two bad picks forced in
    B = 50.0
    u0 = book_utility(net[start].sum(axis=0), B)
    sel, info = improve_book(net, B, group, start)
    assert info["utility_after"] >= info["utility_before"] - 1e-9 and info["utility_before"] == pytest.approx(u0)
    assert info["swaps"] >= 2 and 6 not in sel and 7 not in sel and len(set(sel)) == 3
    assert book_utility(net[sel].sum(axis=0), B) == pytest.approx(info["utility_after"])
    # on an already-optimal book the pass does nothing
    sel2, info2 = improve_book(net, B, group, sel)
    assert sel2 == sel and info2["swaps"] == 0


def test_choose_book_spreads_the_book_and_puts_steady_lineups_where_they_pay():
    """
    Two contests on the same worlds: a flat-payout one (a steady casher is worth +0.8 fee in half the worlds there) and a
    top-heavy one (the same lineup only cashes a little there, tails pay 25x). The pool holds both kinds in both contests.
    The book should hold both kinds overall (the log spreads across worlds), and the steady lineups should land in the
    flat contest more than in the top-heavy one.
    """
    rng = np.random.default_rng(2)
    W = 600
    steady = _rows(rng, 8, W, "steady"); tail = _rows(rng, 8, W, "tail")
    # the flat contest pays the steady lineups as drawn; the tails cash rarely and small there
    flat = np.vstack([steady, np.where(tail > -1, 0.8, -1.0).astype(np.float32)])
    # the top-heavy contest: the steady lineups' hits are worth little, the tails' hits 25x
    heavy = np.vstack([np.where(steady > -1, 0.1, -1.0).astype(np.float32), tail])
    c_flat = _contest(1, "Double Up", 10.0, 6, flat, rng)
    c_heavy = _contest(2, "Millionaire", 10.0, 6, heavy, rng)
    out = SB.choose_book([c_flat, c_heavy], bankroll=300.0, weights={"cash_pct": 0.5, "avg_roi_win": 0.5}, shared={"method": "mc", "draw": "slate"})
    picks_flat, picks_heavy = out["picks"]
    assert len(picks_flat) == 6 and len(picks_heavy) == 6
    assert len(set(picks_flat)) == 6 and len(set(picks_heavy)) == 6               # a lineup once per contest
    steady_ids = set(range(1, 9))
    n_steady_flat = len(steady_ids & set(picks_flat)); n_steady_heavy = len(steady_ids & set(picks_heavy))
    assert n_steady_flat > n_steady_heavy                                         # the steady ones go where they pay
    assert n_steady_heavy < 6                                                     # and the top-heavy contest carries tails
    book = out["book"]
    assert book["method"] == "slate" and book["entries"] == 12 and book["fees"] == 120.0 and book["worlds"] == W
    assert book["chosen"]["scenarios"] == W and book["chosen"]["bankroll"] == 300.0
    assert book["improve"]["log_growth_after"] >= book["improve"]["log_growth_before"] - 1e-12
    assert book["lineups_distinct"] <= 12 and book["lineups_reused"] == sum(1 for p in book["placements"] if p["n"] > 1)
    assert {c["contest_name"] for c in book["contests"]} == {"Double Up", "Millionaire"}
    for c in book["contests"]:
        assert c["entries"] == 6 and c["cash_mean"] is not None and c["p_loss"] is not None
    assert book["cross_fit"] is not None and book["cross_fit"]["worlds_fit"] == W // 2
    # per-contest stats carry what the exports write, with the slate book under the viz
    st = out["stats"][0]
    assert st["chosen"] == 6 and st["book"]["method"] == "slate" and st["viz"]["slate"] is book and len(st["trace"]) == 6
    assert set(st["by_lineup"]) == set(picks_flat)


def test_choose_book_checks_the_world_axis_and_handles_an_empty_contest():
    rng = np.random.default_rng(3)
    a = _contest(1, "A", 5.0, 2, rng.normal(0, 1, (5, 100)).astype(np.float32), rng)
    b = _contest(2, "B", 5.0, 2, rng.normal(0, 1, (5, 80)).astype(np.float32), rng)
    with pytest.raises(ValueError, match="world counts"):
        SB.choose_book([a, b], 100.0)
    empty = SB.ContestCandidates(contest_id=3, contest_name="C", fee=5.0, slots=3, lineup_ids=[], roi_raw=np.zeros((0, 0), dtype=np.float32),
                                 fpts=np.zeros((0, 0)), cash=np.zeros(0), counts=(0, 0, 0, 0, 0))
    out = SB.choose_book([a, empty], 100.0)
    assert len(out["picks"][0]) == 2 and out["picks"][1] == [] and out["stats"][1]["chosen"] == 0
    assert out["book"]["entries"] == 2 and [c["contest_name"] for c in out["book"]["contests"]] == ["A"]
    assert SB.describe_book(out["book"]) and "slate book: 2 entries" in SB.describe_book(out["book"])[0]
    assert SB.describe_book(None) == []
    with pytest.raises(ValueError):
        SB.bankroll_dollars_of({"bankroll_dollars": -1})
    assert SB.bankroll_dollars_of({}) == SB.DEFAULT_BANKROLL_DOLLARS
