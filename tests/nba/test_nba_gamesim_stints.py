"""The stint measurements (ivan.sports.nba.gamesim.stints): panels, swap sets, the EM over them, the star-out nights."""
import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import stints as T


def _game(gid="g1", date="2026-01-05"):
    """One team: five starters play 0-360 and 720-2880; at 360 'a' and 'b' go out for 'x' and 'y', who play to the break."""
    rows = []
    for nm in ("a", "b"):
        rows += [(nm, 1, 0.0, 360.0, 0, 1), (nm, 2, 720.0, 2880.0, 1, 1)]
    for nm in ("c", "d", "e"):
        rows += [(nm, 1, 0.0, 2880.0, 1, 1)]
    for nm in ("x", "y"):
        rows += [(nm, 1, 360.0, 720.0, 0, 0)]
    df = pd.DataFrame(rows, columns=["player_name", "stint_no", "start_sec", "end_sec", "to_buzzer", "starter"])
    df["game_id"] = gid; df["game_date"] = pd.Timestamp(date); df["team"] = "AAA"
    return df


def _empty_fouls():
    return pd.DataFrame(columns=["game_id", "player_name", "elapsed"])


def _scores(gid="g1"):
    return pd.DataFrame({"game_id": gid, "elapsed": [100.0, 400.0], "home_margin": [2.0, -3.0], "home": "AAA"})


def test_on_panel_has_five_on_the_floor_every_step_and_marks_the_exits():
    st = _game()
    usual = pd.DataFrame({"game_id": "g1", "player_name": list("abcdexy"), "usual": [30, 30, 30, 30, 30, 10, 10]})
    on = T.on_panel(st, _empty_fouls(), _scores(), usual, {})
    assert (on.groupby("step").size() == 5).all() and on.step.nunique() == T.N_STEPS
    a = on[on.player_name == "a"]
    assert a.out.sum() == 1 and a[a.out == 1].step.iloc[0] == 17 and a[a.out == 1].tis.iloc[0] == 360.0   # out in the step that ends at 360 s
    x = on[on.player_name == "x"]
    assert x.out.sum() == 1 and x[x.out == 1].brk.iloc[0] == 1                                            # x leaves at the period break
    assert on[on.player_name == "c"].out.sum() == 0                                                       # to the buzzer: never an exit
    assert on[(on.player_name == "c") & (on.step == 25)].margin.iloc[0] == -3.0                           # the home margin after the last score


def test_bench_panel_counts_rest_from_the_exit_and_from_the_tip():
    st = _game()
    usual = pd.DataFrame({"game_id": "g1", "player_name": list("abcdexy"), "usual": [30, 30, 30, 30, 30, 10, 10]})
    be = T.bench_panel(st, _empty_fouls(), _scores(), usual, {})
    a = be[be.player_name == "a"]
    assert a.enter.sum() == 1 and a[a.enter == 1].rest.iloc[0] == 360.0 and a.fresh.max() == 0
    x = be[be.player_name == "x"]
    assert x[x.fresh == 1].enter.sum() == 1 and x[x.enter == 1].rest.iloc[0] == 360.0                     # not yet used: rest runs from the tip
    assert x[x.fresh == 0].enter.sum() == 0 and x.step.max() == T.N_STEPS - 1                             # never returned: on the bench to the end


def test_swap_moments_are_sets_with_the_bench_that_was_there():
    mo = T.swap_moments(_game())
    assert list(mo.t) == [360.0, 720.0]
    first = mo.iloc[0]
    assert set(first.outs) == {"a", "b"} and set(first.ins) == {"x", "y"} and set(first.bench) == {"x", "y"} and not first.brk
    second = mo.iloc[1]
    assert set(second.outs) == {"x", "y"} and set(second.ins) == {"a", "b"} and second.brk
    assert dict(zip(second.bench, second.rest)) == {"a": 360.0, "b": 360.0}


def test_em_splits_a_double_swap_by_what_the_single_swaps_say():
    # x always replaces a, y always replaces b in single swaps; the scorer types the double swap the wrong way round
    singles = [(("a",), ("x",))] * 6 + [(("b",), ("y",))] * 6
    counts = T.em_pairs(singles + [(("a", "b"), ("y", "x"))] * 4)
    assert counts[("a", "x")] > 9.5 and counts[("b", "y")] > 9.5
    assert counts.get(("a", "y"), 0.0) < 0.5 and counts.get(("b", "x"), 0.0) < 0.5
    even = T.em_pairs([(("a", "b"), ("x", "y"))] * 4)                     # nothing to tell them apart: an even split
    assert np.isclose(even[("a", "x")], even[("a", "y")])
    lop = T.em_pairs([(("a",), ("x", "y"))])                              # one out, two in: half a credit each
    assert np.isclose(lop[("a", "x")], 0.5) and np.isclose(lop[("a", "y")], 0.5)


def _box():
    """Twelve games; the star 's' (34 min) sits the last one and the deep man 'z' takes 30 of his minutes."""
    rows = []
    for g in range(12):
        gid = f"g{g:02d}"; date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=g)
        last = g == 11
        mins = {"s": 0 if last else 34, "p1": 34, "p2": 34, "p3": 34, "p4": 34, "r1": 20, "r2": 20, "r3": 14 if not last else 18, "r4": 12, "z": 30 if last else 4}
        for nm, m in mins.items():
            rows.append(dict(game_id=gid, game_date=date, player_name=nm, team="AAA", minutes=float(m), dk_fpts=float(m), played=m > 0,
                             starter=1.0 if nm in ("s", "p1", "p2", "p3", "p4") and m > 0 else (1.0 if last and nm == "r1" else 0.0)))
    b = pd.DataFrame(rows); b["season"] = 2025
    return b


def test_star_out_finds_the_night_and_who_took_the_minutes():
    nights, heirs = T.star_out(_box(), {})
    assert len(nights) == 1
    n = nights.iloc[0]
    assert n.star == "s" and n.freed == 34.0 and n.top == "z" and np.isclose(n.top_gain, 26.0) and n.night == 1
    assert n.has_new_starter and not n.top_is_new_starter                 # r1 starts in his place, z takes the minutes
    assert np.isclose(heirs.gain.sum(), 34.0 - 4.0)                       # the gains add up to what he left (the team played 236 without him)
    quiet = T.quiet_nights(_box())
    assert len(quiet) == 3 and quiet.top_gain.max() < 1e-9                # the three nights before, nobody sat, nobody gained


# ----------------------------------------------------------------------------- the model
def _params(out=-2.2, enter=-1.5, brk=0.0):
    """Flat tables: every on-floor man has the same out-chance per step, every bench man the same return chance."""
    zeros = lambda terms: {"|".join(t): np.zeros([len(T.DOMAINS[c]) for c in t]).tolist() for t in terms}   # noqa: E731
    return {"out": {"intercept": out, "terms": zeros(T.OUT_TERMS)}, "in": {"intercept": enter, "terms": zeros(T.IN_TERMS)},
            "brk": {"intercept": brk, "terms": zeros(T.BRK_TERMS)}, "flow": {g: {h: 1.0 for h in T.DOMAINS["group"]} for g in T.DOMAINS["group"]}}


def _team(k=9, pairs=None):
    names = [f"p{i}" for i in range(k)]
    return T.TeamIn(names, [1] * 5 + [0] * (k - 5), [30] * 5 + [15] * (k - 5), ["guard", "guard", "wing", "wing", "big"] + ["guard", "wing", "big", "wing"][:k - 5], pairs)


def test_draw_keeps_five_on_the_floor_and_240_minutes():
    d = T.draw(_team(), _params(), 200, np.random.default_rng(0))
    assert (d["on"].sum(axis=1) == 5).all()
    assert np.allclose(d["minutes"].sum(axis=1), 240.0)
    assert d["on"][:, :5, 0].all() and not d["on"][:, 5:, 0].any()         # the starters open the game
    again = T.draw(_team(), _params(), 200, np.random.default_rng(0))
    assert (again["on"] == d["on"]).all()                                  # one seed, one draw


def test_replacement_follows_the_pair_history_and_falls_through_when_the_usual_man_is_gone():
    k = 9; pairs = np.zeros((k, k)); pairs[0, 5] = 1.0                    # p5 always comes in for p0
    p = _params()
    s = T.Stepper(_team(pairs=pairs), p, 4000, np.random.default_rng(1))
    s.step(0, np.zeros(4000))
    came = s.on[:, 5:] & (~s.on[:, 0])[:, None]                           # worlds where p0 went out in the first step
    only = (~s.on[:, 0]) & (s.on[:, :5].sum(axis=1) == 4)
    assert only.sum() > 100 and came[only, 0].mean() > 0.65               # mostly p5 (one in four by chance)
    gone = T.Stepper(_team(pairs=pairs), p, 4000, np.random.default_rng(1))
    gone.fouls[:, 5] = 6                                                  # p5 unavailable: someone else must come in
    gone.step(0, np.zeros(4000))
    assert (gone.on.sum(axis=1) == 5).all() and not gone.on[:, 5].any()


def test_calibrate_puts_the_minutes_on_the_targets():
    team = _team(); p = _params()
    targets = [36, 34, 30, 28, 26, 24, 22, 20, 20]
    c = T.calibrate(team, p, targets)
    assert np.isclose(c["scale"], 1.0) and np.abs(c["resid"]).max() < 1.5
    d = T.draw(team, p, 600, np.random.default_rng(3))
    assert np.abs(np.median(d["minutes"], axis=0) - np.array(targets)).max() < 2.5
    short = T.calibrate(_team(), p, [30, 30, 28, 26, 24, 20, 18, 16, 8])   # 200 minutes asked for: scaled to the 240 the team plays
    assert np.isclose(short["scale"], 1.2) and np.abs(short["tilt"]).max() < 6.0
