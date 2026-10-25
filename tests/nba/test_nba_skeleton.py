"""The slot skeleton (ivan.sports.nba.projection.skeleton): five on the floor, targets hit, locks held, vacated slots inherited by group."""
import numpy as np
import pandas as pd

from ivan.sports.nba.projection import skeleton as S


def curve_of(spec):
    """{name: [(start, end), ...]} -> a curve frame like rotation.curves (shares 0/1, minutes = the total)."""
    rows = {}
    for n, stints in spec.items():
        row = {m: 0.0 for m in range(48)}
        for a, b in stints:
            for m in range(a, b):
                row[m] = 1.0
        row.update(games=10, minutes=float(sum(b - a for a, b in stints)), stints=float(len(stints)), played_share=1.0, last_game="2026-01-01")
        rows[n] = row
    return pd.DataFrame(rows).T


# a ten-man rotation: two bigs (C1 36, C2 12), two guards (G1 34, G2 14), wings; the team adds to 240
ROTATION = {
    "c1": [(0, 9), (14, 24), (24, 33), (38, 48)],      # 36
    "g1": [(0, 8), (14, 24), (26, 34), (40, 48)],       # 34
    "w1": [(0, 10), (15, 24), (27, 36), (40, 48)],      # 36
    "w2": [(0, 7), (13, 24), (24, 31), (37, 48)],       # 36
    "g3": [(0, 6), (12, 24), (24, 30), (36, 48)],       # 36
    "c2": [(9, 14), (33, 38)],                          # 10 (the backup big: the exact complement of c1 over 48... plus 2 spare)
    "g2": [(8, 14), (34, 40)],                          # 12
    "w3": [(10, 15), (36, 40)],                         # 9
    "w4": [(7, 13), (31, 37)],                          # 12
    "g4": [(6, 12), (30, 36)],                          # 12
}
GROUPS = {"c1": "big", "c2": "big", "g1": "guard", "g2": "guard", "g3": "guard", "g4": "guard", "w1": "wing", "w2": "wing", "w3": "wing", "w4": "wing"}


def test_full_roster_lands_on_the_targets_with_five_on_the_floor():
    curve = curve_of(ROTATION)
    targets = {n: float(curve.loc[n, "minutes"]) for n in ROTATION}
    scale = 240.0 / sum(targets.values()); targets = {n: v * scale for n, v in targets.items()}
    grid = S.build_grid(curve, targets, list(ROTATION), starters=["c1", "g1", "w1", "w2", "g3"], groups=GROUPS)
    assert set(S.column_counts(grid)) == {5}
    dev = grid.attrs["deviation"]
    assert max(abs(v) for v in dev.values()) <= 1.5, dev                       # an integer grid against fractional targets, plus one minute a line alignment may trade
    assert all(grid.loc[n, 0] for n in ["c1", "g1", "w1", "w2", "g3"])         # starters at the tip
    assert list(grid.index[:5]) == sorted(["c1", "g1", "w1", "w2", "g3"], key=lambda n: (-targets[n], n))


def test_vacated_big_slot_is_inherited_by_the_big_who_is_owed_minutes():
    curve = curve_of(ROTATION)
    avail = [n for n in ROTATION if n != "c1"]
    # the solve gave c1's 36 minutes mostly to c2 (big) and a little to everyone
    targets = {n: float(curve.loc[n, "minutes"]) for n in avail}
    targets["c2"] += 24.0
    rest = 240.0 - sum(targets.values())
    for n in avail:
        if n != "c2":
            targets[n] += rest / (len(avail) - 1)
    grid = S.build_grid(curve, targets, avail, starters=["c2", "g1", "w1", "w2", "g3"], groups=GROUPS)
    assert set(S.column_counts(grid)) == {5}
    assert abs(grid.loc["c2"].sum() - targets["c2"]) <= 1.0
    inh = grid.attrs["inherited"]["c1"]
    assert set(inh) == {"c2"}                                                   # a big's slot goes to the big alone
    # c2 now plays c1's stretches: on at the tip and in c1's second-half run
    assert grid.loc["c2", 0] and grid.loc["c2", list(range(40, 48))].sum() >= 6


def test_guard_slot_prefers_guards_but_is_shared_and_a_newcomer_takes_what_he_inherits():
    curve = curve_of(ROTATION)
    avail = [n for n in ROTATION if n != "g1"] + ["new guy"]
    targets = {n: float(curve.loc[n, "minutes"]) for n in ROTATION if n != "g1"}
    targets["g2"] += 10.0; targets["w3"] += 6.0; targets["new guy"] = 18.0
    grid = S.build_grid(curve, targets, avail, starters=["c1", "g2", "w1", "w2", "g3"], groups={**GROUPS, "new guy": "guard"})
    assert set(S.column_counts(grid)) == {5}
    inh = grid.attrs["inherited"]["g1"]
    assert inh["g2"] > inh["w3"] and inh["new guy"] > inh["w3"]                 # same group weighs more
    assert abs(grid.loc["new guy"].sum() - 18) <= 1.0
    assert max(abs(v) for v in grid.attrs["deviation"].values()) <= 2.0          # coherent stints can leave one row a little off


def test_locked_rows_are_held_exactly_and_the_rest_fill_around_them():
    curve = curve_of(ROTATION)
    targets = {n: float(curve.loc[n, "minutes"]) for n in ROTATION}
    scale = 240.0 / sum(targets.values()); targets = {n: v * scale for n, v in targets.items()}
    painted = pd.DataFrame([[True] * 20 + [False] * 28], index=["c2"], columns=list(range(48)))     # c2 painted: the first 20 minutes straight
    grid = S.build_grid(curve, targets, list(ROTATION), starters=["c1", "g1", "w1", "w2", "g3"], groups=GROUPS, locks=painted)
    assert grid.loc["c2"].tolist() == painted.loc["c2"].tolist()
    assert set(S.column_counts(grid)) == {5}
    assert grid.attrs["locked"] == ["c2"]
    off = pd.DataFrame([[False] * 48], index=["w3"], columns=list(range(48)))                        # painted off: no minutes, stays off
    grid = S.build_grid(curve, targets, list(ROTATION), starters=["c1", "g1", "w1", "w2", "g3"], groups=GROUPS, locks=off)
    assert grid.loc["w3"].sum() == 0 and set(S.column_counts(grid)) == {5}


def test_zero_target_players_get_no_minutes_and_no_history_works():
    names = [f"p{i}" for i in range(13)]
    targets = {n: t for n, t in zip(names, [36, 34, 32, 30, 28, 24, 20, 16, 12, 8, 0, 0, 0])}
    grid = S.build_grid(None, targets, names, starters=names[:5])
    assert set(S.column_counts(grid)) == {5}
    assert all(grid.loc[n].sum() == 0 for n in names[10:])
    assert max(abs(v) for v in grid.attrs["deviation"].values()) <= 1.0


def test_a_big_and_a_guard_stay_on_the_floor_every_minute():
    curve = curve_of(ROTATION)
    targets = {n: float(curve.loc[n, "minutes"]) for n in ROTATION}
    scale = 240.0 / sum(targets.values()); targets = {n: v * scale for n, v in targets.items()}
    grid = S.build_grid(curve, targets, list(ROTATION), starters=["c1", "g1", "w1", "w2", "g3"], groups=GROUPS)
    G = grid[list(range(48))]
    bigs = [n for n in grid.index if GROUPS[n] == "big"]; guards = [n for n in grid.index if GROUPS[n] == "guard"]
    assert all(G.loc[bigs, m].any() for m in range(48))
    assert all(G.loc[guards, m].any() for m in range(48))
    assert set(S.column_counts(grid)) == {5} and max(abs(v) for v in grid.attrs["deviation"].values()) <= 2.0


def test_align_lines_snaps_a_same_line_swap_a_minute_apart():
    # five on: c1 g1 w1 w2 g3; at minute 8 the backup big c2 enters for the wing w2; at minute 9 the big c1 leaves and the wing w3 enters
    names = ["c1", "g1", "w1", "w2", "g3", "c2", "w3"]
    g = np.zeros((7, 48), dtype=bool)
    g[0, :9] = True; g[1, :] = True; g[2, :] = True; g[4, :] = True
    g[3, :8] = True; g[5, 8:] = True; g[6, 9:] = True
    groups = ["big", "guard", "wing", "wing", "guard", "big", "wing"]
    assert set(g.sum(axis=0)) == {5}
    out = S.align_lines(g, groups, [True] * 7, np.array([36, 36, 36, 20, 36, 20, 16.0]))
    assert set(out.sum(axis=0)) == {5}
    assert not out[5, 8] and out[5, 9] and out[6, 8]            # c2 now enters at 9, when c1 leaves; w3 at 8, when w2 left


def test_apply_depth_keeps_the_top_players_and_the_team_total():
    targets = {"a": 34, "b": 32, "c": 30, "d": 28, "e": 26, "f": 22, "g": 18, "h": 14, "i": 12, "j": 10, "k": 6, "l": 4, "m": 2, "n": 2.0}
    out = S.apply_depth(targets, 10, keep=["m"])                 # m is pinned by hand: he stays whatever his target, and counts toward the ten
    assert sum(1 for v in out.values() if v > 0) == 10 and out["m"] == 2.0 and out["j"] == 0 and out["k"] == 0 and out["l"] == 0 and out["n"] == 0
    assert abs(sum(out.values()) - sum(targets.values())) < 1e-6
    assert out["a"] > 34 and abs(out["a"] / out["i"] - 34 / 12) < 1e-6             # the freed minutes are spread in proportion
