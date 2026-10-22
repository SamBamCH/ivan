"""The rotation grid (ivan.sports.nba.projection.rotation): proposal invariants, the stint sampler's conservation and garbage rule, drawn grids."""
import numpy as np
import pandas as pd

from ivan.sports.nba.projection import rotation as R


def fake_curve(names, minutes):
    """A curve frame where each player is on for the first `minutes` of the game (crude but valid)."""
    rows = {}
    for n, m in zip(names, minutes):
        row = {k: (1.0 if k < m else 0.0) for k in range(R.MINUTES)}
        row.update(games=10, minutes=float(m), stints=2.0, played_share=1.0, last_game="2026-01-01")
        rows[n] = row
    return pd.DataFrame(rows).T


def test_scale_targets_takes_surplus_from_the_bottom_and_caps():
    t = R.scale_targets(np.array([38, 36, 34, 30, 28, 24, 20, 14, 10, 8, 6, 4.0]))
    assert abs(t.sum() - 240) < 0.1 and t[0] >= 37 and t[-1] < 4.0 and t.max() <= R.MAX_MINUTES
    t = R.scale_targets(np.array([46, 40, 30, 30, 30, 30, 30.0]))
    assert t.max() <= R.MAX_MINUTES and abs(t.sum() - 240) < 0.2            # two capped at 40, the rest share the remainder


def test_propose_keeps_five_on_the_floor_and_respects_starters_and_ceiling():
    names = [f"p{i}" for i in range(12)]
    curve = fake_curve(names, [40, 38, 36, 34, 32, 26, 22, 18, 14, 10, 6, 4])
    grid = R.propose(curve, names, starters=names[5:10])
    assert set(R.column_counts(grid)) == {5} and grid.shape == (12, 48)
    assert all(grid.loc[n, 0] for n in names[5:10])                           # the named starters open the game
    assert grid.sum(axis=1).max() <= R.MAX_MINUTES
    assert max(b - a for n in names for a, b in R.typical_stints(grid.loc[n].values.astype(float))) <= R.MAX_STINT + 2


def test_sample_worlds_conserves_minutes_and_pulls_closers_in_blowouts():
    names = [f"p{i}" for i in range(10)]
    curve = fake_curve(names, [36, 34, 32, 30, 28, 24, 20, 16, 12, 8])
    grid = R.propose(curve, names, starters=names[:5])
    rng = np.random.default_rng(0)
    d = R.sample_worlds(grid, {n: 1.5 for n in names}, 3000, rng, spread=0.0, curve=curve)
    M = d["minutes"]
    assert M.shape == (3000, 10) and np.all(M >= 0)
    no_ot = d["overtime"] == 0
    assert np.allclose(M[no_ot].sum(axis=1), 240.0, atol=1e-6)                 # every regulation world adds to 240
    base = grid.sum(axis=1).values
    close = (d["garbage_at"] < 0) & no_ot
    assert np.abs(M[close].mean(axis=0) - base).max() < 2.5                     # close games sit on the grid (jitter clipping drifts a little)
    blown = d["garbage_at"] >= 0
    assert blown.mean() > 0.05
    assert M[blown][:, :5].sum(axis=1).mean() < M[close][:, :5].sum(axis=1).mean() - 3   # closers lose minutes in blowouts
    far = R.sample_worlds(grid, {n: 1.5 for n in names}, 3000, rng, spread=-15.0, curve=curve)
    assert (far["garbage_at"] >= 0).mean() > blown.mean()                        # a big favourite blows out more often
    out = R.sample_worlds(grid, {n: 1.5 for n in names}, 500, rng, spread=0.0, p_out={names[0]: 1.0}, curve=curve)
    assert np.all(out["minutes"][:, 0] == 0) and np.allclose(out["minutes"][out["overtime"] == 0].sum(axis=1), 240.0, atol=1e-6)


def test_drawn_grid_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROTATIONS_DIR", tmp_path)
    names = ["a b", "c d"]
    grid = pd.DataFrame([[True] * 24 + [False] * 24, [False] * 24 + [True] * 24], index=names, columns=list(range(48)))
    R.save_drawn("XYZ", grid)
    back = R.load_drawn("XYZ")
    assert back.equals(grid) and R.drawn_minutes("XYZ") == {"a b": 24.0, "c d": 24.0} and R.stints_of(back)["c d"] == [(24, 48)]
    assert R.clear_drawn("XYZ") and R.load_drawn("XYZ") is None
