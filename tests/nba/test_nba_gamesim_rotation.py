"""Layer B, the rotation draw under game events (ivan.sports.nba.gamesim.rotation)."""
import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import rotation as R


def _grid():
    """Ten players: five starters 0-18 and 24-42, five bench 18-24 and 42-48 (every row 36 or 12 minutes, five on the floor at every minute)."""
    names = [f"s{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
    g = np.zeros((10, 48), dtype=int)
    g[:5, 0:18] = 1; g[:5, 24:42] = 1; g[5:, 18:24] = 1; g[5:, 42:48] = 1
    return pd.DataFrame(g, index=names, columns=list(range(48)))


def _draw(n=400, seed=0, **kw):
    grid = _grid()
    rng = np.random.default_rng(seed)
    path = kw.pop("path", np.zeros((n, 48)))
    ot = kw.pop("overtime", np.zeros(n, dtype=int))
    return R.draw(grid, {nm: 1.5 for nm in grid.index}, n, rng, path, ot, **kw)


def test_five_on_the_floor_and_minutes_conserved():
    d = _draw()
    F = d["onfloor"]
    assert F.shape == (400, 10, 48)
    assert np.allclose(F.sum(axis=1), 5.0, atol=1e-6)                       # five every minute of every world
    assert np.allclose(d["minutes"].sum(axis=1), 240.0, atol=1e-6)
    assert abs(d["minutes"][:, 0].mean() - 36.0) < 1.0 and abs(d["minutes"][:, 5].mean() - 12.0) < 1.0
    assert (d["garbage_at"] == -1).all() and d["fouls"].sum() == 0


def test_scratch_hands_minutes_to_heirs_by_group_with_room():
    groups = {**{f"s{i}": "wing" for i in range(5)}, **{f"b{i}": "wing" for i in range(5)}}
    groups["s0"] = "big"; groups["b0"] = "big"
    d = _draw(p_out={"s0": 1.0}, groups=groups)
    m = pd.Series(d["minutes"].mean(axis=0), index=d["names"])
    assert m["s0"] == 0.0 and abs(d["minutes"].sum(axis=1).mean() - 240.0) < 1e-6
    # the big's minutes went first to the other big (he has room when s0 plays); everyone else a little more than the plan
    assert m["b0"] > 12.0 + 8.0 and m["b1"] > 12.0


def test_fouls_follow_the_rate_and_a_foul_out_leaves():
    d = _draw(foul_rate={nm: 3.0 / 36 for nm in _grid().index}, n=2000)
    f = d["fouls"]
    assert 2.6 < f[:, :5].mean() < 3.4                                       # a 36-minute player at 3 per 36
    assert 0.6 < f[:, 5:].mean() < 1.4                                       # a 12-minute player
    fo = d["fouled_out"]
    assert 0.0 < fo[:, :5].mean() < 0.05                                       # rare
    assert d["minutes"][:, :5][fo[:, :5]].mean() < d["minutes"][:, :5][~fo[:, :5]].mean() - 3.0   # and a fouled-out starter lost minutes
    assert np.allclose(d["onfloor"].sum(axis=1), 5.0, atol=1e-6)
    assert R.sit_share(4, 15) < 1.0 and R.sit_share(4, 2) == 1.0 and R.sit_share(5, 10) < R.sit_share(4, 10) and R.level_rate(35) < R.level_rate(5)


def test_garbage_time_pulls_closers_only_while_the_lead_holds():
    n = 300
    blow = np.tile(np.linspace(0, 30, 48), (n, 1))                             # a lead that grows to 30 and stays
    d = _draw(path=blow, n=n)
    assert ((d["garbage_at"] >= R.GARBAGE_FROM) & (d["garbage_at"] <= 42)).all()      # a 26-point lead with seven minutes left trips the pull
    m = pd.Series(d["minutes"].mean(axis=0), index=d["names"])
    assert m["s0"] < 36.0 - 1.0 and m["b0"] > 12.0 + 1.0                       # the plan already rests the starters from 42: the pull costs them the minute before
    back = blow.copy(); back[:, 45:] = 8.0                                     # a lead that comes back to eight: the closers return
    d2 = _draw(path=back, n=n)
    assert (d2["garbage_at"] == -1).all()
    assert np.allclose(d["onfloor"].sum(axis=1), 5.0, atol=1e-6)


def test_overtime_goes_to_the_five_at_the_buzzer():
    d = _draw(overtime=np.ones(100, dtype=int), n=100)
    assert np.allclose(d["ot_minutes"].sum(axis=1), 25.0, atol=1e-6)
    assert d["ot_minutes"][:, 5:].sum() > d["ot_minutes"][:, :5].sum()        # the bench closes this grid
    assert np.allclose(d["minutes"].sum(axis=1), 265.0, atol=1e-6)
