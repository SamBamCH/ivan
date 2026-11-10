"""The soft pool (ivan.sports.nba.projection.cuts; the draw's cuts and fringe in gamesim.rotation): the table, the lookups, conservation."""
from datetime import date

import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import rotation as RO
from ivan.sports.nba.projection import cuts as C


def test_bins_roles_and_the_users_calls():
    t = C.DEFAULT_TABLE
    assert C.role_of(9) == "deep" and C.role_of(18) == "bench" and C.role_of(25) == "starter" and C.role_of(34) == "star"
    assert C.bin_of(0.0) == "0" and C.bin_of(0.2) == "1-2" and C.bin_of(0.35) == "3-4" and C.bin_of(0.9) == "5+" and C.bin_of(None) == "1-2"
    assert C.cut_probability(t, 18, 0.4, mode="all") == t["p_cut"]["bench"]["3-4"] == C.cut_probability(t, 18, 0.4, held=True, mode="all")
    assert C.cut_probability(t, 18, 0.4, starter=True, mode="all") == 0.0 and C.cut_probability(t, 18, 0.4, held=True, mode="unpinned") == 0.0
    assert C.cut_probability(t, 18, 0.4, mode="unpinned") > 0 and C.cut_probability(t, 18, 0.4, mode="off") == 0.0 and C.cut_probability(t, 0, 0.0, mode="all") == 0.0
    p, mu, sd = C.fringe_profile(t, 0.8)
    assert 0 < p < 1 and mu > 0 and sd > 0


def test_fit_reads_cut_rates_off_a_box():
    rows = []
    d0 = date(2026, 1, 1)
    for g in range(40):
        gd = date.fromordinal(d0.toordinal() + g)
        rows.append(dict(player_name="steady", team="AAA", game_date=gd, game_id=f"g{g}", minutes=30.0, played=True))
        rows.append(dict(player_name="flaky", team="AAA", game_date=gd, game_id=f"g{g}", minutes=0.0 if g % 2 else 16.0, played=g % 2 == 0))
    t = C.fit(pd.DataFrame(rows), window=10)
    assert t["rows"] == 60 and t["p_cut"]["star"]["0"] == 0.0 or t["p_cut"]["starter"]["0"] == 0.0
    # flaky is a bench player (16 when he plays) cut half the time: his bin is 5+ and the rate is read off him (the cell has 30 rows, under MIN_ROWS: the role's pooled value, 0.5)
    assert abs(t["p_cut"]["bench"]["5+"] - 0.5) < 0.05 and t["p_dnp_given_cut"]["bench"]["5+"] == 1.0


def _grid(names, starters=5, bench_minutes=(18, 24)):
    g = pd.DataFrame(False, index=names, columns=range(48))
    for i, nm in enumerate(names):
        if i < starters:
            g.loc[nm, list(range(0, 18)) + list(range(24, 42))] = True
        else:
            g.loc[nm, list(range(bench_minutes[0], bench_minutes[1])) + list(range(42, 48))] = True
    return g


def test_cuts_and_fringe_conserve_five_on_the_floor_and_move_the_minutes():
    names = [f"p{i}" for i in range(10)]
    grid = _grid(names)
    grid.loc["fringe1"] = False                       # the plan leaves him at zero
    spreads = {n: 1.5 for n in grid.index}
    n = 600; rng = np.random.default_rng(3)
    d = RO.draw(grid, spreads, n, rng, np.zeros((n, 48)), np.zeros(n, dtype=int), groups={nm: "wing" for nm in grid.index}, garbage=False,
                p_cut={"p9": 0.4}, dnp_share={"p9": 0.5}, token_minutes=4.0, fringe={"fringe1": (0.5, 6.0, 2.0)})
    F, M = d["onfloor"], d["minutes"]
    assert np.allclose(F.sum(axis=1), 5.0, atol=1e-3)                                  # five on the floor in every minute of every world
    assert np.allclose(M.sum(axis=1), 240.0, atol=0.5)                                  # the team's minutes are conserved
    j9 = list(d["names"]).index("p9"); jf = list(d["names"]).index("fringe1")
    assert 0.3 < d["cut"][:, j9].mean() < 0.5 and (M[d["cut"][:, j9], j9] < 8.5).all()  # a cut night is under the token or zero
    assert (M[:, j9] <= 0).mean() > 0.1 and (M[:, j9] <= 0).mean() < 0.3               # the DNP share of the cut worlds
    played = d["fringe_played"][:, jf]
    assert 0.4 < played.mean() < 0.6 and (M[played, jf] > 0.9).all() and M[~played, jf].mean() < 1.0      # an heir at his play weight in the other worlds
    assert 2.0 < M[:, jf].mean() < 4.0                                                   # about half of about six
    pool = [j for j in range(len(d["names"])) if j != jf]
    assert 236 < M[:, pool].sum(axis=1).mean() < 239                                   # the pool plays short by what the fringe took


def test_spread_scales_to_the_players_own_sd():
    cells = np.zeros(48, dtype=bool); cells[:18] = True; cells[24:42] = True            # two stints
    base = RO.spread_for(1.5, 36, None, cells)
    wide = RO.spread_for(1.5, 36, 7.4, cells); tight = RO.spread_for(1.5, 36, 3.0, cells)
    assert base == 1.5 and wide > base > tight and 0.8 <= tight and wide <= 6.0
    assert RO.spread_for(None, 20, None, None) == RO.default_spread(20)
