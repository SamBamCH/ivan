"""The World (ivan.sports.nba.gamesim.world) and the shooting profiles (gamesim.profiles) on a synthetic two-team game."""
from datetime import date

import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import profiles as PR, rotation as RO, script as SC, world as W


def _box():
    rows = []
    for g in range(6):
        for nm, m, fga, fg3a, fta, fgm, fg3m, ftm in (("a", 34, 16, 6, 5, 8, 2, 4), ("b", 20, 8, 4, 2, 3, 1, 2)):
            rows.append(dict(game_id=f"g{g}", game_date=date(2026, 1, 1 + g), player=nm, team="AAA", minutes=m, fga=fga, fg3a=fg3a, fta=fta, fgm=fgm, fg3m=fg3m, ftm=ftm, rebounds_off=1, rebounds_def=4))
    return pd.DataFrame(rows)


def test_profiles_shrink_and_fit_to_rates():
    pr = PR.ShootingProfiles.build(_box())
    st = pr.latest()
    a = pr.profile(st, "a"); z = pr.profile(st, "zz")
    assert z == pr.league and 0.3 < a["fga_pm"] < 0.5 and 0.4 < a["fg2_pct"] < 0.6 and 0.1 < a["oreb_frac"] < 0.4
    q = PR.fit_to_rates(a, points_pm=0.9, fg3m_pm=0.1)
    assert abs(PR.points_per_minute(q) - 0.9) < 1e-6 and abs(q["fg3a_pm"] * q["fg3_pct"] - 0.1) < 1e-6
    q2 = PR.fit_to_rates(a, points_pm=5.0, fg3m_pm=None)                       # an impossible target is clipped at the factor's ceiling
    assert q2["fta_pm"] == a["fta_pm"] * 2.0


def _players(team, pid0, home):
    names = [f"{team.lower()}{i}" for i in range(10)]
    cells = []
    for i in range(10):
        row = [0] * 48
        if i < 5:
            row[0:18] = [1] * 18; row[24:42] = [1] * 18
        else:
            row[18:24] = [1] * 6; row[42:48] = [1] * 6
        cells.append(row)
    out = []
    for i, nm in enumerate(names):
        mins = 36.0 if i < 5 else 12.0
        rates = {"points": 0.5 if i < 5 else 0.4, "rebounds": 0.19, "assists": 0.1, "steals": 0.03, "blocks": 0.02, "turnovers": 0.055, "fg3m": 0.06}
        out.append(W.PlayerIn(pid=pid0 + i, name=nm, team=team, minutes=mins, rates=rates, cells=cells[i], boundary_sd=1.5, p_out=0.0, fpts_mean=mins * 1.0, fpts_std=8.0))
    return out


def test_world_draws_a_game_that_adds_up():
    game = W.GameIn(home="AAA", away="BBB", spread_home=-5.0, total=230.0, pace_home=100.0, pace_away=100.0,
                    players={"AAA": _players("AAA", 1, True), "BBB": _players("BBB", 100, False)})
    pr = PR.ShootingProfiles.build(_box()); st = pr.latest()
    fr = RO.FoulRates(pd.DataFrame(columns=["player", "game_date", "game_id", "f_num", "m_den"])); fs = fr.rows.set_index("player") if len(fr.rows) else pd.DataFrame(columns=["f_num", "m_den"])
    params = dict(SC.DEFAULTS, pace_a=0.0, pace_b=1.0, kappa=0.0, lambda_pace=0.0, q_tie=0.0)
    rng = np.random.default_rng(0)
    d = W.draw_game(game, 400, rng, params, W.LEAGUE_DEFAULTS, st, pr, fs, fr, {}, None)
    n = 400
    pts = {t: d["accs"][t].pts().sum(axis=1) for t in ("AAA", "BBB")}
    assert abs(d["total"].mean() - 230.0) < 6.0 and abs(d["margin"].mean() - 5.0) < 3.0           # the reconciliation puts the means on the lines
    assert np.allclose(d["total"], pts["AAA"] + pts["BBB"]) and np.allclose(d["margin"], pts["AAA"] - pts["BBB"])
    for t in ("AAA", "BBB"):
        a = d["accs"][t]
        assert np.allclose(a.minutes.sum(axis=1), 240.0 + 25.0 * d["overtime"], atol=1e-6)        # five on the floor every minute, plus overtime
        assert (a.fg2m <= a.fg2a).all() and (a.fg3m <= a.fg3a).all() and (a.ftm <= a.fta).all()
        assert a.minutes[:, :5].mean() > a.minutes[:, 5:].mean()
    # the misses of one team are the rebounds of both, the turnovers the other team's steals, the missed shots its blocks
    miss = {t: ((d["accs"][t].fg2a + d["accs"][t].fg3a) - (d["accs"][t].fg2m + d["accs"][t].fg3m)).sum(axis=1) for t in ("AAA", "BBB")}
    reb_total = sum((d["accs"][t].oreb + d["accs"][t].dreb).sum(axis=1) for t in ("AAA", "BBB"))
    assert (reb_total <= miss["AAA"] + miss["BBB"] + (d["accs"]["AAA"].fta - d["accs"]["AAA"].ftm).sum(axis=1) + (d["accs"]["BBB"].fta - d["accs"]["BBB"].ftm).sum(axis=1) + 1e-9).all()
    assert (d["accs"]["BBB"].stl.sum(axis=1) <= d["accs"]["AAA"].tov.sum(axis=1)).all() and (d["accs"]["AAA"].blk.sum(axis=1) <= miss["BBB"]).all()
    # a miss the offence does not rebound is a defender's at the defence's share, never all of them: the rest are team rebounds
    for off, de in (("AAA", "BBB"), ("BBB", "AAA")):
        p = d["setups"][de].pie["p_dreb"]
        assert W.P_DREB_LO <= p <= W.P_DREB_HI
        left = miss[off] + (d["accs"][off].fta - d["accs"][off].ftm).sum(axis=1) - d["accs"][off].oreb.sum(axis=1)       # every missed free throw: an upper bound
        got = d["accs"][de].dreb.sum(axis=1)
        assert (got <= left).all() and got.mean() < p * left.mean() + 0.5
        plan = float((d["setups"][de].w_dreb * d["setups"][de].base).sum())
        if W.P_DREB_LO < p < W.P_DREB_HI:                                                           # unclipped: the draw's mean is the plan's
            assert abs(got.mean() - plan) < 0.08 * plan + 1.0
    assert 0.0 <= (d["garbage_at"] >= 0).mean() <= 1.0


def test_dk_points_and_bonuses():
    stats = {s: np.zeros((2, 1)) for s in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")}
    stats["pts"][0, 0] = 20; stats["reb"][0, 0] = 10; stats["ast"][0, 0] = 10; stats["fg3m"][0, 0] = 2; stats["tov"][0, 0] = 2
    fp = W.dk_points(stats)
    assert abs(fp[0, 0] - (20 + 1 + 12.5 + 15 - 1 + 1.5 + 3.0)) < 1e-9 and fp[1, 0] == 0.0
