"""Layer 2 of the projection (ivan.sports.nba.projection.worlds): a synthetic game drawn through simulate_game, the seed, the fpts pin."""
from datetime import date

import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import profiles as PR, rotation as RO, script as SC, world as W
from ivan.sports.nba.projection import store as PS
from ivan.sports.nba.projection import worlds as WS


def _box():
    rows = []
    for g in range(6):
        for nm, m, fga, fg3a, fta, fgm, fg3m, ftm in (("a", 34, 16, 6, 5, 8, 2, 4), ("b", 20, 8, 4, 2, 3, 1, 2)):
            rows.append(dict(game_id=f"g{g}", game_date=date(2026, 1, 1 + g), player=nm, team="AAA", minutes=m, fga=fga, fg3a=fg3a, fta=fta, fgm=fgm, fg3m=fg3m, ftm=ftm, rebounds_off=1, rebounds_def=4))
    return pd.DataFrame(rows)


def _players(team, pid0):
    out = []
    for i in range(10):
        row = [0] * 48
        if i < 5:
            row[0:18] = [1] * 18; row[24:42] = [1] * 18
        else:
            row[18:24] = [1] * 6; row[42:48] = [1] * 6
        mins = 36.0 if i < 5 else 12.0
        rates = {"points": 0.5 if i < 5 else 0.4, "rebounds": 0.19, "assists": 0.1, "steals": 0.03, "blocks": 0.02, "turnovers": 0.055, "fg3m": 0.06}
        out.append(W.PlayerIn(pid=pid0 + i, name=f"{team.lower()}{i}", team=team, minutes=mins, rates=rates, cells=row, boundary_sd=1.5, p_out=0.0, fpts_mean=mins, fpts_std=8.0))
    return out


def _ctx(garbage=False):
    pr = PR.ShootingProfiles.build(_box())
    fr = RO.FoulRates(pd.DataFrame(columns=["player", "game_date", "game_id", "f_num", "m_den"]))
    fs = fr.rows.set_index("player") if len(fr.rows) else pd.DataFrame(columns=["f_num", "m_den"])
    params = dict(SC.DEFAULTS, pace_a=0.0, pace_b=1.0, kappa=0.0, lambda_pace=0.0, q_tie=0.0, garbage=garbage)
    return W.WorldContext(params=params, teams=W.LEAGUE_DEFAULTS, groups={}, profiles=pr, prof_state=pr.latest(), foul_rates=fr, foul_state=fs)


def _game():
    return W.GameIn(home="AAA", away="BBB", spread_home=-20.0, total=230.0, pace_home=100.0, pace_away=100.0,
                    players={"AAA": _players("AAA", 1), "BBB": _players("BBB", 100)})


def test_simulate_game_reads_every_player_back_and_is_seeded_by_the_pair():
    ctx = _ctx()
    sim = WS.simulate_game(None, ctx, _game(), n=300)
    assert sim.seed == WS.game_seed("BBB", "AAA") == WS.game_seed("aaa", "bbb") and sim.n == 300
    for t in ("AAA", "BBB"):
        f = sim.frames[t]
        assert len(f) == 10 and set(f.columns) >= {"fpts_mean", "fpts_std", "quantiles", "minutes_sim", "minutes_p10", "minutes_p90", "sim_points", "sim_rebounds"}
        q = np.array(f.loc["aaa0" if t == "AAA" else "bbb0", "quantiles"])
        assert len(q) == 101 and (np.diff(q) >= 0).all() and q[0] <= f["fpts_mean"].iloc[0] <= q[-1]
        s = sim.samples[t]
        assert s["fpts"].shape == (300, 10) and s["minutes"].shape == (300, 10) and s["fpts"].dtype == np.float32
        assert np.allclose(s["fpts"].mean(axis=0), f["fpts_mean"].to_numpy(), atol=0.05)          # the frame is the samples' summary
        assert abs(f["minutes_sim"].sum() - 240.0) < 3.0                                            # five on the floor for 48 minutes
    assert sim.game["garbage"] == 0.0 and not sim.game["garbage_on"]                                 # off: a 20-point spread pulls nobody
    again = WS.simulate_game(None, ctx, _game(), n=300)
    assert np.array_equal(again.samples["AAA"]["fpts"], sim.samples["AAA"]["fpts"])                 # the same inputs draw the same worlds


def test_garbage_time_off_keeps_the_starters_on_the_plan():
    on = WS.simulate_game(None, _ctx(garbage=True), _game(), n=400).frames["AAA"]
    off = WS.simulate_game(None, _ctx(garbage=False), _game(), n=400).frames["AAA"]
    starters = [f"aaa{i}" for i in range(5)]
    assert abs(off.loc[starters, "minutes_sim"].mean() - 36.0) < 1.0
    assert on.loc[starters, "minutes_sim"].mean() < off.loc[starters, "minutes_sim"].mean()         # with it on, a 20-point favourite's closers sit


def test_fpts_pin_moves_the_mean_and_scales_the_samples():
    sim = WS.simulate_game(None, _ctx(), _game(), n=200)
    f, s = sim.frames["AAA"], sim.samples["AAA"]
    old = float(f.loc["aaa0", "fpts_mean"]); old_sd = float(f.loc["aaa0", "fpts_std"])
    WS.apply_fpts_pin(f, s, "aaa0", old * 1.5)
    assert abs(f.loc["aaa0", "fpts_mean"] - old * 1.5) < 1e-9 and abs(f.loc["aaa0", "fpts_std"] - old_sd * 1.5) < 1e-6
    assert abs(f.loc["aaa0", "fpts_model"] - old) < 1e-9 and abs(float(s["fpts"][:, 0].mean()) - old * 1.5) < 0.1
    assert abs(f.loc["aaa1", "fpts_mean"] - float(s["fpts"][:, 1].mean())) < 0.05                     # nobody else moved


def test_store_merged_lays_the_draw_over_the_mapping():
    frame = pd.DataFrame({"minutes": [30.0, 20.0, 5.0], "fpm": [1.0, 1.0, 1.0], "fpts_mean": [30.0, 20.0, 5.0]}, index=pd.Index(["a", "b", "c"], name="player_name"))
    rec = PS.record("AAA", frame, frame.copy(), {}, 1, 0)
    m0 = PS.merged(rec)
    assert list(m0["fpts_mean"]) == [30.0, 20.0, 5.0] and m0["fpts_std"].isna().all()
    sim = pd.DataFrame({"fpts_mean": [33.0, 18.0], "fpts_std": [10.0, 8.0], "quantiles": [[1.0] * 101, [2.0] * 101], "minutes_sim": [29.0, 21.0],
                        "minutes_p10": [20.0, 10.0], "minutes_p90": [38.0, 30.0], "sim_points": [15.0, 8.0]}, index=pd.Index(["a", "b"], name="player_name"))
    PS.set_sim(rec, sim, {"names": ["a", "b"]}, {"factor": {}}, 500)
    m = PS.merged(rec)
    assert list(m["fpts_mean"]) == [33.0, 18.0, 5.0] and m.loc["c", "fpts_std"] != m.loc["c", "fpts_std"] and m.loc["a", "minutes_p90"] == 38.0
    assert rec["n_worlds"] == 500 and m.loc["c", "quantiles"] is None
