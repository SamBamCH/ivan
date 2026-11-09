"""Layer C, usage shares (ivan.sports.nba.gamesim.usage)."""
from datetime import date

import pandas as pd

from ivan.sports.nba.gamesim import usage as U


def _pg(game_id, day, fga_a=10, fga_b=5):
    """One game, team AAA with five players a..e sharing 100 possessions; the on-floor denominators as if all five played every possession."""
    rows = []
    for nm, fga in zip("abcde", (fga_a, fga_b, fga_b, fga_b, fga_b)):
        rows.append(dict(game_id=game_id, game_date=day, team="AAA", player=nm, pts=0, fga=fga, fgm=fga // 2, fg3a=fga // 2, fg3m=1, fta=2, ftm=1, oreb=2, dreb=6, ast=4, stl=1, blk=1, tov=2, pf=2,
                         off_poss=100, def_poss=100, team_fga_on=30 - fga, team_fgm_on=15, team_miss_on=15, opp_fga_on=90, opp_miss_on=45))
    return pd.DataFrame(rows)


def test_rates_shrink_toward_the_group_prior_and_the_pie_renormalises():
    pg = pd.concat([_pg("g1", date(2026, 1, 1)), _pg("g2", date(2026, 1, 3))], ignore_index=True)
    minutes = pd.DataFrame([dict(game_id=g, player=p, minutes=30.0) for g in ("g1", "g2") for p in "abcde"])
    groups = {p: "wing" for p in "abcde"}
    u = U.build_usage(pg, minutes, groups)
    st = u.before(date(2026, 1, 2))
    assert set(st.index) == set("abcde") and st.loc["a", "games"] == 1
    # a takes 10 of 100 possessions, the others 5: shrunk toward the league (0.06) with 150 pseudo-possessions
    ra, rb = u.rate(st, "a", "fga_r"), u.rate(st, "b", "fga_r")
    assert 0.06 < ra < 0.10 and 0.05 < rb < 0.06 and ra > rb
    assert u.rate(st, "zz", "fga_r") == u.priors["wing"]["fga_r"]                       # no history: the prior
    assert abs(u.per_minute(st, "a", "fga") - (10 + 60 * u.pm_league["fga"]) / 90.0) < 1e-9
    split = U.split_pie({"a": 0.2, "b": 0.1, "c": 0.1}, 40.0)
    assert abs(split["a"] - 20.0) < 1e-9 and abs(sum(split.values()) - 40.0) < 1e-9
    assert U.split_pie({"a": 0.0, "b": 0.0}, 10.0) == {"a": 5.0, "b": 5.0}


def test_predict_game_replays_the_real_fives():
    pg = pd.concat([_pg("g1", date(2026, 1, 1)), _pg("g2", date(2026, 1, 3))], ignore_index=True)
    minutes = pd.DataFrame([dict(game_id=g, player=p, minutes=30.0) for g in ("g1", "g2") for p in "abcde"])
    u = U.build_usage(pg, minutes, {p: "wing" for p in "abcde"})
    st = u.before(date(2026, 1, 5))
    five = "a|b|c|d|e"; opp = "v|w|x|y|z"
    ev = pd.DataFrame([dict(game_id="g3", team="AAA", opp="BBB", on_off=five, on_def=opp, poss_end=True, poss_team="AAA", kind="tov", made=False, trip_i=0, trip_n=0)] * 10
                      + [dict(game_id="g3", team="AAA", opp="BBB", on_off=five, on_def=opp, poss_end=False, poss_team=None, kind="fg", made=False, trip_i=0, trip_n=0)] * 4
                      + [dict(game_id="g3", team="AAA", opp="BBB", on_off=five, on_def=opp, poss_end=True, poss_team="AAA", kind="fg", made=True, trip_i=0, trip_n=0)] * 6)
    prof = {"AAA": dict(fga_pp=1.0, fg3a_pp=0.4, fta_pp=0.2, tov_pp=0.1, oreb_share=0.25, dreb_share=0.75, ast_share=0.5),
            "BBB": dict(fga_pp=1.0, fg3a_pp=0.4, fta_pp=0.2, tov_pp=0.1, oreb_share=0.25, dreb_share=0.75, ast_share=0.5)}
    p = U.predict_game(ev, u, st, prof).set_index(["team", "player"])
    # 16 possessions ended by AAA: 16 attempts, 1.6 turnovers, 3.2 free throws over the five; a takes the biggest share
    assert abs(p.loc["AAA"].pred_fga.sum() - 16.0) < 1e-6 and abs(p.loc["AAA"].pred_tov.sum() - 1.6) < 1e-6
    assert p.loc[("AAA", "a"), "pred_fga"] > p.loc[("AAA", "b"), "pred_fga"]
    # 4 misses: AAA's 25% offensive-rebound share, BBB's 75% defensive; 6 makes: half assisted
    assert abs(p.loc["AAA"].pred_oreb.sum() - 1.0) < 1e-6 and abs(p.loc["BBB"].pred_dreb.sum() - 3.0) < 1e-6 and abs(p.loc["AAA"].pred_ast.sum() - 3.0) < 1e-6
    # BBB's five face 16 possessions and 10 attempts: steals and blocks at the (prior) rates
    assert abs(p.loc[("BBB", "v"), "pred_stl"] - 16 * u.priors["wing"]["stl_r"]) < 1e-9 and abs(p.loc[("BBB", "v"), "pred_blk"] - 10 * u.priors["wing"]["blk_r"]) < 1e-9


def test_team_profile_uses_player_stats_for_the_pies():
    seg = pd.DataFrame([dict(game_id="g1", game_date=date(2026, 1, 1), team=t, opp=o, segment=0, poss=100, fga=90, fg3a=35, fta=20, fgm=45, tov=14, oreb=12, dreb=35)
                        for t, o in (("AAA", "BBB"), ("BBB", "AAA"))])
    pg = pd.DataFrame([dict(game_id="g1", team=t, tov=12, oreb=10, dreb=33, ast=25) for t in ("AAA", "BBB")])
    tp = U.build_team_profile(seg, pg)
    prof = tp.before("AAA", date(2026, 1, 2))
    assert abs(prof["fga_pp"] - 0.9) < 1e-9 and abs(prof["tov_pp"] - 0.12) < 1e-9 and abs(prof["oreb_share"] - 10 / 45) < 1e-9 and abs(prof["ast_share"] - 25 / 45) < 1e-9
    assert tp.before("AAA", date(2026, 1, 1)) == tp.league                               # nothing before: the league
