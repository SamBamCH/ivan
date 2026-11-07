"""Layer A, the game script (ivan.sports.nba.gamesim.script)."""
import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import script as S


def _seg(game_id="g1", date="2026-01-05", home="AAA", away="BBB", poss=12, pts_h=14, pts_a=13, segments=8):
    rows = []
    for s in range(segments):
        for team, opp, is_home, pts in ((home, away, True, pts_h), (away, home, False, pts_a)):
            rows.append(dict(game_id=game_id, game_date=pd.Timestamp(date).date(), team=team, opp=opp, home=is_home, segment=s, poss=poss, pts=pts,
                             fga=10, fg3a=4, fgm=5, fg3m=2, fta=3, ftm=2, tov=1, oreb=1, dreb=4, seconds=360.0 if s < 8 else 300.0))
    return pd.DataFrame(rows)


def test_efficiencies_and_expected_possessions():
    eh, ea = S.efficiencies(230.0, -6.0, 100.0)
    assert abs(eh - 1.18) < 1e-9 and abs(ea - 1.12) < 1e-9 and abs((eh + ea) * 100 - 230) < 1e-9 and abs((eh - ea) * 100 - 6) < 1e-9
    assert S.expected_possessions(98.0, 102.0, dict(pace_a=0.0, pace_b=1.0)) == 100.0


def test_draw_centres_on_the_lines_and_reports_the_game():
    p = dict(S.DEFAULTS, kappa=0.0, lambda_pace=0.0, q_tie=0.0)
    d = S.draw(20000, 100.0, 1.18, 1.12, p, np.random.default_rng(0))
    assert d["poss"].shape == (20000, 8) and d["path"].shape == (20000, 8)
    assert abs(d["total"].mean() - 230.0) < 1.5 and abs(d["margin"].mean() - 6.0) < 0.6
    assert abs(d["pace"].mean() - 100.0) < 0.3 and 8 < d["total"].std() < 30 and 8 < d["margin"].std() < 20
    assert (d["overtime"] > 0).mean() < 0.08 and ((d["garbage_at"] == -1) | (d["garbage_at"] >= 42)).all()
    assert (d["margin"][d["overtime"] > 0] != 0).all()                                   # a game that went to overtime was settled
    # the pull-back narrows the margin, the tie bump raises the overtime rate, the pace coupling narrows the total
    d2 = S.draw(20000, 100.0, 1.18, 1.12, dict(p, kappa=0.01), np.random.default_rng(0))
    assert d2["margin"].std() < d["margin"].std() - 1.0
    d3 = S.draw(20000, 100.0, 1.18, 1.12, dict(p, q_tie=0.5), np.random.default_rng(0))
    assert (d3["overtime"] > 0).mean() > (d["overtime"] > 0).mean() + 0.03
    d4 = S.draw(20000, 100.0, 1.18, 1.12, dict(p, lambda_pace=1.0), np.random.default_rng(0))
    assert d4["total"].std() < d["total"].std()


def test_trailing_pace_and_lined_games():
    seg = pd.concat([_seg("g1", "2026-01-01", poss=12), _seg("g2", "2026-01-03", poss=13), _seg("g3", "2026-01-05", poss=11)], ignore_index=True)
    gp = S.game_pace(seg)
    assert gp.poss.tolist() == [96, 96, 104, 104, 88, 88]                                  # 8 segments x the segment's possessions
    tp = S.trailing_pace(gp, league=100.0, prior_games=0.0)
    assert np.isclose(tp[gp.game_id == "g1"], 100.0).all()                                  # no history: the league
    assert np.isclose(tp[gp.game_id == "g2"], 96.0).all() and (tp[gp.game_id == "g3"] > 96).all()
    lines = pd.DataFrame([dict(game_id="g3", spread_home=-4.0, total=216.0)])
    g = S.lined_games(seg, lines, dict(S.DEFAULTS, pace_a=0.0, pace_b=1.0))
    assert len(g) == 1 and g.iloc[0].margin == 8 and g.iloc[0].final_total == 216 and abs(g.iloc[0].e_home * g.iloc[0].P - 110.0) < 1e-6
    shape = S.segment_shape(seg)
    assert len(shape) == 8 and abs(sum(shape) - 1.0) < 1e-9


def test_check_runs_and_pit_is_a_probability():
    seg = _seg("g1", "2026-01-05")
    lines = pd.DataFrame([dict(game_id="g1", spread_home=-6.0, total=230.0)])
    r = S.check(seg, lines, dict(S.DEFAULTS), n=500)
    assert r["summary"]["games"] == 1 and 0.0 <= r["games"].pit_total.iloc[0] <= 1.0 and 0.0 <= r["games"].pit_margin.iloc[0] <= 1.0
    assert S.pit(np.array([1.0, 2.0, 3.0, 4.0]), 2.5) == 0.5 and S.pit(np.array([1.0, 2.0, 3.0]), 2.0) == 0.5


def test_match_is_a_bisection_on_a_monotone_function():
    assert abs(S._match(lambda x: 2 * x, 1.0, 0.0, 2.0) - 0.5) < 1e-3
    assert S._match(lambda x: 2 * x, -1.0, 0.0, 2.0) == 0.0 and S._match(lambda x: 2 * x, 10.0, 0.0, 2.0) == 2.0
