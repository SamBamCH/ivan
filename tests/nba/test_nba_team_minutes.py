"""The team minutes draw (ivan.sports.nba.projection.team_minutes) and the game-lines consensus (projection.lines)."""
import numpy as np
import pandas as pd

from ivan.sports.nba.projection import team_minutes as TM
from ivan.sports.nba.projection.lines import GameLines, consensus


def test_team_draw_sums_to_the_budget_and_scales_noise_by_role():
    rng = np.random.default_rng(0)
    mu = [36, 34, 30, 28, 26, 22, 18, 14, 12, 10, 6, 4]
    roles = [TM.role_of(m, m >= 26) for m in mu]
    d = TM.sample_team_minutes(mu, roles, 4000, rng, spread=0.0, params=TM.DEFAULTS)
    total = d.minutes.sum(axis=1)
    assert np.allclose(total, 240 + 25 * d.overtime, atol=1e-6)
    cv = d.minutes.std(axis=0) / d.minutes.mean(axis=0)
    assert cv[0] < cv[5] < cv[-1]                                  # star tighter than rotation tighter than deep bench
    assert abs(d.minutes.mean(axis=0)[0] - 36) < 2.0               # the mean rotation is roughly preserved
    assert 0.03 < d.overtime.mean() < 0.09


def test_blowouts_move_starter_minutes_to_the_bench_and_pins_hold():
    rng = np.random.default_rng(1)
    mu = [36, 34, 30, 28, 26, 22, 18, 14, 12, 10, 6, 4]
    roles = [TM.role_of(m, m >= 26) for m in mu]
    close = TM.sample_team_minutes(mu, roles, 3000, rng, params=TM.DEFAULTS, margin=np.zeros(3000), overtime=np.zeros(3000, dtype=int))
    blow = TM.sample_team_minutes(mu, roles, 3000, rng, params=TM.DEFAULTS, margin=np.full(3000, 25.0), overtime=np.zeros(3000, dtype=int))
    assert blow.minutes[:, :5].sum(axis=1).mean() < close.minutes[:, :5].sum(axis=1).mean() - 5
    assert blow.minutes[:, 5:].sum(axis=1).mean() > close.minutes[:, 5:].sum(axis=1).mean() + 5
    pinned = [False] * 12; pinned[0] = True
    d = TM.sample_team_minutes(mu, roles, 500, rng, params=TM.DEFAULTS, pinned=pinned, overtime=np.zeros(500, dtype=int))
    assert np.allclose(d.minutes[:, 0], 36.0) and np.allclose(d.minutes.sum(axis=1), 240.0)
    out = TM.sample_team_minutes(mu, roles, 500, rng, params=TM.DEFAULTS, p_out=[1.0] + [0.0] * 11, overtime=np.zeros(500, dtype=int))
    assert (out.minutes[:, 0] == 0).all() and np.allclose(out.minutes.sum(axis=1), 240.0)   # a scratch's minutes go to the others


def test_scratch_probability_and_roles():
    assert TM.scratch_probability("Q", 1.0, 30, TM.DEFAULTS) == 0.35 and TM.scratch_probability(None, 0.4, 8, TM.DEFAULTS) == 0.0   # play rate is in the allocation
    assert TM.scratch_probability("gtd", params=TM.DEFAULTS) == 0.35 and TM.scratch_probability(None, 0.4, 30, TM.DEFAULTS) == 0.0
    assert [TM.role_of(m, s) for m, s in ((34, True), (22, True), (22, False), (12, False), (4, False))] == ["star", "starter", "rotation", "rotation", "deep"]


def test_lines_consensus_and_team_lookup():
    rows = pd.DataFrame([
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="spreads", side="home", point=-4.5, price=-110),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="b", market="spreads", side="home", point=-5.5, price=-110),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="totals", side="over", point=239.0, price=-110),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="h2h", side="home", point=None, price=-180),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="h2h", side="away", point=None, price=150),
    ])
    c = consensus(rows)
    assert len(c) == 1 and c.iloc[0].spread_home == -5.0 and c.iloc[0].total == 239.0 and 0.6 < c.iloc[0].p_home < 0.65
    gl = GameLines(c)
    assert gl.team_spread("WAS") == -5.0 and gl.team_spread("MIL") == 5.0 and gl.team_total("MIL") == 239.0 and gl.team_spread("BOS") is None
