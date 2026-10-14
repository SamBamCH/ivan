# tests/nfl/test_usage.py
"""Layer 2 of the game model: the usage allocation of unpriced volume."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from ivan.sports.nfl.refactor import usage as U


def P(pid, name, team, pos, salary):
    return SimpleNamespace(id=pid, name=name, team=team, position=pos, salary=salary)


EFF = {"yards_per_dropback": 6.5, "completions_per_dropback": 0.6, "yards_per_completion": 11.0, "yards_per_carry": 4.3}
ENV = pd.DataFrame([{"team": "AAA", "plays_mean": 64.0, "pass_rate_mean": 0.60},
                    {"team": "BBB", "plays_mean": 60.0, "pass_rate_mean": 0.50}])


def model(env=ENV):
    return U.UsageModel(environment=env, priors={k: dict(v) for k, v in U.DEFAULT_PRIORS.items()}, efficiency=EFF)


def test_residual_after_the_qb_line_is_split_by_depth_rank():
    m = model()
    qb = P(1, "QB A", "AAA", "QB", 10000)
    wr1, wr2, wr3, wr4, wr5 = (P(pid, f"WR{i}", "AAA", "WR", s) for i, (pid, s) in enumerate(zip(range(2, 7), (9000, 7000, 5000, 3000, 2500)), start=1))
    te1 = P(7, "TE1", "AAA", "TE", 4000)
    for p in (wr1, wr2, wr3, wr4, wr5, te1):
        m.add_candidate(p, 0.2)
    m.add_fixed(qb, "passing_yards", 250.0)
    m.add_fixed(wr1, "receiving_yards", 90.0); m.add_fixed(wr2, "receiving_yards", 60.0); m.add_fixed(te1, "receiving_yards", 40.0)
    # team passing yards come from the QB line; 60 is left for WR3, WR4, WR5
    assert m.predict(wr1, "receiving_yards") is None                   # priced: the props are the marginals
    got = {p.name: m.predict(p, "receiving_yards") for p in (wr3, wr4, wr5)}
    assert sum(got.values()) == pytest.approx(60.0)
    assert got["WR3"] > got["WR4"] > got["WR5"]                         # salary order is depth order
    pri = U.DEFAULT_PRIORS
    assert got["WR3"] / got["WR4"] == pytest.approx(pri[("WR", 3)]["receiving_yards"] / pri[("WR", 4)]["receiving_yards"])
    tu = m.teams["AAA"]
    assert tu.total_source["receiving_yards"] == "qb line" and tu.totals["receiving_yards"] == 250.0
    assert tu.n_unpriced["receiving_yards"] == 3 and not tu.floored["receiving_yards"]
    # receptions follow the same line through yards per completion
    assert tu.totals["receptions"] == pytest.approx(250.0 / 11.0)


def test_environment_supplies_the_total_without_a_qb_line_and_the_floor_holds():
    m = model()
    rb1, rb2, rb3 = P(1, "RB1", "BBB", "RB", 8000), P(2, "RB2", "BBB", "RB", 5000), P(3, "RB3", "BBB", "RB", 3000)
    qb = P(9, "QB B", "BBB", "QB", 9000)
    for p in (rb1, rb2, rb3):
        m.add_candidate(p, 0.3)
    m.add_fixed(rb1, "rushing_yards", 80.0); m.add_fixed(rb2, "rushing_yards", 45.0)
    m.add_fixed(qb, "rushing_yards", 20.0)                              # the QB's salary-model rushing is settled too
    tu_total = 60.0 * 0.5 * EFF["yards_per_carry"]                     # rushes x yards per carry = 129
    assert m.predict(rb3, "rushing_yards") == pytest.approx(max(tu_total - 145.0, U.RESIDUAL_FLOOR * tu_total))
    tu = m.teams["BBB"]
    assert tu.total_source["rushing_yards"] == "environment" and tu.floored["rushing_yards"]
    # receiving with no QB line: dropbacks x yards per dropback, everything unpriced shares it
    rec = {p.name: m.predict(p, "receiving_yards") for p in (rb1, rb2, rb3)}
    assert sum(rec.values()) == pytest.approx(60.0 * 0.5 * EFF["yards_per_dropback"])
    assert rec["RB1"] > rec["RB2"] > rec["RB3"]


def test_league_fallback_and_scope():
    m = model(env=None)
    wr = P(1, "WR", "CCC", "WR", 5000); qb = P(2, "QB", "CCC", "QB", 6000); k = P(3, "K", "CCC", "K", 3000)
    m.add_candidate(wr, 0.1); m.add_candidate(qb, 0.1); m.add_candidate(k, None)
    assert m.predict(wr, "receiving_yards") == pytest.approx(U.LEAGUE_TEAM["dropbacks"] * EFF["yards_per_dropback"])
    assert m.teams["CCC"].total_source["receiving_yards"] == "league"
    assert m.predict(qb, "rushing_yards") is None and m.predict(wr, "rushing_tds") is None and m.predict(k, "kicker_pts") is None
    with pytest.raises(RuntimeError):
        m.add_fixed(wr, "receptions", 3.0)                              # too late: already allocated
    assert "usage allocation" in m.describe() and len(m.team_rows()) == 3


def test_priors_fall_back_to_defaults_and_ranks_cap():
    m = U.UsageModel(environment=ENV, priors={}, efficiency=EFF)
    wrs = [P(i, f"WR{i}", "AAA", "WR", 9000 - 500 * i) for i in range(1, 10)]
    for p in wrs:
        m.add_candidate(p, 0.1)
    got = [m.predict(p, "receptions") for p in wrs]
    assert got[5] == pytest.approx(got[6]) == pytest.approx(got[8])    # WR6 and beyond share the cap's prior
    assert got[0] > got[1] > got[2]
    assert sum(got) == pytest.approx(64.0 * 0.6 * EFF["completions_per_dropback"])
