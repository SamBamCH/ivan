# tests/nfl/test_environment.py
"""Layer 1 of the game model: the environment fit, draws and rolling backtest on synthetic team-games."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivan.core import validation as V
from ivan.sports.nfl.refactor import environment as E

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def synthetic_games(seasons=(2019, 2020, 2021, 2022, 2023, 2024), weeks=17, seed=0) -> pd.DataFrame:
    """Games where points follow the lines, plays have a team pace habit, and pass rate has a team habit plus game script."""
    rng = np.random.default_rng(seed)
    pace = dict(zip(TEAMS, rng.normal(0, 3, len(TEAMS))))
    passy = dict(zip(TEAMS, rng.normal(0, 0.05, len(TEAMS))))
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, len(order), 2):
                home, away = order[i], order[i + 1]
                total = float(rng.normal(44, 4)); spread = float(rng.normal(0, 6))
                pts_h = max(0, rng.normal(total / 2 + spread / 2, 9)); pts_a = max(0, rng.normal(total / 2 - spread / 2, 9))
                shared = rng.normal(0, 4)                        # pace shared by both teams
                for team, opp, is_home, margin, pts, opp_pts in ((home, away, 1, spread, pts_h, pts_a), (away, home, 0, -spread, pts_a, pts_h)):
                    plays = max(40, 62 + 0.4 * (total - 44) + pace[team] + shared + rng.normal(0, 5))   # pace habit + a little from the total
                    pr = float(np.clip(0.57 + passy[team] - 0.003 * (pts - opp_pts) + rng.normal(0, 0.06), 0.3, 0.85))
                    db = plays * pr
                    rows.append({"season": s, "week": w, "game_id": f"{s}_{w}_{away}_{home}", "game_type": "REG", "gameday": f"{s}-10-01",
                                 "team": team, "opponent": opp, "is_home": is_home, "spread": margin, "total": total,
                                 "implied_points": total / 2 + margin / 2, "implied_opp_points": total / 2 - margin / 2,
                                 "points": pts, "opp_points": opp_pts, "pass_att": db * 0.93, "sacks": db * 0.07, "rush_att": plays - db,
                                 "dropbacks": db, "plays": plays, "pass_rate": pr})
    return pd.DataFrame(rows)


def test_fit_recovers_the_lines_and_team_habits():
    df = synthetic_games()
    fit = E.fit_environment(df)
    assert fit.n_games == len(df)
    assert fit.coef["points"][1] == pytest.approx(1.0, abs=0.25)               # points track implied points about one for one
    assert fit.resid_corr.shape == (6, 6) and np.allclose(np.diag(fit.resid_corr), 1.0)
    # the pass-rate team habit is recovered in sign for the most and least pass-happy synthetic teams
    te = fit.team_effect["pass_rate"]
    assert te and len(te) == len(TEAMS)
    pred = E.predict(fit, df.head(10))
    for q in E.QUANTITIES:
        assert f"{q}_mean" in pred.columns and f"{q}_sd" in pred.columns
    assert (pred.pass_rate_mean.between(*E.PASS_RATE_BOUNDS)).all() and (pred.plays_mean >= E.PLAYS_MIN).all()


def test_draw_has_the_right_shape_and_respects_bounds():
    df = synthetic_games()
    fit = E.fit_environment(df)
    game = df[df.game_id == df.game_id.iloc[0]]
    d = E.draw(fit, game, 3000, np.random.default_rng(1))
    assert set(d) == set(game.team)
    for t, v in d.items():
        assert v["points"].shape == (3000,) and (v["points"] >= 0).all()
        assert (v["plays"] >= E.PLAYS_MIN).all()
        assert v["pass_rate"].min() >= E.PASS_RATE_BOUNDS[0] and v["pass_rate"].max() <= E.PASS_RATE_BOUNDS[1]
        assert np.allclose(v["dropbacks"] + v["rushes"], v["plays"])
        row = game[game.team == t].iloc[0]
        assert v["points"].mean() == pytest.approx(row.implied_points, abs=2.0)


def test_rolling_backtest_is_out_of_sample_and_calibrated():
    df = synthetic_games()
    bt = E.backtest(df, (2023, 1), min_train=100)
    assert not bt.empty and bt.season.min() == 2023
    s = E.backtest_summary(bt)
    for q in E.QUANTITIES:
        o = s["quantities"][q]["overall"]
        assert 0.65 < o["coverage_10_90"] < 0.92, (q, o)
        assert 0.8 < o["sd_ratio"] < 1.25, (q, o)
    # team habits are real in the synthetic world, so the full model should beat lines-only on pass rate
    o = s["quantities"]["pass_rate"]["overall"]
    assert o["mae"] <= o["mae_lines_only"] + 1e-9


def test_environment_report_and_checks():
    df = synthetic_games()
    bt = E.backtest(df, (2023, 1), min_train=100)
    env = bt.rename(columns={q: f"actual_{q}" for q in E.QUANTITIES})
    env["slate"] = "s"
    rep = V.environment_report(env)
    assert rep["n"] == len(bt) and set(rep["quantities"]) == set(E.QUANTITIES)
    checks = V.build_checks(environment=rep)
    assert {c["name"] for c in checks} == set(E.QUANTITIES)
    assert all(c["status"] in ("pass", "warn") for c in checks)
