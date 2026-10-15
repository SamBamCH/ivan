# tests/nfl/test_dupe_model.py
"""The product dupe model: fit on counts, copies scale with the field, weights follow the same function."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ivan.sports.nfl import dupe_model as DM


def _field(cid, n, k, beta, rng, is_showdown=1.0, rows=400, gamma=0.0):
    slo = rng.uniform(-14, -5, rows)                       # sum of log(own/100) over six players
    left = rng.choice([0, 100, 300, 500, 1000, 1500, 2500, 4000], rows)
    lam = n * k * np.exp(beta * slo) * np.exp(gamma * np.minimum(left, DM.SALARY_KNEE) / 1000)
    cnt = np.maximum(1, rng.poisson(lam))
    return pd.DataFrame({"contest_id": cid, "count": cnt, "entries": n, "is_showdown": is_showdown, "n_players": 6, "sum_log_own": slo,
                         "salary_left": left, "date": pd.Timestamp("2025-11-09 19:00") + pd.Timedelta(days=cid)})


def test_fit_recovers_the_rule_and_copies_scale_with_entries():
    rng = np.random.default_rng(0)
    df = pd.concat([_field(i, n, 0.05, 0.6, rng) for i, n in enumerate((20000, 50000, 80000, 120000, 30000))], ignore_index=True)
    m = DM.fit(df)
    p = m.params["showdown"]
    assert abs(p["beta"] - 0.6) < 0.08 and 0.03 < p["k"] < 0.08 and m.contests["showdown"] == 5
    assert abs(p["gamma"]) < 0.1                                   # no salary effect in the data: the term stays near zero
    assert m.params["classic"] == DM.DEFAULTS["classic"]          # no classic fields: the default stays
    c1 = m.copies(10000, True, np.array([-8.0])); c2 = m.copies(20000, True, np.array([-8.0]))
    assert c2[0] == 2 * c1[0]
    w = m.weights(True, np.array([-8.0, -9.0]))
    assert w[0] > w[1] and np.isclose(w[0] / w[1], np.exp(p["beta"]))
    assert DM.sum_log_own([50.0, 25.0]) == np.log(0.5) + np.log(0.25) and DM.sum_log_own([0.0]) == np.log(DM.MIN_OWN / 100)
    bt = DM.backtest(df)
    assert bt["showdown"]["contests"] == 5 and len(bt["showdown"]["folds"]) == 5
    assert all(abs(f["log_bias"]) < 0.4 for f in bt["showdown"]["folds"])


def test_save_and_load_round_trip(tmp_path):
    m = DM.ProductDupeModel(params={"showdown": {"k": 0.05, "beta": 0.6}, "classic": {"k": 1e-4, "beta": 0.1}}, fitted_on="2026-10-13", contests={"showdown": 27})
    m.save(tmp_path)
    back = DM.load(tmp_path)
    assert back.params["showdown"] == {"k": 0.05, "beta": 0.6, "beta_cpt": None, "gamma": 0.0, "knee": DM.SALARY_KNEE, "stack": {}}   # an older file loads with the extra terms off
    assert back.contests == {"showdown": 27} and "27 fields" in back.describe()
    assert DM.load(tmp_path / "nowhere").params == DM.DEFAULTS


def test_salary_term_is_recovered_and_applied():
    rng = np.random.default_rng(1)
    df = pd.concat([_field(i, n, 0.05, 0.6, rng, gamma=-1.2) for i, n in enumerate((20000, 50000, 80000, 120000, 30000))], ignore_index=True)
    m = DM.fit(df)
    p = m.params["showdown"]
    assert abs(p["gamma"] + 1.2) < 0.15 and abs(p["beta"] - 0.6) < 0.08
    full = m.copies(10000, True, np.array([-8.0, -8.0]), salary_left=np.array([0.0, 2000.0]))
    assert np.isclose(full[0] / full[1], np.exp(-p["gamma"] * 2.0))          # $2k left = the whole effect; past the knee it stops
    assert np.isclose(m.copies(10000, True, np.array([-8.0]), salary_left=np.array([5000.0]))[0], full[1])
    assert np.isclose(m.copies(10000, True, np.array([-8.0]))[0], full[0])   # no salary given: the rule alone
    m0 = DM.fit(df, terms=())
    assert m0.params["showdown"]["gamma"] == 0.0 and "exp(" not in m0.describe() and "exp(" in m.describe()
    bt = DM.backtest(df)
    f = bt["showdown"]["folds"][0]
    assert f["ablation"]["salary"]["log_mae"] < f["ablation"]["none"]["log_mae"]   # the term helps out of sample when the data has it
    w = m.weights(True, np.array([-8.0, -8.0]), salary_left=np.array([0.0, 2000.0]))
    assert w[0] > w[1]


def _field_full(cid, n, rng, rows=600, beta=0.6, beta_cpt=0.9, gamma=-1.0, mates=0.3, two_qb=0.4, no_qb=-0.8):
    flex = rng.uniform(-12, -4, rows)                       # five flex players
    cpt = rng.uniform(-4.5, -1.0, rows)                     # the captain
    left = rng.choice([0, 100, 300, 500, 1000, 1500, 2500, 4000], rows)
    n_qb = rng.choice([0, 1, 2], rows, p=[0.15, 0.6, 0.25])
    qb_mates = np.where(n_qb > 0, rng.choice([0, 1, 2, 3], rows), 0)
    lr = beta * flex + beta_cpt * cpt + gamma * np.minimum(left, DM.SALARY_KNEE) / 1000 + mates * np.minimum(qb_mates, 3) + two_qb * (n_qb >= 2) + no_qb * (n_qb == 0)
    cnt = rng.poisson(n * 0.05 * np.exp(lr))
    keep = cnt > 0                                         # a lineup nobody entered is not in the field (flooring at 1 would bias the slopes)
    return pd.DataFrame({"contest_id": cid, "count": cnt[keep], "entries": n, "is_showdown": 1.0, "n_players": 6, "sum_log_own": (flex + cpt)[keep],
                         "log_own_cpt": cpt[keep], "salary_left": left[keep], "n_qb": n_qb[keep], "qb_mates": qb_mates[keep],
                         "date": pd.Timestamp("2025-11-09 19:00") + pd.Timedelta(days=cid)})


def test_captain_exponent_and_stack_terms_are_recovered():
    rng = np.random.default_rng(2)
    df = pd.concat([_field_full(i, n, rng) for i, n in enumerate((20000, 50000, 80000, 120000, 30000))], ignore_index=True)
    m = DM.fit(df, terms=DM.ALL_TERMS)
    p = m.params["showdown"]
    assert abs(p["beta"] - 0.6) < 0.08 and abs(p["beta_cpt"] - 0.9) < 0.12 and abs(p["gamma"] + 1.0) < 0.15
    assert abs(p["stack"]["qb_mates"] - 0.3) < 0.1 and abs(p["stack"]["two_qb"] - 0.4) < 0.15 and abs(p["stack"]["no_qb"] + 0.8) < 0.2
    assert "cpt ^" in m.describe() and "qb_mates" in m.describe()
    # the terms only apply when their inputs are given; the captain's log ownership is counted once
    base = m.copies(10000, True, np.array([-8.0]), log_own_cpt=np.array([-2.0]))
    plain = m.copies(10000, True, np.array([-8.0]))
    assert np.isclose(base[0] / plain[0], np.exp((p["beta_cpt"] - p["beta"]) * -2.0))
    with_stack = m.copies(10000, True, np.array([-8.0, -8.0]), log_own_cpt=np.array([-2.0, -2.0]), qb_mates=np.array([2, 0]), n_qb=np.array([1, 0]))
    assert np.isclose(with_stack[0] / base[0], np.exp(2 * p["stack"]["qb_mates"])) and np.isclose(with_stack[1] / base[0], np.exp(p["stack"]["no_qb"]))
    bt = DM.backtest(df)
    f = bt["showdown"]["folds"][0]
    assert f["ablation"]["all"]["log_mae"] < f["ablation"]["salary"]["log_mae"] < f["ablation"]["none"]["log_mae"]
    m.save(__import__("tempfile").mkdtemp())


def test_lineup_stack_features():
    L = np.array([[1, 1, 1, 0, 0, 0],     # QB A + WR A + TE A
                  [1, 0, 0, 1, 1, 0],     # QB A + QB B + WR B
                  [0, 1, 0, 0, 1, 1]])    # no QB
    teams = ["A", "A", "A", "B", "B", "B"]
    pos = ["QB", "WR", "TE", "QB", "WR", "RB"]
    n_qb, mates = DM.lineup_stack_features(L, teams, pos)
    assert n_qb.tolist() == [1, 2, 0] and mates.tolist() == [2, 1, 0]
    assert DM.stack_features([("A", "QB"), ("A", "WR"), ("B", "WR")]) == (1, 1)
