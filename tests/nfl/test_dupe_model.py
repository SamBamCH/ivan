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
    assert back.params["showdown"] == {"k": 0.05, "beta": 0.6, "gamma": 0.0, "knee": DM.SALARY_KNEE}   # an older file without the salary term loads as gamma 0
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
    m0 = DM.fit(df, salary_term=False)
    assert m0.params["showdown"]["gamma"] == 0.0 and "exp(" not in m0.describe() and "exp(" in m.describe()
    bt = DM.backtest(df)
    f = bt["showdown"]["folds"][0]
    assert f["log_mae"] < f["log_mae_no_salary"]                             # the term helps out of sample when the data has it
    w = m.weights(True, np.array([-8.0, -8.0]), salary_left=np.array([0.0, 2000.0]))
    assert w[0] > w[1]
