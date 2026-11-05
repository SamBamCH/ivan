# tests/nfl/test_dupe_model.py
"""The product dupe model: fit on counts, copies scale with the field, the salary / projection-gap / entry-cap terms, weights follow the same function."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ivan.sports.nfl import dupe_model as DM


def _field(cid, n, k, beta, rng, is_showdown=1.0, rows=400, gamma=0.0, dlog=0.0, demax=0.0, emax=20, slo_range=(-14, -5)):
    slo = rng.uniform(*slo_range, rows)                    # sum of log(own/100) over six players
    left = rng.choice([0, 100, 300, 500, 1000, 1500, 2500, 4000], rows)
    gap = rng.uniform(0, 25, rows)
    lam = n * k * np.exp(beta * slo + gamma * np.minimum(left, DM.SALARY_KNEE) / 1000 + dlog * np.log1p(gap) + demax * np.log(emax))
    cnt = np.maximum(1, rng.poisson(lam))
    return pd.DataFrame({"contest_id": cid, "count": cnt, "entries": n, "is_showdown": is_showdown, "n_players": 6, "sum_log_own": slo,
                         "salary_left": left, "proj_gap": gap, "max_entries": float(emax), "fpts_proj": 100.0 - gap,
                         "date": pd.Timestamp("2025-11-09 19:00") + pd.Timedelta(days=cid)})


def test_fit_recovers_the_rule_and_copies_scale_with_entries():
    rng = np.random.default_rng(0)
    df = pd.concat([_field(i, n, 0.05, 0.6, rng) for i, n in enumerate((20000, 50000, 80000, 120000, 30000))], ignore_index=True)
    m = DM.fit(df)
    p = m.params["showdown"]
    assert abs(p["beta"] - 0.6) < 0.08 and 0.03 < p["k"] < 0.08 and m.contests["showdown"] == 5
    assert abs(p["gamma"]) < 0.1 and abs(p["delta_logproj"]) < 0.05           # no salary or projection effect in the data: the terms stay near zero
    assert p["delta_emax"] == 0.0                                              # one cap everywhere: nothing to fit, the term is left out
    assert m.params["classic"] == DM.DEFAULTS["classic"]                      # no classic fields: the default stays
    c1 = m.copies(10000, True, np.array([-8.0])); c2 = m.copies(20000, True, np.array([-8.0]))
    assert c2[0] == 2 * c1[0]
    w = m.weights(True, np.array([-8.0, -9.0]))
    assert w[0] > w[1] and np.isclose(w[0] / w[1], np.exp(p["beta"]))
    assert DM.sum_log_own([50.0, 25.0]) == np.log(0.5) + np.log(0.25) and DM.sum_log_own([0.0]) == np.log(DM.MIN_OWN / 100)
    bt = DM.backtest(df)
    assert bt["showdown"]["contests"] == 5 and len(bt["showdown"]["folds"]) == 5
    assert all(abs(f["log_bias"]) < 0.4 for f in bt["showdown"]["folds"]) and set(bt["showdown"]["folds"][0]["ablation"]) == {a[0] for a in DM.ABLATION}


def test_save_and_load_round_trip_and_older_files(tmp_path):
    m = DM.ProductDupeModel(params={"showdown": {"k": 0.05, "beta": 0.6}, "classic": {"k": 1e-4, "beta": 0.1}}, fitted_on="2026-10-13", contests={"showdown": 27})
    m.save(tmp_path)
    back = DM.load(tmp_path)
    assert back.params["showdown"] == {"k": 0.05, "beta": 0.6, "gamma": 0.0, "knee": DM.SALARY_KNEE, "delta_logproj": 0.0, "delta_emax": 0.0, "delta_logn": 0.0}
    assert back.contests == {"showdown": 27} and "27 fields" in back.describe()
    assert DM.load(tmp_path / "nowhere").params == DM.DEFAULTS
    # a file from before 2026-11-05 carries the captain exponent, the linear gap and the stack deltas: ignored, the rest read
    (tmp_path / DM.FILE).write_text('{"params": {"showdown": {"k": 0.035, "beta": 0.22, "beta_cpt": 0.12, "gamma": -1.0, "knee": 2000.0, "delta_proj": 0.0, '
                                    '"delta_logproj": -1.14, "stack": {}, "method": "wls"}, "classic": {"k": 3.6e-05, "beta": 0.07}}, "fitted_on": "2026-10-15", "contests": {"showdown": 140}}')
    old = DM.load(tmp_path)
    assert old.params["showdown"] == {"k": 0.035, "beta": 0.22, "gamma": -1.0, "knee": 2000.0, "delta_logproj": -1.14, "delta_emax": 0.0, "delta_logn": 0.0}
    assert old.fitted_on == "2026-10-15"


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
    two = df[df.contest_id < 2]                                               # two slates only: the extra terms are not trusted, ownership only
    assert DM.fit(two).params["showdown"]["gamma"] == 0.0 and DM.fit(two).params["showdown"]["delta_logproj"] == 0.0
    m0 = DM.fit(df, terms=())
    assert m0.params["showdown"]["gamma"] == 0.0 and "exp(" not in m0.describe() and "exp(" in m.describe()


def test_projection_gap_and_entry_cap_terms_are_recovered_and_applied():
    rng = np.random.default_rng(2)
    parts = []
    # the counts stay above the floor of one (a larger k, chalkier lineups), or the floor flattens every slope
    for i, (n, emax) in enumerate(((60000, 1), (180000, 20), (450000, 150), (90000, 1), (240000, 20), (360000, 150), (120000, 5), (270000, 20))):
        parts.append(_field(i, n, 2.0, 0.6, rng, gamma=-1.0, dlog=-1.1, demax=-0.3, emax=emax, rows=600, slo_range=(-9, -5)))
    df = pd.concat(parts, ignore_index=True)
    m = DM.fit(df)
    p = m.params["showdown"]
    assert abs(p["delta_logproj"] + 1.1) < 0.12 and abs(p["delta_emax"] + 0.3) < 0.08 and abs(p["gamma"] + 1.0) < 0.15 and abs(p["beta"] - 0.6) < 0.08
    assert "pts below best" in m.describe() and "entries per user" in m.describe()
    base = m.copies(10000, True, np.array([-8.0]))
    gap = m.copies(10000, True, np.array([-8.0, -8.0]), proj_gap=np.array([0.0, 10.0]))
    assert np.isclose(gap[0], base[0]) and np.isclose(gap[0] / gap[1], np.exp(-np.log1p(10) * p["delta_logproj"]))
    caps = m.copies(10000, True, np.array([-8.0, -8.0, -8.0]), max_entries=np.array([1.0, 150.0, np.nan]))
    assert np.isclose(caps[0] / caps[1], np.exp(-p["delta_emax"] * np.log(150.0)))
    assert np.isclose(caps[2], base[0])                                       # unknown cap = the default cap = no cap given
    assert np.isclose(caps[2] / caps[0], np.exp(p["delta_emax"] * np.log(DM.EMAX_DEFAULT)))
    w = m.weights(True, np.array([-8.0, -8.0]), max_entries=np.array([1.0, 150.0]))
    assert np.isclose(w[0] / w[1], caps[0] / caps[1])                        # the sampler's weights follow the same function
    # the fit with the live terms is what predict() scores
    pr = DM.predict(p, df.head(5))
    assert np.allclose(pr, m.copies(df.head(5).entries.values[0], True, df.head(5).sum_log_own.values, salary_left=df.head(5).salary_left.values,
                                    proj_gap=df.head(5).proj_gap.values, max_entries=df.head(5).max_entries.values))


def test_candidate_contest_size_term_is_off_unless_asked():
    rng = np.random.default_rng(3)
    df = pd.concat([_field(i, n, 0.05, 0.6, rng) for i, n in enumerate((5000, 20000, 50000, 120000, 300000))], ignore_index=True)
    assert DM.fit(df).params["showdown"]["delta_logn"] == 0.0
    m = DM.fit(df, terms=DM.ALL_TERMS)
    assert abs(m.params["showdown"]["delta_logn"]) < 0.1                       # copies scale with N in the data: the exponent stays near zero
    assert "logn" in DM.ALL_TERMS and "logn" not in DM.TERMS
