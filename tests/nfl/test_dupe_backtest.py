# tests/nfl/test_dupe_backtest.py
"""The season dupe backtest: forward-only fits, the entry-cap term, the lineup key that finds our entries in a field."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ivan.sports.nfl import dupe_backtest as DB
from ivan.sports.nfl import dupe_model as DM


def _field(cid, day, n, k, beta, rng, emax=20, delta_emax=0.0, rows=300):
    slo = rng.uniform(-14, -5, rows)
    lam = n * k * np.exp(beta * slo) * np.exp(delta_emax * np.log(emax))
    cnt = np.maximum(1, rng.poisson(lam))
    lineups = [f"CPT p{cid}_{i}|FLEX a|FLEX b{i}|FLEX c|FLEX d|FLEX e" for i in range(rows)]
    return pd.DataFrame({"contest_id": cid, "count": cnt, "entries": n, "is_showdown": 1.0, "n_players": 6, "sum_log_own": slo,
                         "log_own_cpt": slo / 6, "salary_left": rng.choice([0, 300, 1200], rows), "n_qb": 1, "qb_mates": 1, "fpts_proj": 100.0,
                         "proj_gap": rng.uniform(0, 20, rows), "lineup": lineups, "max_entries": float(emax), "entry_fee": 5.0,
                         "date": pd.Timestamp("2026-10-01 20:00") + pd.Timedelta(days=day)})


def _frame(rng, delta_emax=0.0, days=8):
    parts = []
    cid = 0
    for day in range(days):
        for emax, n in ((1, 6000), (20, 60000), (150, 150000)):
            parts.append(_field(cid, day, n, 0.05, 0.6, rng, emax=emax, delta_emax=delta_emax)); cid += 1
    df = pd.concat(parts, ignore_index=True)
    df["slate"] = df.date.dt.strftime("%Y-%m-%d %H")
    df["key"] = df.lineup.map(lambda s: DB.lineup_key(str(s).split("|")))
    df["ours"] = 0
    df["emax_bucket"] = df.max_entries.map(lambda v: DB._bucket(v, DB.EMAX_BUCKETS))
    df["in_window"] = True
    return df


def test_forward_scores_only_slates_with_history_and_never_looks_ahead():
    rng = np.random.default_rng(0)
    df = _frame(rng)
    variants = [DB.Variant("asof", DM.TERMS), DB.Variant("asof own only", ())]
    slates, preds = DB.forward(df, variants)
    # the first slate has no history, the second has 3 fields (< MIN_CONTESTS): both unscored
    scored = sorted(slates.slate.unique())
    assert scored == sorted(df.slate.unique())[2:]
    assert (slates.groupby("variant").size() == len(scored)).all()
    # the training set is strictly before the slate
    first = slates[slates.variant == "asof"].sort_values("lock").iloc[0]
    assert first.train_contests == 6
    last = slates[slates.variant == "asof"].sort_values("lock").iloc[-1]
    assert last.train_contests == 3 * (len(df.slate.unique()) - 1)
    # predictions exist exactly on the scored rows and the rule is recovered
    assert np.isfinite(preds["asof"]).sum() == df.slate.isin(scored).sum()
    assert all(abs(b - 0.6) < 0.1 for b in slates[slates.variant == "asof own only"].beta)
    assert slates.log_bias.abs().max() < 0.5
    s = DB.summarize(slates, n_boot=200)
    assert set(s.variant) == {"asof", "asof own only"} and (s.log_bias_lo <= s.log_bias).all() and (s.log_bias <= s.log_bias_hi).all()


def test_fixed_model_is_scored_only_after_its_fitted_on_date():
    rng = np.random.default_rng(1)
    df = _frame(rng)
    m = DM.ProductDupeModel(params={"showdown": {"k": 0.05, "beta": 0.6, "beta_cpt": None, "gamma": 0.0, "delta_logproj": 0.0, "delta_emax": 0.0, "stack": {}}, "classic": dict(DM.DEFAULTS["classic"])},
                            fitted_on="2026-10-04", contests={"showdown": 12})
    slates, _ = DB.forward(df, [DB.Variant("live file", None, model=m)])
    assert slates.lock.min() > pd.Timestamp("2026-10-04 23:59")
    assert slates.slate.nunique() == 4 and slates.log_bias.abs().max() < 0.3          # Oct 5..8 of the 8-day frame


def test_entry_cap_term_is_recovered_and_shows_up_by_bucket():
    rng = np.random.default_rng(2)
    df = _frame(rng, delta_emax=-0.4, days=10)
    slates, preds = DB.forward(df, [DB.Variant("asof", DM.TERMS), DB.Variant("asof +emax", tuple(DM.TERMS) + ("emax",))])
    with_term = slates[slates.variant == "asof +emax"]
    assert abs(with_term.delta_emax.iloc[-1] + 0.4) < 0.1
    # without the term the single-entry fields are under-predicted and the 150-max fields over-predicted; with it both are flat
    bem_without = DB.by_entry_max(df, preds["asof"]).set_index("entry_max")
    bem_with = DB.by_entry_max(df, preds["asof +emax"]).set_index("entry_max")
    assert bem_without.loc["single entry", "log_bias_w"] > 0.3 and bem_without.loc["21-150", "log_bias_w"] < -0.3
    assert bem_with.log_bias_w.abs().max() < 0.25          # the small single-entry fields keep some count>=1 truncation bias
    cal = DB.calibration(df, preds["asof +emax"])
    assert cal.copies.sum() == df[np.isfinite(preds["asof +emax"])]["count"].sum()


def test_lineup_key_matches_entry_strings_to_field_rows():
    k1 = DB.key_of_entry_string("CPT Bhayshul Tuten FLEX Trevor Lawrence FLEX Will Shipley FLEX Jakobi Meyers FLEX Brenton Strange FLEX Jaguars ")
    k2 = DB.lineup_key("CPT Bhayshul Tuten|FLEX Jaguars|FLEX Brenton Strange|FLEX Trevor Lawrence|FLEX Will Shipley|FLEX Jakobi Meyers".split("|"))
    assert k1 == k2 and k1[0] == "bhayshul tuten" and len(k1[1]) == 5
    assert DB.key_of_entry_string("FLEX a FLEX b") is None
    # a field frame with our entries marked: the ours table weighs an entry made twice twice
    rng = np.random.default_rng(3)
    df = _frame(rng, days=4)
    df.loc[df.index[:3], "ours"] = [2, 1, 1]
    pred = np.full(len(df), np.nan); pred[:3] = [4.0, 1.0, 10.0]
    df.loc[df.index[:3], "count"] = [2, 1, 10]
    t = DB.ours_table(df, pred)
    assert len(t) == 1 and t.entries.iloc[0] == 4 and abs(t.pred_copies_mean.iloc[0] - (4 * 2 + 1 + 10) / 4) < 1e-9
    assert abs(t.log_bias.iloc[0] - (2 * np.log(2 / 4) + 0 + 0) / 4) < 1e-9
