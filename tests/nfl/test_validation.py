# tests/nfl/test_validation.py
"""The projected-vs-actual layer: PIT arithmetic, the pool check, the run artifacts' round trip and discovery."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ivan.core import artifacts, validation as V
from ivan.sports.nfl import run_artifacts as RA


# --------------------------------------------------------------------------- PIT
def test_pit_interpolates_inside_the_grid_and_extrapolates_outside():
    q = np.array([[0, 5, 10, 25, 50, 75, 90, 95, 99]], dtype=float)      # quantile values equal to their percentiles
    assert V.pit_values(np.array([50.0]), q)[0] == pytest.approx(0.50)
    assert V.pit_values(np.array([37.5]), q)[0] == pytest.approx(0.375)
    assert V.pit_values(np.array([-100.0]), q)[0] == 0.0
    assert V.pit_values(np.array([1000.0]), q)[0] == 1.0
    assert np.isnan(V.pit_values(np.array([np.nan]), q)[0])


def test_pit_histogram_is_flat_for_a_calibrated_model():
    rng = np.random.default_rng(0)
    mean = rng.normal(15, 3, 5000); sd = np.full(5000, 5.0)
    actual = rng.normal(mean, sd)
    pit = V.pit_normal(actual, mean, sd)
    hist = V.pit_histogram(pit)
    assert len(hist) == 10
    assert all(abs(h["share"] - 0.1) < 0.03 for h in hist)
    assert V.pit_shape(pit) == "calibrated"
    too_narrow = V.pit_normal(actual, mean, sd / 2)        # the model claims half the spread it has
    assert V.pit_shape(too_narrow) == "too_narrow"


def test_player_report_bias_coverage_and_baseline():
    rng = np.random.default_rng(1)
    n = 400
    salary = rng.integers(200, 12000, n)
    proj = 2 + salary / 600 + rng.normal(0, 1, n)
    actual = proj + 1.0 + rng.normal(0, 4, n)              # projections run 1 point low, sd 4
    df = pd.DataFrame({"slot": "FLEX", "position": rng.choice(["QB", "RB", "WR"], n), "salary": salary, "proj_mean": proj, "proj_sd": 4.0,
                       "actual_fpts": actual, "own_proj": rng.uniform(0, 50, n), "actual_own": rng.uniform(0, 50, n)})
    rep = V.player_report(df)
    o = rep["overall"]
    assert rep["pit_method"] == "normal"
    assert 0.6 < o["bias"] < 1.4
    assert 0.72 < o["coverage_10_90"] < 0.9
    assert 0.85 < o["sd_ratio"] < 1.15
    assert o["baseline_mae"] is not None
    assert len(rep["by_position"]) == 3 and len(rep["by_salary_tier"]) == 4
    assert rep["ownership"]["n"] == n


# --------------------------------------------------------------------------- pool
def test_pool_report_sees_a_sim_that_ranks_lineups_and_picks_that_beat_random():
    rng = np.random.default_rng(2)
    n = 300
    true_roi = rng.normal(-0.3, 1.0, n)
    sim_roi = true_roi + rng.normal(0, 0.5, n)             # the sim is informative
    picked = np.zeros(n, dtype=bool); picked[np.argsort(-sim_roi)[:20]] = True
    df = pd.DataFrame({"contest_id": 1, "contest_name": "c", "slate": "s", "entry_fee": 1.0, "sim_avg_roi": sim_roi, "sim_cash_pct": 0.2,
                       "actual_payout": 1.0 + true_roi, "picked": picked, "n_picked": picked.astype(int)})
    rep = V.pool_report(df, n_boot=300)
    r = rep["rows"][0]
    assert r["spearman_sim_roi_vs_actual"] > 0.5
    assert r["picked_actual_roi"] > r["random_k_actual_roi"]
    assert r["p_random_beats_picked"] < 0.05
    assert len(rep["deciles"]) == 10 and rep["deciles"][-1]["actual_roi"] > rep["deciles"][0]["actual_roi"]
    checks = V.build_checks(pool=rep)
    assert {c["name"] for c in checks} == {"sim ROI ranks lineups", "picks vs random from pool"}
    assert all(c["status"] == "pass" for c in checks)


def test_pool_report_sees_a_sim_that_is_noise():
    rng = np.random.default_rng(3)
    n = 300
    true_roi = rng.normal(-0.3, 1.0, n)
    sim_roi = rng.normal(0, 1, n)
    picked = np.zeros(n, dtype=bool); picked[np.argsort(-sim_roi)[:20]] = True
    df = pd.DataFrame({"contest_id": 1, "entry_fee": 1.0, "sim_avg_roi": sim_roi, "actual_payout": 1.0 + true_roi, "picked": picked, "n_picked": picked.astype(int)})
    rep = V.pool_report(df, n_boot=300)
    assert abs(rep["rows"][0]["spearman_sim_roi_vs_actual"]) < 0.2
    statuses = {c["name"]: c["status"] for c in V.build_checks(pool=rep)}
    assert statuses["sim ROI ranks lineups"] in ("warn", "fail")


# --------------------------------------------------------------------------- artifacts
def test_sim_artifact_round_trip(tmp_path):
    n_sims, M = 20, 7
    u_lids = [10, 11, 12]
    starts = np.array([0, 2, 5]); ends = np.array([2, 5, 7])              # dupes 2, 3, 2
    pid_lists = {10: [1, 2, 3], 11: [1, 4, 5], 12: [2, 3, 6]}
    rng = np.random.default_rng(0)
    scores = rng.normal(100, 10, (n_sims, M)).astype(np.float32)
    payouts = np.zeros_like(scores); payouts[:, 0] = 5.0
    rois = (payouts - 1) / 1
    ranks = np.tile(np.arange(1, M + 1), (n_sims, 1)).astype(np.int32)
    raw = {p: rng.normal(15, 5, n_sims) for p in range(1, 7)}
    players = {p: {"name": f"p{p}", "team": "A", "pos": "WR", "slot": "FLEX", "salary": 5000, "fpts_mean": 15.0, "ownership_proj": 20.0} for p in range(1, 7)}
    arrays, meta = RA.build_sim_artifact(
        contest={"id": 99, "name": "test", "entry_fee": 1.0, "total_entries": 1000}, u_lids=u_lids, starts=starts, ends=ends,
        pid_lists_by_lid=pid_lists, full_scores=scores, full_payouts=payouts, full_rois=rois, full_ranks=ranks, raw_samples=raw,
        players=players, pay_by_rank=np.array([5.0] + [0.0] * 6), tiers=[{"min": 1, "max": 1, "amount": 5.0}], paid_places=1,
        pool_in_field=5.0, sampler_arrays={"lineup_id": np.array([10, 11, 12, 13]), "predicted_dupes": np.array([2.0, 3.0, 2.5, 1.0])},
        sampler_meta={"sampler": "x"}, seed=7)
    assert arrays["dupes"].tolist() == [2, 3, 2]
    assert arrays["predicted_dupes"].tolist() == [2.0, 3.0, 2.5]
    assert arrays["cand_lineup_id"].tolist() == [10, 11, 12, 13]
    assert arrays["lineup_players"].shape == (3, 3)
    assert arrays["player_q"].shape == (6, len(RA.QGRID))
    assert arrays["player_exposure_pct"][0] == pytest.approx(100 * 5 / 7)     # player 1 in lineups 10 (2 copies) and 11 (3)
    assert arrays["cash_pct"][0] == 1.0 and arrays["cash_pct"][1] == 0.0
    assert arrays["score_pct"].shape == (n_sims, len(RA.QGRID))
    npz, js = artifacts.contest_artifact_paths(tmp_path / "sim", 99)
    RA.write_sim_artifact(npz, js, arrays, meta)
    back, meta2 = RA.load_sim_artifact(npz)
    assert back["lineup_id"].tolist() == u_lids and meta2["contest"]["id"] == 99 and meta2["seed"] == 7
    assert len(meta2["players"]) == 6


def test_artifact_paths_follow_the_run_directory(tmp_path, monkeypatch):
    out = tmp_path / "output"
    monkeypatch.delenv(artifacts.ENV, raising=False)
    when = datetime(2026, 10, 11, 9, 30, 0)
    assert artifacts.projections_path(out, when) == out / "projections" / "projections.20261011-093000.csv"
    assert artifacts.sim_dir(out, when) == out / "sim" / "20261011-093000"
    monkeypatch.setenv(artifacts.ENV, str(tmp_path / "rep_000"))
    assert artifacts.projections_path(out) == tmp_path / "rep_000" / "projections.csv"
    assert artifacts.sim_dir(out) == tmp_path / "rep_000" / "sim"


def test_discovery_dates_live_files_by_stamp_and_replayed_files_by_lock(tmp_path):
    live = tmp_path / "live"
    (live / "projections").mkdir(parents=True)
    (live / "projections" / "projections.20261011-093000.csv").write_text("player_id\n1\n")
    (live / "sim" / "20261011-094500").mkdir(parents=True)
    (live / "sim" / "20261011-094500" / "contest_5.npz").write_bytes(b"x")
    slate = tmp_path / "replay" / "20251103-1915_showdown_1"
    rep = slate / "rep_000"
    (rep / "sim").mkdir(parents=True)
    (slate / "replay.json").write_text(json.dumps({"lock": "2025-11-03T19:15:00"}))
    (slate / "projections.csv").write_text("player_id\n1\n")
    (rep / "sim" / "contest_7.npz").write_bytes(b"x")

    projs = artifacts.find_projection_files([live, slate])
    assert [p.when for p in projs] == [datetime(2025, 11, 3, 19, 14), datetime(2026, 10, 11, 9, 30)]
    sims = artifacts.find_sim_runs([live, rep])
    assert [(s.when, s.contests()) for s in sims] == [(datetime(2025, 11, 3, 19, 14), [7]), (datetime(2026, 10, 11, 9, 45), [5])]
    assert artifacts.latest_before(sims, datetime(2026, 10, 11, 9, 40)).contests() == [7]
    assert artifacts.latest_before(sims, None).contests() == [5]


# --------------------------------------------------------------------------- field raking
def test_raking_moves_field_exposure_onto_projected_ownership():
    from scipy.sparse import csr_matrix
    from ivan.sports.nfl.sample_field import rake_weights_to_ownership
    rng = np.random.default_rng(0)
    n_lineups, n_players = 400, 12
    # candidate lineups lean on players 0-2 (the "chalk"), each lineup has 6 players
    L = np.zeros((n_lineups, n_players))
    for i in range(n_lineups):
        p = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        w = np.array([5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        L[i, rng.choice(p, 6, replace=False, p=w / w.sum())] = 1
    weights = np.ones(n_lineups)
    targets = np.array([40, 35, 30, 70, 65, 60, 55, 55, 50, 50, 45, 45], dtype=float)   # sums to 6 players x 100 = 600
    before = 100 * L.T @ (weights / weights.sum())
    w, info = rake_weights_to_ownership(weights, csr_matrix(L), targets, iterations=40, damping=0.7, max_ratio=4.0)
    after = 100 * L.T @ w
    # the candidate pool limits how far exposures can move (few lineups without the chalk), so test the direction and size of the move
    assert np.abs(before - targets).mean() > 10
    assert np.abs(after - targets).mean() < np.abs(before - targets).mean() / 3
    assert after[:3].mean() < before[:3].mean() - 30            # the chalk comes down
    assert info["exposure_mae_after"] < info["exposure_mae_before"]
    assert abs(info["chalk_bias_after"]) < abs(info["chalk_bias_before"])
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_calibrate_ownership_maps_and_renormalises():
    from types import SimpleNamespace
    from ivan.sports.nfl import ownership as O

    class FakeQuery:
        def __init__(self, rows): self.rows = rows
        def all(self): return self.rows

    class FakeSession:
        def __init__(self, rows): self.rows = rows
        def query(self, model): return FakeQuery(self.rows)
        def commit(self): pass

    rows = [SimpleNamespace(id=1, name="rams", position="DST", roster_slot_id=512, fpts_mean=4.8, ownership_proj=1.0),
            SimpleNamespace(id=2, name="qb", position="QB", roster_slot_id=512, fpts_mean=25.0, ownership_proj=70.0),
            SimpleNamespace(id=3, name="wr", position="WR", roster_slot_id=512, fpts_mean=15.0, ownership_proj=429.0),
            SimpleNamespace(id=4, name="rams", position="DST", roster_slot_id=511, fpts_mean=4.8, ownership_proj=0.1),
            SimpleNamespace(id=5, name="qb", position="QB", roster_slot_id=511, fpts_mean=25.0, ownership_proj=99.9)]
    cal = {"contests": 138, "fitted_on": "2026-10-12", "maps": {"512:DST": {"on": "fpts_proj", "a": -9.14, "b": 4.14},
                                                                 "512:QB": {"on": "ownership_proj", "a": 9.79, "b": 0.70},
                                                                 "511:DST": {"on": "fpts_proj", "a": -4.31, "b": 1.18}}}
    out = O.calibrate_ownership(FakeSession(rows), {r.id: r.ownership_proj for r in rows}, cal)
    flex = {r.id: r.ownership_proj for r in rows if r.roster_slot_id == 512}
    assert abs(sum(flex.values()) - 500.0) < 0.05 and abs(sum(r.ownership_proj for r in rows if r.roster_slot_id == 511) - 100.0) < 0.05
    assert rows[0].ownership_proj > 9.0                      # the DST moved up to the points-based line (10.7 before renormalising)
    assert rows[1].ownership_proj < 70.0                     # the QB came down
    assert out[1] == rows[0].ownership_proj
    # no maps: untouched
    assert O.calibrate_ownership(FakeSession(rows), {}, {"maps": {}}) == {}


def test_learned_ownership_features_and_renormalisation():
    import numpy as np
    import pandas as pd
    from ivan.sports.nfl import ownership_model as OM
    rows = []
    for cid in (1, 2):
        for slot in (511, 512):
            for i, (pos, sal, proj, own) in enumerate((("QB", 11000, 24.0, 60.0), ("WR", 9000, 15.0, 30.0), ("DST", 4000, 5.0, 1.0), ("K", 4500, 8.0, 10.0))):
                rows.append({"contest_id": cid, "date": pd.Timestamp("2025-11-01") + pd.Timedelta(days=cid), "total_entries": 20000, "entry_fee": 5.0,
                             "player_id": i, "slot": slot, "salary": sal, "position": pos, "team": "AAA" if i % 2 else "BBB",
                             "own_sampler": own if slot == 512 else own / 5, "fpts_proj": proj, "ownership_actual": own if slot == 512 else own / 5})
    d = OM.add_features(pd.DataFrame(rows), group=["contest_id"])
    assert set(OM.FEATURES) <= set(d.columns) and d.team_share.between(0, 1).all()
    assert d[(d.contest_id == 1) & (d.slot == 512)].salary_rank.tolist() == [1.0, 2.0, 4.0, 3.0]

    class Const:
        def predict(self, X): return np.full(len(X), 7.0)
    m = OM.OwnershipModel(Const(), list(OM.FEATURES), "2026-10-12", None, 2, len(d))
    pred = m.predict_frame(d)
    flex = pred[(d.slot == 512) & (d.contest_id == 1)]; cpt = pred[(d.slot == 511) & (d.contest_id == 1)]
    assert abs(flex.sum() - 500.0) < 1e-6 and abs(cpt.sum() - 100.0) < 1e-6


def test_confidence_rolls_up_the_stages():
    from ivan.core import validation as V
    reports = {"players": {"overall": {"n": 781, "bias": -0.24, "mae": 4.63, "baseline_mae": 4.34, "coverage_10_90": 0.86, "sd_ratio": 0.98},
                           "ownership": {"n": 748, "mae": 2.16, "spearman": 0.89, "chalk_bias": 5.8}, "ownership_field": {"n": 748, "mae": 2.09, "chalk_bias": 3.9}},
               "field": {"cash_line": {"n": 36, "in_band_share": 1.0}, "winning_score": {"n": 36, "in_band_share": 0.92}, "dupes": {"n": 18, "top_share_ratio_median": 0.59}},
               "lineups": {"settled": 1386, "sim_cash_pct": 0.32, "actual_cash_rate": 0.35, "dupes": {"ratio": 0.58}},
               "pool": {"spearman_median": 0.08}}
    c = V.confidence_report(reports)
    assert set(c["stages"]) == {"projections", "ownership", "field", "dupes", "simulator"} and not c["missing"]
    assert 0 < c["score"] <= 100 and c["grade"] in "ABCDF" and c["weakest"] in c["stages"]
    assert c["stages"]["projections"]["metrics"]["coverage_10_90"]["score"] == 0.7      # 86% vs 80% +- 20
    assert c["stages"]["projections"]["metrics"]["skill_vs_salary"]["score"] < 0.35     # MAE worse than the salary line (ratio 0.94)
    assert c["stages"]["field"]["metrics"]["cash_line_in_band"]["score"] == round(1 - 0.2 / 0.3, 3)
    assert c["weakest"] == min(c["stages"], key=lambda k: c["stages"][k]["score"])
    # thin samples cap it, missing stages are listed
    thin = V.confidence_report({"players": {"overall": {"n": 50, "bias": 0.0, "mae": 4.0, "baseline_mae": 5.0, "coverage_10_90": 0.8, "sd_ratio": 1.0}}})
    assert thin["score"] <= 60 and set(thin["missing"]) == {"ownership", "field", "dupes", "simulator"} and thin["grade"] in "ABCDF"
    assert V.confidence_report({})["score"] is None
