"""The Monte Carlo contest simulation's scoring chunk: every simulation pays out exactly the pool, ties split."""
import numpy as np

from ivan.sports.nfl import simulate_contest as SC


def test_simulate_chunk_conserves_the_pool_and_splits_ties():
    rng = np.random.default_rng(0)
    n_sims, n_players = 20, 6
    P = rng.normal(10, 3, (n_sims, n_players)).astype(np.float32)
    lineup_ix = [[0, 1], [0, 1], [2, 3], [4, 5], [1, 2]]          # entries 0 and 1 are copies of one lineup
    payouts = {1: 100.0, 2: 50.0, 3: 20.0}
    part = SC._simulate_chunk(0, n_sims, lineup_ix, P, 10.0, payouts)
    assert part["scores"].shape == (n_sims, 5) and part["stats"]["sims_off"] == 0 and part["stats"]["tie_anomalies"] == 0
    assert np.allclose(part["payouts"].sum(axis=1), 170.0)
    assert np.allclose(part["payouts"][:, 0], part["payouts"][:, 1])           # the copies tie and share
    assert np.allclose(part["rois"], (part["payouts"] - 10.0) / 10.0)
    assert SC.sim_method({}) == "montecarlo" and callable(SC.simulate_one_contest_mc)


def test_expand_sampled_field_uses_the_slates_player_lists():
    pid_lists = {7: [1, 2, 3], 9: [2, 3, 4]}
    u_lids, by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends = SC._expand_sampled_field(
        None, {9: 2, 7: 1, 5: 0}, pid_lists=pid_lists)                  # no session needed: every lineup is in the lists
    assert u_lids == [7, 9] and by_lid == {7: [1, 2, 3], 9: [2, 3, 4]} and unique_pids == [1, 2, 3, 4]
    assert starts.tolist() == [0, 1] and ends.tolist() == [1, 3]
    assert lineup_ix == [[0, 1, 2], [1, 2, 3], [1, 2, 3]]


def test_player_samples_matrix_reuses_the_slate_draw(monkeypatch):
    def no_draw(*a, **k):
        raise AssertionError("a shared draw must not be redrawn")
    monkeypatch.setattr(SC, "generate_correlated_samples_gaussian", no_draw)
    draw = SC.SlateDraw(raw={1: np.arange(5, dtype=np.float32), 2: np.ones(5, dtype=np.float32)}, diag={"games": []}, n_samples=5, seed=3)
    P, raw, diag, missing = SC._player_samples_matrix(None, {1: 0, 2: 1, 3: 2}, 4, draw=draw)
    assert P.shape == (4, 3) and P[:, 0].tolist() == [0, 1, 2, 3] and P[:, 1].tolist() == [1, 1, 1, 1] and missing == [3]
    assert raw is draw.raw and diag is draw.diag
