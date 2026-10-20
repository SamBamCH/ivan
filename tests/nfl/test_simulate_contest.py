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
