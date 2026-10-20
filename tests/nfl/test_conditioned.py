"""The conditioned contest evaluation (ivan.sports.nfl.conditioned): closed-form cells that agree with brute force."""
import math

import numpy as np
from scipy.special import ndtr

from ivan.sports.nfl import conditioned as C


def toy_ladder(F=4000, paid=800):
    pay = np.zeros(F)
    pay[0] = 2000.0; pay[1:5] = 500.0; pay[5:30] = 100.0; pay[30:200] = 20.0; pay[200:paid] = 6.0
    return C.Ladder(pay, F)


def toy_slate(N=300, W=12, F=4000, seed=0):
    """Lineups with world-varying means, a spread of conditional sds and lumpy expected copies summing to F."""
    rng = np.random.default_rng(seed)
    base = rng.normal(100, 6, N)
    m = (base[:, None] + rng.normal(0, 8, (1, W)) + rng.normal(0, 3, (N, W))).astype(np.float32)
    s = rng.uniform(12, 20, N).astype(np.float32)
    own = rng.lognormal(0.5, 1.2, N)
    own[:3] = (120.0, 80.0, 40.0)                      # a few heavily duplicated lineups
    own = own / own.sum() * F
    return m, s, own


def test_ladder_share_and_tiers():
    L = toy_ladder()
    assert L.pool == 2000 + 4 * 500 + 25 * 100 + 170 * 20 + 600 * 6
    assert L.paid_places == 800 and L.top_prize == 2000
    assert L.share(0, 1) == 2000 and L.share(0, 2) == 1250 and math.isclose(L.share(0, 1.5), (2000 + 0.5 * 500) / 1.5)
    assert L.share(5, 3) == 100 and L.share(5000, 1) == 0
    # the Normal closed form at vanishing variance is the ladder itself (a hair off at the rank-0 kink)
    lam = np.array([0.0, 3.0, 50.0, 1000.0])
    got = L.expected_share_normal(lam, np.full(4, 1e-6), 1.0)
    assert np.allclose(got, [2000, 500, 20, 0], rtol=2e-3, atol=1e-3)


def test_blur_is_the_residual_integral():
    grid = np.arange(0, 300, 0.5)
    T = np.exp(-grid / 20.0)[None, :]
    z, w = C._hermite(60)
    for kern in ("phi", "phi_sf"):
        B = C._blur(T, 15.0, 0.5, kern)[0]
        for m in (100.0, 150.0):
            weight = w if kern == "phi" else w * ndtr(-z)
            exact = (weight * np.exp(-(m + 15.0 * z) / 20.0)).sum()
            assert math.isclose(np.interp(m, grid, B), exact, rel_tol=2e-3)


def test_mixture_tails_match_direct_sums():
    m, s, own = toy_slate()
    grid = C.score_grid(m, s)
    lam = C.mixture_tails(m, s, grid, own, "sf")
    for w in (0, 5):
        for x in (110.0, 140.0, 160.0):
            exact = (own * ndtr(-(x - m[:, w].astype(float)) / s)).sum()
            assert math.isclose(np.interp(x, grid, lam[w]), exact, rel_tol=0.03, abs_tol=1e-3)


def test_exact_window_matches_brute_force_count():
    m, s, own = toy_slate(W=4)
    L = toy_ladder()
    T = C.contest_tables(m, s, own, L)
    rng = np.random.default_rng(1)
    w = 1
    j = T.i_top[w] + 20
    x = T.grid[j]
    p = ndtr(-(x - m[:, w].astype(float)) / s)
    Ks = np.array([(rng.random(len(own)) < p) @ rng.poisson(own) for _ in range(4000)])
    assert math.isclose(T.lam[w, j], Ks.mean(), rel_tol=0.08)
    assert abs(T.win[w, j] - (Ks == 0).mean()) < 0.03
    # our prize as a unique entry with k entries ahead
    unique = int(np.argmin(T.own_vals))
    brute = np.mean([L.share(k, 1 + n) for k, n in zip(Ks, rng.poisson(T.own_vals[unique], Ks.size))])
    assert math.isclose(T.H[unique, w, j], brute, rel_tol=0.1)


def test_field_collects_the_pool_and_fills_the_paid_places():
    m, s, own = toy_slate()
    L = toy_ladder()
    ev = C.evaluate_contest(m, s, own, L, rng=np.random.default_rng(2))
    cons = C.pool_conservation(ev, own, L)
    assert abs(cons["rel_error_mean"]) < 0.04, cons
    cashing = (own * ev.cash.mean(axis=1)).sum()
    # ties let a lump straddling the cash line cash as a whole, so a little more than `paid` entries cash
    assert 0.98 * L.paid_places < cashing < 1.15 * L.paid_places, cashing
    assert 0.5 < (own * ev.win).sum() < 200          # one winning lineup, with its copies tied for first
    assert np.all(ev.pay >= 0) and np.all((ev.cash >= 0) & (ev.cash <= 1))
    # realized draws follow the same law
    assert math.isclose((own[:, None] * ev.draw).sum(axis=0).mean(), L.pool, rel_tol=0.1)


def test_cell_matches_monte_carlo_of_its_own_law():
    m, s, own = toy_slate(N=150, W=3)
    L = toy_ladder()
    ev = C.evaluate_contest(m, s, own, L, rng=np.random.default_rng(3))
    rng = np.random.default_rng(4)
    for c in (int(np.argmax(own)), int(np.argmin(np.abs(own - 5.0))), int(np.argmin(own))):
        w = 2
        mc, sc, oc = float(m[c, w]), float(s[c]), float(own[c])
        others = np.ones(len(own), bool); others[c] = False
        mo, so, oo = m[others, w].astype(float), s[others], own[others]
        n = 6000
        xs = mc + sc * rng.standard_normal(n)
        prizes = np.empty(n)
        for i in range(n):
            pj = ndtr(-(xs[i] - mo) / so)
            K = (rng.random(pj.size) < pj) @ rng.poisson(oo)
            prizes[i] = L.share(K, 1 + rng.poisson(oc))
        se = prizes.std() / math.sqrt(n)
        assert abs(ev.pay[c, w] - prizes.mean()) < 4 * se + 0.05 * prizes.mean() + 0.2, (c, oc, ev.pay[c, w], prizes.mean(), se)
        assert abs(ev.cash[c, w] - (prizes > 0).mean()) < 0.04


def test_unique_lineup_beats_its_duplicated_twin_at_the_top():
    """Two lineups with the same score distribution: the one the field duplicates 100 times shares its prizes."""
    m, s, own = toy_slate(N=200, W=6)
    m[0] = m[1] = m[:2].mean(axis=0); s[0] = s[1] = 15.0
    own[0], own[1] = 0.2, 100.0
    own = own / own.sum() * 4000
    L = toy_ladder()
    ev = C.evaluate_contest(m, s, own, L, rng=np.random.default_rng(5))
    assert ev.pay[0].mean() > ev.pay[1].mean()
    assert math.isclose(ev.cash[0].mean(), ev.cash[1].mean(), abs_tol=0.03)


def test_lineup_moments_from_a_world_model():
    W = 5
    mean = np.array([[10.0] * W, [5.0] * W, [20.0] * W], dtype=np.float32)
    cov_a = np.array([[4.0, 1.0], [1.0, 9.0]])
    wm = C.WorldModel(n_worlds=W, seed=0, randomness=1.0, player_ids=np.array([1, 2, 3]), mean=mean,
                      blocks=[(np.array([0, 1]), cov_a), (np.array([2]), np.array([[16.0]]))], teams=[], team_points=np.zeros((W, 0)))
    pool = C.build_lineup_pool(wm, {10: [1, 2], 11: [1, 3], 12: [1, 2, 3, 99]})
    m, s = C.lineup_moments(wm, pool.L)
    assert np.allclose(m[:, 0], [15.0, 30.0, 35.0])
    assert np.allclose(s, [math.sqrt(4 + 9 + 2), math.sqrt(4 + 16), math.sqrt(4 + 9 + 2 + 16)])
    assert pool.missing_players == [99]


def test_antithetic_worlds_have_zero_mean():
    rng = np.random.default_rng(0)
    Z = C._antithetic_normal(rng, np.array([[1.0, 0.3], [0.3, 1.0]]), 100)
    assert Z.shape == (100, 2) and abs(Z.mean()) < 1e-12
    assert C._antithetic_normal(rng, np.eye(2), 101).shape == (101, 2)
