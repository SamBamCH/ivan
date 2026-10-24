# src/ivan/sports/nba/post_lock/ingame.py
"""
In-game update of a player's fantasy-point distribution: what he has banked plus a
draw of the rest of the game, given the live state of his game.

    final = fpts_so_far + remaining

In each world:
  game script   final margin = MARGIN_SLOPE * live margin + N(0, sd(time left)), one draw per
                game shared by both teams (sign flipped), so teammates and opponents move together
  minutes       remaining = prior_minutes * time_left_share + PACE[role] * (minutes so far - prior pace),
                times the team model's blowout factor exp(beta[role] * (|final margin| - E|margin|) / time_left_share),
                plus N(0, MIN_SD[role] * sqrt(time_left_share / 0.5)), clipped to [0, clock left]
  rate          per-minute rate = w * observed + (1 - w) * prior, w = minutes so far / (minutes so far + K)
  points        the player's own pre-game world draw S (keeps the copula and the tails), shrunk to the
                remaining share r = remaining minutes / prior minutes: r * mu + sqrt(r) * (S - mu), times
                the rate update, floored at 0

Fitted 2026-10-23 on 645 games of play-by-play (stat rebuild within 1 pt for 99% of 14,014
player-games), held out from 2026-03-01: the rate weight K is ~300 minutes, so a hot first
quarter moves the per-minute rate by ~3% (held-out remaining-points rmse 5.13 vs 5.15 prior-only,
8.0 trusting the observed rate); final-points rmse 9.8 pre-game -> 8.3 after Q1 -> 6.7 at the half
-> 4.8 after Q3, almost all from banking the points and the minutes. Final margin ~0.92 x the
live margin; its change has sd 13.6 / 11.3 / 8.4 after Q1 / Q2 / Q3. The blowout betas are the
team minutes model's (data/nba/models/team_minutes.json, fitted on final margin).

Validation 2026-10-23 on held-out games (2,745 player-checkpoints from 2026-03-01), with the
per-checkpoint calibration in CAL_BY_LEFT fitted on the earlier games:
                       after Q1   half    after Q3
  rmse, banked + pace    8.43     6.85     5.16
  rmse, this update      8.19     6.59     4.79
  10-90 coverage         0.81     0.78     0.87
  mean bias             +1.39    +0.95    +0.22     (banked + pace: +1.95 +1.29 +0.71)
The remaining bias is the test prior's (a trailing average, high late in the season), not the
update's; live, the prior is the published projection. Blowouts: starters with |margin| >= 20
after Q3 are +4.1 high with the blowout term, +5.8 without.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

REGULATION_MIN = 48.0
K_BY_LEFT = ((0.75, 330.0), (0.5, 306.6), (0.25, 207.3))       # (share of regulation left, K minutes)
MARGIN_SLOPE = 0.92
MARGIN_SD_BY_LEFT = ((1.0, 15.38), (0.75, 13.56), (0.5, 11.29), (0.25, 8.45), (0.0, 0.0))
PACE = {"starter": 0.248, "bench": 0.204}                      # remaining minutes per minute ahead of pace
MIN_SD = {"starter": 4.54, "bench": 4.72}                      # residual sd of remaining minutes at the half
DEFAULT_BETA = {"star": -0.0064, "starter": -0.0019, "rotation": 0.0047, "deep": 0.0191}
E_ABS_MARGIN = 11.8                                            # E|margin| for a N(0, 14.75) game
# calibration of the remaining part, fitted on games before 2026-03-01: mean scale a, spread scale b (share left, a, b)
CAL_BY_LEFT = ((1.0, 1.0, 1.0), (0.75, 0.962, 0.95), (0.5, 0.962, 0.95), (0.25, 0.869, 0.90))


def _interp(table, x: float) -> float:
    xs = np.array([t[0] for t in table]); ys = np.array([t[1] for t in table])
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], ys[order]))


def role(prior_minutes: float, starter: bool) -> str:
    return "star" if prior_minutes >= 32 else "starter" if (starter or prior_minutes >= 24) else "rotation" if prior_minutes >= 12 else "deep"


def margin_draws(live_margin: float, left: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Final margin draws for one team given its live margin and the share of regulation left."""
    if left <= 0:
        return np.full(n, float(live_margin))
    return MARGIN_SLOPE * float(live_margin) + rng.normal(0.0, _interp(MARGIN_SD_BY_LEFT, left), n)


def update(samples: np.ndarray, mu: float, prior_minutes: float, fpts_so_far: float, minutes_so_far: float, left: float,
           final_margin: np.ndarray, starter: bool, rng: np.random.Generator, beta: Optional[Dict[str, float]] = None,
           final: bool = False) -> np.ndarray:
    """
    One player's final-points draws. samples: his pre-game draws (n worlds); mu their mean; prior_minutes his
    pre-game projected minutes; left the share of regulation still to play (0..1); final_margin his team's
    final-margin draw per world (margin_draws, shared with his teammates).
    """
    samples = np.asarray(samples, dtype=np.float64); n = samples.shape[0]
    if final or left <= 0:
        return np.full(n, float(fpts_so_far))
    beta = beta or DEFAULT_BETA
    elapsed = 1.0 - left
    pm = max(float(prior_minutes), 0.0)
    grp = "starter" if (starter or pm >= 24) else "bench"
    base = pm * left + PACE[grp] * (float(minutes_so_far) - pm * elapsed)
    blow = np.exp(beta[role(pm, starter)] * (np.abs(final_margin) - E_ABS_MARGIN) / max(left, 0.25))
    rem_min = base * blow + rng.normal(0.0, MIN_SD[grp] * np.sqrt(left / 0.5), n)
    rem_min = np.clip(rem_min, 0.0, REGULATION_MIN * left + 2.0)       # a little room for overtime
    if pm <= 0 or mu <= 0:
        return float(fpts_so_far) + np.zeros(n)
    prior_rate = mu / pm
    K = _interp(K_BY_LEFT, left)
    w = float(minutes_so_far) / (float(minutes_so_far) + K) if minutes_so_far > 0 else 0.0
    obs_rate = float(fpts_so_far) / float(minutes_so_far) if minutes_so_far > 0 else prior_rate
    rate_mult = (w * obs_rate + (1.0 - w) * prior_rate) / prior_rate
    r = rem_min / pm
    remaining = rate_mult * (r * mu + np.sqrt(r) * (samples - mu))
    ca = _interp([(t[0], t[1]) for t in CAL_BY_LEFT], left); cb = _interp([(t[0], t[2]) for t in CAL_BY_LEFT], left)
    m = float(remaining.mean())
    remaining = ca * m + cb * (remaining - m)
    return float(fpts_so_far) + np.maximum(remaining, 0.0)
