"""
Fantasy points from minutes and rates by Monte Carlo.

Minutes are drawn from a normal around the projection and clipped; each stat
is a negative binomial with mean minutes * rate and a per-stat dispersion
(variance = mean + mean^2 / k). Double-double and triple-double bonuses come
out of the joint draw, which is the only honest way to get them.
"""
from __future__ import annotations

import json

import numpy as np

from ivan.sports.nba.paths import MODELS_DIR
from .history import STATS
from .rates import DK_WEIGHTS

CALIBRATION_JSON = MODELS_DIR / "projection_calibration.json"

# Dispersion k per stat (method of moments on 2025-26 rotation players, see
# scripts/nba/backtest_projection.py --calibrate). Larger = closer to Poisson.
DEFAULT_DISPERSION = {"points": 12.0, "rebounds": 10.0, "assists": 8.0,
                      "steals": 50.0, "blocks": 30.0, "turnovers": 20.0, "fg3m": 10.0}
MIN_SD_FLOOR = 2.5
MIN_SD_SLOPE = 0.12   # minutes sd grows with projected minutes


def load_calibration() -> dict:
    """Fitted dispersion and minutes-sd line; defaults when the file is missing."""
    cal = {"dispersion": dict(DEFAULT_DISPERSION), "min_sd_floor": MIN_SD_FLOOR, "min_sd_slope": MIN_SD_SLOPE}
    if CALIBRATION_JSON.exists():
        try:
            cal.update(json.loads(CALIBRATION_JSON.read_text()))
        except Exception:
            pass
    return cal


_CAL = load_calibration()


def minutes_sd(mu: float, hist_sd: float | None = None) -> float:
    model = _CAL["min_sd_floor"] + _CAL["min_sd_slope"] * mu
    if hist_sd is None or not np.isfinite(hist_sd):
        return model
    return float(0.5 * model + 0.5 * max(hist_sd, 1.0))


def _nbinom(rng: np.random.Generator, mean: np.ndarray, k: float) -> np.ndarray:
    mean = np.maximum(mean, 1e-9)
    p = k / (k + mean)
    return rng.negative_binomial(k, p)


def simulate_fpts(
    minutes_mu: float,
    rates: dict[str, float],
    n: int = 4000,
    min_sd: float | None = None,
    dispersion: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    k = dispersion or _CAL["dispersion"]
    sd = minutes_sd(minutes_mu) if min_sd is None else min_sd
    mins = np.clip(rng.normal(minutes_mu, sd, n), 0.0, 48.0)
    draws = {s: _nbinom(rng, mins * rates[s], k[s]) for s in STATS}
    fp = sum(DK_WEIGHTS[s] * draws[s] for s in STATS).astype(float)
    cats = np.stack([draws["points"], draws["rebounds"], draws["assists"],
                     draws["steals"], draws["blocks"]]) >= 10
    n10 = cats.sum(axis=0)
    fp += 1.5 * (n10 >= 2) + 3.0 * (n10 >= 3)
    return fp


def summarize(samples: np.ndarray) -> tuple[float, float]:
    return float(samples.mean()), float(samples.std())
