"""
Predictive spread calibration for the fitted fantasy-point distributions.

A player's fitted distribution describes the game-to-game spread of his stats
given the lines. What the simulation needs is the spread of *outcomes around
the projection*, which also carries the projection's own error (the lines
move, roles change, the salary model misses). The backtest
(scripts/nfl/backtest_projections.py) measures that gap per position and
writes it here; ingest widens every sample about its mean and stores both the
model sd and the calibrated sd, so the next backtest measures the model
again, not the widening.

Two forms, the second a strict generalisation of the first:
  factor   realized sd = f x model sd (one number per position, the sd of z)
  power    realized sd = a x model sd^b (two numbers per position, fitted
           log-log on the model-sd quartiles). The 2026 backtest found the
           model's spread growing faster than reality's across the projection
           range (b about 0.6-0.8): a flat factor then leaves the high-sd
           players lineups are built from 15-25% too wide, which a convex
           payout ladder turns into a 5-7x overclaim at the top rungs.
Positions with no entry (or no backtest history) keep their model spread.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ivan.sports.nfl.paths import MODELS_DIR, models_dir

CALIBRATION_FILE = "sd_calibration.json"
MIN_ROWS = 100            # scored projections before a position's factor is trusted
MIN_ROWS_POWER = 300      # scored projections before a position's power law is trusted over its flat factor
FACTOR_RANGE = (0.7, 2.0)
POWER_B_RANGE = (0.4, 1.2)


def load_sd_calibration(model_dir: Path | None = None) -> dict:
    path = Path(models_dir() if model_dir is None else model_dir) / CALIBRATION_FILE
    if not path.exists():
        return {"factors": {}, "power": {}, "fitted_on": None, "window": None, "n": {}}
    cal = json.loads(path.read_text())
    cal.setdefault("factors", {})
    cal.setdefault("power", {})
    cal.setdefault("n", {})
    return cal


def write_sd_calibration(factors: dict[str, float], n: dict[str, int], window: tuple[str, str],
                         model_dir: Path = MODELS_DIR, powers: dict[str, dict] | None = None) -> dict:
    """
    Merge new per-position factors (and power laws {pos: {"a", "b"}}) into the
    file; positions not given keep their old value. A position given a factor
    but no power law drops any old power law, so the file always describes
    what the next ingest applies.
    """
    cal = load_sd_calibration(model_dir)
    powers = powers or {}
    for pos, f in factors.items():
        cal["factors"][pos] = round(float(np.clip(f, *FACTOR_RANGE)), 3)
        cal["n"][pos] = int(n.get(pos, 0))
        if pos in powers:
            cal["power"][pos] = {"a": round(float(powers[pos]["a"]), 4), "b": round(float(np.clip(powers[pos]["b"], *POWER_B_RANGE)), 3)}
        else:
            cal["power"].pop(pos, None)
    cal["fitted_on"] = datetime.now().strftime("%Y-%m-%d")
    cal["window"] = list(window)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / CALIBRATION_FILE).write_text(json.dumps(cal, indent=1))
    return cal


def spread_factor(cal: dict, position: str, model_sd: float | None = None) -> float:
    """
    The multiplier on the model spread: a x sd^(b-1) under the position's
    power law when one is on file and the model sd is known, else the flat
    factor. Clipped to FACTOR_RANGE either way.
    """
    pw = (cal.get("power") or {}).get(position)
    if pw and model_sd is not None and model_sd > 0:
        return float(np.clip(float(pw["a"]) * float(model_sd) ** (float(pw["b"]) - 1.0), *FACTOR_RANGE))
    return float(cal.get("factors", {}).get(position, 1.0))


def calibrate_spread(samples: np.ndarray, position: str, cal: dict) -> np.ndarray:
    """Widen (or narrow) samples about their mean by the position's calibration at this player's model sd."""
    mu = float(np.mean(samples))
    f = spread_factor(cal, position, float(np.std(samples)))
    if f == 1.0:
        return samples
    return mu + (samples - mu) * f


def describe_sd_calibration(cal: dict) -> str:
    if not cal.get("factors"):
        return "predictive sd calibration: none on file (model spread used as is; run backtest_projections.py --write-calibration)"
    pw = cal.get("power") or {}
    parts = " ".join((f"{p} {pw[p]['a']:.2f}*sd^{pw[p]['b']:.2f}" if p in pw else f"{p} x{f:.2f}") for p, f in sorted(cal["factors"].items()))
    return f"predictive sd calibration: {parts} (backtest {cal.get('window', ['?', '?'])[0]}..{cal.get('window', ['?', '?'])[1]}, written {cal.get('fitted_on')})"
