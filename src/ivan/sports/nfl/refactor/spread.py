"""
Predictive spread calibration for the fitted fantasy-point distributions.

A player's fitted distribution describes the game-to-game spread of his stats
given the lines. What the simulation needs is the spread of *outcomes around
the projection*, which also carries the projection's own error (the lines
move, roles change, the salary model misses). The backtest
(scripts/nfl/backtest_projections.py) measures that gap per position as the
sd of z = (actual - projected) / model sd, and writes it here; ingest widens
every sample about its mean by that factor and stores both the model sd and
the calibrated sd, so the next backtest measures the model again, not the
widening.

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
FACTOR_RANGE = (0.7, 2.0)


def load_sd_calibration(model_dir: Path | None = None) -> dict:
    path = Path(models_dir() if model_dir is None else model_dir) / CALIBRATION_FILE
    if not path.exists():
        return {"factors": {}, "fitted_on": None, "window": None, "n": {}}
    cal = json.loads(path.read_text())
    cal.setdefault("factors", {})
    cal.setdefault("n", {})
    return cal


def write_sd_calibration(factors: dict[str, float], n: dict[str, int], window: tuple[str, str],
                         model_dir: Path = MODELS_DIR) -> dict:
    """Merge new per-position factors into the file (positions not given keep their old value)."""
    cal = load_sd_calibration(model_dir)
    for pos, f in factors.items():
        cal["factors"][pos] = round(float(np.clip(f, *FACTOR_RANGE)), 3)
        cal["n"][pos] = int(n.get(pos, 0))
    cal["fitted_on"] = datetime.now().strftime("%Y-%m-%d")
    cal["window"] = list(window)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / CALIBRATION_FILE).write_text(json.dumps(cal, indent=1))
    return cal


def spread_factor(cal: dict, position: str) -> float:
    return float(cal.get("factors", {}).get(position, 1.0))


def calibrate_spread(samples: np.ndarray, position: str, cal: dict) -> np.ndarray:
    """Widen (or narrow) samples about their mean by the position's factor."""
    f = spread_factor(cal, position)
    if f == 1.0:
        return samples
    mu = float(np.mean(samples))
    return mu + (samples - mu) * f


def describe_sd_calibration(cal: dict) -> str:
    if not cal.get("factors"):
        return "predictive sd calibration: none on file (model spread used as is; run backtest_projections.py --write-calibration)"
    parts = " ".join(f"{p} x{f:.2f}" for p, f in sorted(cal["factors"].items()))
    return f"predictive sd calibration: {parts} (backtest {cal.get('window', ['?', '?'])[0]}..{cal.get('window', ['?', '?'])[1]}, written {cal.get('fitted_on')})"
