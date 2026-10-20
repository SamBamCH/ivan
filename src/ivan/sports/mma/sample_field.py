# src/ivan/sports/mma/sample_field.py
"""
The opposing field: `field_size` entries drawn from the candidate pool with
replacement, weighted by how often the field duplicates a lineup. Lineups
that spend the cap are duplicated far more (an exponential from 1 copy at
`salary_threshold` left to `dupes_at_zero_remaining` copies at 0 left), and
lineups full of favourites a little more (a linear boost in the summed win
probability, clipped). Ported from mma_dfs Contest.build_field.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from ivan.sports.mma.paths import MMA_DATA_DIR

DEFAULTS = {"size": 5000, "dupes_at_zero_remaining": 200.0, "salary_threshold": 1500,
            "win_boost_slope": 0.2, "win_boost_center": 3.0, "win_boost_min": 0.8, "win_boost_max": 1.5,
            "model": "original", "dupe_model_path": str(MMA_DATA_DIR / "models" / "dupe_model.json")}
# model: "original" = the hand-set exponential + win boost above; "fitted" = the Poisson log-linear model
# scripts/mma/build_dupe_model.py fits on real fields (exp(b_left x min(left, cap)/100 + b_zero x [left = 0] + b_win x sum win)).


def load_dupe_model(path: str | Path) -> Optional[dict]:
    p = Path(path)
    if not p.is_absolute():
        from ivan.core.utils import PROJECT_ROOT
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fitted_weights(lineups: list[list[int]], salary: dict[int, int], win_prob: dict[int, float], model: dict,
                   fights: Optional[dict[int, str]] = None, salary_cap: int = 50000, rng: Optional[np.random.Generator] = None,
                   main_event: Optional[set] = None) -> np.ndarray:
    """
    exp(linear predictor) of the fitted model per candidate; the field size
    and intercept cancel in the sampling. A version-2 (universe) model adds
    the lineup's rank by summed win probability among the candidates and,
    with rng, a Gamma(alpha, 1/alpha) multiplier per lineup: the fitted
    lumpiness of the top (which of several near-identical builds becomes the
    chalk). Without rng the weights are the expectation.
    """
    c = model["coef"]
    cap = float(model.get("left_cap", 2000.0))
    w = np.empty(len(lineups), dtype=float)
    for i, ids in enumerate(lineups):
        left = float(salary_cap - sum(salary[p] for p in ids))
        eta = c["b_left_per_100"] * min(left, cap) / 100.0 + c["b_zero_left"] * (1.0 if left <= 0 else 0.0)
        eta += c["b_win"] * sum(win_prob.get(p, 0.5) for p in ids)
        if fights is not None and len({fights.get(p, p) for p in ids}) < len(ids):
            eta += c.get("b_same_fight", 0.0)
        if main_event:
            eta += c.get("b_main_event", 0.0) * sum(1 for p in ids if p in main_event)
        w[i] = eta
    if "b_log_rank" in c:
        from ivan.sports.mma.universe import rank_within
        w += c["b_log_rank"] * np.log(rank_within(lineups, win_prob, salary, salary_cap))
    w = np.exp(w - w.max())
    alpha = float(model.get("alpha") or 0)
    if rng is not None and 0 < alpha < 1e6:
        w *= rng.gamma(alpha, 1.0 / alpha, size=len(lineups))
    return w


def expected_copies(lineups: list[list[int]], salary: dict[int, int], win_prob: dict[int, float], model: dict, field_size: int,
                    fights: Optional[dict[int, str]] = None, salary_cap: int = 50000, main_event: Optional[set] = None) -> np.ndarray:
    """Each candidate's expected copies in a real field of field_size entries (the model without the Gamma noise)."""
    w = fitted_weights(lineups, salary, win_prob, model, fights, salary_cap, rng=None, main_event=main_event)
    return field_size * w / w.sum()


def dupe_weights(lineups: list[list[int]], salary: dict[int, int], win_prob: dict[int, float], cfg: dict,
                 salary_cap: int = 50000, fights: Optional[dict[int, str]] = None, rng: Optional[np.random.Generator] = None,
                 main_event: Optional[set] = None) -> np.ndarray:
    """Relative frequency of each candidate in the field, by the configured model."""
    f = {**DEFAULTS, **cfg}
    if str(f.get("model", "original")).lower() == "fitted":
        model = load_dupe_model(f["dupe_model_path"])
        if model is None:
            raise FileNotFoundError(f"field.model is 'fitted' but {f['dupe_model_path']} does not exist; run scripts/mma/build_dupe_model.py")
        return fitted_weights(lineups, salary, win_prob, model, fights, salary_cap, rng=rng, main_event=main_event)
    thr = float(f["salary_threshold"])
    k = np.log(float(f["dupes_at_zero_remaining"])) / thr if thr > 0 else 0.0
    w = np.empty(len(lineups), dtype=float)
    for i, ids in enumerate(lineups):
        remaining = salary_cap - sum(salary[p] for p in ids)
        base = 1.0 if remaining >= thr else float(np.exp(k * (thr - remaining)))
        total_win = sum(win_prob.get(p, 0.5) for p in ids)
        boost = 1.0 + (total_win - float(f["win_boost_center"])) * float(f["win_boost_slope"])
        w[i] = base * float(np.clip(boost, f["win_boost_min"], f["win_boost_max"]))
    return w


def sample_field(lineups: list[list[int]], salary: dict[int, int], win_prob: dict[int, float], field_size: int,
                 cfg: dict, rng: np.random.Generator, salary_cap: int = 50000, fights: Optional[dict[int, str]] = None,
                 main_event: Optional[set] = None) -> tuple[np.ndarray, dict]:
    """
    Copies of each candidate in a field of `field_size` entries (array
    aligned with `lineups`), plus diagnostics: the duplication histogram and
    the number of unique lineups in the field.
    """
    if not lineups:
        raise ValueError("no candidate lineups to sample the field from")
    w = dupe_weights(lineups, salary, win_prob, cfg, salary_cap, fights, rng=rng, main_event=main_event)
    probs = w / w.sum()
    draws = rng.choice(len(lineups), size=int(field_size), replace=True, p=probs)
    counts = np.bincount(draws, minlength=len(lineups))
    hist = Counter(int(c) for c in counts if c > 0)
    return counts, {"field_size": int(field_size), "unique": int((counts > 0).sum()), "model": str({**DEFAULTS, **cfg}.get("model", "original")),
                    "dupe_hist": [{"copies": c, "lineups": n, "entries": c * n} for c, n in sorted(hist.items())],
                    "max_dupes": int(counts.max())}


def format_field_diagnostics(meta: dict, exposures: list[dict]) -> str:
    """The duplication table and fighter exposures as the original printed them."""
    lines = [f"field: {meta['field_size']:,} entries, {meta['unique']:,} unique lineups, max copies {meta['max_dupes']}",
             f"{'copies':>7} {'lineups':>8} {'entries':>8} {'% field':>8}"]
    for h in meta["dupe_hist"]:
        lines.append(f"{h['copies']:>7} {h['lineups']:>8} {h['entries']:>8} {100 * h['entries'] / meta['field_size']:>7.1f}%")
    lines.append(f"\n{'fighter':<24} {'salary':>7} {'win%':>6} {'exposure':>9}")
    for e in exposures:
        lines.append(f"{e['name']:<24} {e['salary']:>7} {100 * e['win_prob']:>5.1f}% {100 * e['exposure']:>8.1f}%")
    return "\n".join(lines)
