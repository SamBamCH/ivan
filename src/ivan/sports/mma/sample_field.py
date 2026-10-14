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

from collections import Counter

import numpy as np

DEFAULTS = {"size": 5000, "dupes_at_zero_remaining": 200.0, "salary_threshold": 1500,
            "win_boost_slope": 0.2, "win_boost_center": 3.0, "win_boost_min": 0.8, "win_boost_max": 1.5}


def dupe_weights(lineups: list[list[int]], salary: dict[int, int], win_prob: dict[int, float], cfg: dict,
                 salary_cap: int = 50000) -> np.ndarray:
    """Relative frequency of each candidate in the field."""
    f = {**DEFAULTS, **cfg}
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
                 cfg: dict, rng: np.random.Generator, salary_cap: int = 50000) -> tuple[np.ndarray, dict]:
    """
    Copies of each candidate in a field of `field_size` entries (array
    aligned with `lineups`), plus diagnostics: the duplication histogram and
    the number of unique lineups in the field.
    """
    if not lineups:
        raise ValueError("no candidate lineups to sample the field from")
    w = dupe_weights(lineups, salary, win_prob, cfg, salary_cap)
    probs = w / w.sum()
    draws = rng.choice(len(lineups), size=int(field_size), replace=True, p=probs)
    counts = np.bincount(draws, minlength=len(lineups))
    hist = Counter(int(c) for c in counts if c > 0)
    return counts, {"field_size": int(field_size), "unique": int((counts > 0).sum()),
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
