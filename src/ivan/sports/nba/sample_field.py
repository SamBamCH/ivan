# src/ivan/sports/nba/sample_field.py
"""
The field sampler: which of the generated lineups the contest's other
entrants hold, and how many copies of each.

projection_model   (the default) the pickled field distribution model
                   (data/nba/models/field_distribution_model.pkl, fitted on
                   last season's contest bins by historical.data_calculation):
                   the contest's fee, size and max entries predict a
                   20-bin distribution over normalised lineup projection,
                   bins are drawn for every entry and a lineup is drawn
                   uniformly inside the bin.
projection_weighted  every lineup with probability proportional to its
                   projection sum ^ field.projection_exponent.

Both return counts_by_lineup (lineup id -> copies) and a meta dict for the
sim report. expand_sampled_field turns the counts into the arrays the
simulator scores. Draws are seeded from IVAN_SEED (ivan.core.seed).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from ivan.core.utils import coerce_player_ids
from ivan.sports.nba.models import Lineup, NBAPlayer
from ivan.sports.nba.paths import FIELD_MODEL_PKL
from ivan.sports.nba.utils import _cfg

SAMPLERS = ("projection_model", "projection_weighted")


def lineup_projections(session: Session) -> Tuple[List[int], np.ndarray, Dict[int, List[int]]]:
    """Every generated lineup's id, its projection sum, and its player ids (one query for the players)."""
    means = {int(pid): float(m) for pid, m in session.query(NBAPlayer.id, NBAPlayer.fpts_mean).all() if m is not None}
    ids, sums, pids_by_lid = [], [], {}
    for lid, raw in session.query(Lineup.id, Lineup.player_ids_list).all():
        pids = coerce_player_ids(raw)
        pids_by_lid[int(lid)] = pids
        ids.append(int(lid))
        sums.append(sum(means.get(p, 0.0) for p in pids))
    return ids, np.asarray(sums, dtype=np.float64), pids_by_lid


def sample_field_projection_model(session: Session, field_size: int, contest_params: dict,
                                  model_path: str | Path = FIELD_MODEL_PKL, rng: Optional[np.random.Generator] = None) -> Tuple[Dict[int, int], dict]:
    rng = rng or np.random.default_rng()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler, proj_model, bin_edges = model["scaler"], model["proj_model"], model["bin_edges"]
    projection_min = float(model.get("projection_min", 200.0))
    X = np.array([[float(contest_params.get("entry_fee", 4.0)), int(contest_params.get("total_entries", 20000)),
                   int(contest_params.get("multi_entry", 150))]])
    proj_dist = np.asarray(proj_model.predict(scaler.transform(X))[0], dtype=np.float64)

    ids, sums, _ = lineup_projections(session)
    if not ids:
        raise ValueError("no lineups in the database; run the generate step first")
    keep = sums >= projection_min
    if not keep.any():
        raise ValueError(f"no lineups with projection >= {projection_min:.0f}")
    ids = [i for i, k in zip(ids, keep) if k]
    sums = sums[keep]
    max_proj = sums.max()
    norm = (sums - projection_min) / (max_proj - projection_min) if max_proj > projection_min else np.zeros_like(sums)
    norm = np.clip(norm, 0, 1 - 1e-8)
    n_bins = len(proj_dist)
    bin_idx = np.clip(np.floor(norm * (len(bin_edges) - 1)).astype(int), 0, len(bin_edges) - 2)
    by_bin: Dict[int, np.ndarray] = {b: np.flatnonzero(bin_idx == b) for b in np.unique(bin_idx)}
    probs = proj_dist / proj_dist.sum() if proj_dist.sum() > 0 else np.full(n_bins, 1.0 / n_bins)
    drawn_bins = rng.choice(n_bins, size=int(field_size), replace=True, p=probs)
    counts: Dict[int, int] = {}
    fallback = 0
    for b, n in zip(*np.unique(drawn_bins, return_counts=True)):
        members = by_bin.get(int(b))
        if members is None or len(members) == 0:
            picks = rng.choice(len(ids), size=int(n), replace=True)
            fallback += int(n)
        else:
            picks = members[rng.integers(0, len(members), size=int(n))]
        for i in picks:
            counts[ids[int(i)]] = counts.get(ids[int(i)], 0) + 1
    meta = {"sampler": "projection_model", "model": str(model_path), "projection_min": projection_min,
            "candidates": int(len(ids)), "dropped_below_min": int((~keep).sum()), "bin_probs": probs.tolist(),
            "entries_outside_populated_bins": fallback, "unique": len(counts), "sampled": int(sum(counts.values()))}
    return counts, meta


def sample_field_projection_weighted(session: Session, field_size: int, exponent: float = 1.0,
                                     rng: Optional[np.random.Generator] = None) -> Tuple[Dict[int, int], dict]:
    rng = rng or np.random.default_rng()
    ids, sums, _ = lineup_projections(session)
    if not ids:
        raise ValueError("no lineups in the database; run the generate step first")
    w = sums - sums.min() + 1.0 if sums.min() <= 0 else sums.copy()
    p = w / w.sum()
    if exponent != 1.0:
        p = np.power(p, exponent); p /= p.sum()
    picks = rng.choice(len(ids), size=int(field_size), replace=True, p=p)
    u, c = np.unique(picks, return_counts=True)
    counts = {ids[int(i)]: int(n) for i, n in zip(u, c)}
    meta = {"sampler": "projection_weighted", "exponent": float(exponent), "candidates": int(len(ids)),
            "unique": len(counts), "sampled": int(field_size)}
    return counts, meta


def sample_field(session: Session, field_size: int, config: dict, contest_params: dict,
                 rng: Optional[np.random.Generator] = None) -> Tuple[Dict[int, int], dict]:
    """The configured sampler (config field.sampler, default projection_model)."""
    which = str(_cfg(config, "field.sampler", "projection_model"))
    if which not in SAMPLERS:
        raise ValueError(f"unknown field.sampler {which!r}; expected one of {SAMPLERS}")
    if which == "projection_weighted":
        return sample_field_projection_weighted(session, field_size, float(_cfg(config, "field.projection_exponent", 1.0)), rng)
    return sample_field_projection_model(session, field_size, contest_params, _cfg(config, "field.model_path", FIELD_MODEL_PKL), rng)


def expand_sampled_field(session: Session, counts_by_lineup: Dict[int, int]):
    """
    (u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends):
    the field's unique lineups in id order, each one's players, one player-index
    list per field entry (copies adjacent, entry a..b of lineup u), and the
    index of every player on the field.
    """
    u_lids = sorted(int(l) for l, c in counts_by_lineup.items() if c > 0)
    cnts = np.array([int(counts_by_lineup[l]) for l in u_lids], dtype=np.int64)
    starts = np.cumsum(np.r_[0, cnts[:-1]]) if len(cnts) else np.zeros(0, dtype=np.int64)
    ends = starts + cnts
    pid_lists_by_lid: Dict[int, List[int]] = {}
    for i in range(0, len(u_lids), 900):
        for lid, raw in session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.id.in_(u_lids[i:i + 900])).all():
            pid_lists_by_lid[int(lid)] = coerce_player_ids(raw)
    unique_pids = sorted({p for l in u_lids for p in pid_lists_by_lid.get(l, [])})
    pid_to_idx = {p: i for i, p in enumerate(unique_pids)}
    lineup_ix: List[List[int]] = [None] * int(cnts.sum())
    for u, lid in enumerate(u_lids):
        idxs = [pid_to_idx[p] for p in pid_lists_by_lid.get(lid, []) if p in pid_to_idx]
        for pos in range(int(starts[u]), int(ends[u])):
            lineup_ix[pos] = idxs
    return u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends
