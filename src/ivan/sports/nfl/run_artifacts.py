# src/ivan/sports/nfl/run_artifacts.py
"""
What the NFL pipeline writes down about its own predictions, for the
validation layer (see ivan.core.artifacts for where the files go).

projections.csv     one row per player-slot after the projection step: the
                    fitted distribution's moments and quantiles plus the
                    projected ownership. Written by scripts/nfl/ingest.py
                    (live) and the replay adapter.

sim/contest_<id>    what the simulator assumed about one contest:
  .json             the contest, the sampled field's shape, payouts, the
                    players (ids, slots, salaries) and the sampler's notes
  .npz              arrays: every candidate lineup in the field with its
                    sampled copies, predicted dupes and simulated cash % /
                    ROI / rank; the players' sampled score quantiles and
                    their exposure in the field; the field's score
                    distribution per simulation (cash line, winning score,
                    percentiles); the payout by rank at the simulated size

Both are read back by ivan.sports.nfl.validate.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ivan.sports.nfl.models import NFLPlayer

ARTIFACT_VERSION = 1
QGRID = (1, 5, 10, 25, 50, 75, 90, 95, 99)                 # percentiles stored for every distribution

PROJECTION_COLUMNS = ["player_id", "roster_slot_id", "slot_label", "name", "team", "opponent", "position", "salary", "status",
                      "gametime", "fpts_mean", "fpts_std", "fpts_std_model", "fpts_p01", "fpts_p05", "fpts_p10", "fpts_p25",
                      "fpts_p50", "fpts_p75", "fpts_p90", "fpts_p95", "fpts_p99", "ownership_proj", "fpts_dist_name",
                      "volume_source", "unpriced_stats"]


# --------------------------------------------------------------------------- projections
def projections_frame(session) -> pd.DataFrame:
    rows = []
    for p in session.query(NFLPlayer).all():
        rows.append({"player_id": int(p.id), "roster_slot_id": int(p.roster_slot_id), "slot_label": p.slot_label, "name": p.name,
                     "team": p.team, "opponent": p.opponent, "position": p.position, "salary": int(p.salary), "status": p.status,
                     "gametime": p.gametime.isoformat(timespec="minutes") if p.gametime else None,
                     "fpts_mean": p.fpts_mean, "fpts_std": p.fpts_std, "fpts_std_model": p.fpts_std_model,
                     "fpts_p01": p.fpts_p01, "fpts_p05": p.fpts_p05, "fpts_p10": p.fpts_p10, "fpts_p25": p.fpts_p25,
                     "fpts_p50": p.fpts_p50, "fpts_p75": p.fpts_p75, "fpts_p90": p.fpts_p90, "fpts_p95": p.fpts_p95,
                     "fpts_p99": p.fpts_p99, "ownership_proj": p.ownership_proj, "fpts_dist_name": p.fpts_dist_name,
                     "volume_source": getattr(p, "volume_source", None), "unpriced_stats": getattr(p, "unpriced_stats", None)})
    return pd.DataFrame(rows, columns=PROJECTION_COLUMNS)


def write_projections(session, path: Path) -> int:
    """Write the slate's projections to `path`; returns the row count (0 writes nothing)."""
    df = projections_frame(session)
    if df.empty:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def load_projections(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def environment_path(projections_path: Path) -> Path:
    """The environment file that goes with a projections file: projections.<stamp>.csv -> environment.<stamp>.csv (projections.csv -> environment.csv)."""
    p = Path(projections_path)
    return p.with_name(p.name.replace("projections", "environment", 1))


def write_environment(session, projections_path: Path) -> int:
    """Write the slate's projected game environment next to the projections file (ivan.sports.nfl.refactor.environment); returns rows."""
    from ivan.sports.nfl.refactor.environment import slate_environment
    df = slate_environment(session)
    if df.empty:
        return 0
    path = environment_path(projections_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def load_environment(projections_path: Path) -> pd.DataFrame:
    path = environment_path(projections_path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


USAGE_COLUMNS = ["team", "stat", "total", "total_source", "fixed", "residual", "n_unpriced", "floored", "coverage", "unpriced_volume"]


def usage_path(projections_path: Path) -> Path:
    """projections.<stamp>.csv -> usage.<stamp>.csv: the team totals the usage model (layer 2) saw, priced lines against them."""
    p = Path(projections_path)
    return p.with_name(p.name.replace("projections", "usage", 1))


def write_usage(rows, projections_path: Path, unpriced_volume: str = "") -> int:
    """Write fit_fpts_distributions' usage team rows next to the projections file; returns rows written."""
    df = pd.DataFrame(list(rows or []))
    if df.empty:
        return 0
    df["coverage"] = df.fixed / df.total.replace(0, np.nan)
    df["unpriced_volume"] = unpriced_volume
    path = usage_path(projections_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[USAGE_COLUMNS].to_csv(path, index=False)
    return len(df)


def load_usage(projections_path: Path) -> pd.DataFrame:
    path = usage_path(projections_path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- sim artifacts
def _pad(lists: Sequence[Sequence[int]], fill: int = -1) -> np.ndarray:
    k = max((len(l) for l in lists), default=0)
    out = np.full((len(lists), k), fill, dtype=np.int64)
    for i, l in enumerate(lists):
        out[i, :len(l)] = list(l)
    return out


def build_sim_artifact(*, contest: dict, u_lids: Sequence[int], starts: np.ndarray, ends: np.ndarray,
                       pid_lists_by_lid: Dict[int, List[int]], full_scores: np.ndarray, full_payouts: np.ndarray,
                       full_rois: np.ndarray, full_ranks: np.ndarray, raw_samples: Dict[int, np.ndarray],
                       players: Dict[int, dict], pay_by_rank: np.ndarray, tiers: List[dict], paid_places: int,
                       pool_in_field: float, sampler_arrays: Optional[dict] = None, sampler_meta: Optional[dict] = None,
                       seed: Optional[int] = None) -> tuple[dict, dict]:
    """
    Assemble (arrays, meta) for one simulated contest from the simulator's
    working state. `players` is sim_report.player_meta's dict; `raw_samples`
    the per-player sampled scores (captain rows already x1.5).
    """
    n_sims, M = full_scores.shape
    U = len(u_lids)
    dupes = np.asarray([int(ends[u] - starts[u]) for u in range(U)], dtype=np.int64)
    cash_pct = np.zeros(U); avg_roi = np.zeros(U); mean_rank = np.zeros(U); mean_score = np.zeros(U); win_pct = np.zeros(U)
    for u in range(U):
        a, b = int(starts[u]), int(ends[u])
        cash_pct[u] = (full_payouts[:, a] > 0).mean()
        avg_roi[u] = full_rois[:, a:b].mean()
        mean_rank[u] = full_ranks[:, a:b].mean()
        mean_score[u] = full_scores[:, a:b].mean()
        win_pct[u] = (full_ranks[:, a:b] == 1).any(axis=1).mean()
    lineup_players = _pad([pid_lists_by_lid.get(int(l), []) for l in u_lids])

    # predicted dupes for the field's lineups (and every candidate the sampler saw)
    pred_by_lid: Dict[int, float] = {}
    cand_lids = np.zeros(0, dtype=np.int64); cand_pred = np.zeros(0)
    if sampler_arrays and "lineup_id" in sampler_arrays and "predicted_dupes" in sampler_arrays:
        cand_lids = np.asarray(sampler_arrays["lineup_id"], dtype=np.int64)
        cand_pred = np.asarray(sampler_arrays["predicted_dupes"], dtype=np.float64)
        pred_by_lid = dict(zip(cand_lids.tolist(), cand_pred.tolist()))
    predicted_dupes = np.asarray([pred_by_lid.get(int(l), np.nan) for l in u_lids], dtype=np.float64)

    # players: sampled quantiles for everyone the sampler drew; field exposure for those in the field
    exposure: Dict[int, int] = {}
    for u, lid in enumerate(u_lids):
        for pid in pid_lists_by_lid.get(int(lid), []):
            exposure[pid] = exposure.get(pid, 0) + int(dupes[u])
    pids = sorted(set(raw_samples) | set(exposure))
    q = np.full((len(pids), len(QGRID)), np.nan); mean = np.full(len(pids), np.nan); sd = np.full(len(pids), np.nan)
    for i, pid in enumerate(pids):
        v = raw_samples.get(pid)
        if isinstance(v, np.ndarray) and v.size:
            q[i] = np.percentile(v, QGRID); mean[i] = v.mean(); sd[i] = v.std()
    exp_pct = np.asarray([100.0 * exposure.get(pid, 0) / max(1, M) for pid in pids])
    own_proj = np.asarray([(players.get(pid, {}).get("ownership_proj") if players.get(pid, {}).get("ownership_proj") is not None else np.nan)
                           for pid in pids], dtype=np.float64)

    # the field's score distribution per simulation
    score_pct = np.percentile(full_scores, QGRID, axis=1).T.astype(np.float32)      # (n_sims, len(QGRID))
    winning = full_scores.max(axis=1)
    sorted_scores = -np.sort(-full_scores, axis=1)
    cash_line = sorted_scores[:, min(max(paid_places, 1), M) - 1] if M else np.zeros(n_sims)

    arrays = {
        "lineup_id": np.asarray(u_lids, dtype=np.int64), "dupes": dupes, "predicted_dupes": predicted_dupes,
        "cash_pct": cash_pct, "avg_roi": avg_roi, "mean_rank": mean_rank, "mean_score": mean_score, "win_pct": win_pct,
        "lineup_players": lineup_players,
        "cand_lineup_id": cand_lids, "cand_predicted_dupes": cand_pred,
        "player_id": np.asarray(pids, dtype=np.int64), "player_q": q, "player_mean": mean, "player_sd": sd,
        "player_exposure_pct": exp_pct, "player_ownership_proj": own_proj,
        "score_pct": score_pct, "cash_line": cash_line.astype(np.float32), "winning_score": winning.astype(np.float32),
        "pay_by_rank": np.asarray(pay_by_rank, dtype=np.float64), "qgrid": np.asarray(QGRID, dtype=np.int64),
    }
    meta = {
        "version": ARTIFACT_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "contest": contest, "n_sims": int(n_sims), "simulated_size": int(M), "unique_lineups": int(U),
        "paid_places": int(paid_places), "pool_in_field": float(pool_in_field), "tiers": tiers,
        "players": [{"id": int(pid), **{k: players.get(pid, {}).get(k) for k in ("name", "team", "pos", "slot", "salary", "fpts_mean")}}
                    for pid in pids],
        "sampler": sampler_meta or {},
    }
    return arrays, meta


def write_sim_artifact(npz_path: Path, json_path: Path, arrays: dict, meta: dict) -> None:
    from ivan.sports.nfl.sim_report import jsonable
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    Path(json_path).write_text(json.dumps(jsonable(meta), indent=1, default=str))


def load_sim_artifact(npz_path: Path, json_path: Optional[Path] = None) -> tuple[dict, dict]:
    """(arrays, meta) of one contest's sim artifact."""
    npz_path = Path(npz_path)
    json_path = Path(json_path) if json_path else npz_path.with_suffix(".json")
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    return arrays, meta
