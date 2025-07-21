# src/ivan/sports/mlb/selection.py

from typing import List
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.sports.mlb.models import LineupSimulationResult, MLBPlayer
from ivan.core.utils import show_player_exposures, show_team_exposures

def choose_lineup_ids(
    session: Session,
    weights: dict[str, float],
    num_lineups: int,
    contest_name: str
) -> List[int]:
    """
    Select lineup IDs by cash_pct + avg_roi_win + diversity,
    using only simulation results (no date filtering).
    """
    # 1) pull all simulation-result rows
    sims = session.query(LineupSimulationResult) \
                  .filter(LineupSimulationResult.contest_name == contest_name) \
                  .all()
    print(f'pulled {len(sims)} lineups from lineup_simulation_results')

    # 2) optionally drop negative-ROI sims if too many
    if len(sims) > 5000:
        sims = [
            res for res in sims
            if np.array(res.roi, dtype=float).mean() >= 0.0
        ]
    if not sims:
        return []

    # 3) drop any sims whose lineup contains missing players
    all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
    existing = {
        pid for (pid,) in
        session.query(MLBPlayer.id).filter(MLBPlayer.id.in_(all_pids)).all()
    }
    filtered = []
    for res in sims:
        missing = set(res.lineup.player_ids_list) - existing
        if missing:
            print(f"Skipping lineup {res.lineup_id}: missing players {missing}")
        else:
            filtered.append(res)
    sims = filtered
    if not sims:
        print("No valid lineups remain after dropping those with missing players.")
        return []

    # 4) remove exact duplicates by sorted composition
    unique = []
    seen_comps = set()
    for res in sims:
        comp = tuple(sorted(res.lineup.player_ids_list))
        if comp not in seen_comps:
            seen_comps.add(comp)
            unique.append(res)
    sims = unique
    if not sims:
        print("No unique lineups remain after removing duplicates.")
        return []

    N = len(sims)
    print(f'choosing from top {N} unique lineups')

    # 5) build cash% & ROI features
    raw = np.zeros((N, 2), dtype=float)
    roi_rows = []
    for i, res in enumerate(sims):
        payouts = np.array(res.payout, dtype=float)
        rois    = np.array(res.roi,    dtype=float)
        raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0
        raw[i, 1] = rois.mean()            if rois.size   else 0.0
        roi_rows.append(rois)
    roi_matrix = np.vstack(roi_rows)

    # 6) build fpts trace matrix for diversity
    fpts_rows = [np.array(res.fpts, dtype=float) for res in sims]
    fpts_matrix = np.vstack(fpts_rows)

    # 7) normalize & weigh cash%/ROI
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm       = (raw - mins) / (maxs - mins + 1e-6)
    w          = np.array([weights["cash_pct"], weights["avg_roi_win"]])
    comp       = (norm * w)[:, 1]  # ROI component for ranking

    # 8) compute diversity distances
    dist_mat = squareform(pdist(fpts_matrix, metric="euclidean"))

    # 9) greedy select
    remaining = set(range(N))
    selected  = []

    first = int(np.argmax(comp))
    selected.append(first)
    remaining.remove(first)

    λ = weights.get("diversity_lambda", 0.0)
    while len(selected) < num_lineups and remaining:
        best_score, best_idx = -np.inf, None
        for j in remaining:
            score = comp[j] + λ * dist_mat[j, selected].sum()
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(best_idx)
        remaining.remove(best_idx)

    # 10) fill by pure ROI if still short
    if len(selected) < num_lineups:
        for j in np.argsort(-comp):
            if j not in selected:
                selected.append(j)
            if len(selected) >= num_lineups:
                break

    return [sims[i].lineup_id for i in selected]
