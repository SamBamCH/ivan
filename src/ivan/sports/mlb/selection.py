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
    sims = session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_name == contest_name).all()
    total_count = len(sims)

    print(f'pulled {total_count} lineups from lineup_simulation_results')

    # if we’ve got too many, drop the truly bad ones up front
    THRESHOLD = 10000
    if total_count > THRESHOLD:
        # keep only sims whose avg roi ≥ 0
        sims = [
            res for res in sims
            if np.array(res.roi, dtype=float).mean() >= 0.0
        ]
    N = len(sims)
    if N == 0:
        return []

    print(f'choosing from top {N} lineups')

# now drop any sims whose lineup contains missing players
    # 1) collect every player_id across our sims
    all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
    # 2) fetch the subset that actually exists in MLBPlayer
    existing = {
        pid for (pid,) in
        session.query(MLBPlayer.id).filter(MLBPlayer.id.in_(all_pids)).all()
    }
    # 3) filter out sims with any missing players
    filtered = []
    for res in sims:
        missing = set(res.lineup.player_ids_list) - existing
        if missing:
            print(f"Skipping lineup {res.lineup_id}: missing players {missing}")
        else:
            filtered.append(res)
    sims = filtered

    N = len(sims)
    if N == 0:
        print("No valid lineups remain after dropping those with missing players.")
        return []
    
    

    # 2) build raw features + ROI trace matrix
    raw = np.zeros((N, 2), dtype=float)
    roi_rows = []
    lineup_ids = []

    for i, res in enumerate(sims):
        payouts = np.array(res.payout, dtype=float)
        rois    = np.array(res.roi,    dtype=float)

        raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0
        raw[i, 1] = rois.mean()            if rois.size   else 0.0

        roi_rows.append(rois)
        lineup_ids.append(res.lineup_id)

    roi_matrix = np.vstack(roi_rows)  # shape = (N lineups) × (S sim-runs)

     # 4) build fpts trace matrix instead of roi for diversity
    fpts_rows = []
    for res in sims:
        fpts = np.array(res.fpts, dtype=float)  # make sure this exists
        fpts_rows.append(fpts)
    fpts_matrix = np.vstack(fpts_rows)  # shape = (N lineups) × (S sim‐runs)

    # 3) normalize & weight features
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)
    w    = np.array([weights["cash_pct"], weights["avg_roi_win"]])
    comp = (norm * w)[:, 1]  # our tie-breaker

    # 4) diversity distances
    dist_mat = squareform(pdist(fpts_matrix, metric="euclidean"))

    # 5) greedy selection
    remaining = set(range(N))
    selected  = []

    # seed with best ROI-component
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

    # fill by pure ROI if still short
    if len(selected) < num_lineups:
        for j in np.argsort(-comp):
            if j not in selected:
                selected.append(j)
            if len(selected) >= num_lineups:
                break

    chosen_id_lists = [
        sims[i].lineup.player_ids_list
        for i in selected
    ]

    # now print exposures
    show_player_exposures(session, chosen_id_lists)
    show_team_exposures(session, chosen_id_lists, stack_size=3)

    # finally return the lineup_ids
    return [sims[i].lineup_id for i in selected]

