from typing import List
import numpy as np
from sqlalchemy.orm import Session

from ivan.sports.nba.models import LineupSimulationResult, NBAPlayer
from scipy.spatial.distance import pdist, squareform

def choose_lineup_ids(
    session: Session,
    weights: dict[str, float],
    num_lineups: int,
    contest_name: str
) -> List[int]:

    sims = session.query(LineupSimulationResult) \
                  .filter(LineupSimulationResult.contest_name == contest_name) \
                  .all()
    print(f'pulled {len(sims)} lineups from lineup_simulation_results')

    if len(sims) >  5000:
        sims = [
            res for res in sims
            if np.array(res.roi, dtype=float).mean() >= 0.0
        ]
    if not sims:
        return []
    
    all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
    existing = {
        pid for (pid,) in 
        session.query(NBAPlayer.id).filter(NBAPlayer.id.in_(all_pids)).all()
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
        print("no valid lineups after dropping missing players")
        return []
    
    unique = []
    seen_comps = set()
    for res in sims: 
        comp = tuple(sorted(res.lineup.player_ids_list))
        if comp not in seen_comps:
            seen_comps.add(comp)
            unique.append(res)

    sims = unique
    if not sims:
        print("no unique lineups remain after removing duplicates.")
        return []
    
    N = len(sims)
    print(f'choosing from top {N} unique lineups')
    
    raw = np.zeros((N, 2), dtype=float)
    roi_rows = []
    for i, res in enumerate(sims):
        payouts = np.array(res.payout, dtype=float)
        rois    = np.array(res.roi,    dtype=float)
        raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0 #cash%
        raw[i, 1] = rois.mean()
        roi_rows.append(rois)
    roi_matrix = np.vstack(roi_rows)

    fpts_rows = [np.array(res.fpts, dtype=float) for res in sims]
    fpts_matrix = np.vstack(fpts_rows)

    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)

    #blend into single score q [0,1]
    w_feat = np.array([weights.get("cash_pct", 0),
                       weights.get("avg_roi_win", 1)], dtype=float)
    q = (norm * w_feat).sum(axis=1)

    dist_mat = squareform(pdist(fpts_matrix, metric="cosine"))

    lam = float(weights.get("diversity_lambda", 0.5))

    remaining = set(range(N))
    selected = []

    first = int(np.argmax(q))
    selected.append(first)
    remaining.remove(first)

    def lineup_score(j: int, S: list[int]) -> float:
        if S:
            diversity_gain = np.mean(dist_mat[j, S])
        else:
            diversity_gain = 0.0
        return q[j] + lam * diversity_gain
    
    while len(selected) < num_lineups and remaining:
        best_j = max(remaining, key=lambda j: lineup_score(j, selected))
        selected.append(best_j)
        remaining.remove(best_j)

    if len(selected) < num_lineups:
        print('not enough lineups in diversity choosing algo, just selecting top lineups...')
        for j in np.argsort(-q):
            if j not in selected:
                selected.append(j)
            if len(selected) >= num_lineups:
                break
    
    chosen_ids = [sims[i].lineup_id for i in selected]


    #TODO: could add recap shit here if we want to, like plots/exposures/etc.

    return chosen_ids


    
