# src/ivan/sports/mlb/selection.py

from typing import List
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import LineupSimulationResult, NFLPlayer, Lineup

# put near your imports
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

def _plot_lineup_space(
    fpts_matrix: np.ndarray,
    selected: list[int],
    avg_roi: np.ndarray,
    out_dir: str = "mlb_debug_plots",
    max_points: int = 1500,
):
    """
    Save three debug plots:
      1) PCA 2D scatter of all (subsampled) lineups with selected highlighted
      2) Explained variance (top 10 PCs)
      3) Pairwise-distance heatmap among selected lineups
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    N, T = fpts_matrix.shape

    # --- Subsample (keep all selected, plus random others) to keep plots light
    keep = np.arange(N)
    if N > max_points:
        mask_sel = np.zeros(N, dtype=bool)
        mask_sel[selected] = True
        pool = keep[~mask_sel]
        np.random.seed(0)
        extra = np.random.choice(pool, size=max(0, max_points - len(selected)), replace=False)
        keep = np.concatenate([np.array(selected, dtype=int), extra])
    X = fpts_matrix[keep]
    sel_mask = np.isin(keep, selected)

    # --- PCA via SVD on column-centered data
    Xc = X - X.mean(axis=0, keepdims=True)
    # full_matrices=False gives thin SVD; efficient and plenty for PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords2 = U[:, :2] * S[:2]  # 2D PCA coordinates
    evr = (S**2) / np.sum(S**2)  # explained variance ratio

    # 1) PCA scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(coords2[~sel_mask, 0], coords2[~sel_mask, 1], s=8, alpha=0.35)
    plt.scatter(coords2[sel_mask, 0], coords2[sel_mask, 1], s=60, marker='x')
    plt.title("PCA of lineup fpts traces (X = selected)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "pca_scatter.png")
    plt.savefig(p1, dpi=140); plt.close()

    # 2) Explained variance (top 10 PCs)
    k = min(10, len(evr))
    plt.figure(figsize=(7, 3.2))
    plt.plot(np.arange(1, k+1), evr[:k], marker='o')
    plt.title("Explained variance ratio (top PCs)")
    plt.xlabel("PC index"); plt.ylabel("EVR")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "pca_evr.png")
    plt.savefig(p2, dpi=140); plt.close()

    # 3) Distance heatmap among selected
    if len(selected) >= 2:
        Dsel = squareform(pdist(fpts_matrix[selected], metric="euclidean"))
        plt.figure(figsize=(5, 4))
        plt.imshow(Dsel, aspect='auto', interpolation='nearest')
        plt.title("Pairwise distance among SELECTED lineups")
        plt.colorbar()
        plt.tight_layout()
        p3 = os.path.join(out_dir, "selected_distance_heatmap.png")
        plt.savefig(p3, dpi=160); plt.close()
    else:
        p3 = None

    # Optional: ROI vs PC1 to see quality gradient
    if avg_roi is not None:
        roi_keep = avg_roi[keep]
        plt.figure(figsize=(7, 3.2))
        plt.scatter(coords2[:, 0], roi_keep, s=10, alpha=0.5)
        plt.scatter(coords2[sel_mask, 0], roi_keep[sel_mask], s=50, marker='x')
        plt.title("Avg ROI vs PC1 (X = selected)")
        plt.xlabel("PC1"); plt.ylabel("avg ROI")
        plt.tight_layout()
        p4 = os.path.join(out_dir, "roi_vs_pc1.png")
        plt.savefig(p4, dpi=140); plt.close()
    else:
        p4 = None

    print("\n[debug plots]")
    print("  PCA scatter:", p1)
    print("  PCA EVR:    ", p2)
    if p3: print("  Sel heatmap:", p3)
    if p4: print("  ROI vs PC1: ", p4)

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
        session.query(NFLPlayer.id).filter(NFLPlayer.id.in_(all_pids)).all()
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
        raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0  # cash%
        raw[i, 1] = rois.mean()            if rois.size   else 0.0  # avg ROI
        roi_rows.append(rois)
    roi_matrix = np.vstack(roi_rows)

    # 6) build fpts trace matrix for diversity
    fpts_rows = [np.array(res.fpts, dtype=float) for res in sims]
    fpts_matrix = np.vstack(fpts_rows)

    # Normalize features to [0,1]
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)

    # Blend into a single quality score q in [0,1]
    # (Use your existing weights; works even if cash weight is 0.)
    w_feat = np.array([weights.get("cash_pct", 0),
                       weights.get("avg_roi_win", 1)], dtype=float)
    q = (norm * w_feat).sum(axis=1)  # quality score in [0,1]

    # Diversity distances on fpts traces
    dist_mat = squareform(pdist(fpts_matrix, metric="euclidean"))
    # Normalize distance matrix to [0, 1] to match quality score scale
    dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min() + 1e-6)
    
    # Debug: print distance matrix statistics
    print(f"Distance matrix stats - min: {dist_mat.min():.6f}, "
          f"max: {dist_mat.max():.6f}, mean: {dist_mat.mean():.6f}")
    print(f"Quality score (q) stats - min: {q.min():.6f}, max: {q.max():.6f}, "
          f"range: {q.max() - q.min():.6f}")

    # --- simple quality + diversity trade-off ---
    lam = float(weights.get("diversity_lambda", 0.5))  # diversity weight

    # Print average ROI of top num_lineups before diversity
    top_indices = np.argsort(-q)[:num_lineups]
    top_roi_means = [roi_matrix[i].mean() for i in top_indices]
    avg_top_roi = np.mean(top_roi_means)
    print(f"Average ROI of top {num_lineups} lineups (before diversity): {avg_top_roi:.4f}")

    # Greedy selection
    remaining = set(range(N))
    selected = []

    # anchor with the best quality item
    first = int(np.argmax(q))
    selected.append(first)
    remaining.remove(first)
    
    # Debug: track first few selections
    debug_count = 0

    def lineup_score(j: int, S: list[int]) -> float:
        # quality + weighted diversity (average cosine distance to selected)
        if S:
            diversity_gain = np.mean(dist_mat[j, S])
        else:
            diversity_gain = 0.0
        return q[j] + lam * diversity_gain

    while len(selected) < num_lineups and remaining:
        # pick item with max quality + diversity trade-off
        best_j = max(remaining, key=lambda j: lineup_score(j, selected))
        
        if debug_count < 5:
            div_gain = np.mean(dist_mat[best_j, selected]) if selected else 0.0
            total_score = q[best_j] + lam * div_gain
            print(f"  Selection {len(selected)}: lineup {best_j}, q={q[best_j]:.6f}, "
                  f"div_gain={div_gain:.6f}, total={total_score:.6f}")
            debug_count += 1
        
        selected.append(best_j)
        remaining.remove(best_j)

    # if we still need to fill, do it by descending q (quality) 
    #TODO: figure out if this loop is needed. I don't think we want to be choosing lineups that aren't diverse. 
    if len(selected) < num_lineups:
        for j in np.argsort(-q):
            if j not in selected:
                selected.append(j)
            if len(selected) >= num_lineups:
                break

    chosen_ids = [sims[i].lineup_id for i in selected]

    # Print average ROI of chosen lineups after diversity
    chosen_roi_means = [roi_matrix[i].mean() for i in selected]
    avg_chosen_roi = np.mean(chosen_roi_means)
    print(f"Average ROI of {len(selected)} chosen lineups (after diversity): {avg_chosen_roi:.4f}")

    # === Verification printout: ROIs and cash% for chosen lineups ===
    print("\n=== Selected lineup verification ===")
    print(f"diversity_lambda = {lam}")
    per_lineup_stats = []
    for k, idx in enumerate(selected):
        res = sims[idx]
        rois = np.array(res.roi, dtype=float)
        payouts = np.array(res.payout, dtype=float)
        avg_roi = rois.mean() if rois.size else 0.0
        cash_pct = (payouts > 0).mean() if payouts.size else 0.0
        per_lineup_stats.append((avg_roi, cash_pct))
        print(f"{k+1:2d}. lineup_id={res.lineup_id:>8} | avg_roi={avg_roi: .4f} | cash%={cash_pct: .3f} | sims={rois.size}")

    if per_lineup_stats:
        avg_roi_all = float(np.mean([r for r, _ in per_lineup_stats]))
        cash_all = float(np.mean([c for _, c in per_lineup_stats]))
        print(f"\nSummary over {len(selected)} selected lineups:")
        print(f"  mean(avg_roi) = {avg_roi_all: .4f}")
        print(f"  mean(cash%)   = {cash_all: .3f}")



    # build avg ROI vector to annotate plots
    avg_roi = np.array([np.array(res.roi, dtype=float).mean() for res in sims], dtype=float)

    # OPTIONAL: turn on plotting via a weight flag
    if weights.get("debug_plots", True):
        _plot_lineup_space(
            fpts_matrix=fpts_matrix,
            selected=selected,
            avg_roi=avg_roi,
            out_dir='data/mlb/debug',
            max_points=int(weights.get("debug_plots_max_points", 1500)),
        )

    return chosen_ids