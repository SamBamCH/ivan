from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import scipy.stats as st
from scipy.stats import ks_2samp
from sqlalchemy import text
from sqlalchemy.orm import Session

# use the new sampler (rank-impose or direct, whatever you aliased)
from ivan.sports.nfl.correlated_samples import (
    generate_correlated_samples_impose,
    TEAM10,
)

# ---------- pretty print helpers ----------
def _print_corr_wide(corr, rows, cols, title):
    import shutil
    width = shutil.get_terminal_size(fallback=(240, 24)).columns
    with np.printoptions(precision=3, suppress=True, linewidth=max(120, width)):
        pd.set_option("display.width", max(120, width))
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", None)
        df = pd.DataFrame(corr, index=rows, columns=cols)
        print(f"\n[{title}] Spearman:")
        print(df.to_string())

def _spark(vals: np.ndarray, bins: int = 20, width: int = 40) -> str:
    h, _ = np.histogram(vals, bins=bins)
    h = (h / h.max() * width).astype(int)
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks)-1, x * (len(blocks)-1) // max(1, width))] if x > 0 else " " for x in h)

# ---------- HGV-20 loading / uniforms ----------
def _load_team10(model_dir: Path):
    obj = joblib.load(model_dir / "team10_vine.pkl")
    with open(model_dir / "game_factors.json", "r") as f:
        fac = json.load(f)
    vine = obj.vine
    order = list(obj.cols)
    if fac.get("order") and fac["order"] != order:
        order = list(fac["order"])
    lam_off = np.asarray(fac["lam_off"], dtype=float)
    lam_def = np.asarray(fac["lam_def"], dtype=float)
    return vine, order, lam_off, lam_def

def _apply_two_factors(UA: np.ndarray, UB: np.ndarray, lam_off: np.ndarray, lam_def: np.ndarray, rng: np.random.Generator):
    zA = st.norm.ppf(UA); zB = st.norm.ppf(UB)
    n, d = zA.shape
    G_off = rng.standard_normal((n,1)); G_def = rng.standard_normal((n,1))
    lo = lam_off.reshape(1,d); ld = lam_def.reshape(1,d)
    scale = np.sqrt(np.clip(1.0 - lo**2 - ld**2, 1e-4, 1.0))
    zA = lo*G_off + ld*G_def + scale*zA
    zB = lo*G_off + ld*G_def + scale*zB
    return st.norm.cdf(zA).astype(np.float32), st.norm.cdf(zB).astype(np.float32)

def _spearman_cols(U: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, U).astype(float)
    return np.corrcoef(ranks, rowvar=False)

def print_hgv20_spearman(model_dir: str | Path, n: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate uniforms from HGV-20 and print within-team + cross-team Spearman."""
    model_dir = Path(model_dir)
    vine, order, lam_off, lam_def = _load_team10(model_dir)
    idx = {s: order.index(s) for s in TEAM10}

    UA0 = vine.simulate(n); UB0 = vine.simulate(n)
    UA, UB = _apply_two_factors(UA0, UB0, lam_off, lam_def, np.random.default_rng(7))

    A = np.column_stack([UA[:, idx[s]] for s in TEAM10])
    B = np.column_stack([UB[:, idx[s]] for s in TEAM10])

    Ra = _spearman_cols(A)
    _print_corr_wide(Ra, TEAM10, TEAM10, title="HGV-20 within-team (A)")

    S = np.empty((len(TEAM10), len(TEAM10)), dtype=float)
    for i in range(len(TEAM10)):
        for j in range(len(TEAM10)):
            S[i, j] = float(pd.Series(A[:, i]).corr(pd.Series(B[:, j]), method="spearman"))
    _print_corr_wide(S, TEAM10, [f"OPP{c}" for c in TEAM10], title="HGV-20 cross-team (A vs B)")
    return Ra, S

# ---------- Historical Spearman (fast subset) ----------
def print_historical_slot_spearman(session: Session, max_rows: int = 30000) -> Tuple[np.ndarray, np.ndarray]:
    cols_a = TEAM10
    cols_b = [f"OPP{c}" for c in TEAM10]
    sql = f"SELECT {','.join(cols_a + cols_b)} FROM nfl_team_week_positional_fpts LIMIT :n"
    with session.bind.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"n": int(max_rows)})
    for c in cols_a + cols_b:
        if c not in df.columns:
            df[c] = 0.0
    X = df[cols_a + cols_b].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Ra = X[cols_a].corr(method="spearman").to_numpy(dtype=float)
    _print_corr_wide(Ra, cols_a, cols_a, title=f"Historical within-team (n≤{len(X)})")

    S = np.empty((len(cols_a), len(cols_b)), dtype=float)
    for i, ai in enumerate(cols_a):
        for j, bj in enumerate(cols_b):
            S[i, j] = float(pd.Series(X[ai]).corr(pd.Series(X[bj]), method="spearman"))
    _print_corr_wide(S, cols_a, cols_b, title=f"Historical cross-team (n≤{len(X)})")
    return Ra, S

# ---------- NEW: Sample-based Spearman ----------
def compute_samples_spearman(
    session: Session,
    samples: Dict[int, np.ndarray],
    model_dir: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, List[int]]]]:
    """
    Compute Spearman correlations from actual player samples.
    Returns within-team and cross-team correlation matrices plus player mappings.
    """
    from ivan.sports.nfl.models import NFLPlayer
    
    # Build player index to map players to teams and positions
    pids = list(samples.keys())
    q = session.query(NFLPlayer).filter(NFLPlayer.id.in_(pids))
    
    # Map players to their team/position/slot
    team_slots = {}  # {team: {slot: [player_ids]}}
    player_meta = {}  # {pid: (team, opp, pos, proj)}
    
    for p in q.all():
        team = str(getattr(p, "team", "")).strip().upper()
        opp = str(getattr(p, "opponent", "")).strip().upper()
        pos = str(getattr(p, "position", "")).strip().upper()
        proj = float(getattr(p, "fpts_mean", 0) or 0)
        name = getattr(p, "name", None) or getattr(p, "player_display_name", None)
        
        if team and pos in {"QB", "RB", "WR", "TE", "DST", "K"}:
            player_meta[p.id] = (team, opp, pos, proj, name)
            if team not in team_slots:
                team_slots[team] = {s: [] for s in TEAM10}
    
    # Assign players to slots based on projection rank within position
    for team in team_slots:
        # Get all players for this team by position
        team_players = {pos: [] for pos in ["QB", "RB", "WR", "TE", "DST", "K"]}
        
        for pid, (t, opp, pos, proj, name) in player_meta.items():
            if t == team:
                team_players[pos].append((pid, proj, name))
        
        # Sort by projection and assign to slots
        for pos, players in team_players.items():
            players.sort(key=lambda x: x[1], reverse=True)  # Sort by proj descending
            
            if pos == "QB" and len(players) > 0:
                team_slots[team]["QB"].append(players[0][0])
            elif pos == "RB":
                if len(players) > 0:
                    team_slots[team]["RB1"].append(players[0][0])
                if len(players) > 1:
                    team_slots[team]["RB2"].extend([p[0] for p in players[1:]])
            elif pos == "WR":
                if len(players) > 0:
                    team_slots[team]["WR1"].append(players[0][0])
                if len(players) > 1:
                    team_slots[team]["WR2"].append(players[1][0])
                if len(players) > 2:
                    # WR3+ all go to WR3 slot
                    team_slots[team]["WR3"].extend([p[0] for p in players[2:]])
            elif pos == "TE":
                if len(players) > 0:
                    team_slots[team]["TE1"].append(players[0][0])
                if len(players) > 1:
                    team_slots[team]["TE2"].extend([p[0] for p in players[1:]])
            elif pos == "K" and len(players) > 0:
                team_slots[team]["K"].append(players[0][0])
            elif pos == "DST" and len(players) > 0:
                team_slots[team]["DST"].append(players[0][0])
    
    # Build unique matchup pairs (de-dupe A–B vs B–A)
    matchups: dict[str, str] = {}
    for pid, (team, opp, _, _, _) in player_meta.items():
        if team and opp:
            matchups[team] = opp

    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for team, opp in matchups.items():
        if not team or not opp:
            continue
        if team not in team_slots or opp not in team_slots:
            continue
        key = frozenset((team, opp))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((team, opp))

    if not pairs:
        print("Warning: No complete matchups found for correlation analysis")
        return np.zeros((10, 10)), np.zeros((10, 10)), team_slots
    
    def build_slot_matrix(team: str) -> Optional[np.ndarray]:
        """Build n_samples x 10 matrix of slot values"""
        slot_data = []
        has_data = False
        
        for slot in TEAM10:
            pids_in_slot = team_slots[team][slot]
            if pids_in_slot and pids_in_slot[0] in samples:
                slot_data.append(samples[pids_in_slot[0]])
                has_data = True
            else:
                n = len(next(iter(samples.values())))
                slot_data.append(np.full(n, np.nan))
        
        return np.column_stack(slot_data) if has_data else None
    
    # CRITICAL CHANGE: Compute correlations for EACH game separately
    within_team_corrs = []
    cross_team_corrs = []
    
    for team_a, team_b in pairs:
        A = build_slot_matrix(team_a)
        B = build_slot_matrix(team_b)
        
        if A is None or B is None:
            continue
        
        # Within-team correlation for this game
        Ra_game = np.full((10, 10), np.nan)
        for i in range(10):
            for j in range(10):
                if not np.all(np.isnan(A[:, i])) and not np.all(np.isnan(A[:, j])):
                    Ra_game[i, j] = pd.Series(A[:, i]).corr(pd.Series(A[:, j]), method="spearman")
        
        # Cross-team correlation for this game
        S_game = np.full((10, 10), np.nan)
        for i in range(10):
            for j in range(10):
                if not np.all(np.isnan(A[:, i])) and not np.all(np.isnan(B[:, j])):
                    S_game[i, j] = pd.Series(A[:, i]).corr(pd.Series(B[:, j]), method="spearman")
        
        within_team_corrs.append(Ra_game)
        cross_team_corrs.append(S_game)
        print(f"  Computed correlations for {team_a} vs {team_b}")
    
    if not within_team_corrs or not cross_team_corrs:
        print("Warning: No valid games produced matrices; returning zeros")
        return np.zeros((10, 10)), np.zeros((10, 10)), team_slots
    Ra_avg = np.nanmean(np.stack(within_team_corrs, axis=0), axis=0)
    S_avg  = np.nanmean(np.stack(cross_team_corrs,  axis=0), axis=0)
    
    # Replace NaNs with 0
    Ra_avg = np.nan_to_num(Ra_avg, 0)
    S_avg = np.nan_to_num(S_avg, 0)
    
    print(f"\nAveraged correlations across {len(within_team_corrs)} games")
    _print_corr_wide(Ra_avg, TEAM10, TEAM10, title=f"SAMPLES within-team (avg {len(within_team_corrs)} games)")
    _print_corr_wide(S_avg, TEAM10, [f"OPP{c}" for c in TEAM10], title=f"SAMPLES cross-team (avg {len(cross_team_corrs)} games)")
    
    return Ra_avg, S_avg, team_slots

# ---------- Independent marginals for reference ----------
def _independent_samples(session: Session, model_dir: Path, n: int, pids: List[int], seed: int | None):
    # Reuse the same marginal machinery used in the sampler to ensure apples-to-apples
    try:
        from ivan.sports.nfl.fpts_distribution_calculations import load_models as _load, simulate_player_fpts as _sim
    except Exception:
        from ivan.sports.nfl.stat_distribution_calculations import load_models as _load, simulate_player_fpts as _sim

    from ivan.sports.nfl.models import NFLPlayer
    from sqlalchemy import and_

    player_models = _load(model_dir)
    rng = np.random.default_rng(seed)

    q = session.query(NFLPlayer).filter(NFLPlayer.id.in_(pids))
    meta = {}
    for p in q.all():
        dname = getattr(p, "fpts_dist_name", None)
        dparam = getattr(p, "fpts_dist_params", None)
        if isinstance(dparam, list): dparam = tuple(dparam)
        meta[p.id] = (dname, dparam)

    ref = {}
    for pid in pids:
        dname, dparam = meta.get(pid, (None, None))
        if dname and dparam:
            try:
                dist = getattr(st, str(dname))
                u = rng.random(n, dtype=np.float32)
                ref[pid] = dist.ppf(u, *dparam).astype(np.float32)
                continue
            except Exception:
                pass
        # fallback simulate
        sim = _sim(None, pid, player_models[0], player_models[1], n_samples=max(2*n, 64))
        s = np.asarray(sim, dtype=np.float32); s.sort()
        u = rng.random(n, dtype=np.float32)
        idx = np.minimum((u * (len(s)-1)).astype(np.int32), len(s)-1)
        ref[pid] = s[idx]
    return ref

# ---------- Per-player marginal fidelity ----------
def _check_player_marginals(samples: Dict[int, np.ndarray], ref: Dict[int, np.ndarray], top_k: int = 20):
    pids = [pid for pid in samples if pid in ref]
    ks_vals = []; mean_delta = []; qd = []
    for pid in pids:
        x = samples[pid]; y = ref[pid]
        ks = ks_2samp(x, y).statistic
        ks_vals.append(ks)
        mean_delta.append(float(np.mean(x) - np.mean(y)))
        qd.append([float(np.quantile(x, q) - np.quantile(y, q)) for q in (0.1, 0.5, 0.9)])

    ks_arr = np.array(ks_vals); md = np.array(mean_delta); qd = np.array(qd)
    print("\n[Per-player marginals]")
    print(f"players={len(pids)}  KS: median={np.median(ks_arr):.4f}  p95={np.percentile(ks_arr,95):.4f}  max={ks_arr.max():.4f}")
    print(f"abs(meanΔ): median={np.median(np.abs(md)):.4f}  p95={np.percentile(np.abs(md),95):.4f}  max={np.max(np.abs(md)):.4f}")

    worst_idx = np.argsort(-ks_arr)[:min(top_k, len(pids))]
    print("\nTop KS offenders (pid, KS, meanΔ, q10Δ/q50Δ/q90Δ):")
    for w in worst_idx:
        pid = pids[w]
        q10,q50,q90 = qd[w]
        print(f"  {pid:>7}  KS={ks_arr[w]:.4f}  meanΔ={md[w]:+.4f}  qΔ={q10:+.3f}/{q50:+.3f}/{q90:+.3f}")

    # quick text viz
    print("\nKS distribution (spark):")
    print(_spark(ks_arr, bins=30, width=60))

# ---------- Correlation comparison ----------
def compare_correlations(Ra_samples, Ra_hgv, Ra_hist, S_samples, S_hgv, S_hist):
    """Compare correlation matrices and print differences."""
    print("\n" + "="*80)
    print("CORRELATION COMPARISON")
    print("="*80)
    
    # Within-team differences
    print("\n[Within-team correlation differences]")
    print(f"Samples vs HGV-20: RMSE = {np.sqrt(np.mean((Ra_samples - Ra_hgv)**2)):.4f}")
    print(f"Samples vs Historical: RMSE = {np.sqrt(np.mean((Ra_samples - Ra_hist)**2)):.4f}")
    print(f"HGV-20 vs Historical: RMSE = {np.sqrt(np.mean((Ra_hgv - Ra_hist)**2)):.4f}")
    
    # Cross-team differences  
    print("\n[Cross-team correlation differences]")
    print(f"Samples vs HGV-20: RMSE = {np.sqrt(np.mean((S_samples - S_hgv)**2)):.4f}")
    print(f"Samples vs Historical: RMSE = {np.sqrt(np.mean((S_samples - S_hist)**2)):.4f}")
    print(f"HGV-20 vs Historical: RMSE = {np.sqrt(np.mean((S_hgv - S_hist)**2)):.4f}")
    
    # Find largest discrepancies
    within_diff = Ra_samples - Ra_hgv
    cross_diff = S_samples - S_hgv
    
    # Mask diagonal for within-team
    np.fill_diagonal(within_diff, 0)
    
    max_within_idx = np.unravel_index(np.argmax(np.abs(within_diff)), within_diff.shape)
    max_cross_idx = np.unravel_index(np.argmax(np.abs(cross_diff)), cross_diff.shape)
    
    print(f"\nLargest within-team discrepancy: {TEAM10[max_within_idx[0]]} vs {TEAM10[max_within_idx[1]]}")
    print(f"  Samples: {Ra_samples[max_within_idx]:.3f}, HGV-20: {Ra_hgv[max_within_idx]:.3f}, Diff: {within_diff[max_within_idx]:.3f}")
    
    print(f"\nLargest cross-team discrepancy: {TEAM10[max_cross_idx[0]]} vs OPP{TEAM10[max_cross_idx[1]]}")
    print(f"  Samples: {S_samples[max_cross_idx]:.3f}, HGV-20: {S_hgv[max_cross_idx]:.3f}, Diff: {cross_diff[max_cross_idx]:.3f}")

# ---------- Optional plots ----------
def _maybe_plots(out_dir: Path, 
                 Ra_samples: np.ndarray, S_samples: np.ndarray,
                 Ra_hgv: np.ndarray, S_hgv: np.ndarray,
                 Ra_hist: np.ndarray, S_hist: np.ndarray, 
                 ks_vals: np.ndarray | None = None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib not available; skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def heat(mat, title, fname):
        plt.figure()
        plt.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap='RdBu_r')
        plt.xticks(range(len(TEAM10)), TEAM10, rotation=45, ha="right")
        plt.yticks(range(len(TEAM10)), TEAM10)
        plt.title(title)
        plt.colorbar()
        path = out_dir / fname
        plt.tight_layout(); plt.savefig(path); plt.close()
        print(f"[plots] wrote {path}")

    # All correlation matrices
    heat(Ra_samples, "Samples Within-Team (Spearman)", "within_samples.png")
    heat(Ra_hgv, "HGV-20 Within-Team (Spearman)", "within_hgv.png")
    heat(Ra_hist, "Historical Within-Team (Spearman)", "within_hist.png")
    
    heat(S_samples, "Samples Cross-Team (Spearman)", "cross_samples.png")
    heat(S_hgv, "HGV-20 Cross-Team (Spearman)", "cross_hgv.png")
    heat(S_hist, "Historical Cross-Team (Spearman)", "cross_hist.png")
    
    # Difference plots
    heat(Ra_samples - Ra_hgv, "Within-Team Δ (Samples - HGV)", "within_delta_samples_hgv.png")
    heat(S_samples - S_hgv, "Cross-Team Δ (Samples - HGV)", "cross_delta_samples_hgv.png")

    if ks_vals is not None:
        plt.figure()
        plt.hist(ks_vals, bins=40)
        plt.title("Per-player KS vs independent")
        path = out_dir / "player_ks_hist.png"
        plt.tight_layout(); plt.savefig(path); plt.close()
        print(f"[plots] wrote {path}")

# ---------- One-call runner ----------
def run_end_to_end_and_validate(
    session: Session,
    model_dir: str | Path,
    n_samples: int = 1000,
    hist_max_rows: int = 1000,
    plot_dir: str | Path | None = None,
    seed: int | None = 7,
) -> Dict[int, np.ndarray]:
    """Samples correlated values, checks per-player marginals vs independent, and prints HGV-20 vs historical Spearman."""
    model_dir = Path(model_dir)

    # 1) correlated samples (your normal path)
    print("\n" + "="*80)
    print("GENERATING CORRELATED SAMPLES")
    print("="*80)
    samples = generate_correlated_samples_impose(session, model_dir=model_dir, n_samples=n_samples, seed=seed)
    print(f"Generated {len(samples)} player samples with n={n_samples}")

    # 2) independent reference for the same players
    print("\n" + "="*80)
    print("MARGINAL FIDELITY CHECK")
    print("="*80)
    pids = list(samples.keys())
    ref = _independent_samples(session, model_dir, n_samples, pids, seed)
    _check_player_marginals(samples, ref, top_k=20)

    # 3) Compute correlations from actual samples
    print("\n" + "="*80)
    print("SAMPLE CORRELATIONS")
    print("="*80)
    Ra_samples, S_samples, team_slots = compute_samples_spearman(session, samples, model_dir)

    # 4) HGV-20 uniforms (theoretical model)
    print("\n" + "="*80)
    print("HGV-20 MODEL CORRELATIONS")
    print("="*80)
    Ra_hgv, S_hgv = print_hgv20_spearman(model_dir, n=max(10000, n_samples))

    # 5) Historical correlations
    print("\n" + "="*80)
    print("HISTORICAL CORRELATIONS")
    print("="*80)
    Ra_hist, S_hist = print_historical_slot_spearman(session, max_rows=hist_max_rows)

    # 6) Compare all three
    compare_correlations(Ra_samples, Ra_hgv, Ra_hist, S_samples, S_hgv, S_hist)

    # 7) optional plots
    if plot_dir is not None:
        ks_vals = []
        for pid in pids:
            if pid in ref:
                ks_vals.append(ks_2samp(samples[pid], ref[pid]).statistic)
        _maybe_plots(Path(plot_dir), Ra_samples, S_samples, Ra_hgv, S_hgv, 
                     Ra_hist, S_hist, np.array(ks_vals))

    return samples