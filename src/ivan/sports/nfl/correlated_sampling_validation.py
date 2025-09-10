from __future__ import annotations
from typing import Dict, Tuple, List
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
    generate_correlated_samples,
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

# ---------- Optional plots ----------
def _maybe_plots(out_dir: Path, Ra_sim: np.ndarray, S_sim: np.ndarray, Ra_hist: np.ndarray, S_hist: np.ndarray, ks_vals: np.ndarray | None = None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib not available; skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def heat(mat, title, fname):
        plt.figure()
        plt.imshow(mat, aspect="auto")
        plt.xticks(range(len(TEAM10)), TEAM10, rotation=45, ha="right")
        plt.yticks(range(len(TEAM10)), TEAM10)
        plt.title(title)
        plt.colorbar()
        path = out_dir / fname
        plt.tight_layout(); plt.savefig(path); plt.close()
        print(f"[plots] wrote {path}")

    heat(Ra_sim, "HGV-20 Within-Team (Sim Spearman)", "within_sim.png")
    heat(Ra_hist, "Historical Within-Team (Spearman)", "within_hist.png")
    heat(S_sim, "HGV-20 Cross-Team (Sim Spearman)", "cross_sim.png")
    heat(S_hist, "Historical Cross-Team (Spearman)", "cross_hist.png")
    heat(S_sim - S_hist, "Cross-Team Δ (Sim - Hist)", "cross_delta.png")

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
    samples = generate_correlated_samples(session, model_dir=model_dir, n_samples=n_samples, seed=seed)

    # 2) independent reference for the same players
    pids = list(samples.keys())
    ref = _independent_samples(session, model_dir, n_samples, pids, seed)

    # 3) per-player marginal fidelity
    _check_player_marginals(samples, ref, top_k=20)

    # 4) HGV-20 uniforms (sim) vs historical
    Ra_sim, S_sim = print_hgv20_spearman(model_dir, n=max(10000, n_samples))
    Ra_hist, S_hist = print_historical_slot_spearman(session, max_rows=hist_max_rows)

    # 5) optional plots
    if plot_dir is not None:
        # reuse the KS stats we just computed
        ks_vals = []
        for pid in pids:
            if pid in ref:
                ks_vals.append(ks_2samp(samples[pid], ref[pid]).statistic)
        _maybe_plots(Path(plot_dir), Ra_sim, S_sim, Ra_hist, S_hist, np.array(ks_vals))

    return samples
