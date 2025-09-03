from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import os
import numpy as np
import pandas as pd
import joblib
import scipy.stats as st
from sqlalchemy.orm import Session
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed

from ivan.sports.nfl.models import NFLPlayer
from ivan.sports.nfl.fpts_distribution_calculations import (
    load_models as load_player_models,
    simulate_player_fpts,
)

try:
    # pip install copulas
    from copulas.multivariate import VineCopula
except Exception:  # pragma: no cover
    VineCopula = object  # type: ignore

from types import SimpleNamespace
from typing import Any, Literal, Mapping

_DEFAULTS: dict[str, Any] = {
    "n_samples": 2000,
    "alpha": 0.95,
    "jitter": 0.01,
    "seed": None,
    "rho_game": 0.60,
    "kicker_rho": 0.45,
    "kicker_shape": 5.5,
    "kicker_scale": 1.2,
    "n_jobs": 0,           # 0 = auto threads across teams; 1 = single-thread
    "print_sanity": True,
}

def _merge_dicts(*layers: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer in layers:
        if isinstance(layer, Mapping):
            out.update(layer)
    return out
# ---- Canonical roster slots ----
QB_SLOT, RB_SLOT, WR_SLOT, TE_SLOT, DST_SLOT = 66, 67, 68, 69, 71
_VINE_POSITIONS = ["QB", "RB", "WR", "TE", "OppDST"]
_POS_FROM_SLOT = {
    QB_SLOT: "QB",
    RB_SLOT: "RB",
    WR_SLOT: "WR",
    TE_SLOT: "TE",
    DST_SLOT: "DST",
}


# ---------- utilities ----------

def _load_team_vines(model_dir: Path) -> Dict[str, VineCopula]:
    """Load per-team vine copulas trained on (QB, RB, WR, TE, OppDST) uniforms."""
    vines: Dict[str, VineCopula] = {}
    vine_dir = Path(model_dir) / "team_vines"
    if not vine_dir.exists():
        return vines
    for p in vine_dir.glob("*_vine.pkl"):
        team = p.stem.replace("_vine", "").upper()
        try:
            vines[team] = joblib.load(p)
        except Exception:
            continue
    return vines


def _norm_to_uniform(z: np.ndarray) -> np.ndarray:
    return st.norm.cdf(z)


def _coerce_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s or None

def _cfg_from_dict(
    config: Mapping[str, Any] | None,
    mode: Literal["showdown", "classic", "default"] = "showdown",
    process: Literal["lineup_generation", "sampling", "simulation", "sampling_field", "default"] = "sampling",
    overrides: Mapping[str, Any] | None = None,
) -> SimpleNamespace:
    """
    Merge precedence (lowest → highest):
      1) _DEFAULTS
      2) config.get("global", {})
      3) legacy flat section config.get(process, {})         # optional old style
      4) mode section       config.get(mode, {})
      5) mode+process       config.get(mode, {}).get(process, {})
      6) overrides          (per-call hot swaps)
    Returns attr-accessible namespace.
    """
    d_global  = (config or {}).get("global", {})
    d_legacy  = (config or {}).get(process, {})
    d_mode    = ((config or {}).get(mode, {}) if config else {})
    d_mode_p  = d_mode.get(process, {}) if isinstance(d_mode, Mapping) else {}

    merged = _merge_dicts(
        _DEFAULTS,
        d_global,
        d_legacy,
        d_mode,
        d_mode_p,
        overrides,
    )

    # Ensure every default exists
    for k, v in _DEFAULTS.items():
        merged.setdefault(k, v)

    return SimpleNamespace(**merged)


def _pos_label_from_slot(slot_id: Optional[int], fallback_pos: Optional[str]) -> Optional[str]:
    if slot_id in _POS_FROM_SLOT:
        return _POS_FROM_SLOT[slot_id]
    fb = _coerce_str(fallback_pos)
    if fb in {"K", "DST", "QB", "RB", "WR", "TE"}:
        return fb
    return None

# Lightweight player meta to avoid getattr in hot loops
class _PM:
    __slots__ = ("id","team","opp","slot","pos","dist","params")
    def __init__(self, id:int, team:str|None, opp:str|None, slot:int|None, pos:str|None, dist:str|None, params:tuple|None):
        self.id=id; self.team=team; self.opp=opp; self.slot=slot; self.pos=pos
        self.dist=dist; self.params=params


def _build_player_index(session: Session) -> Tuple[Dict[int,_PM], Dict[str, Dict[str, List[_PM]]]]:
    idx: Dict[int,_PM] = {}
    by_team: Dict[str, Dict[str, List[_PM]]] = {}
    for p in session.query(NFLPlayer).all():
        team = _coerce_str(getattr(p, "team", None))
        if not team:
            continue
        slot = getattr(p, "roster_slot_id", None)
        pos = _pos_label_from_slot(int(slot) if slot is not None else None, getattr(p, "position", None))
        if pos not in {"QB", "RB", "WR", "TE", "DST", "K"}:
            continue
        opp = _coerce_str(getattr(p, "opponent", None))
        dname = getattr(p, "fpts_dist_name", None)
        dparam = getattr(p, "fpts_dist_params", None)
        if isinstance(dparam, list):
            dparam = tuple(dparam)
        meta = _PM(p.id, team, opp, int(slot) if slot is not None else None, pos, str(dname) if dname else None, dparam if dname else None)
        idx[p.id] = meta
        by_team.setdefault(team, {}).setdefault(pos, []).append(meta)
    return idx, by_team


def _bulk_dst_histories(session: Session) -> Tuple[Dict[str,np.ndarray], np.ndarray]:
    """Load all DST game logs once. Return (per-team sorted arrays, league fallback)."""
    with session.bind.connect() as conn:
        df = pd.read_sql(text("SELECT team, dst_points FROM nfl_dst_game WHERE dst_points IS NOT NULL"), conn)
    if df.empty:
        league = np.zeros(1, dtype=np.float32)
        return {}, league
    df["team"] = df["team"].str.upper()
    league = df["dst_points"].astype(np.float32).to_numpy()
    league.sort()
    per: Dict[str,np.ndarray] = {}
    for t, g in df.groupby("team"):
        arr = g["dst_points"].astype(np.float32).to_numpy()
        arr.sort()
        per[t] = arr
    return per, league


def _player_ppf(dist_name: Optional[str], params: Optional[tuple], u: np.ndarray) -> Optional[np.ndarray]:
    if not dist_name or params is None:
        return None
    dist = getattr(st, str(dist_name))
    x = dist.ppf(u, *params).astype(np.float32)
    return x

# ---------- correlation engines ----------

def _nearest_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals[eigvals < eps] = eps
    return (eigvecs * eigvals) @ eigvecs.T


def _fallback_corr_matrix() -> np.ndarray:
    #        QB   RB   WR   TE   ODST
    C = np.array([
        [ 1.00, 0.15, 0.55, 0.40,-0.35],
        [ 0.15, 1.00, 0.20, 0.15,-0.20],
        [ 0.55, 0.20, 1.00, 0.50,-0.35],
        [ 0.40, 0.15, 0.50, 1.00,-0.30],
        [-0.35,-0.20,-0.35,-0.30, 1.00],
    ], dtype=np.float32)
    return _nearest_psd(C)

_DEF_CORR = _fallback_corr_matrix()


def _sample_team_uniforms(rng: np.random.Generator, vine: Optional[VineCopula], n: int) -> np.ndarray:
    if isinstance(vine, VineCopula):
        U = np.asarray(vine.sample(n), dtype=np.float32)
        return np.clip(U, 1e-6, 1 - 1e-6)
    Z = rng.multivariate_normal(mean=np.zeros(len(_VINE_POSITIONS), dtype=np.float32), cov=_DEF_CORR, size=n)
    U = _norm_to_uniform(Z)
    return np.clip(U, 1e-6, 1 - 1e-6)

# ---------- main sampler ----------

def generate_correlated_samples(
    session: Session,
    model_dir: Path | str = "data/nfl/models",
    config: dict | None = None,
    mode: Literal["showdown", "classic"] = "showdown", 
    process: Literal["lineup_generation", "sampling", "simulation", "default"] = "default",
    overrides: dict | None = None
) -> Dict[int, np.ndarray]:

    model_dir = Path(model_dir)
    cfg = _cfg_from_dict(config, mode=mode, process=process, overrides=overrides)
    rng = np.random.default_rng()

    print(f"[generate_correlated_samples] process='{process}' using cfg: "
        f"n_samples={cfg.n_samples}, alpha={cfg.alpha}, jitter={cfg.jitter}, "
        f"rho_game={cfg.rho_game}, kicker_rho={cfg.kicker_rho}, "
        f"kicker_shape={cfg.kicker_shape}, kicker_scale={cfg.kicker_scale}, "
        f"n_jobs={cfg.n_jobs}, print_sanity={cfg.print_sanity}")

    vines = _load_team_vines(model_dir)
    try:
        player_models = load_player_models(model_dir)
    except Exception:
        player_models = None

    # Build player index and team groupings (fast attr access later)
    pidx, teams = _build_player_index(session)

    # Team -> opponent map
    team_to_opp: Dict[str, Optional[str]] = {t: (teams[t].get("QB") or teams[t].get("WR") or teams[t].get("RB") or teams[t].get("TE") or [None])[0].opp if any(teams[t].get(k) for k in ("QB","WR","RB","TE")) else None for t in teams}

    # Shared game factor per matchup
    game_env_u: Dict[Tuple[str, str], np.ndarray] = {}
    seen_pairs = set()
    for t, opp in team_to_opp.items():
        if not opp:
            continue
        key = tuple(sorted((t, opp)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        z = rng.standard_normal(cfg.n_samples).astype(np.float32)
        game_env_u[key] = _norm_to_uniform(z)

    # OppDST uniforms buffer populated by team workers
    oppdst_uniforms: Dict[Tuple[str, str], np.ndarray] = {}

    # Preload DST histories once
    dst_hist, dst_league = _bulk_dst_histories(session)

    pos_to_col = {p: i for i, p in enumerate(_VINE_POSITIONS)}

    # Optional: precompute fallback fpts sims ONCE per player missing dist (to avoid recomputation)
    fallback_cache: Dict[int, np.ndarray] = {}
    if player_models is not None:
        missing = [pm.id for pm in pidx.values() if (pm.pos not in {"DST","K"}) and (pm.dist is None)]
        # If many missing, this is the slow leg. We compute once here.
        for pid in missing:
            sim = simulate_player_fpts(session, pid, player_models[0], player_models[1], n_samples=cfg.n_samples*2)
            s = np.asarray(sim, dtype=np.float32)
            s.sort()
            fallback_cache[pid] = s

    def _team_worker(team: str) -> Tuple[str, Dict[int, np.ndarray]]:
        vine = vines.get(team)
        U_team = _sample_team_uniforms(rng, vine, cfg.n_samples)

        opponent = team_to_opp.get(team)
        if opponent:
            key = tuple(sorted((team, opponent)))
            z_shared = game_env_u.get(key)
            if z_shared is not None and cfg.rho_game > 0:
                for p in ("QB", "RB", "WR", "TE"):
                    c = pos_to_col.get(p)
                    if c is not None:
                        U_team[:, c] = (1.0 - cfg.rho_game) * U_team[:, c] + cfg.rho_game * z_shared
                c_od = pos_to_col.get("OppDST")
                if c_od is not None:
                    u_base = U_team[:, c_od]
                    U_team[:, c_od] = 0.5 * (u_base + (1.0 - z_shared))
                U_team = np.clip(U_team, 1e-6, 1 - 1e-6)

        # Stash opponent DST uniforms (opponent's DST vs this offense)
        if opponent and "OppDST" in pos_to_col:
            opp_col = pos_to_col["OppDST"]
            oppdst_uniforms[(opponent, team)] = U_team[:, opp_col].copy()

        out_local: Dict[int, np.ndarray] = {}

        # Fan out uniforms to players on this team
        def emit_player(pm: _PM, u_base: Optional[np.ndarray]) -> None:
            # Build u for this player
            if pm.pos == "DST":
                # DST: use pre-stashed or anti-corr with shared game factor, then map via empirical dst history
                u = oppdst_uniforms.get((pm.team, pm.opp))
                if u is None:
                    if pm.opp and pm.team:
                        key = tuple(sorted((pm.team, pm.opp)))
                        z = game_env_u.get(key)
                    else:
                        z = None
                    u = (1.0 - z) if z is not None else np.random.random(cfg.n_samples).astype(np.float32)
                # Map with team-specific or league fallback
                vals = dst_hist.get(pm.team)
                if vals is None or (isinstance(vals, np.ndarray) and vals.size == 0):
                    vals = dst_league
                idx = np.minimum((u * (len(vals) - 1)).astype(int), len(vals) - 1)
                x = vals[idx]
                out_local[pm.id] = x.astype(np.float32)
                return

            if pm.pos == "K":
                cols = [pos_to_col[p] for p in ("QB", "RB", "WR", "TE") if p in pos_to_col]
                if cols:
                    u_off = np.mean(U_team[:, cols], axis=1)
                    u = cfg.kicker_rho * u_off + (1.0 - cfg.kicker_rho) * np.random.random(cfg.n_samples).astype(np.float32)
                else:
                    u = np.random.random(cfg.n_samples).astype(np.float32)
                x = st.gamma.ppf(np.clip(u,1e-6,1-1e-6), cfg.kicker_shape, scale=cfg.kicker_scale).astype(np.float32)
                out_local[pm.id] = x
                return

            # Offense positions
            if u_base is None:
                u = np.random.random(cfg.n_samples).astype(np.float32)
            else:
                u = cfg.alpha * u_base + (1.0 - cfg.alpha) * np.random.random(cfg.n_samples).astype(np.float32)

            if cfg.jitter > 0:
                u = u + np.random.normal(0.0, cfg.jitter, size=u.shape).astype(np.float32)
            u = np.clip(u, 1e-6, 1 - 1e-6)

            # Prefer closed-form PPF if available; else use cached fallback sims
            x = _player_ppf(pm.dist, pm.params, u)
            if x is None:
                s = fallback_cache.get(pm.id)
                if s is None:
                    # Last resort (should be rare now)
                    sim = simulate_player_fpts(session, pm.id, player_models[0], player_models[1], n_samples=cfg.n_samples*2)
                    s = np.asarray(sim, dtype=np.float32)
                    s.sort()
                    fallback_cache[pm.id] = s
                idx = np.minimum((u * (len(s) - 1)).astype(int), len(s) - 1)
                x = s[idx]
            out_local[pm.id] = x.astype(np.float32)

        # Emit per position
        for pos in ("QB","RB","WR","TE","DST","K"):
            lst = teams[team].get(pos)
            if not lst:
                continue
            col = pos_to_col.get(pos)
            u_pos_base = U_team[:, col] if (pos in ("QB","RB","WR","TE") and col is not None) else None
            for pm in lst:
                emit_player(pm, u_pos_base)

        return team, out_local

    # Decide threading
    if cfg.n_jobs == 1:
        results = [_team_worker(t) for t in teams.keys()]
    else:
        max_workers = (os.cpu_count() or 4) if cfg.n_jobs == 0 else cfg.n_jobs
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_team_worker, t): t for t in teams.keys()}
            results = [f.result() for f in as_completed(futs)]

    out: Dict[int, np.ndarray] = {}
    for _team, local in results:
        out.update(local)

    # --- Optional sanity print ---
    if cfg.print_sanity:
        try:
            _print_sanity_checks(out, session)
        except Exception:
            pass

    return out

# ---------- diagnostics ----------

def _print_sanity_checks(samples: Dict[int, np.ndarray], session: Session) -> None:
    from scipy.stats import spearmanr
    pids = list(samples.keys())
    players = {p.id: p for p in session.query(NFLPlayer).filter(NFLPlayer.id.in_(pids)).all()}
    by_team: Dict[str, Dict[str, List[int]]] = {}
    for pid, p in players.items():
        team = _coerce_str(getattr(p, "team", None))
        if not team:
            continue
        slot = getattr(p, "roster_slot_id", None)
        pos = _pos_label_from_slot(int(slot) if slot is not None else None, getattr(p, "position", None))
        if pos not in {"QB", "RB", "WR", "TE", "DST"}:
            continue
        by_team.setdefault(team, {}).setdefault(pos, []).append(pid)

    qb_wr_rhos, qb_te_rhos, game_rhos, dst_rhos = [], [], [], []
    opp_map: Dict[str, Optional[str]] = {}
    for t, d in by_team.items():
        opp = None
        for pos in d.values():
            for pid in pos:
                o = _coerce_str(getattr(players[pid], "opponent", None))
                if o: opp = o; break
            if opp: break
        opp_map[t] = opp

    for team, d in by_team.items():
        if "QB" not in d or ("WR" not in d and "TE" not in d):
            continue
        qb = samples[d["QB"][0]]
        if d.get("WR"):
            wrs = np.mean([samples[i] for i in d["WR"]], axis=0)
            qb_wr_rhos.append(spearmanr(qb, wrs).correlation)
        if d.get("TE"):
            tes = np.mean([samples[i] for i in d["TE"]], axis=0)
            qb_te_rhos.append(spearmanr(qb, tes).correlation)
        opp = opp_map.get(team)
        if opp and opp in by_team and ("WR" in by_team[opp] or "TE" in by_team[opp]):
            opp_off = []
            if by_team[opp].get("WR"):
                opp_off.append(np.mean([samples[i] for i in by_team[opp]["WR"]], axis=0))
            if by_team[opp].get("TE"):
                opp_off.append(np.mean([samples[i] for i in by_team[opp]["TE"]], axis=0))
            if opp_off:
                opp_sum = np.mean(opp_off, axis=0)
                game_rhos.append(spearmanr(qb, opp_sum).correlation)
        if d.get("DST"):
            dst = samples[d["DST"][0]]
            dst_rhos.append(spearmanr(qb, dst).correlation)

    if qb_wr_rhos:
        print(f"[corr] mean Spearman(QB, team WR) = {np.nanmean(qb_wr_rhos):.3f}")
    if qb_te_rhos:
        print(f"[corr] mean Spearman(QB, team TE) = {np.nanmean(qb_te_rhos):.3f}")
    if game_rhos:
        print(f"[corr] mean Spearman(QB, opp pass-catchers) = {np.nanmean(game_rhos):.3f}")
    if dst_rhos:
        print(f"[corr] mean Spearman(QB, own DST) = {np.nanmean(dst_rhos):.3f} (expect negative)")
