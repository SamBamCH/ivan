# src/ivan/sports/nfl/correlated_samples.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Mapping, Any, Literal

import numpy as np
import scipy.stats as st
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ivan.sports.nfl.models import NFLPlayer

# Player marginal models (robust import for repo naming)
try:
    from ivan.sports.nfl.fpts_distribution_calculations import (
        load_models as load_player_models,
        simulate_player_fpts,
    )
except Exception:
    from ivan.sports.nfl.stat_distribution_calculations import (
        load_models as load_player_models,
        simulate_player_fpts,
    )

# -----------------------------------------------------------------------------
# Config defaults
# -----------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "n_samples": 2000,
    "alpha": 0.80,          # shared slot-U weight
    "jitter": 0.10,         # small noise in U-space
    "seed": None,
    "print_sanity": True,
    "min_fpts": None,       # DB filter for players
    "wr3_proj_max": 5.0,    # pool WRs < threshold into WR3 (after top-2)
    "validation_samples": 100,  # rows to use for Spearman matrix print
}

def _cfg_from_dict(
    config: Mapping[str, Any] | None,
    mode: Literal["showdown", "classic", "default"] = "showdown",
    process: Literal["lineup_generation", "sampling", "simulation", "default"] = "default",
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    merged = dict(_DEFAULTS)
    for layer in (config or {}).get("global", {}), (config or {}).get(process, {}), (config or {}).get(mode, {}), ((config or {}).get(mode, {}) or {}).get(process, {}), (overrides or {}):
        if isinstance(layer, Mapping):
            merged.update(layer)
    return type("Cfg", (), merged)

# -----------------------------------------------------------------------------
# Roster slot → position helpers (DK-style IDs kept for compatibility)
# -----------------------------------------------------------------------------
QB_SLOT, RB_SLOT, WR_SLOT, TE_SLOT, DST_SLOT = 66, 67, 68, 69, 71
_POS_FROM_SLOT = {QB_SLOT: "QB", RB_SLOT: "RB", WR_SLOT: "WR", TE_SLOT: "TE", DST_SLOT: "DST"}

# Slots modeled by the global vine (K is intentionally excluded → independent)
_DEF_SLOTS = ("QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2","DST")

# Mapping from own slots → opponent columns in the vine draw
_OPP_MAP = {
    "QB": "OPPQB",
    "RB1": "OPPRB1", "RB2": "OPPRB2",
    "WR1": "OPPWR1", "WR2": "OPPWR2", "WR3": "OPPWR3",
    "TE1": "OPPTE1", "TE2": "OPPTE2",
    "DST": "OPPDST",
}

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _coerce_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s or None

def _pos_label_from_slot(slot_id: Optional[int], fallback_pos: Optional[str]) -> Optional[str]:
    if slot_id in _POS_FROM_SLOT:
        return _POS_FROM_SLOT[slot_id]
    fb = _coerce_str(fallback_pos)
    return fb if fb in {"QB", "RB", "WR", "TE", "DST", "K"} else None

@dataclass
class _PM:
    id: int
    team: Optional[str]
    opp: Optional[str]
    slot: Optional[int]
    pos: Optional[str]
    dist: Optional[str]
    params: Optional[tuple]
    proj: Optional[float]
    name: Optional[str]

# Build a player index and bucket by team/position

def _build_player_index(session: Session, min_fpts: Optional[float]) -> Tuple[Dict[int,_PM], Dict[str, Dict[str, List[_PM]]]]:
    idx: Dict[int,_PM] = {}
    by_team: Dict[str, Dict[str, List[_PM]]] = {}

    q = session.query(NFLPlayer)
    if min_fpts is not None:
        q = q.filter(and_(NFLPlayer.fpts_mean.isnot(None), NFLPlayer.fpts_mean >= float(min_fpts)))

    for p in q.all():
        team = _coerce_str(getattr(p, "team", None))
        if not team:
            continue
        slot = getattr(p, "roster_slot_id", None)
        pos  = _pos_label_from_slot(int(slot) if slot is not None else None, getattr(p, "position", None))
        if pos not in {"QB", "RB", "WR", "TE", "DST", "K"}:
            continue
        opp    = _coerce_str(getattr(p, "opponent", None))
        dname  = getattr(p, "fpts_dist_name", None)
        dparam = getattr(p, "fpts_dist_params", None)
        if isinstance(dparam, list):
            dparam = tuple(dparam)
        proj   = getattr(p, "fpts_mean", None)
        name   = getattr(p, "name", None) or getattr(p, "player_display_name", None)

        meta = _PM(
            id=p.id, team=team, opp=opp,
            slot=int(slot) if slot is not None else None,
            pos=str(pos) if pos else None,
            dist=str(dname) if dname else None,
            params=dparam if dname else None,
            proj=float(proj) if proj is not None else None,
            name=str(name) if name else None,
        )
        idx[p.id] = meta
        by_team.setdefault(team, {}).setdefault(pos, []).append(meta)
    return idx, by_team

# Assign players to slots by projection (WR3 is a pool, can contain many)

def _assign_slot_groups(team_bucket: Dict[str, List[_PM]], *, wr3_threshold: float) -> Dict[str, List[_PM]]:
    out: Dict[str, List[_PM]] = {s: [] for s in _DEF_SLOTS}

    def top(lst: List[_PM], k: int) -> List[_PM]:
        return sorted(lst, key=lambda pm: (pm.proj if pm.proj is not None else -1e12), reverse=True)[:k]

    # QB
    qbs = top(team_bucket.get("QB", []), 1)
    if qbs: out["QB"] = [qbs[0]]

    # RB1/RB2
    rbs = top(team_bucket.get("RB", []), 2)
    if len(rbs) >= 1: out["RB1"] = [rbs[0]]
    if len(rbs) >= 2: out["RB2"] = [rbs[1]]

    # WR1..WR3
    wrs_sorted = sorted(team_bucket.get("WR", []), key=lambda pm: (pm.proj if pm.proj is not None else -1e12), reverse=True)
    if len(wrs_sorted) >= 1: out["WR1"] = [wrs_sorted[0]]
    if len(wrs_sorted) >= 2: out["WR2"] = [wrs_sorted[1]]
    wr3_pool = [pm for pm in wrs_sorted[2:] if (pm.proj is not None and pm.proj < wr3_threshold)]
    if wr3_pool:
        out["WR3"] = wr3_pool
    elif len(wrs_sorted) >= 3:
        out["WR3"] = [wrs_sorted[2]]

    # TE1/TE2
    tes = top(team_bucket.get("TE", []), 2)
    if len(tes) >= 1: out["TE1"] = [tes[0]]
    if len(tes) >= 2: out["TE2"] = [tes[1]]

    # DST (single; vine-correlated)
    dsts = top(team_bucket.get("DST", []), 1)
    if dsts: out["DST"] = [dsts[0]]

    return out

# -----------------------------------------------------------------------------
# Global vine loader
# -----------------------------------------------------------------------------

@dataclass
class _GlobalPkg:
    cols: List[str]
    vine: Any
    n_obs: int

def _load_global_vine(model_dir: Path | str) -> _GlobalPkg:
    path = Path(model_dir) / "positional_global_vine.pkl"
    pkg = joblib.load(path)
    return _GlobalPkg(cols=list(getattr(pkg, "cols", [])), vine=getattr(pkg, "vine", None), n_obs=int(getattr(pkg, "n_obs", 0)))

# -----------------------------------------------------------------------------
# Marginal helpers
# -----------------------------------------------------------------------------

def _player_ppf_fast(dist_name: Optional[str], params: Optional[tuple], u: np.ndarray) -> Optional[np.ndarray]:
    if not dist_name or params is None:
        return None
    try:
        dist = getattr(st, str(dist_name))
    except AttributeError:
        return None
    return dist.ppf(u, *params).astype(np.float32)

# Emit samples for all slotted players on a team, including DST (correlated) and K (independent)

def _emit_players_from_u(
    rng: np.random.Generator,
    n: int,
    alpha: float,
    jitter: float,
    team: str,
    grouped_slots: Dict[str, List[_PM]],
    u_cols: Dict[str, np.ndarray],
    teams: Dict[str, Dict[str, List[_PM]]],
    player_models,
    fallback_cache: Dict[int, np.ndarray],
    out_local: Dict[int, np.ndarray],
) -> None:
    def u_for(slot_name: str, fallback_pos: Optional[str] = None) -> np.ndarray:
        if slot_name in u_cols:
            return u_cols[slot_name].astype(np.float32, copy=False)
        if fallback_pos and fallback_pos in u_cols:
            return u_cols[fallback_pos].astype(np.float32, copy=False)
        return rng.random(n, dtype=np.float32)

    grouped = grouped_slots or {}

    # 1) Correlated slots via vine U (QB, RB1/2, WR1/2/3, TE1/2, DST)
    for slot_name, pm_list in grouped.items():
        if not pm_list:
            continue
        base_pos = slot_name[:2] if slot_name.startswith(("RB","WR","TE")) else slot_name
        if base_pos not in {"QB","RB","WR","TE","DST"}:
            base_pos = None
        u_slot = u_for(slot_name, fallback_pos=base_pos)
        for pm in pm_list:
            noise = rng.random(n, dtype=np.float32)
            u = (alpha * u_slot + (1.0 - alpha) * noise).astype(np.float32)
            if jitter > 0:
                u += rng.normal(0.0, jitter, size=n).astype(np.float32)
                np.clip(u, 1e-6, 1-1e-6, out=u)
            x = _player_ppf_fast(pm.dist, pm.params, u)
            if x is None:
                s = fallback_cache.get(pm.id)
                if s is None:
                    if player_models is None:
                        out_local[pm.id] = np.zeros(n, dtype=np.float32); continue
                    sim = simulate_player_fpts(None, pm.id, player_models[0], player_models[1], n_samples=n*2)
                    s = np.asarray(sim, dtype=np.float32); s.sort(); fallback_cache[pm.id] = s
                idx = np.minimum((u * (len(s) - 1)).astype(np.int32), len(s) - 1)
                x = s[idx]
            out_local[pm.id] = x.astype(np.float32, copy=False)

    # 2) Unslotted players for the main four positions use independent u, so they still get values
    for pos in ("QB","RB","WR","TE","DST"):
        pm_all = teams.get(team, {}).get(pos) or []
        slotted_ids = {pm.id for pms in grouped.values() for pm in pms}
        for pm in pm_all:
            if pm.id in slotted_ids:
                continue
            u = rng.random(n, dtype=np.float32)
            x = _player_ppf_fast(pm.dist, pm.params, u)
            if x is None:
                s = fallback_cache.get(pm.id)
                if s is None:
                    if player_models is None:
                        out_local[pm.id] = np.zeros(n, dtype=np.float32); continue
                    sim = simulate_player_fpts(None, pm.id, player_models[0], player_models[1], n_samples=n*2)
                    s = np.asarray(sim, dtype=np.float32); s.sort(); fallback_cache[pm.id] = s
                idx = np.minimum((u * (len(s) - 1)).astype(np.int32), len(s) - 1)
                x = s[idx]
            out_local[pm.id] = x.astype(np.float32, copy=False)

    # 3) Kickers: always independent (explicitly no correlation)
    for pm in teams.get(team, {}).get("K", []) or []:
        u = rng.random(n, dtype=np.float32)
        x = _player_ppf_fast(pm.dist, pm.params, u)
        if x is None:
            s = fallback_cache.get(pm.id)
            if s is None:
                if player_models is None:
                    out_local[pm.id] = np.zeros(n, dtype=np.float32); continue
                sim = simulate_player_fpts(None, pm.id, player_models[0], player_models[1], n_samples=n*2)
                s = np.asarray(sim, dtype=np.float32); s.sort(); fallback_cache[pm.id] = s
            idx = np.minimum((u * (len(s) - 1)).astype(np.int32), len(s) - 1)
            x = s[idx]
        out_local[pm.id] = x.astype(np.float32, copy=False)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_correlated_samples(
    session: Session,
    model_dir: Path | str = "data/nfl/models",
    config: dict | None = None,
    mode: Literal["showdown", "classic"] = "showdown",
    process: Literal["lineup_generation", "sampling", "simulation", "default"] = "default",
    overrides: dict | None = None,
) -> Dict[int, np.ndarray]:
    """Use a single global positional vine to sample both teams in each matchup from the same U draw.

    - Slots in _DEF_SLOTS (including DST) use the vine-correlated uniforms.
    - Kickers (K) are sampled **independently** by design (no correlation).
    - Returns: {player_id -> np.ndarray[n_samples]} across the slate.
    """
    cfg = _cfg_from_dict(config, mode=mode, process=process, overrides=overrides)

    # ensure Path for downstream loaders
    model_dir = Path(model_dir)

    ss = np.random.SeedSequence(cfg.seed)
    rng = np.random.default_rng(ss)

    if cfg.print_sanity:
        print(f"[global corr] n={cfg.n_samples} alpha={cfg.alpha} jitter={cfg.jitter} wr3_max={cfg.wr3_proj_max}")

    gvine = _load_global_vine(model_dir)
    player_models = load_player_models(model_dir)

    _pidx, teams = _build_player_index(session, cfg.min_fpts)
    if not teams:
        return {}

    # team → opponent mapping (best-effort from any available pos)
    team_to_opp: Dict[str, Optional[str]] = {}
    for t, d in teams.items():
        opp = None
        for key in ("QB","WR","RB","TE","DST"):
            if d.get(key):
                opp = d[key][0].opp
                if opp:
                    break
        team_to_opp[t] = opp

    # dedupe matchups
    seen = set(); matchups: List[Tuple[str,str]] = []
    for t, o in team_to_opp.items():
        if not o:
            continue
        a, b = sorted([t, o])
        if (a, b) in seen:
            continue
        seen.add((a, b)); matchups.append((a, b))

    # assign slots per team
    slot_groups: Dict[str, Dict[str, List[_PM]]] = {t: _assign_slot_groups(bucket, wr3_threshold=float(cfg.wr3_proj_max)) for t, bucket in teams.items()}
    fallback_cache: Dict[int, np.ndarray] = {}

    def _do_pair(a: str, b: str, seed: int) -> Dict[int, np.ndarray]:
        rng_i = np.random.default_rng(seed)
        n = int(cfg.n_samples)

        # draw once per game from the global vine; DataFrame with named cols
        U_df = gvine.vine.sample(n)
        U_df = U_df.clip(lower=1e-6, upper=1-1e-6)
        u_cols = {c: U_df[c].to_numpy(dtype=np.float32) for c in gvine.cols if c in U_df.columns}

        # map this joint U to each side's slots
        uA = {s: u_cols.get(s, rng_i.random(n, dtype=np.float32)) for s in _DEF_SLOTS}
        uB = {s: u_cols.get(_OPP_MAP[s], rng_i.random(n, dtype=np.float32)) for s in _DEF_SLOTS}

        out_local: Dict[int, np.ndarray] = {}
        _emit_players_from_u(rng_i, n, float(cfg.alpha), float(cfg.jitter), a, slot_groups.get(a, {}), uA, teams, player_models, fallback_cache, out_local)
        _emit_players_from_u(rng_i, n, float(cfg.alpha), float(cfg.jitter), b, slot_groups.get(b, {}), uB, teams, player_models, fallback_cache, out_local)
        return out_local

    out: Dict[int, np.ndarray] = {}
    for a, b in matchups:
        local = _do_pair(a, b, int(rng.integers(0, 2**31-1)))
        out.update(local)

    # Optional small validation: Spearman of vine uniforms actually used
    if cfg.print_sanity and int(cfg.validation_samples) > 0:
        nval = int(cfg.validation_samples)
        V_df = gvine.vine.sample(nval)
        V_df = V_df.clip(lower=1e-6, upper=1-1e-6)
        V = V_df.to_numpy(dtype=np.float64)
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, V)
        corr = np.corrcoef(ranks, rowvar=False)
        with np.printoptions(precision=3, suppress=True):
            print("[global vine] Spearman corr (cols order):", gvine.cols)
            print(corr)

    return out
