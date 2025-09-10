from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import json, joblib, numpy as np
import scipy.stats as st
from sqlalchemy.orm import Session
from sqlalchemy import and_
import pyvinecopulib as pv

from ivan.sports.nfl.models import NFLPlayer

# Player marginals
try:
    from ivan.sports.nfl.fpts_distribution_calculations import load_models as load_player_models, simulate_player_fpts
except Exception:
    from ivan.sports.nfl.stat_distribution_calculations import load_models as load_player_models, simulate_player_fpts

TEAM10 = ["QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2","K","DST"]
QB_SLOT, RB_SLOT, WR_SLOT, TE_SLOT, DST_SLOT = 66, 67, 68, 69, 71
_POS_FROM_SLOT = {QB_SLOT:"QB", RB_SLOT:"RB", WR_SLOT:"WR", TE_SLOT:"TE", DST_SLOT:"DST"}

def _coerce_str(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    s = str(x).strip().upper()
    return s or None

def _pos_label_from_slot(slot_id: Optional[int], fb: Optional[str]) -> Optional[str]:
    if slot_id in _POS_FROM_SLOT: return _POS_FROM_SLOT[slot_id]
    fb = _coerce_str(fb)
    return fb if fb in {"QB","RB","WR","TE","DST","K"} else None

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

def _build_player_index(session: Session, min_fpts: Optional[float]) -> tuple[Dict[int,_PM], Dict[str, Dict[str, List[_PM]]]]:
    idx: Dict[int,_PM] = {}
    by_team: Dict[str, Dict[str, List[_PM]]] = {}
    q = session.query(NFLPlayer)
    if min_fpts is not None:
        q = q.filter(and_(NFLPlayer.fpts_mean.isnot(None), NFLPlayer.fpts_mean >= float(min_fpts)))
    for p in q.all():
        team = _coerce_str(getattr(p, "team", None)); 
        if not team: continue
        slot = getattr(p, "roster_slot_id", None)
        pos  = _pos_label_from_slot(int(slot) if slot is not None else None, getattr(p, "position", None))
        if pos not in {"QB","RB","WR","TE","DST","K"}: continue
        opp    = _coerce_str(getattr(p, "opponent", None))
        dname  = getattr(p, "fpts_dist_name", None)
        dparam = getattr(p, "fpts_dist_params", None)
        if isinstance(dparam, list): dparam = tuple(dparam)
        proj   = getattr(p, "fpts_mean", None)
        name   = getattr(p, "name", None) or getattr(p, "player_display_name", None)
        meta = _PM(p.id, team, opp, int(slot) if slot is not None else None, pos,
                   str(dname) if dname else None, dparam if dname else None,
                   float(proj) if proj is not None else None, str(name) if name else None)
        idx[p.id] = meta
        by_team.setdefault(team, {}).setdefault(pos, []).append(meta)
    return idx, by_team

def _independent_samples(session: Session, model_dir: Path, n: int, seed: int | None):
    rng = np.random.default_rng(seed)
    player_models = load_player_models(model_dir)
    _pidx, teams = _build_player_index(session, None)
    out: Dict[int, np.ndarray] = {}
    for team, bypos in teams.items():
        for pos, lst in bypos.items():
            for pm in lst:
                u = rng.random(n, dtype=np.float32)
                x = _player_ppf_fast(pm.dist, pm.params, u)
                if x is None:
                    sim = simulate_player_fpts(None, pm.id, player_models[0], player_models[1], n_samples=max(2*n, 64))
                    s = np.asarray(sim, dtype=np.float32)
                    # use empirical quantile
                    s.sort()
                    idx = np.minimum((u * (len(s)-1)).astype(np.int32), len(s)-1)
                    x = s[idx]
                out[pm.id] = x.astype(np.float32, copy=False)
    return out, teams

def _impose_rank(values: np.ndarray, target_u: np.ndarray, pid: int) -> np.ndarray:
    # map the order of 'values' to the rank order of 'target_u'
    n = len(values)
    v_sorted = np.sort(values, kind="mergesort")  # stable
    # small per-player rotation so WR3 pools etc. aren't identical
    k = (pid * 2654435761) % max(1, n)
    order_u = np.argsort(np.argsort(np.roll(target_u, k)))
    out = np.empty_like(values)
    out[order_u] = v_sorted
    return out

def _assign_slot_groups(team_bucket: Dict[str, List[_PM]], *, wr3_threshold: float) -> Dict[str, List[_PM]]:
    out = {s: [] for s in TEAM10}
    def top(lst: List[_PM], k: int) -> List[_PM]:
        return sorted(lst, key=lambda pm: (pm.proj if pm.proj is not None else -1e12), reverse=True)[:k]
    # QB
    qbs = top(team_bucket.get("QB", []), 1);        out["QB"]  = qbs[:1]
    # RB1/RB2
    rbs = top(team_bucket.get("RB", []), 2);        out["RB1"] = rbs[:1]; out["RB2"] = rbs[1:2]
    # WR1..WR3
    wrs = sorted(team_bucket.get("WR", []), key=lambda pm: (pm.proj if pm.proj is not None else -1e12), reverse=True)
    if len(wrs) >= 1: out["WR1"] = [wrs[0]]
    if len(wrs) >= 2: out["WR2"] = [wrs[1]]
    wr3_pool = [pm for pm in wrs[2:] if (pm.proj is not None and pm.proj < wr3_threshold)]
    out["WR3"] = wr3_pool if wr3_pool else ([wrs[2]] if len(wrs) >= 3 else [])
    # TE1/TE2
    tes = top(team_bucket.get("TE", []), 2);        out["TE1"] = tes[:1]; out["TE2"] = tes[1:2]
    # K / DST
    ks  = top(team_bucket.get("K", []), 1);         out["K"]   = ks[:1]
    dst = top(team_bucket.get("DST", []), 1);       out["DST"] = dst[:1]
    return out

def _player_ppf_fast(dist_name: Optional[str], params: Optional[tuple], u: np.ndarray) -> Optional[np.ndarray]:
    if not dist_name or params is None: return None
    try:
        dist = getattr(st, str(dist_name))
    except AttributeError:
        return None
    return dist.ppf(u, *params).astype(np.float32)

# -------- HGV-20 loaders
def _load_team10_vine(model_dir: Path) -> tuple[pv.Vinecop, list[str]]:
    obj = joblib.load(model_dir / "team10_vine.pkl")
    return obj.vine, list(obj.cols)

def _load_game_factors(model_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = json.loads((model_dir / "game_factors.json").read_text())
    return np.asarray(d["lam_off"], float), np.asarray(d["lam_def"], float), d.get("order", TEAM10)

def _apply_two_factors(UA: np.ndarray, UB: np.ndarray, lam_off: np.ndarray, lam_def: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray,np.ndarray]:
    zA = st.norm.ppf(UA); zB = st.norm.ppf(UB)
    n, d = zA.shape
    G_off = rng.standard_normal((n,1)); G_def = rng.standard_normal((n,1))
    lo = lam_off.reshape(1,d); ld = lam_def.reshape(1,d)
    scale = np.sqrt(np.clip(1.0 - lo**2 - ld**2, 1e-4, 1.0))
    zA = lo*G_off + ld*G_def + scale*zA
    zB = lo*G_off + ld*G_def + scale*zB
    return st.norm.cdf(zA).astype(np.float32), st.norm.cdf(zB).astype(np.float32)

def _emit_players_from_u(
    rng: np.random.Generator,
    n: int,
    team: str,
    grouped_slots: Dict[str, List[_PM]],
    u_cols: Dict[str, np.ndarray],
    teams: Dict[str, Dict[str, List[_PM]]],
    player_models,
    fallback_cache: Dict[int, np.ndarray],
    out_local: Dict[int, np.ndarray],
) -> None:
    def u_for(slot_name: str) -> np.ndarray:
        return u_cols.get(slot_name, rng.random(n, dtype=np.float32))
    grouped = grouped_slots or {}

    # Correlated starters (all ten slots incl. K/DST)
    for slot_name, pm_list in grouped.items():
        if not pm_list: continue
        u_slot = u_for(slot_name)
        for pm in pm_list:
            u = u_slot  # no extra noise — dependence comes from the model
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

    # Depth players at each position: independent draws
    for pos in ("QB","RB","WR","TE","K","DST"):
        pm_all = teams.get(team, {}).get(pos) or []
        slotted_ids = {pm.id for pms in grouped.values() for pm in pms}
        for pm in pm_all:
            if pm.id in slotted_ids: continue
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

# --- public: impose vine correlation onto independent samples ----------------
def generate_correlated_samples_impose(
    session: Session,
    model_dir: str | Path = "data/nfl/models",
    n_samples: int = 2000,
    seed: int | None = None,
    wr3_proj_max: float = 5.0,
) -> Dict[int, np.ndarray]:
    """
    1) draw independent samples per-player from their marginals,
    2) sample team10 vine + two game factors,
    3) impose the vine ranks onto the independent vectors (per slot).
    """
    model_dir = Path(model_dir)
    rng = np.random.default_rng(seed)

    indep, teams = _independent_samples(session, model_dir, n_samples, seed)
    if not teams:
        return {}

    # team↔opp & slot groups
    team_to_opp = {}
    for t, d in teams.items():
        opp = None
        for key in ("QB","WR","RB","TE","DST","K"):
            if d.get(key):
                opp = d[key][0].opp
                if opp: break
        team_to_opp[t] = opp

    seen = set(); matchups = []
    for t, o in team_to_opp.items():
        if not o: continue
        a, b = sorted([t, o])
        if (a, b) in seen: continue
        seen.add((a, b)); matchups.append((a, b))

    slot_groups = {t: _assign_slot_groups(bucket, wr3_threshold=float(wr3_proj_max)) for t, bucket in teams.items()}

    vine10, order = _load_team10_vine(model_dir)
    lam_off, lam_def, order_f = _load_game_factors(model_dir)
    if order_f != order: order = order_f
    idx_map = {s: order.index(s) for s in TEAM10}

    out: Dict[int, np.ndarray] = {}
    # start from independent samples everywhere
    out.update(indep)

    for a, b in matchups:
        UA0 = vine10.simulate(n_samples); UB0 = vine10.simulate(n_samples)
        UA, UB = _apply_two_factors(UA0, UB0, lam_off, lam_def, rng)

        uA = {s: UA[:, idx_map[s]].astype(np.float32, copy=False) for s in TEAM10}
        uB = {s: UB[:, idx_map[s]].astype(np.float32, copy=False) for s in TEAM10}

        for side, team, ucols in (("A", a, uA), ("B", b, uB)):
            grouped = slot_groups.get(team, {})
            for slot_name, pms in (grouped or {}).items():
                if not pms: continue
                u_slot = ucols.get(slot_name)
                if u_slot is None: continue
                for pm in pms:
                    v = indep.get(pm.id)
                    if v is None: continue
                    out[pm.id] = _impose_rank(v, u_slot, pm.id)

    return out
