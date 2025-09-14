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

def _shrink_u(u: np.ndarray, randomness: float) -> np.ndarray:
    """
    Linearly shrink quantiles toward the median (0.5).
    randomness in [0,1]: 0 -> all 0.5 (pure median), 1 -> unchanged.
    """
    r = float(np.clip(randomness, 0.0, 1.0))
    if r >= 1.0: 
        return u
    return 0.5 + r * (u - 0.5)


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

def _independent_samples(
    session: Session,
    model_dir: Path,
    n: int,
    seed: int | None,
    randomness: float = 1.0,   # NEW
):
    rng = np.random.default_rng(seed)
    player_models = load_player_models(model_dir)
    _pidx, teams = _build_player_index(session, None)
    out: Dict[int, np.ndarray] = {}
    for team, bypos in teams.items():
        for pos, lst in bypos.items():
            for pm in lst:
                u = rng.random(n, dtype=np.float32)
                u = _shrink_u(u, randomness)  # NEW: tighten around median

                x = _player_ppf_fast(pm.dist, pm.params, u)
                if x is None:
                    sim = simulate_player_fpts(None, pm.id, player_models[0], player_models[1], n_samples=max(2*n, 64))
                    s = np.asarray(sim, dtype=np.float32)
                    s.sort()
                    idx = np.minimum((u * (len(s)-1)).astype(np.int32), len(s)-1)  # uses shrunk u
                    x = s[idx]
                out[pm.id] = np.asarray(x, dtype=np.float32)

    return out, teams

def _impose_rank(values: np.ndarray, target_u: np.ndarray) -> np.ndarray:
    """FIXED: No rotation, no pid parameter"""
    v_sorted = np.sort(values, kind="mergesort")
    order_u = np.argsort(np.argsort(target_u))  # NO ROLL!
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
    n_samples: int = 1000,
    seed: int | None = None,
    wr3_proj_max: float = 5.0,
    # --- NEW: optional hard caps by slot/depth ---
    caps: dict[str, float] | None = None,       # e.g., {"RB2": 25, "WR3": 22, "QB": 45, "*": 60},
    config: dict | None = None,        # optional config dict to pull "randomness" from
    randomness: float | None = None,   # [0,1], 1=full spread, 0=median only
) -> Dict[int, np.ndarray]:
    """
    1) draw independent samples per-player from their marginals,
    2) sample team10 vine + two game factors,
    3) impose the vine ranks onto the independent vectors (per slot).
    4) (NEW) hard-cap samples by slot/depth if caps are provided.
    """
    def _resolve_cap(slot_name: str) -> float | None:
        if not caps:
            return None
        # Exact -> base (e.g., "RB") -> aliases (TE1<->TE) -> "*" fallback
        if slot_name in caps: return float(caps[slot_name])
        # split base + depth
        base, num = None, None
        for i, ch in enumerate(slot_name):
            if ch.isdigit():
                base, num = slot_name[:i], slot_name[i:]
                break
        base = base or slot_name
        if base in caps: return float(caps[base])
        # common aliases (treat TE/TE1, WR/WR1, RB/RB1 as synonyms)
        if num is None:
            alias1 = base + "1"
            if alias1 in caps: return float(caps[alias1])
        else:
            if base in caps: return float(caps[base])
        if "*" in caps: return float(caps["*"])
        return None

    if randomness is None:
        if config is not None and "randomness" in config:
            randomness = float(config["randomness"])
            print(f"Using 'randomness' from config {randomness}")
        else:
            print("WARNING: no 'randomness' provided; defaulting to 1.0")
            randomness = 1.0
    randomness = float(np.clip(randomness, 0.0, 1.0))

    model_dir = Path(model_dir)
    rng = np.random.default_rng(seed)

    # pass randomness into the independent sampler
    indep, teams = _independent_samples(session, model_dir, n_samples, seed, randomness=randomness)
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

    out: Dict[int, np.ndarray] = {}  # Start empty!
    processed_pids = set()

    # (NEW) keep a mapping from player->slot for final safety pass
    pid_to_slot: dict[int, str] = {}

    for a, b in matchups:
        UA0 = vine10.simulate(n_samples); UB0 = vine10.simulate(n_samples)
        UA, UB = _apply_two_factors(UA0, UB0, lam_off, lam_def, rng)

        uA = {s: UA[:, idx_map[s]].astype(np.float32, copy=False) for s in TEAM10}
        uB = {s: UB[:, idx_map[s]].astype(np.float32, copy=False) for s in TEAM10}

        for team, ucols in [(a, uA), (b, uB)]:
            grouped = slot_groups.get(team, {})
            for slot_name, pms in grouped.items():
                if not pms: continue
                u_slot = ucols.get(slot_name)
                if u_slot is None: continue
                cap_val = _resolve_cap(slot_name)  # may be None
                for pm in pms:
                    if pm.id in processed_pids:
                        continue
                    v = indep.get(pm.id)
                    if v is None:
                        continue
                    
                    # FIXED: No pid parameter
                    x = _impose_rank(v, u_slot)
                    
                    # ... apply caps ...
                    out[pm.id] = x
                    processed_pids.add(pm.id)

    for pid, samples in indep.items():
        if pid not in out:
            out[pid] = samples
    
    return out

def _assign_slot_groups_pooled(team_bucket: Dict[str, List[_PM]], *, wr3_threshold: float) -> Dict[str, List[_PM]]:
    """
    Assign ALL players to correlation slots, pooling depth players into the deepest available slot.
    For example, WR3/WR4/WR5 all share WR3 correlation structure.
    """
    out = {s: [] for s in TEAM10}
    
    # Sort by projected points (highest first)
    def sort_by_proj(lst: List[_PM]) -> List[_PM]:
        return sorted(lst, key=lambda pm: (pm.proj if pm.proj is not None else -1e12), reverse=True)
    
    # QB - all QBs share QB correlation (usually just 1-2 per team)
    qbs = sort_by_proj(team_bucket.get("QB", []))
    out["QB"] = qbs  # ALL QBs get QB correlation
    
    # RB - split into RB1, RB2, rest get RB2 correlation
    rbs = sort_by_proj(team_bucket.get("RB", []))
    if len(rbs) >= 1:
        out["RB1"] = [rbs[0]]  # Top RB gets RB1 correlation
    if len(rbs) >= 2:
        out["RB2"] = rbs[1:]   # RB2, RB3, RB4... all get RB2 correlation
    
    # WR - split by threshold and rank
    wrs = sort_by_proj(team_bucket.get("WR", []))
    if len(wrs) >= 1:
        out["WR1"] = [wrs[0]]  # Top WR
    if len(wrs) >= 2:
        out["WR2"] = [wrs[1]]  # Second WR
    if len(wrs) >= 3:
        # WR3 slot gets all remaining WRs with proj < threshold
        # If none meet threshold, just use WR3
        wr3_candidates = [pm for pm in wrs[2:] if pm.proj is not None and pm.proj < wr3_threshold]
        if wr3_candidates:
            out["WR3"] = wr3_candidates  # All low-projection WRs
        else:
            out["WR3"] = wrs[2:]  # All remaining WRs
    
    # TE - split into TE1 and TE2 (rest)
    tes = sort_by_proj(team_bucket.get("TE", []))
    if len(tes) >= 1:
        out["TE1"] = [tes[0]]  # Top TE
    if len(tes) >= 2:
        out["TE2"] = tes[1:]   # TE2, TE3... all get TE2 correlation
    
    # K - usually just one, but if multiple, all share K correlation
    ks = sort_by_proj(team_bucket.get("K", []))
    out["K"] = ks
    
    # DST - usually just one
    dst = sort_by_proj(team_bucket.get("DST", []))
    out["DST"] = dst
    
    return out


def _impose_rank_with_noise(values: np.ndarray, target_u: np.ndarray, 
                            rng: np.random.Generator, 
                            noise_level: float = 0.1) -> np.ndarray:
    """
    Impose rank with optional noise for players sharing the same slot.
    This helps differentiate players in the same correlation group.
    """
    n = len(values)
    v_sorted = np.sort(values, kind="mergesort")
    
    # Add small noise to target_u to break ties for multiple players in same slot
    if noise_level > 0:
        noise = rng.normal(0, noise_level, size=n)
        # Keep in [0,1] range
        noisy_u = np.clip(target_u + noise, 0, 1)
        order_u = np.argsort(np.argsort(noisy_u))
    else:
        order_u = np.argsort(np.argsort(target_u))
    
    out = np.empty_like(values)
    out[order_u] = v_sorted
    return out


def generate_correlated_samples_impose_pooled(
    session: Session,
    model_dir: str | Path = "data/nfl/models",
    n_samples: int = 1000,
    seed: int | None = None,
    wr3_proj_max: float = 5.0,
    caps: dict[str, float] | None = None,
    config: dict | None = None,
    randomness: float | None = None,
) -> Dict[int, np.ndarray]:
    """
    Version that assigns ALL players to correlation slots by pooling depth players.
    """
    def _resolve_cap(slot_name: str) -> float | None:
        if not caps:
            return None
        # Exact -> base (e.g., "RB") -> aliases (TE1<->TE) -> "*" fallback
        if slot_name in caps: return float(caps[slot_name])
        # split base + depth
        base, num = None, None
        for i, ch in enumerate(slot_name):
            if ch.isdigit():
                base, num = slot_name[:i], slot_name[i:]
                break
        base = base or slot_name
        if base in caps: return float(caps[base])
        # common aliases (treat TE/TE1, WR/WR1, RB/RB1 as synonyms)
        if num is None:
            alias1 = base + "1"
            if alias1 in caps: return float(caps[alias1])
        else:
            if base in caps: return float(caps[base])
        if "*" in caps: return float(caps["*"])
        return None

    if randomness is None:
        if config is not None and "randomness" in config:
            randomness = float(config["randomness"])
            print(f"Using 'randomness' from config {randomness}")
        else:
            print("WARNING: no 'randomness' provided; defaulting to 1.0")
            randomness = 1.0
    randomness = float(np.clip(randomness, 0.0, 1.0))
    
    model_dir = Path(model_dir)
    rng = np.random.default_rng(seed)
    
    # Get independent samples
    indep, teams = _independent_samples(session, model_dir, n_samples, seed, randomness=randomness)
    if not teams:
        return {}
    
    # Build team matchups
    team_to_opp = {}
    for t, d in teams.items():
        opp = None
        for key in ("QB","WR","RB","TE","DST","K"):
            if d.get(key):
                opp = d[key][0].opp
                if opp: break
        team_to_opp[t] = opp
    
    seen = set()
    matchups = []
    for t, o in team_to_opp.items():
        if not o: continue
        a, b = sorted([t, o])
        if (a, b) in seen: continue
        seen.add((a, b))
        matchups.append((a, b))
    
    # Use pooled assignment
    slot_groups = {t: _assign_slot_groups_pooled(bucket, wr3_threshold=float(wr3_proj_max)) 
                   for t, bucket in teams.items()}
    
    # Load vine and factors
    vine10, order = _load_team10_vine(model_dir)
    lam_off, lam_def, order_f = _load_game_factors(model_dir)
    if order_f != order: 
        order = order_f
    idx_map = {s: order.index(s) for s in TEAM10}
    
    # Start with independent samples
    out: Dict[int, np.ndarray] = {}
    pid_to_slot: dict[int, str] = {}
    
    for a, b in matchups:
        # Generate correlated uniforms
        UA0 = vine10.simulate(n_samples)
        UB0 = vine10.simulate(n_samples)
        UA, UB = _apply_two_factors(UA0, UB0, lam_off, lam_def, rng)
        
        # Extract uniforms for each slot
        uA = {s: UA[:, idx_map[s]].astype(np.float32, copy=False) for s in TEAM10}
        uB = {s: UB[:, idx_map[s]].astype(np.float32, copy=False) for s in TEAM10}
        
        for side, team, ucols in (("A", a, uA), ("B", b, uB)):
            grouped = slot_groups.get(team, {})
            
            for slot_name, pms in grouped.items():
                if not pms: 
                    continue
                    
                u_slot = ucols.get(slot_name)
                if u_slot is None: 
                    continue
                    
                cap_val = _resolve_cap(slot_name) if caps else None
                
                # Handle multiple players in the same slot
                if len(pms) == 1:
                    # Single player - direct correlation
                    pm = pms[0]
                    v = indep.get(pm.id)
                    if v is not None:
                        x = _impose_rank(v, u_slot)
                        if cap_val is not None:
                            x = np.minimum(x, cap_val).astype(np.float32, copy=False)
                        out[pm.id] = x
                        pid_to_slot[pm.id] = slot_name
                else:
                    # Multiple players sharing the slot - add slight noise to differentiate
                    for i, pm in enumerate(pms):
                        v = indep.get(pm.id)
                        if v is None:
                            continue
                        
                        # Option 1: Add small noise to break perfect correlation
                        noise_level = 0.05 * (i / max(1, len(pms)-1))  # Increasing noise for deeper players
                        x = _impose_rank_with_noise(v, u_slot, rng, noise_level)
                        
                        # Option 2: Alternative - use slightly shifted quantiles
                        # shift = i * 0.02  # Small shift per depth
                        # shifted_u = np.clip(u_slot + shift * rng.standard_normal(n_samples), 0, 1)
                        # x = _impose_rank(v, shifted_u)
                        
                        if cap_val is not None:
                            x = np.minimum(x, cap_val).astype(np.float32, copy=False)
                        out[pm.id] = x
                        pid_to_slot[pm.id] = slot_name
    
    # Any players not assigned (shouldn't happen with pooled approach)
    for pid in indep:
        if pid not in out:
            print(f"Warning: Player {pid} not assigned to any correlation slot")
            out[pid] = indep[pid]  # Fallback to independent
    
    return out

