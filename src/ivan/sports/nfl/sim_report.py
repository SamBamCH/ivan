# src/ivan/sports/nfl/sim_report.py
"""
The per-contest simulation report: what the simulator sampled, scaled and
computed, and the invariants it checked, as one JSON document.

Built piece by piece by simulate_contest.simulate_one_contest, stored in the
contest_sim_report table (one row per contest, replaced on re-run) and as
report.json under data/nfl/diagnostics/, rendered by the web UI's Contests
page and summarised on the terminal by format_summary().

Sections: contest, field, payouts, simulation, correlation, checks.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import ContestSimReport, NFLPlayer
from ivan.sports.nfl.paths import DIAG_DIR

REPORT_VERSION = 1
SLOT_NAMES = {511: "CPT", 512: "FLEX", 66: "QB", 67: "RB", 68: "WR", 69: "TE", 70: "FLEX", 71: "DST"}
CPT_SLOT = 511
CPT_MULTIPLIER = 1.5
TOP_N = 10


# ----------------------------------------------------------------------------- helpers
def _f(x) -> Optional[float]:
    """float or None, never NaN/inf (JSON has no spelling for those)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _stats(x, quantiles=(10, 50, 90)) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    out = {"n": int(x.size), "min": _f(x.min()), "mean": _f(x.mean()), "max": _f(x.max()), "std": _f(x.std())}
    for q, v in zip(quantiles, np.percentile(x, quantiles)):
        out[f"p{q}"] = _f(v)
    return out


def jsonable(obj):
    """Recursively turn numpy scalars/arrays, tuples and NaN into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _f(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sanitize_contest_name(name: str) -> str:
    """Contest name as a filesystem-safe directory name."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = re.sub(r"_+", "_", safe)
    return safe[:80].strip("_")


def player_meta(session: Session, pids) -> Dict[int, dict]:
    """Display data for the players the report mentions, keyed by DK draftable id."""
    pids = [int(p) for p in pids]
    meta: Dict[int, dict] = {}
    for i in range(0, len(pids), 900):
        for r in session.query(NFLPlayer).filter(NFLPlayer.id.in_(pids[i:i + 900])).all():
            meta[int(r.id)] = {
                "id": int(r.id), "name": r.name, "team": r.team, "opponent": r.opponent, "pos": r.position,
                "roster_slot": r.roster_slot_id, "slot": SLOT_NAMES.get(r.roster_slot_id, str(r.roster_slot_id)),
                "depth_slot": r.slot_label, "salary": r.salary,
                "fpts_mean": _f(r.fpts_mean), "fpts_std": _f(r.fpts_std),
                "ownership_proj": _f(r.ownership_proj),
            }
    return meta


def _label(meta: Dict[int, dict], pid: int, with_slot: bool = True) -> str:
    m = meta.get(pid)
    if not m:
        return f"#{pid}"
    return f"{m['name']} ({m['slot']})" if with_slot and m["roster_slot"] in (511, 512) else m["name"]


# ----------------------------------------------------------------------------- sections
def field_section(counts_by_lineup: Dict[int, int], pid_lists_by_lid: Dict[int, List[int]],
                  meta: Dict[int, dict], sampler_meta: dict, n_candidates: int) -> dict:
    """Duplication profile and player exposures of the sampled field, next to projected ownership."""
    u_lids = sorted(lid for lid, c in counts_by_lineup.items() if c > 0)
    dupes = np.array([int(counts_by_lineup[l]) for l in u_lids], dtype=np.int64)
    M = int(dupes.sum())
    unique = int(len(u_lids))
    solo = int((dupes == 1).sum())

    # histogram: 1..10 exact, then ranges
    edges = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 15), (16, 25), (26, 50), (51, 100), (101, 250), (251, None)]
    hist = []
    for lo, hi in edges:
        sel = (dupes >= lo) & ((dupes <= hi) if hi else True)
        if sel.any() or hi:
            hist.append({"label": f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi else f"{lo}+"),
                         "lineups": int(sel.sum()), "entries": int(dupes[sel].sum())})
    while hist and hist[-1]["lineups"] == 0:
        hist.pop()

    # exposures, dupe-weighted (an entry counts once per copy in the field)
    counts: Dict[int, int] = {}
    for lid, c in zip(u_lids, dupes):
        for pid in pid_lists_by_lid.get(lid, []):
            counts[pid] = counts.get(pid, 0) + int(c)
    rows = []
    for pid, cnt in counts.items():
        m = meta.get(pid, {})
        own = m.get("ownership_proj")
        exp_pct = 100.0 * cnt / max(1, M)
        rows.append({"id": pid, "name": m.get("name", f"#{pid}"), "team": m.get("team"), "pos": m.get("pos"),
                     "slot": m.get("slot"), "depth_slot": m.get("depth_slot"), "salary": m.get("salary"),
                     "fpts_mean": m.get("fpts_mean"), "count": cnt, "exposure_pct": _f(exp_pct),
                     "ownership_pct": own, "delta_pct": _f(exp_pct - own) if own is not None else None})
    rows.sort(key=lambda r: (-r["exposure_pct"], r["name"]))

    with_own = [r for r in rows if r["ownership_pct"] is not None]
    calib = {"players": len(with_own)}
    if len(with_own) >= 3:
        e = np.array([r["exposure_pct"] for r in with_own]); o = np.array([r["ownership_pct"] for r in with_own])
        calib.update(corr=_f(np.corrcoef(e, o)[0, 1]) if e.std() > 0 and o.std() > 0 else None,
                     mean_abs_diff_pct=_f(np.abs(e - o).mean()), max_abs_diff_pct=_f(np.abs(e - o).max()),
                     exposure_sum_pct=_f(e.sum()), ownership_sum_pct=_f(o.sum()))

    top = []
    for lid, c in sorted(zip(u_lids, dupes), key=lambda t: -t[1])[:TOP_N]:
        pids = pid_lists_by_lid.get(lid, [])
        top.append({"lineup_id": int(lid), "dupes": int(c),
                    "salary": int(sum((meta.get(p, {}).get("salary") or 0) for p in pids)),
                    "players": [_label(meta, p) for p in pids]})

    q = np.percentile(dupes, [50, 75, 90, 95, 99]) if dupes.size else [0] * 5
    return {
        "entries": M, "candidates": int(n_candidates), "unique": unique, "unique_share": _f(unique / max(1, M)),
        "candidates_scored_share": _f(unique / max(1, n_candidates)),
        "solo": solo, "dupe_rate": _f(1.0 - unique / max(1, M)), "max_dupes": int(dupes.max()) if dupes.size else 0,
        "dupe_quantiles": {"p50": _f(q[0]), "p75": _f(q[1]), "p90": _f(q[2]), "p95": _f(q[3]), "p99": _f(q[4])},
        "dupe_hist": hist, "top_lineups": top, "exposures": rows, "ownership_calibration": calib,
        "sampler": jsonable(sampler_meta),
    }


def _compress_ranks(scaled: Dict[int, float]) -> List[dict]:
    """{rank: amount} -> tiers of consecutive ranks paying the same cents; amounts keep full precision in `pool`."""
    tiers: List[dict] = []
    for r in sorted(scaled):
        a = float(scaled[r])
        if tiers and round(tiers[-1]["amount"], 2) == round(a, 2) and tiers[-1]["max"] == r - 1:
            tiers[-1]["max"] = r
            tiers[-1]["pool"] += a
        else:
            tiers.append({"min": r, "max": r, "amount": round(a, 2), "pool": a})
    return tiers


def _prize_curve(pay_by_rank: np.ndarray, n: int, pool: float, tops=(0.001, 0.01, 0.05, 0.10, 0.20)) -> Dict[str, float]:
    """Share of the pool paid to the top x% of the field (x% of n places, at least one)."""
    cum = np.cumsum(pay_by_rank)
    out = {}
    for t in tops:
        k = max(1, int(math.ceil(t * n)))
        out[f"{t * 100:g}"] = _f(cum[min(k, len(cum)) - 1] / pool) if pool and len(cum) else None
    return out


def payout_section(orig_tiers: List[dict], scaled: Dict[int, float], entry_fee: float,
                   field_size: int, simulated_size: int) -> dict:
    """Original DK payout table next to the table the simulator actually paid from."""
    def by_rank(tiers: List[dict], n: int) -> np.ndarray:
        arr = np.zeros(n, dtype=float)
        for t in tiers:
            arr[t["min"] - 1:min(t["max"], n)] = t["amount"]
        return arr

    def summarise(tiers: List[dict], n: int, pool: float) -> dict:
        paid = max((t["max"] for t in tiers), default=0)
        first = next((t["amount"] for t in tiers if t["min"] == 1), 0.0)
        return {"field_size": n, "entry_fee": _f(entry_fee), "prize_pool": _f(pool), "buy_ins": _f(entry_fee * n),
                "rake": _f(1 - pool / (entry_fee * n)) if entry_fee and n else None,
                "paid_places": int(paid), "paid_share": _f(paid / n) if n else None,
                "first_place": _f(first), "first_share": _f(first / pool) if pool else None,
                "min_cash": _f(min((t["amount"] for t in tiers), default=0)), "tiers": tiers}

    orig_pool = sum((t["max"] - t["min"] + 1) * t["amount"] for t in orig_tiers)
    scaled_pool = float(sum(scaled.values()))
    scaled_tiers = _compress_ranks(scaled)
    orig = summarise(orig_tiers, field_size, orig_pool)
    sc = summarise(scaled_tiers, simulated_size, scaled_pool)
    orig["curve"] = _prize_curve(by_rank(orig_tiers, field_size), field_size, orig_pool)
    scaled_by_rank = np.zeros(simulated_size)
    for r, a in scaled.items():
        if 1 <= r <= simulated_size:
            scaled_by_rank[r - 1] = a
    sc["curve"] = _prize_curve(scaled_by_rank, simulated_size, scaled_pool)

    ranks = sorted(scaled)
    contiguous = bool(ranks and ranks[0] == 1 and ranks == list(range(1, len(ranks) + 1)))
    target_pool = orig_pool * simulated_size / field_size if field_size else 0.0
    return {
        "original": orig, "scaled": sc,
        "target_pool": _f(target_pool),
        "pool_error": _f((scaled_pool - target_pool) / target_pool) if target_pool else None,
        "real_places_per_rank": _f(field_size / simulated_size) if simulated_size else None,
        "first_place_represents": int(math.ceil(field_size / simulated_size)) if simulated_size else 0,
        "curve_max_diff": _f(max(abs((sc["curve"][k] or 0) - (orig["curve"][k] or 0)) for k in orig["curve"])) if orig["curve"] else None,
        "ranks_contiguous": contiguous, "max_rank": int(max(ranks)) if ranks else 0,
        "ranks_beyond_field": int(sum(1 for r in ranks if r > simulated_size)),
    }


def simulation_section(full_scores: np.ndarray, full_payouts: np.ndarray, full_rois: np.ndarray, full_ranks: np.ndarray,
                       starts: np.ndarray, ends: np.ndarray, u_lids: List[int], pid_lists_by_lid: Dict[int, List[int]],
                       meta: Dict[int, dict], entry_fee: float, pool_in_field: float, paid_places: int,
                       chunk_stats: dict, seconds: float, missing_sample_pids: List[int]) -> dict:
    """Invariants over the simulated outcomes and the best candidate lineups."""
    n_sims, M = full_scores.shape
    expected_roi = pool_in_field / (M * entry_fee) - 1 if M and entry_fee else None
    sorted_scores = -np.sort(-full_scores, axis=1)            # desc per sim
    winning = sorted_scores[:, 0]
    cash_line = sorted_scores[:, min(max(paid_places, 1), M) - 1] if M else np.array([])

    # per unique lineup, aggregated over its copies in the field (what gets persisted)
    per: List[dict] = []
    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        pay = full_payouts[:, a:b]
        per.append({"lineup_id": int(lid), "dupes": b - a,
                    "cash_pct": _f((pay[:, 0] > 0).mean()),
                    "avg_roi": _f(full_rois[:, a:b].mean()),
                    "mean_rank": _f(full_ranks[:, a:b].mean()),
                    "mean_score": _f(full_scores[:, a:b].mean()),
                    "win_pct": _f((full_ranks[:, a:b] == 1).any(axis=1).mean())})
    for r in per:
        r["players"] = [_label(meta, p) for p in pid_lists_by_lid.get(r["lineup_id"], [])]
    rois = np.array([r["avg_roi"] for r in per], dtype=float)
    cash = np.array([r["cash_pct"] for r in per], dtype=float)

    return {
        "n_sims": int(n_sims), "entries": int(M), "seconds": _f(seconds), "workers": chunk_stats.get("workers"),
        "conservation": {"expected_pool": _f(pool_in_field), "max_abs_error": _f(chunk_stats.get("max_abs_error", 0.0)),
                         "sims_off": int(chunk_stats.get("sims_off", 0))},
        "tie_anomalies": int(chunk_stats.get("tie_anomalies", 0)),
        "payouts_above_top_prize": int(chunk_stats.get("over_top", 0)),
        "field_roi": _f(full_rois.mean()), "expected_field_roi": _f(expected_roi),
        "entry_score": _stats(full_scores.ravel()),
        "winning_score": _stats(winning), "cash_line_score": _stats(cash_line), "paid_places": int(paid_places),
        "lineup_roi": _stats(rois), "lineup_cash_pct": _stats(cash),
        "lineups_positive_roi": int((rois > 0).sum()),
        "top_by_roi": sorted(per, key=lambda r: -(r["avg_roi"] or -9))[:TOP_N],
        "top_by_cash": sorted(per, key=lambda r: -(r["cash_pct"] or 0))[:TOP_N],
        "players_without_samples": [{"id": p, "name": meta.get(p, {}).get("name", f"#{p}")} for p in missing_sample_pids],
    }


def correlation_section(raw: Dict[int, np.ndarray], diag: dict, meta: Dict[int, dict]) -> dict:
    """Compare what the correlated sampler drew against the targets it was given."""
    n = int(diag.get("n_samples") or 0)
    randomness = float(diag.get("randomness", 1.0))
    se = 1.0 / math.sqrt(n) if n > 1 else None
    games_out = []
    all_err: List[float] = []
    all_z: List[float] = []
    all_std_ratio: List[float] = []
    n_correlated = 0

    for g in diag.get("games", []):
        ids = [int(p) for p in g["player_ids"]]
        n_correlated += len(ids)
        X = np.column_stack([np.asarray(raw[p], dtype=float) for p in ids])
        with np.errstate(invalid="ignore", divide="ignore"):
            emp = np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
        emp = np.nan_to_num(emp, nan=0.0)
        target = np.asarray(g["corr"], dtype=float)
        requested = np.asarray(g["requested_corr"], dtype=float)

        # players: expected moments vs sampled
        players = []
        for j, pid in enumerate(ids):
            m = meta.get(pid, {})
            mult = CPT_MULTIPLIER if m.get("roster_slot") == CPT_SLOT else 1.0
            exp_mean, exp_std = g["mu"][j] * mult, g["sigma"][j] * randomness * mult
            s_mean, s_std = float(X[:, j].mean()), float(X[:, j].std(ddof=1)) if n > 1 else 0.0
            z = (s_mean - exp_mean) / (exp_std / math.sqrt(n)) if exp_std > 0 and n > 1 else 0.0
            ratio = s_std / exp_std if exp_std > 0 else None
            all_z.append(abs(z))
            if ratio is not None:
                all_std_ratio.append(ratio)
            players.append({"id": pid, "name": m.get("name", f"#{pid}"), "team": m.get("team"), "slot": m.get("slot"),
                            "depth_slot": g["slots"][j], "expected_mean": _f(exp_mean), "sample_mean": _f(s_mean),
                            "expected_std": _f(exp_std), "sample_std": _f(s_std), "mean_z": _f(z), "std_ratio": _f(ratio)})

        # one column per person for the matrix: showdown has CPT and FLEX rows of the same player
        keep: List[int] = []
        first_by_name: Dict[str, int] = {}
        for j, pid in enumerate(ids):
            nm = meta.get(pid, {}).get("name", str(pid))
            if nm not in first_by_name:
                first_by_name[nm] = j
                keep.append(j)
            elif meta.get(pid, {}).get("roster_slot") != CPT_SLOT and meta.get(ids[first_by_name[nm]], {}).get("roster_slot") == CPT_SLOT:
                keep[keep.index(first_by_name[nm])] = j
                first_by_name[nm] = j
        keep.sort()
        emp_c, tgt_c = emp[np.ix_(keep, keep)], target[np.ix_(keep, keep)]
        k = len(keep)
        off = ~np.eye(k, dtype=bool)
        err = np.abs(emp_c - tgt_c)
        # DST rows built from the opponent's drawn score carry their dependence structurally; the
        # requested matrix is not their target, so they stay out of the error statistics
        structural = {int(x) for x in g.get("structural_ids", [])}
        is_struct = np.array([ids[j] in structural for j in keep])
        scored = off & ~is_struct[:, None] & ~is_struct[None, :]
        all_err.extend(err[scored].tolist())
        labels = []
        for j in keep:
            m = meta.get(ids[j], {})
            labels.append({"id": ids[j], "name": m.get("name", f"#{ids[j]}"), "team": m.get("team"), "depth_slot": g["slots"][j]})

        pairs = []
        iu = np.triu_indices(k, 1)
        for a, b in zip(*iu):
            st_pair = bool(is_struct[a] or is_struct[b])
            pairs.append({"a": labels[a]["name"], "a_team": labels[a]["team"], "a_slot": labels[a]["depth_slot"],
                          "b": labels[b]["name"], "b_team": labels[b]["team"], "b_slot": labels[b]["depth_slot"],
                          "target": _f(tgt_c[a, b]), "sampled": _f(emp_c[a, b]),
                          "error": None if st_pair else _f(err[a, b]), "structural": st_pair})
        strongest = sorted(pairs, key=lambda p: -abs(p["sampled"] or 0))[:TOP_N]
        worst = sorted([p for p in pairs if not p["structural"]], key=lambda p: -(p["error"] or 0))[:TOP_N]
        # different people sharing a team and a depth slot (pooled deep players)
        pooled = [p for p in pairs if p["a_team"] == p["b_team"] and p["a_slot"] == p["b_slot"]]

        games_out.append({
            "teams": g["teams"], "sources": g.get("sources", {}), "n_players": len(ids), "n_people": k,
            "spd_adjustment_max": _f(np.abs(target - requested).max()) if target.size else 0.0,
            "max_abs_error": _f(err[scored].max()) if scored.any() else 0.0,
            "mean_abs_error": _f(err[scored].mean()) if scored.any() else 0.0,
            "players": players, "labels": labels,
            "latent": g.get("latent"), "n_structural": int(len(structural)),
            "target": np.round(tgt_c, 3).tolist(), "sampled": np.round(emp_c, 3).tolist(),
            "strongest_pairs": strongest, "largest_errors": worst,
            "pooled_pairs": sorted(pooled, key=lambda p: -abs(p["sampled"] or 0)),
        })

    # CPT row must be exactly 1.5x the FLEX row of the same player
    by_name: Dict[str, dict] = {}
    for pid, s in raw.items():
        m = meta.get(pid)
        if not m or m.get("roster_slot") not in (511, 512):
            continue
        by_name.setdefault(m["name"], {})[m["roster_slot"]] = np.asarray(s, dtype=float)
    cpt_flex = {"pairs": 0}
    ratios, corrs, off_names = [], [], []
    for nm, d in by_name.items():
        if 511 in d and 512 in d and d[512].std() > 0:
            ratio = float(np.median(d[511] / np.where(d[512] == 0, np.nan, d[512]))) if np.any(d[512] != 0) else float("nan")
            corr = float(np.corrcoef(d[511], d[512])[0, 1])
            ratios.append(ratio)
            corrs.append(corr)
            if not (corr > 0.999 and math.isfinite(ratio) and abs(ratio - CPT_MULTIPLIER) < 1e-3):
                off_names.append(nm)
    if corrs:
        ratios = [r for r in ratios if math.isfinite(r)]
        cpt_flex = {"pairs": len(corrs), "min_corr": _f(min(corrs)),
                    "ratio_min": _f(min(ratios)) if ratios else None, "ratio_max": _f(max(ratios)) if ratios else None,
                    "off": off_names}
    pooled_all = [p for g in games_out for p in g["pooled_pairs"]]

    independent = []
    for d in diag.get("independent", []):
        m = meta.get(int(d["id"]), {})
        independent.append({"id": int(d["id"]), "name": m.get("name", f"#{d['id']}"), "team": m.get("team"),
                            "slot": m.get("slot"), "depth_slot": m.get("depth_slot"), "reason": d["reason"]})

    return {
        "n_samples": n, "randomness": randomness, "method": diag.get("method"), "standard_error": _f(se),
        "n_correlated": n_correlated, "n_independent": len(independent), "independent": independent,
        "games": games_out,
        "max_abs_error": _f(max(all_err)) if all_err else None, "mean_abs_error": _f(np.mean(all_err)) if all_err else None,
        "max_mean_z": _f(max(all_z)) if all_z else None,
        "std_ratio_min": _f(min(all_std_ratio)) if all_std_ratio else None,
        "std_ratio_max": _f(max(all_std_ratio)) if all_std_ratio else None,
        "cpt_flex": cpt_flex,
        "pooled": {"pairs": len(pooled_all),
                   "max_sampled": _f(max(abs(p["sampled"] or 0) for p in pooled_all)) if pooled_all else None,
                   "max_target": _f(max(abs(p["target"] or 0) for p in pooled_all)) if pooled_all else None},
    }


# ----------------------------------------------------------------------------- checks
def build_checks(report: dict) -> List[dict]:
    """Pass/warn/fail verdicts a person can scan. Each: {status, name, detail}."""
    checks: List[dict] = []

    def add(status: str, name: str, detail: str):
        checks.append({"status": status, "name": name, "detail": detail})

    c, f, p, s, r = (report.get(k, {}) for k in ("contest", "field", "payouts", "simulation", "correlation"))

    # --- field
    if f:
        add("ok" if f["entries"] == c.get("simulated_size") else "bad", "Field size",
            f"{f['entries']:,} entries sampled, {c.get('simulated_size', 0):,} requested")
        add("ok", "Field coverage",
            f"{f['unique']:,} unique lineups ({(f['unique_share'] or 0):.1%} of the field) from {f['candidates']:,} candidates; "
            f"only these {f['unique']:,} get simulation results and can be chosen")
        own = f.get("sampler", {}).get("ownership", {})
        src = own.get("source")
        if src == "projected":
            add("ok", "Ownership input", f"projected ownership on {f['ownership_calibration'].get('players', 0)} players, "
                                          f"mean {own.get('mean', 0):.1f}%")
        else:
            add("warn", "Ownership input", "no projected ownership on this slate; the field was built from a heuristic")
        cal = f.get("ownership_calibration", {})
        if cal.get("corr") is not None:
            st = "ok" if cal["corr"] >= 0.8 else ("warn" if cal["corr"] >= 0.5 else "bad")
            add(st, "Field exposure vs projected ownership",
                f"correlation {cal['corr']:.2f}, mean |difference| {cal['mean_abs_diff_pct']:.1f} pts, "
                f"largest {cal['max_abs_diff_pct']:.1f} pts")

    # --- payouts
    if p:
        o, sc = p["original"], p["scaled"]
        pe = abs(p["pool_error"] or 0)
        add("ok" if pe <= 0.005 else ("warn" if pe <= 0.02 else "bad"), "Prize pool scaled",
            f"${o['prize_pool']:,.0f} for {o['field_size']:,} entries -> ${sc['prize_pool']:,.2f} for {sc['field_size']:,} "
            f"(target ${p['target_pool']:,.2f}, off by {pe:.2%}); rake {(o['rake'] or 0):.1%} -> {(sc['rake'] or 0):.1%}")
        d = abs((sc["paid_share"] or 0) - (o["paid_share"] or 0)) * 100
        add("ok" if d <= 1 else ("warn" if d <= 3 else "bad"), "Paid places preserved",
            f"{o['paid_places']:,} of {o['field_size']:,} ({(o['paid_share'] or 0):.1%}) -> "
            f"{sc['paid_places']:,} of {sc['field_size']:,} ({(sc['paid_share'] or 0):.1%})")
        d = (p["curve_max_diff"] or 0) * 100
        curve = ", ".join(f"top {k}%: {(o['curve'][k] or 0):.1%} -> {(sc['curve'][k] or 0):.1%}" for k in ("1", "10", "20") if k in o["curve"])
        add("ok" if d <= 2 else ("warn" if d <= 5 else "bad"), "Prize curve preserved",
            f"share of the pool paid to the {curve} (largest difference {d:.1f} pts); "
            f"first place ${sc['first_place']:,.2f} is the average of the real top {p['first_place_represents']} prizes")
        add("ok" if p["ranks_contiguous"] and not p["ranks_beyond_field"] else "bad", "Scaled payout ranks",
            f"ranks 1..{p['max_rank']:,} {'contiguous' if p['ranks_contiguous'] else 'with gaps or overlaps'}"
            + (f", {p['ranks_beyond_field']} beyond the field" if p["ranks_beyond_field"] else ""))

    # --- simulation
    if s:
        cons = s["conservation"]
        add("ok" if (cons["max_abs_error"] or 0) <= 0.01 else "bad", "Payout conservation",
            f"every simulation pays out ${cons['expected_pool']:,.2f}; largest deviation ${cons['max_abs_error'] or 0:.4f}"
            + (f" in {cons['sims_off']} sims" if cons["sims_off"] else ""))
        add("ok" if s["tie_anomalies"] == 0 else "bad", "Tie handling",
            "tied entries split their pooled prizes" if s["tie_anomalies"] == 0 else f"{s['tie_anomalies']} tie groups paid outside their range")
        add("ok" if s["payouts_above_top_prize"] == 0 else "bad", "Top prize bound",
            "no entry paid more than first place" if s["payouts_above_top_prize"] == 0 else f"{s['payouts_above_top_prize']} sims paid above first place")
        if s["expected_field_roi"] is not None:
            d = abs((s["field_roi"] or 0) - s["expected_field_roi"])
            add("ok" if d <= 0.001 else "bad", "Field ROI equals the rake",
                f"mean ROI across the whole field {s['field_roi']:+.2%}, expected {s['expected_field_roi']:+.2%}")
        if s["players_without_samples"]:
            names = ", ".join(x["name"] for x in s["players_without_samples"][:8])
            add("bad", "Players without samples", f"{len(s['players_without_samples'])} players in the field have no projection "
                                                  f"(removed from the slate after lineups were generated?): {names}")
        else:
            add("ok", "Player samples", "every player in the field has a sampled score")

    # --- correlation
    if r:
        n_c, n_i = r["n_correlated"], r["n_independent"]
        if n_c and not n_i:
            add("ok", "Correlation applied", f"all {n_c} player rows sampled inside {len(r['games'])} game-level covariance(s)")
        elif n_c:
            reasons = {}
            for x in r["independent"]:
                reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
            add("warn", "Correlation applied", f"{n_c} rows correlated, {n_i} sampled independently ("
                + "; ".join(f"{v} {k}" for k, v in reasons.items()) + ")")
        else:
            add("bad", "Correlation applied", f"no player was correlated: all {n_i} rows sampled independently "
                                              "(slot labels missing? run ingest again)")
        for g in r["games"]:
            src = g.get("sources", {})
            teams = "-".join(g["teams"])
            identity = [t for t, v in src.items() if v == "identity"]
            if identity:
                add("warn", f"Within-team correlation {teams}",
                    f"no history in nfl_stats for {', '.join(identity)}: identity matrix (players on that team sampled independently of each other)")
            if src.get("cross") == "default":
                add("info", f"Cross-team correlation {teams}",
                    "neither team has 10 games with an opponent line in history: default cross-team assumptions (+0.15 offence-offence, -0.20 offence vs opposing DST)")
            lat = g.get("latent")
            if lat:
                dst_rho = [p for p in g["players"] if p.get("depth_slot") == "DST"]
                rho_txt = ""
                if lat.get("corr"):
                    rhos = []
                    for p in dst_rho:
                        c = lat["corr"].get(str(p["id"])) or lat["corr"].get(p["id"])
                        if c:
                            rhos.append(f"{p['team']} DST vs opponent points {c['opp_pts']:+.2f}")
                    rho_txt = "; " + ", ".join(dict.fromkeys(rhos)) if rhos else ""
                scores = ", ".join(f"{t} {lat['implied'][t]:.1f} implied -> {lat['points_mean'][t]:.1f} +- {lat['points_sd'][t]:.1f}"
                                   for t in g["teams"] if t in lat.get("implied", {}))
                add("ok", f"Game score latent {teams}",
                    f"each universe draws both scores from the lines (total {lat['lines']['total']}); {scores}{rho_txt}; "
                    f"DST = points-allowed bracket of that score + takeaways")
            else:
                add("warn", f"Game score latent {teams}",
                    "no game lines or no DST model: DST sampled from its fitted distribution, not from the opponent's score")
        n = r.get("n_samples") or 0
        if r.get("max_abs_error") is not None and r.get("standard_error"):
            se = r["standard_error"]
            e = r["max_abs_error"]
            add("ok" if e <= 4 * se + 0.01 else ("warn" if e <= 8 * se + 0.02 else "bad"), "Sampled correlations match targets",
                f"largest |sampled - target| {e:.3f}, mean {r['mean_abs_error']:.3f}, over {n} samples "
                f"(sampling noise ~{se:.3f})")
        if r.get("max_mean_z") is not None:
            z = r["max_mean_z"]
            add("ok" if z <= 4 else ("warn" if z <= 6 else "bad"), "Sampled means match projections",
                f"largest deviation {z:.1f} standard errors from the projected mean")
        if r.get("std_ratio_min") is not None and n > 1:
            lo, hi = r["std_ratio_min"], r["std_ratio_max"]
            se_sd = 1.0 / math.sqrt(2 * n)
            dev = max(abs(lo - 1), abs(hi - 1))
            add("ok" if dev <= 4 * se_sd + 0.005 else ("warn" if dev <= 8 * se_sd + 0.01 else "bad"), "Sampled spreads match projections",
                f"sampled/expected standard deviation between {lo:.3f} and {hi:.3f} (noise ~{se_sd:.3f}, randomness {r['randomness']:.2f})")
        pooled = r.get("pooled", {})
        if pooled.get("pairs"):
            m = pooled["max_sampled"] or 0
            add("ok" if m < 0.9 else "bad", "Pooled depth-chart players",
                f"{pooled['pairs']} pairs of different players share a team and depth slot; "
                f"largest correlation between them {m:.2f} (target up to {(pooled['max_target'] or 0):.2f})")
        cf = r.get("cpt_flex", {})
        if cf.get("pairs"):
            off = cf.get("off") or []
            add("ok" if not off else "bad", "Captain = 1.5 x FLEX",
                f"{cf['pairs']} players: CPT/FLEX correlation >= {cf['min_corr']:.4f}, ratio {cf['ratio_min']:.3f}-{cf['ratio_max']:.3f}"
                + (f"; off for {', '.join(off[:6])}" + (" ..." if len(off) > 6 else "") + " (captain row fitted separately; re-run ingest)" if off else ""))

    return checks


def summarise_checks(checks: List[dict]) -> dict:
    return {k: sum(1 for c in checks if c["status"] == k) for k in ("ok", "warn", "bad", "info")}


# ----------------------------------------------------------------------------- output
def format_summary(report: dict) -> str:
    """A few lines per contest for the terminal / job log."""
    c, f, p, s, r = (report.get(k, {}) for k in ("contest", "field", "payouts", "simulation", "correlation"))
    lines = [f"[{c['id']}] {c['name']}",
             f"  {c['mode']} · ${c['entry_fee']:.2f} · field {c['total_entries']:,} -> {c['simulated_size']:,} simulated · "
             f"{c['num_sims']} sims · {f.get('candidates', 0):,} candidates · {c.get('seconds', 0):.0f}s"]
    if f:
        q = f["dupe_quantiles"]
        cal = f.get("ownership_calibration", {})
        lines.append(f"  field:    {f['unique']:,} unique ({(f['unique_share'] or 0):.1%}), dupe rate {(f['dupe_rate'] or 0):.1%}, "
                     f"max dupes {f['max_dupes']}, p50/p90/p99 = {q['p50']:.0f}/{q['p90']:.0f}/{q['p99']:.0f}"
                     + (f"; exposure vs ownership corr {cal['corr']:.2f}, mean |diff| {cal['mean_abs_diff_pct']:.1f} pts"
                        if cal.get("corr") is not None else ""))
    if p:
        o, sc = p["original"], p["scaled"]
        lines.append(f"  payouts:  pool ${o['prize_pool']:,.0f} -> ${sc['prize_pool']:,.2f} (rake {(o['rake'] or 0):.1%}), "
                     f"pays {o['paid_places']:,}/{o['field_size']:,} ({(o['paid_share'] or 0):.1%}) -> "
                     f"{sc['paid_places']:,}/{sc['field_size']:,} ({(sc['paid_share'] or 0):.1%}), "
                     f"1st ${o['first_place']:,.0f} -> ${sc['first_place']:,.2f} (avg of real top {p['first_place_represents']})")
    if s:
        lines.append(f"  sim:      payout error ${s['conservation']['max_abs_error'] or 0:.4f}, field ROI {s['field_roi']:+.1%} "
                     f"(expected {s['expected_field_roi']:+.1%}), winning score p50 {s['winning_score'].get('p50', 0):.1f}, "
                     f"cash line p50 {s['cash_line_score'].get('p50', 0):.1f}, best lineup ROI {s['lineup_roi'].get('max', 0):+.1%}")
    if r:
        srcs = "; ".join(f"{'-'.join(g['teams'])}: " + ", ".join(f"{t} {v}" for t, v in g.get("sources", {}).items())
                         for g in r["games"][:3])
        lines.append(f"  corr:     {r['n_correlated']} rows correlated, {r['n_independent']} independent; "
                     + (f"max |dRho| {r['max_abs_error']:.3f} (noise ~{r['standard_error']:.3f})" if r.get("max_abs_error") is not None else "n/a")
                     + (f"; CPT/FLEX ratio {r['cpt_flex']['ratio_min']:.3f}" if r.get("cpt_flex", {}).get("pairs") else "")
                     + (f" · {srcs}" if srcs else ""))
    checks = report.get("checks", [])
    if checks:
        n = summarise_checks(checks)
        lines.append(f"  checks:   {n['ok']} ok · {n['warn']} warn · {n['bad']} bad" + (f" · {n['info']} note" if n.get("info") else ""))
        for ch in checks:
            if ch["status"] in ("warn", "bad"):
                lines.append(f"    {'!!' if ch['status'] == 'bad' else ' !'} {ch['name']}: {ch['detail']}")
    return "\n".join(lines)


def save_report(session: Session, report: dict, diag_dir: str | Path = DIAG_DIR) -> Path:
    """Upsert the contest's row and write report.json next to the old diagnostics location."""
    report = jsonable(report)
    cid, name = int(report["contest"]["id"]), report["contest"]["name"]
    row = session.get(ContestSimReport, cid)
    if row is None:
        row = ContestSimReport(contest_id=cid, contest_name=name, report=report)
        session.add(row)
    else:
        row.contest_name, row.report, row.created_at = name, report, datetime.utcnow()
    session.flush()
    out = Path(diag_dir) / f"contest_{cid}_{sanitize_contest_name(name)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return out / "report.json"


def load_reports(session: Session) -> List[ContestSimReport]:
    return session.query(ContestSimReport).order_by(ContestSimReport.contest_name).all()
