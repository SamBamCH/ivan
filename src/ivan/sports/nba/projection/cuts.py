# src/ivan/sports/nba/projection/cuts.py
"""
Who plays tonight: the pool is soft.

The desk's mapping names the rotation (the pool: who plays, and the grid row each plays). Reality is softer on both sides:
a healthy bench player is cut to under CUT_MIN minutes one night in four (one in seven a straight DNP), and a player the
mapping left at zero plays one night in three for a few minutes. Measured on the 2025-26 box scores (every roster row, DNP
= 0), a player's own recent cut history is the signal: bench players (trailing 12-22 minutes) cut 0 of their last 10 roster
games are cut again 10% of the time, 1-2 → 19%, 3-4 → 39%, 5+ → 53%; the deep bench 43-67%; starters 4%
(docs/nba/notes/minutes_distribution_2026-11-10.md).

    fit(box)                         the table from a box-score frame (history.load_box_scores): per role (planned minutes level)
                                     and own-cut bin, P(cut), P(DNP | cut), the token minutes when cut but used; the fringe profile
                                     (planned zero, available): P(plays) and the minutes when he does, by own-cut bin
    fringe_from_tips(session, box)   the fringe profile from its own population: the zero-minute players of every tip record
    load_table() / save_table()      data/nba/models/cuts.json
    cut_probability(table, planned, own_share, *, starter, held, status)   the chance tonight's rotation does not hold for him
    dnp_share(table, planned, own_share)                                   of his cut nights, the share with no minutes at all
    fringe_profile(table, own_share)                                       (p_play, mean_minutes, sd_minutes) for a zero-minute player
    cuts_mode()                      config gamesim.cuts: "all" (default: every row but a named starter carries his rate; a typed pin or a
                                     locked row says how long he plays when he plays, not whether the coach uses him), "unpinned" (a
                                     typed minutes pin is certain too), "off" (the pool is hard)

The mapping (projection.engine) writes p_cut / fringe_p / fringe_minutes per player; the World (gamesim.rotation.draw)
draws them: a cut world hands the row to the heirs (the DNP share) or keeps a token stint (the rest); a fringe player's
play-world gives him a short stint at the end of a half, the five on the floor giving it up.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ivan.sports.nba.paths import MODELS_DIR

TABLE_JSON = MODELS_DIR / "cuts.json"
CUT_MIN = 8.0                       # under this many minutes the night counts as a cut
WINDOW = 10                         # roster games the own share is read over
ROLES = (("deep", 0.0, 12.0), ("bench", 12.0, 22.0), ("starter", 22.0, 30.0), ("star", 30.0, 99.0))
BINS = (("0", 0.0, 0.0), ("1-2", 0.1, 0.2), ("3-4", 0.3, 0.4), ("5+", 0.5, 1.0))      # own cut share over the window
MIN_ROWS = 40                       # a cell with fewer rows takes its role's pooled value
# the fallback table when the file is missing: the 2025-26 fit of 2026-11-10, rounded
DEFAULT_TABLE = {
    "p_cut": {"deep": {"0": 0.33, "1-2": 0.49, "3-4": 0.43, "5+": 0.67}, "bench": {"0": 0.10, "1-2": 0.19, "3-4": 0.39, "5+": 0.53},
              "starter": {"0": 0.04, "1-2": 0.09, "3-4": 0.21, "5+": 0.50}, "star": {"0": 0.03, "1-2": 0.09, "3-4": 0.35, "5+": 0.50}},
    "p_dnp_given_cut": {"deep": {"0": 0.3, "1-2": 0.3, "3-4": 0.5, "5+": 0.58}, "bench": {"0": 0.5, "1-2": 0.58, "3-4": 0.59, "5+": 0.58},
                        "starter": {"0": 0.8, "1-2": 0.8, "3-4": 0.8, "5+": 0.8}, "star": {"0": 0.85, "1-2": 0.85, "3-4": 0.85, "5+": 0.85}},
    "token_minutes": 4.0,
    "fringe": {"0": {"p_play": 0.7, "mean": 9.0, "sd": 5.0}, "1-2": {"p_play": 0.6, "mean": 8.0, "sd": 5.0}, "3-4": {"p_play": 0.5, "mean": 7.0, "sd": 4.5},
               "5+": {"p_play": 0.5, "mean": 8.0, "sd": 5.0}, "10": {"p_play": 0.25, "mean": 6.0, "sd": 4.0}},
}


def role_of(planned: float) -> str:
    for name, lo, hi in ROLES:
        if planned < hi:
            return name
    return "star"


def bin_of(own_share: Optional[float]) -> str:
    """The own-cut bin from the share of his last WINDOW roster games under CUT_MIN minutes; None (no history) reads as 1-2."""
    if own_share is None or not np.isfinite(own_share):
        return "1-2"
    k = int(round(float(own_share) * WINDOW))
    return "0" if k <= 0 else "1-2" if k <= 2 else "3-4" if k <= 4 else "5+"


def cuts_mode() -> str:
    try:
        from ivan.core.config import cfg, load_config
        return str(cfg(load_config("nba"), "gamesim.cuts", "all"))
    except Exception:   # noqa: BLE001
        return "all"


def load_table(path: Path = TABLE_JSON) -> dict:
    t = json.loads(json.dumps(DEFAULT_TABLE))
    if Path(path).exists():
        try:
            t.update(json.loads(Path(path).read_text()))
        except Exception:   # noqa: BLE001 - a broken file is the defaults
            pass
    return t


def save_table(table: dict, path: Path = TABLE_JSON) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(table, indent=1))
    return Path(path)


def cut_probability(table: dict, planned: float, own_share: Optional[float], *, starter: bool = False, held: bool = False,
                    status: Optional[str] = None, mode: Optional[str] = None) -> float:
    """
    P(the rotation does not hold for him tonight: under CUT_MIN minutes, DNP included), beyond the injury scratch the status
    gives (team_minutes.scratch_probability). A named starter is the user's call and certain in every mode (the five are named,
    not guessed); a typed minutes pin (`held`) is certain under mode "unpinned" only: by default a pin, like a locked row, says how
    long he plays when he plays, and the cut rate says whether the coach uses him. "off" rates nobody.
    """
    mode = mode or cuts_mode()
    if mode == "off" or planned <= 0 or starter:
        return 0.0
    if mode == "unpinned" and held:
        return 0.0
    return float(table["p_cut"][role_of(planned)][bin_of(own_share)])


def dnp_share(table: dict, planned: float, own_share: Optional[float]) -> float:
    return float(table["p_dnp_given_cut"][role_of(planned)][bin_of(own_share)])


def token_minutes(table: dict) -> float:
    return float(table.get("token_minutes", 4.0))


def fringe_bin(own_share: Optional[float]) -> str:
    """The fringe's own bin: a player cut in every one of his last WINDOW games ("10", the DNP-CD regular) apart from the rest."""
    if own_share is not None and np.isfinite(own_share) and own_share >= 0.95:
        return "10"
    return bin_of(own_share)


def fringe_profile(table: dict, own_share: Optional[float]) -> Tuple[float, float, float]:
    """A player the mapping left at zero but available: (P(plays), minutes when he plays, their sd)."""
    f = table["fringe"].get(fringe_bin(own_share)) or table["fringe"][bin_of(own_share)]
    return float(f["p_play"]), float(f["mean"]), float(f["sd"])


# ----------------------------------------------------------------------------- the fringe from the tip records
def fringe_from_tips(session, box: pd.DataFrame, min_rows: int = 150) -> Optional[dict]:
    """
    The fringe profile from the population it is for: every player a tip record (backtest.tips, one per slate, the first tip)
    had at zero minutes and not out, against what he played that night (nba_stats). By own-cut bin as of the day: P(plays),
    the minutes when he does, their sd. None when fewer than min_rows such rows exist (the deep-role rows stand in).
    """
    from ivan.sports.nba import slates as S
    from ivan.sports.nba.backtest import tips as T
    from ivan.sports.nba.names import normalize_name
    from ivan.sports.nba.projection.history import build_history
    from ivan.sports.nba.projection.live import OUT_STATUSES
    rows = []
    box = box.copy(); box["key"] = box["player_name"].map(normalize_name)
    by_day = {d: g for d, g in box.groupby("game_date")}
    for dg in S.held():
        tips = T.tips_of(dg)
        if not tips:
            continue
        rec = T.read_tip(dg, tips[0])
        if not rec:
            continue
        day = tips[0].date()
        if day not in by_day:
            continue
        hist = build_history(box[box.game_date < day]); state = hist.before(day)
        played = by_day[day].drop_duplicates("key").set_index("key")["minutes"].fillna(0.0)
        for p in rec["players"]:
            if (p.get("minutes_proj") or 0) > 0 or str(p.get("status") or "").upper() in OUT_STATUSES or not p.get("team"):
                continue
            nm = normalize_name(str(p.get("name") or ""))
            own = float(state.loc[nm, "cut10"]) if nm in state.index and "cut10" in state.columns and not pd.isna(state.loc[nm, "cut10"]) else None
            rows.append((fringe_bin(own), float(played.get(nm, 0.0))))
    if len(rows) < min_rows:
        return None
    r = pd.DataFrame(rows, columns=["fbin", "act"])
    out = {"_source": f"{len(r)} zero-minute players across the tip records", "_p_play_all": round(float((r.act > 0).mean()), 3),
           "_mean_all": round(float(r[r.act > 0].act.mean()), 2) if (r.act > 0).any() else None}
    pooled = (float((r.act > 0).mean()), float(r[r.act > 0].act.mean()) if (r.act > 0).any() else 6.0, float(r[r.act > 0].act.std()) if (r.act > 0).sum() > 2 else 4.0)
    for b in [b for b, _, _ in BINS] + ["10"]:
        rb = r[r.fbin == b]; pl = rb[rb.act > 0]
        if len(rb) >= 25 and len(pl) >= 8:
            out[b] = dict(p_play=round(float((rb.act > 0).mean()), 3), mean=round(float(pl.act.mean()), 2), sd=round(float(pl.act.std()), 2), n=int(len(rb)))
        else:
            out[b] = dict(p_play=round(pooled[0], 3), mean=round(pooled[1], 2), sd=round(pooled[2], 2), n=int(len(rb)), pooled=True)
    return out


# ----------------------------------------------------------------------------- the fit
def fit(box: pd.DataFrame, window: int = WINDOW) -> dict:
    """
    From a box-score frame (player_name, team, game_date, game_id, minutes, played): for every roster game after a player's
    first `window` games with his team, the trailing-10 mean of the games he played (his role), the share of the last `window`
    roster games under CUT_MIN (his own bin) and what happened next: under CUT_MIN or not, zero or not, the minutes. Cells
    with under MIN_ROWS rows take the role's pooled value. The fringe profile is the deep role's rows: P(played) and the
    minutes when he did, by bin.
    """
    rows = []
    for (nm, tm), g in box.sort_values(["game_date", "game_id"]).groupby(["player_name", "team"], sort=False):
        m = g["minutes"].fillna(0.0).to_numpy(dtype=float)
        if len(m) <= window:
            continue
        for i in range(window, len(m)):
            w = m[i - window:i]; played = w[w > 0]
            if len(played) < 3:
                continue
            rows.append((played.mean(), float((w < CUT_MIN).mean()), m[i]))
    r = pd.DataFrame(rows, columns=["avg", "own", "act"])
    r["role"] = [role_of(a) for a in r["avg"]]; r["bin"] = [bin_of(o) for o in r["own"]]
    r["cut"] = r["act"] < CUT_MIN; r["dnp"] = r["act"] <= 0
    table = {"p_cut": {}, "p_dnp_given_cut": {}, "fringe": {}, "n": {}, "fitted_on": str(box["game_date"].max()) if len(box) else None, "rows": int(len(r))}
    for role, _, _ in ROLES:
        rr = r[r.role == role]
        pooled_cut = float(rr.cut.mean()) if len(rr) else DEFAULT_TABLE["p_cut"][role]["1-2"]
        pooled_dnp = float(rr[rr.cut].dnp.mean()) if rr.cut.any() else DEFAULT_TABLE["p_dnp_given_cut"][role]["1-2"]
        table["p_cut"][role], table["p_dnp_given_cut"][role], table["n"][role] = {}, {}, {}
        for b, _, _ in BINS:
            rb = rr[rr.bin == b]
            table["n"][role][b] = int(len(rb))
            table["p_cut"][role][b] = round(float(rb.cut.mean()) if len(rb) >= MIN_ROWS else pooled_cut, 3)
            cb = rb[rb.cut]
            table["p_dnp_given_cut"][role][b] = round(float(cb.dnp.mean()) if len(cb) >= MIN_ROWS else pooled_dnp, 3)
    tok = r[r.cut & ~r.dnp]
    table["token_minutes"] = round(float(tok.act.mean()), 2) if len(tok) else 4.0
    deep = r[r.role == "deep"]
    deep = deep.assign(fbin=[fringe_bin(o) for o in deep["own"]])
    for b in [b for b, _, _ in BINS] + ["10"]:
        db = deep[deep.fbin == b]; played = db[db.act > 0]
        if len(db) >= MIN_ROWS and len(played) >= 10:
            table["fringe"][b] = dict(p_play=round(float((db.act > 0).mean()), 3), mean=round(float(played.act.mean()), 2), sd=round(float(played.act.std()), 2), n=int(len(db)))
        else:
            table["fringe"][b] = dict(DEFAULT_TABLE["fringe"][b])
    return table
