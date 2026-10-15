# src/ivan/sports/nfl/dupe_analysis.py
"""
How the dupe model's predicted copies compare with the copies a real field held,
lineup by lineup, and where the misses cluster.

For every scraped contest (the settle job keeps its field under
data/nfl/derived/contests/<id>/lineups.parquet) field_lineups() builds one row
per unique lineup: its actual copies, the model's expected copies from the
ownership we projected before lock (what the pipeline uses) and, separately,
from the ownership the field actually showed (so a dupe miss can be told apart
from an ownership miss upstream), plus the features a miss could cluster on:
the captain, the salary left on the table, the team split, the QB stack, the
lineup's ownership sum. analyze() turns a set of contests into the calibration
table, the residual-by-feature tables and the most duplicated lineups that the
Dupes page and scripts/nfl/dupe_report.py show.

Residuals are log(actual / predicted): +0.69 means twice as duplicated as the
model said. Copy-weighted means weigh a lineup by its copies, which is what the
model is fitted on (the chalk) and what matters to a field sample; the
unweighted mean treats every unique lineup alike.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.core import archive
from ivan.core.utils import normalize_name
from ivan.database.session import read_sql
from ivan.sports.nfl import dupe_model as DM
from ivan.sports.nfl.dupe_model import DERIVED_DIR, MIN_OWN, _slot_id

SOURCES = ("proj", "actual")          # the ownership the prediction was made from
SALARY_CAP = 50000
PASS_CATCHERS = {"WR", "TE"}
RANK_BUCKETS = [(1, 1, "top lineup"), (2, 10, "2-10"), (11, 100, "11-100"), (101, 1000, "101-1000"), (1001, 10 ** 9, "rest")]
SALARY_LEFT_BUCKETS = [(0, 0, "$0"), (1, 500, "$100-500"), (501, 1000, "$600-1000"), (1001, 2000, "$1.1-2k"), (2001, 5000, "$2.1-5k"), (5001, 10 ** 9, "$5k+")]
CPT_OWN_BUCKETS = [(0, 5, "<5%"), (5, 10, "5-10%"), (10, 20, "10-20%"), (20, 35, "20-35%"), (35, 1000, "35%+")]
PRED_BIN_EDGES = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10 ** 9]


# --------------------------------------------------------------------------- what has been scraped
def scraped_contests(session) -> pd.DataFrame:
    """Every contest with a derived field on disk: id, name, type, entries, fee, date, draft group, whether actual ownership is stored."""
    ids = sorted(int(p.parent.name) for p in DERIVED_DIR.glob("*/meta.json")) if DERIVED_DIR.exists() else []
    cols = ["contest_id", "name", "contest_type", "entries", "entry_fee", "date", "draft_group_id", "has_actual_own", "is_showdown"]
    if not ids:
        return pd.DataFrame(columns=cols)
    q = ",".join(map(str, ids))
    h = read_sql(session, text(f"""SELECT h.contest_id, h.name, h.contest_type, h.total_entries AS entries, h.entry_fee, h.date, h.draft_group_id,
                                           (SELECT count(*) FROM showdown_player p WHERE p.contest_id = h.contest_id AND p.ownership_actual IS NOT NULL) AS has_actual_own
                                    FROM historical_contest_information h WHERE h.contest_id IN ({q})"""))
    h["has_actual_own"] = h.has_actual_own.astype(int) > 0
    h["is_showdown"] = h.contest_type.astype(str).str.startswith("Showdown")
    h["date"] = pd.to_datetime(h.date)
    return h.sort_values(["date", "entries"], ascending=[False, False]).reset_index(drop=True)[cols]


# --------------------------------------------------------------------------- one field, lineup by lineup
def _players_of(session, contest_id: int, draft_group_id) -> tuple[dict, dict]:
    """(slot id, normalised name) -> capture info; (slot, team, pos, salary) -> showdown_player ownership/projection."""
    caps = {c.slate_key: c for c in archive.slates(session, "nfl")}
    cap = caps.get(str(draft_group_id))
    info = {}
    if cap is not None:
        for p in archive.load_blobs(session, cap.id).get("players", []):
            info[(int(p["roster_slot_id"]), normalize_name(str(p.get("name", ""))))] = {
                "team": str(p.get("team")), "opponent": str(p.get("opponent") or ""), "position": str(p.get("position", "")).upper(),
                "salary": int(p.get("salary") or 0), "name": str(p.get("name", ""))}
    sp = read_sql(session, text("SELECT roster_slot_id AS slot, salary, position, team, ownership_proj, ownership_actual, fpts_proj "
                                "FROM showdown_player WHERE contest_id = :c").bindparams(c=int(contest_id)))
    own = {(int(r.slot), r.team, str(r.position).upper(), int(r.salary)): (r.ownership_proj, r.ownership_actual, r.fpts_proj) for r in sp.itertuples()}
    return info, own


def _log_own(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(np.log(max(float(v), MIN_OWN) / 100.0))


def field_lineups(session, contest_id: int, model: Optional[DM.ProductDupeModel] = None) -> Optional[pd.DataFrame]:
    """
    One row per unique lineup of a scraped contest with its actual copies, the model's
    expected copies from projected and from actual ownership, and the features. None when
    the field, the contest row or the capture is missing.
    """
    meta_p = DERIVED_DIR / str(contest_id) / "meta.json"
    if not meta_p.exists():
        return None
    h = read_sql(session, text("SELECT name, draft_group_id, contest_type, total_entries, entry_fee, date FROM historical_contest_information WHERE contest_id = :c")
                 .bindparams(c=int(contest_id)))
    if h.empty:
        return None
    h = h.iloc[0]
    is_showdown = str(h.contest_type).startswith("Showdown")
    info, own = _players_of(session, contest_id, h.draft_group_id)
    if not info or not own:
        return None
    model = model or DM.load()
    lp = pd.read_parquet(DERIVED_DIR / str(contest_id) / "lineups.parquet")
    slot_cols = [c for c in lp.columns if c.startswith("slot")]
    if not slot_cols:
        return None
    entries = int(len(lp))
    counts = lp[slot_cols].astype(str).agg("|".join, axis=1).value_counts()

    rows: List[dict] = []
    for lineup, c in counts.items():
        players = []
        ok = True
        for tok in lineup.split("|"):
            if tok in ("None", "nan", ""):
                continue
            slot, _, name = tok.partition(" ")
            sid = _slot_id(slot, is_showdown)
            pi = info.get((sid, normalize_name(name)))
            o = own.get((sid, pi["team"], pi["position"], pi["salary"])) if pi else None
            if pi is None or o is None:
                ok = False
                break
            players.append({**pi, "slot": slot, "own_proj": o[0], "own_actual": o[1], "fpts_proj": o[2]})
        if not ok or not players:
            continue
        lo_p = [_log_own(p["own_proj"]) for p in players]
        lo_a = [_log_own(p["own_actual"]) for p in players]
        teams = Counter(p["team"] for p in players)
        split_counts = sorted(teams.values(), reverse=True)
        cpt = next((p for p in players if p["slot"] == "CPT"), None)
        qbs = [p for p in players if p["position"] == "QB"]
        qb_mates = max((sum(1 for p in players if p is not qb and p["team"] == qb["team"] and p["position"] in PASS_CATCHERS) for qb in qbs), default=None)
        qb_bring = max((sum(1 for p in players if p["team"] == qb["opponent"] and p["position"] in PASS_CATCHERS | {"RB"}) for qb in qbs), default=None)
        salary_used = sum(p["salary"] for p in players)
        if cpt is not None and split_counts:
            cpt_side = "even split" if len(split_counts) > 1 and split_counts[0] == split_counts[1] else (
                "majority" if teams[cpt["team"]] == split_counts[0] else "minority")
        else:
            cpt_side = None
        rows.append({
            "contest_id": int(contest_id), "lineup": " | ".join(f"{p['slot']} {p['name']}" for p in players), "count": int(c), "entries": entries,
            "n_players": len(players), "is_showdown": is_showdown,
            "sum_log_own_proj": float(sum(lo_p)) if all(v is not None for v in lo_p) else np.nan,
            "sum_log_own_actual": float(sum(lo_a)) if all(v is not None for v in lo_a) else np.nan,
            "min_own_proj": min((float(p["own_proj"]) for p in players if p["own_proj"] is not None), default=np.nan),
            "cpt_name": cpt["name"] if cpt else None, "cpt_pos": cpt["position"] if cpt else None, "cpt_team": cpt["team"] if cpt else None,
            "cpt_own_proj": float(cpt["own_proj"]) if cpt and cpt["own_proj"] is not None else np.nan,
            "cpt_own_actual": float(cpt["own_actual"]) if cpt and cpt["own_actual"] is not None else np.nan,
            "salary_used": int(salary_used), "salary_left": int(SALARY_CAP - salary_used),
            "split": "-".join(str(x) for x in split_counts) if len(split_counts) > 1 else f"{split_counts[0]}-0", "cpt_side": cpt_side,
            "n_qb": len(qbs), "qb_mates": qb_mates, "qb_bringback": qb_bring,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["name"] = str(h["name"])
    df["entry_fee"] = float(h.entry_fee)
    df["date"] = pd.to_datetime(h.date)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["share"] = df["count"] / entries
    for s in SOURCES:
        slo = df[f"sum_log_own_{s}"].values
        pred = model.copies(entries, is_showdown, np.where(np.isnan(slo), 0.0, slo))
        pred = np.where(np.isnan(slo), np.nan, pred)
        df[f"pred_{s}"] = pred
        df[f"resid_{s}"] = np.log(df["count"].values / pred)
    return df


# --------------------------------------------------------------------------- buckets
def _bucket(v, buckets) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    for lo, hi, label in buckets:
        if lo <= v <= hi:
            return label
    return None


def _qb_label(n_qb, mates) -> str:
    if n_qb == 0:
        return "no QB"
    if n_qb > 1:
        return "2+ QBs"
    return f"QB +{min(int(mates), 3)}" + ("+" if mates >= 3 else "")


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rank_bucket"] = [_bucket(r, RANK_BUCKETS) for r in d["rank"]]
    d["salary_left_bucket"] = [_bucket(v, SALARY_LEFT_BUCKETS) for v in d["salary_left"]]
    d["cpt_own_bucket"] = [_bucket(v, CPT_OWN_BUCKETS) for v in d["cpt_own_proj"]]
    d["qb_stack"] = [_qb_label(n, m if m is not None else 0) for n, m in zip(d["n_qb"], d["qb_mates"])]
    d["bringback"] = ["no QB" if n == 0 else (f"opp {min(int(b), 2)}" + ("+" if b >= 2 else "")) for n, b in zip(d["n_qb"], d["qb_bringback"].fillna(0))]
    if len(d):
        try:
            d["own_sum_decile"] = pd.qcut(d["sum_log_own_proj"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)]).astype(str)
        except ValueError:
            d["own_sum_decile"] = "D1"
    else:
        d["own_sum_decile"] = []
    return d


FEATURES = [("rank_bucket", "lineup rank by actual copies", [b[2] for b in RANK_BUCKETS]),
            ("own_sum_decile", "ownership-sum decile (D10 = chalkiest)", [f"D{i}" for i in range(1, 11)]),
            ("cpt_pos", "captain position", ["QB", "RB", "WR", "TE", "K", "DST"]),
            ("cpt_own_bucket", "captain's projected ownership", [b[2] for b in CPT_OWN_BUCKETS]),
            ("cpt_side", "captain's side of the split", ["majority", "minority", "even split"]),
            ("split", "team split", ["6-0", "5-1", "4-2", "3-3"]),
            ("qb_stack", "QB stack (same-team WR/TE)", ["no QB", "QB +0", "QB +1", "QB +2", "QB +3+", "2+ QBs"]),
            ("bringback", "bring-back (opponent RB/WR/TE with the QB)", ["opp 0", "opp 1", "opp 2+"]),
            ("salary_left_bucket", "salary left on the table", [b[2] for b in SALARY_LEFT_BUCKETS])]


def total_entries(df: pd.DataFrame) -> int:
    """Entries over the contests in the frame (each contest's field size once)."""
    return int(df.groupby("contest_id")["entries"].first().sum()) if len(df) else 0


def residual_by(df: pd.DataFrame, feature: str, order: Optional[List[str]] = None) -> pd.DataFrame:
    """Per bucket: unique lineups, copies, actual and predicted copies, copy-weighted and plain log residual, both sources."""
    out = []
    n_entries = total_entries(df)
    for key, g in df.groupby(feature, dropna=True):
        row = {"bucket": key, "lineups": int(len(g)), "copies": int(g["count"].sum()), "share_of_entries": float(g["count"].sum() / n_entries) if n_entries else 0.0,
               "actual_mean": float(g["count"].mean())}
        for s in SOURCES:
            ok = g[f"resid_{s}"].notna()
            w = g.loc[ok, "count"].values.astype(float)
            r = g.loc[ok, f"resid_{s}"].values.astype(float)
            row[f"pred_mean_{s}"] = float(g.loc[ok, f"pred_{s}"].mean()) if ok.any() else np.nan
            row[f"bias_w_{s}"] = float((w * r).sum() / w.sum()) if w.sum() > 0 else np.nan
            row[f"bias_{s}"] = float(r.mean()) if ok.any() else np.nan
            row[f"mae_w_{s}"] = float((w * np.abs(r)).sum() / w.sum()) if w.sum() > 0 else np.nan
        out.append(row)
    res = pd.DataFrame(out)
    if order and len(res):
        res["_o"] = [order.index(b) if b in order else len(order) for b in res["bucket"]]
        res = res.sort_values(["_o", "bucket"]).drop(columns="_o")
    return res.reset_index(drop=True)


def calibration(df: pd.DataFrame, source: str = "proj") -> pd.DataFrame:
    """Lineups binned by predicted copies: how many, their actual copies, and the ratio."""
    pred = df[f"pred_{source}"]
    ok = pred.notna()
    d = df[ok]
    bins = pd.cut(d[f"pred_{source}"], PRED_BIN_EDGES, right=True, include_lowest=True)
    out = []
    for b, g in d.groupby(bins, observed=True):
        out.append({"pred_bin": f"{max(b.left, 0):g}-{b.right:g}" if b.right < 10 ** 8 else f"{b.left:g}+", "lineups": int(len(g)), "copies": int(g["count"].sum()),
                    "pred_mean": float(g[f"pred_{source}"].mean()), "actual_mean": float(g["count"].mean()),
                    "ratio": float(g["count"].mean() / g[f"pred_{source}"].mean()) if g[f"pred_{source}"].mean() > 0 else np.nan,
                    "bias_w": float((g["count"] * g[f"resid_{source}"]).sum() / g["count"].sum())})
    return pd.DataFrame(out)


def contest_summary(df: pd.DataFrame) -> dict:
    """One contest's headline numbers: entries, unique lineups, the top lineup, copy-weighted bias/MAE per source."""
    g = df
    entries = int(g["entries"].iloc[0])
    out = {"contest_id": int(g["contest_id"].iloc[0]), "name": str(g["name"].iloc[0]), "date": g["date"].iloc[0], "entry_fee": float(g["entry_fee"].iloc[0]),
           "entries": entries, "unique": int(len(g)), "unique_share": float(len(g) / entries),
           "top1_actual": int(g["count"].iloc[0]), "top10_actual": float(g["count"].iloc[:10].mean()),
           "dupe_share": float(g.loc[g["count"] >= 2, "count"].sum() / entries)}
    for s in SOURCES:
        ok = g[f"resid_{s}"].notna()
        w = g.loc[ok, "count"].values.astype(float)
        r = g.loc[ok, f"resid_{s}"].values.astype(float)
        out[f"has_{s}"] = bool(ok.any())
        out[f"top1_pred_{s}"] = float(g[f"pred_{s}"].iloc[0]) if ok.iloc[0] else np.nan
        out[f"top10_pred_{s}"] = float(g[f"pred_{s}"].iloc[:10].mean()) if ok.iloc[:10].any() else np.nan
        out[f"bias_w_{s}"] = float((w * r).sum() / w.sum()) if w.sum() > 0 else np.nan
        out[f"mae_w_{s}"] = float((w * np.abs(r)).sum() / w.sum()) if w.sum() > 0 else np.nan
        out[f"bias_{s}"] = float(r.mean()) if ok.any() else np.nan
        out[f"corr_{s}"] = float(np.corrcoef(np.log(g.loc[ok, "count"]), np.log(g.loc[ok, f"pred_{s}"]))[0, 1]) if ok.sum() > 2 else np.nan
    return out


# --------------------------------------------------------------------------- a set of contests
def analyze(session, contest_ids: Iterable[int], model: Optional[DM.ProductDupeModel] = None,
            frames: Optional[Dict[int, pd.DataFrame]] = None, top_n: int = 15) -> dict:
    """
    The dupe report for a set of scraped contests. `frames` lets a caller pass
    field_lineups() results it already has (the web page caches them).
    """
    model = model or DM.load()
    frames = dict(frames or {})
    per = []
    for cid in contest_ids:
        f = frames.get(int(cid))
        if f is None:
            f = field_lineups(session, int(cid), model)
            if f is not None:
                frames[int(cid)] = f
        if f is not None and len(f):
            per.append(f)
    if not per:
        return {"model": model.describe(), "contests": [], "lineups": pd.DataFrame(), "calibration": {}, "features": [], "top": [], "totals": {}, "frames": frames}
    df = add_buckets(pd.concat(per, ignore_index=True))
    contests = [contest_summary(f) for f in per]
    totals = {"contests": len(per), "entries": int(sum(c["entries"] for c in contests)), "unique": int(len(df))}
    for s in SOURCES:
        ok = df[f"resid_{s}"].notna()
        w = df.loc[ok, "count"].values.astype(float)
        r = df.loc[ok, f"resid_{s}"].values.astype(float)
        totals[f"bias_w_{s}"] = float((w * r).sum() / w.sum()) if w.sum() > 0 else np.nan
        totals[f"mae_w_{s}"] = float((w * np.abs(r)).sum() / w.sum()) if w.sum() > 0 else np.nan
        totals[f"bias_{s}"] = float(r.mean()) if ok.any() else np.nan
        top1 = [c for c in contests if not np.isnan(c[f"top1_pred_{s}"])]
        totals[f"top1_ratio_median_{s}"] = float(np.median([c["top1_actual"] / c[f"top1_pred_{s}"] for c in top1])) if top1 else np.nan
    features = [{"key": k, "label": lab, "rows": residual_by(df, k, order).to_dict("records")} for k, lab, order in FEATURES
                if df[k].notna().any()]
    top = df.sort_values(["share"], ascending=False).head(top_n)[
        ["contest_id", "name", "lineup", "count", "entries", "share", "pred_proj", "pred_actual", "cpt_name", "split", "salary_left", "qb_stack"]].to_dict("records")
    return {"model": model.describe(), "model_params": model.params, "contests": contests, "lineups": df, "totals": totals,
            "calibration": {s: calibration(df, s).to_dict("records") for s in SOURCES if df[f"pred_{s}"].notna().any()},
            "features": features, "top": top, "frames": frames}


def format_report(rep: dict) -> str:
    """The report as text, for the command line."""
    if not rep["contests"]:
        return "no scraped fields for those contests"
    t = rep["totals"]
    lines = [f"dupe model: {rep['model']}",
             f"{t['contests']} contests, {t['entries']:,} entries, {t['unique']:,} unique lineups",
             f"copy-weighted log residual (actual/pred): from projected ownership bias {t['bias_w_proj']:+.2f} MAE {t['mae_w_proj']:.2f}"
             + (f"; from actual ownership bias {t['bias_w_actual']:+.2f} MAE {t['mae_w_actual']:.2f}" if not np.isnan(t.get("bias_w_actual", np.nan)) else ""),
             f"top lineup actual/pred, median over contests: {t['top1_ratio_median_proj']:.2f} (projected own)"
             + (f", {t['top1_ratio_median_actual']:.2f} (actual own)" if not np.isnan(t.get("top1_ratio_median_actual", np.nan)) else ""), ""]
    lines.append(f"{'contest':60s} {'entries':>8s} {'unique':>7s} {'top1':>6s} {'pred':>7s} {'bias_w':>7s} {'mae_w':>6s} {'corr':>5s}")
    for c in rep["contests"]:
        lines.append(f"{c['name'][:60]:60s} {c['entries']:8,d} {c['unique']:7,d} {c['top1_actual']:6d} {c['top1_pred_proj']:7.1f} {c['bias_w_proj']:+7.2f} {c['mae_w_proj']:6.2f} {c['corr_proj']:5.2f}")
    for s, rows in rep["calibration"].items():
        lines += ["", f"calibration from {'projected' if s == 'proj' else 'actual'} ownership (bins of predicted copies):",
                  f"{'pred':>10s} {'lineups':>8s} {'copies':>8s} {'pred mean':>9s} {'actual':>8s} {'ratio':>6s}"]
        for r in rows:
            lines.append(f"{r['pred_bin']:>10s} {r['lineups']:8,d} {r['copies']:8,d} {r['pred_mean']:9.2f} {r['actual_mean']:8.2f} {r['ratio']:6.2f}")
    for f in rep["features"]:
        lines += ["", f"residual by {f['label']} (copy-weighted log(actual/pred); + = more duplicated than predicted):",
                  f"{'bucket':>14s} {'lineups':>8s} {'copies':>8s} {'share':>6s} {'bias proj':>9s} {'bias actual':>11s}"]
        for r in f["rows"]:
            ba = r.get("bias_w_actual")
            lines.append(f"{str(r['bucket']):>14s} {r['lineups']:8,d} {r['copies']:8,d} {r['share_of_entries']:6.1%} {r['bias_w_proj']:+9.2f} "
                         + (f"{ba:+11.2f}" if ba is not None and not np.isnan(ba) else f"{'':>11s}"))
    lines += ["", "most duplicated lineups:"]
    for r in rep["top"]:
        lines.append(f"  {r['count']:5d} copies ({r['share']:.2%}) pred {r['pred_proj']:6.1f}  {r['lineup']}  [{r['split']}, ${r['salary_left']} left, {r['qb_stack']}]  {r['name'][:40]}")
    return "\n".join(lines)
