"""
The players table of pre-lock and post-lock (GET /players): every player of the slate with his row and his exposure in one set of
lineups (the candidate pool, a simmed contest's field or the chosen entries), and that set's histogram.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import Request
from fastapi.responses import HTMLResponse
from ivan.sports.nba.models import HistoricalContest, Lineup, NBAPlayer
from ivan.sports.nba import slates as S
from .core import _render, desk, LOCK, slate
from .outputs import _slate_contest_ids


@slate.get("/players", response_class=HTMLResponse)
@slate.get("/exposures", response_class=HTMLResponse)                    # the section's address before the two tables were one
def players_section(request: Request, set: str = "", metric: str = "proj", bins: str = "", hist: str = ""):
    """The players section of pre-lock and post-lock (loaded after the board; its switches reload only the section). `set` picks
    whose exposure the table counts and is remembered on the slate's desk; blank = the last one picked, else the chosen entries
    when there are picks, else the candidate pool. hist=1 keeps the histogram open across a switch."""
    from .core import _team_url
    with LOCK:
        if desk.busy:
            return HTMLResponse(f'<div class="loading"><span class="spin"></span> {desk.busy}: the table comes back when it is done</div>')
        if set:
            desk.exp_set = set
        ex = _exposures_view(getattr(desk, "exp_set", "") or "", metric, int(bins) if str(bins).strip().isdigit() else None)
        ex["hist_open"] = bool(hist)
        return _render(request, "_nba_players.html", exposures=ex, team_url=lambda t: _team_url(request, t))


HIST_COLORS = {"field": "#2a78d6", "pool": "#eb6834", "entries": "#1baf7a"}   # the reference palette's first three slots (validated together)


def _field_sample() -> Optional[list]:
    """The sampled public field for the loaded pool, from its cache (written by the sim or the Field + ownership button); None when absent."""
    from ivan.sports.nba.field import public_field as PF
    try:
        path = PF.FIELD_CACHE_DIR / f"{PF.field_key(desk.session)}.json"
        if not path.exists():
            return None
        return [list(x) for x in json.loads(path.read_text())]
    except Exception:   # noqa: BLE001
        return None


def _pool_lineups() -> list:
    from ivan.core.utils import coerce_player_ids
    dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
    q = desk.session.query(Lineup.player_ids_list)
    if dg is not None:
        q = q.filter((Lineup.draft_group_id == int(dg)) | (Lineup.draft_group_id.is_(None)))
    return [coerce_player_ids(ids) for (ids,) in q.all()]


def _entries_lineups(contest: Optional[int] = None) -> Optional[dict]:
    """The chosen entries of the newest run covering this slate: {stamp, when, contests: [(id, name, n)], lineups: [[ids]], frame}."""
    mine = _slate_contest_ids()
    for classic in sorted(S.output_dir().glob("chosen_entries.classic.*.csv"), reverse=True):
        stamp = classic.name.split(".")[2]
        lu = S.output_dir() / f"chosen_entries.lineups.{stamp}.csv"
        if not lu.exists():
            continue
        try:
            L = pd.read_csv(lu)
        except Exception:   # noqa: BLE001
            continue
        L = L[L["contest_id"].astype(int).isin(mine)]
        if not len(L):
            continue
        contests = [(int(cid), str(g["contest_name"].iloc[0]), int(g["entry_id"].nunique())) for cid, g in L.groupby("contest_id")]
        if contest is not None:
            L = L[L["contest_id"].astype(int) == int(contest)]
        lineups = [[int(x) for x in g["player_id"].tolist()] for _, g in L.groupby("entry_id")]
        return dict(stamp=stamp, when=datetime.strptime(stamp, "%Y%m%d-%H%M%S"), contests=contests, lineups=lineups, frame=L)
    return None


def _hist(series: dict, step: float, lo: Optional[float] = None, hi: Optional[float] = None) -> dict:
    """Shared bins over every series: {edges, series: {name: {shares, counts, n, mean}}}; shares are per-series so sizes compare."""
    vals = {k: np.asarray(v, dtype=float) for k, v in series.items() if v is not None and len(v)}
    if not vals:
        return dict(edges=[], series={})
    allv = np.concatenate(list(vals.values()))
    lo = float(np.floor((allv.min() if lo is None else lo) / step) * step); hi = float(np.ceil((allv.max() if hi is None else hi) / step) * step)
    if hi <= lo:
        hi = lo + step
    edges = np.arange(lo, hi + step / 2, step)
    out = {}
    for k, v in vals.items():
        c, _ = np.histogram(v, bins=edges)
        out[k] = dict(counts=[int(x) for x in c], shares=[float(x) / max(1, len(v)) for x in c], n=int(len(v)), mean=float(v.mean()), p90=float(np.percentile(v, 90)))
    return dict(edges=[float(e) for e in edges], series=out, ymax=max((max(d["shares"]) for d in out.values()), default=0.0))


def _sim_fields() -> dict:
    """The sampled field of every simmed contest of this slate, from the newest sim artifacts (unique lineups x copies): {contest_id: {name, lineups, dupes}}."""
    mine = _slate_contest_ids()
    if not mine:
        return {}
    names = {int(c.contest_id): str(c.name or "") for c in desk.session.query(HistoricalContest).filter(HistoricalContest.contest_id.in_(list(mine))).all()}
    base = S.output_dir() / "sim"
    if not base.is_dir():
        return {}
    for d in sorted(base.iterdir(), reverse=True):
        found = {}
        for f in d.glob("contest_*.npz"):
            try:
                cid = int(f.stem.split("_")[1])
            except ValueError:
                continue
            if cid not in mine:
                continue
            try:
                z = np.load(f, allow_pickle=True)
                found[cid] = dict(name=names.get(cid, str(cid)), lineups=z["lineup_players"].astype(int), dupes=z["dupes"].astype(float), stamp=d.name)
            except Exception:   # noqa: BLE001
                continue
        if found:
            return found
    return {}


def _hist1(values: np.ndarray, weights: Optional[np.ndarray], step: float, bins: Optional[int] = None) -> dict:
    """One series: {edges, counts (weighted), shares, n, mean, p10, p50, p90}. `bins` fixes the number of equal bins over the range; else `step` sets the width."""
    if values is None or not len(values):
        return dict(edges=[], counts=[], shares=[], n=0)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if bins:
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + step
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        lo = float(np.floor(values.min() / step) * step); hi = float(np.ceil(values.max() / step) * step)
        if hi <= lo:
            hi = lo + step
        edges = np.arange(lo, hi + step / 2, step)
    c, _ = np.histogram(values, bins=edges, weights=w)
    tot = float(w.sum()) or 1.0
    order = np.argsort(values); cw = np.cumsum(w[order]) / tot
    q = lambda f: float(values[order][min(len(values) - 1, int(np.searchsorted(cw, f)))])   # noqa: E731
    return dict(edges=[float(e) for e in edges], counts=[float(x) for x in c], shares=[float(x) / tot for x in c], n=int(round(tot)),
                mean=float(np.average(values, weights=w)), p10=q(0.10), p50=q(0.50), p90=q(0.90), ymax=float(c.max() / tot) if len(c) else 0.0)


_FIELD_TARGET: dict = {}


def _field_target(metric: str) -> Optional[dict]:
    """
    The real fields the two-group sampler was fitted to (data/nba/derived/contests/<id>: DraftKings standings parsed at
    lock, one row per unique lineup with copies): per-lineup values for the histogram's target line. proj = the lineup's
    projected total as a share of that contest's best projected lineup (slates differ in scale; tonight's best rescales it),
    own = the sum of the eight players' real ownership, sal = salary used. Cached per process. None without derived contests.
    """
    if metric in _FIELD_TARGET:
        return _FIELD_TARGET[metric]
    from ivan.sports.nba.data_clean.standings import DERIVED_DIR
    vals, wts, n_contests, dates = [], [], 0, []
    for d in sorted(DERIVED_DIR.glob("*")) if DERIVED_DIR.is_dir() else []:
        lp, pc = d / "lineups.parquet", d / "players.csv"
        if not lp.exists():
            continue
        try:
            lu = pd.read_parquet(lp)
            if metric == "proj":
                if "proj_sum" not in lu or lu.proj_sum.max() <= 0:
                    continue
                v = lu.proj_sum.values / float(lu.proj_sum.max())
            elif metric == "own":
                if not pc.exists():
                    continue
                pl = pd.read_csv(pc); own = dict(zip(pl["id"].astype(int), pd.to_numeric(pl["drafted_pct"], errors="coerce").fillna(0.0)))
                v = np.array([sum(own.get(int(getattr(r, f"p{j}")), 0.0) for j in range(1, 9)) for r in lu.itertuples()])
            else:
                if "salary" not in lu:
                    continue
                v = lu.salary.values.astype(float)
            w = lu.copies.values.astype(float); v = np.asarray(v, dtype=float)
            ok = np.isfinite(v)                                   # a lineup with an unprojected player has no total
            vals.append(v[ok]); wts.append(w[ok]); n_contests += 1
            try:
                dates.append(json.loads((d / "field.json").read_text()).get("capture_at", "")[:10])
            except Exception:   # noqa: BLE001
                pass
        except Exception:   # noqa: BLE001
            continue
    out = None if not vals else dict(values=np.concatenate(vals), weights=np.concatenate(wts), contests=n_contests,
                                     span=(min(x for x in dates if x), max(x for x in dates if x)) if any(dates) else None)
    _FIELD_TARGET[metric] = out
    return out


def _target_line(target: dict, edges: list, scale: float = 1.0) -> Optional[dict]:
    """The target's share per bin of the histogram's own edges (mass outside the edges is kept in the denominator)."""
    if not target or not edges:
        return None
    v = target["values"] * scale; w = target["weights"]
    c, _ = np.histogram(v, bins=np.asarray(edges), weights=w)
    tot = float(w.sum()) or 1.0
    return dict(shares=[float(x) / tot for x in c], contests=target["contests"], span=target["span"], inside=float(c.sum() / tot))


def _exposures_view(set_key: str = "", metric: str = "proj", bins: Optional[int] = None) -> dict:
    """
    The players table: every player of the slate with the team card's full row, plus his exposure in one set of lineups (the
    candidate pool, a simmed contest's sampled field, or the chosen entries of one contest or all), his leverage against the
    field's ownership (and for entries the projection at pick with the delta). A player the set does not hold has exposure 0
    (in_set False). With it one histogram of a per-lineup metric over the set (projected total under our projection, ownership
    sum under the field's ownership, or salary used). An unknown or blank set = the chosen entries when there are picks, else
    the pool. Rows: the set's players by exposure, then the rest by ownership.
    """
    players = {int(p.id): p for p in desk.session.query(NBAPlayer).all()}
    means = {i: float(p.fpts_mean or 0.0) for i, p in players.items()}
    owns = {i: float(p.ownership or 0.0) for i, p in players.items()}
    sals = {i: float(p.salary or 0.0) for i, p in players.items()}
    fields = _sim_fields()
    ent = _entries_lineups(None)
    options = [dict(key="pool", label=f"candidate pool · {len(_pool_lineups())} lineups", group="pool")]
    if fields:
        options.append(dict(key="field", label=f"sampled fields · all {len(fields)} contests · {int(round(sum(f['dupes'].sum() for f in fields.values())))} entries", group="field"))
    for cid, f in fields.items():
        options.append(dict(key=f"field:{cid}", label=f"sampled field · {f['name'][:46]} · {int(round(f['dupes'].sum()))} entries", group="field"))
    if ent:
        options.append(dict(key="entries", label=f"chosen entries · all contests · {len(ent['lineups'])} entries", group="entries"))
        for cid, name, k in ent["contests"]:
            options.append(dict(key=f"entries:{cid}", label=f"chosen entries · {name[:46]} · {k} entries", group="entries"))
    keys = {o["key"] for o in options}
    if set_key not in keys:
        set_key = "entries" if "entries" in keys else "pool"
    weights = None; label = next(o["label"] for o in options if o["key"] == set_key); then = {}
    if set_key == "pool":
        lineups = [np.asarray(x, dtype=int) for x in _pool_lineups()]
    elif set_key == "field":
        lineups = [row for f in fields.values() for row in f["lineups"]]; weights = np.concatenate([f["dupes"] for f in fields.values()])
    elif set_key.startswith("field:"):
        f = fields[int(set_key.split(":")[1])]
        lineups = [row for row in f["lineups"]]; weights = f["dupes"]
    else:
        cid = int(set_key.split(":")[1]) if ":" in set_key else None
        e = _entries_lineups(cid) if cid is not None else ent
        lineups = [np.asarray(x, dtype=int) for x in (e["lineups"] if e else [])]
        if e is not None:
            then = e["frame"].groupby("player_id")["projection"].first().to_dict()
    n = len(lineups)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    metric = metric if metric in ("proj", "own", "sal") else "proj"
    per = {"proj": (means, 5.0, "projected total per lineup (our projection)", "fpts"),
           "own": (owns, 20.0, "ownership sum per lineup (the field's ownership)", "%"),
           "sal": (sals, 500.0, "salary used per lineup", "$")}
    table, step, title, unit = per[metric]
    vals = np.array([sum(table.get(int(i), 0.0) for i in lu) for lu in lineups], dtype=float)
    bins = int(bins) if bins and 4 <= int(bins) <= 80 else None
    hist = _hist1(vals, w, step, bins=bins) if n else dict(edges=[], counts=[], shares=[], n=0)
    # the dotted target over a sampled field: the real fields' shape, projected totals rescaled to tonight's best lineup
    target = None
    if set_key.startswith("field") and hist.get("edges"):
        tg = _field_target(metric)
        if tg:
            anchor = 1.0
            if metric == "proj":
                pool_sums = [sum(means.get(int(i), 0.0) for i in lu) for lu in _pool_lineups()]
                anchor = float(max(pool_sums)) if pool_sums else float(vals.max())
            target = _target_line(tg, hist["edges"], anchor)
            if target:
                hist["ymax"] = max(float(hist.get("ymax") or 0.0), max(target["shares"], default=0.0))
    counts: dict = {}
    for lu, wt in zip(lineups, w):
        for i in lu:
            counts[int(i)] = counts.get(int(i), 0.0) + float(wt)
    tot = float(w.sum()) or 1.0
    by_id = {}
    for t in sorted({p.team for p in players.values() if p.team}):
        if t not in desk.proj:
            desk.reproject(t)
        for r in desk.team_rows(t):
            by_id[int(r["id"])] = r
    rows = []
    for pid, r in by_id.items():
        k = counts.get(pid, 0.0)
        exp = k / tot if n else 0.0
        x = dict(r); x.update(n_in=int(round(k)), exposure=exp, in_set=k > 0, leverage=None if (not n or r.get("own") is None) else exp * 100 - r["own"],
                             proj_then=None if pid not in then or pd.isna(then[pid]) else float(then[pid]))
        x["delta"] = None if x["proj_then"] is None or r.get("fpts") is None else r["fpts"] - x["proj_then"]
        rows.append(x)
    rows.sort(key=lambda x: (-x["exposure"], -(x["own"] or 0), -(x["fpts"] or 0)))
    return dict(set=set_key, metric=metric, bins=bins, options=options, label=label, n=int(round(tot)) if n else 0, n_in=sum(1 for x in rows if x["in_set"]),
                rows=rows, hist=hist, title=title, unit=unit, target=target,
                is_entries=set_key.startswith("entries"), teams=sorted({x["team"] for x in rows}),
                color=HIST_COLORS["entries" if set_key.startswith("entries") else ("field" if set_key.startswith("field") else "pool")])
