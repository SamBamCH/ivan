# src/ivan/web/dupes_view.py
"""
The Dupes page: the dupe model's predicted copies against the copies real
fields held (ivan.sports.nfl.dupe_analysis), with the charts and tables laid
out for the template. Per-contest lineup frames are cached in memory, keyed
by the field file's mtime and the model parameters, so re-selecting contests
is cheap.
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import pandas as pd

from ivan.sports.nfl import dupe_analysis as DA
from ivan.sports.nfl import dupe_model as DM
from ivan.web import charts

_cache: dict = {}          # contest_id -> (key, frame)
MAX_POINTS = 1500


def _frame(session, cid: int, model) -> Optional[pd.DataFrame]:
    meta = DA.DERIVED_DIR / str(cid) / "lineups.parquet"
    key = (meta.stat().st_mtime if meta.exists() else None, json.dumps(model.params, sort_keys=True))
    hit = _cache.get(cid)
    if hit and hit[0] == key:
        return hit[1]
    f = DA.field_lineups(session, cid, model)
    _cache[cid] = (key, f)
    return f


def slates(session) -> list[dict]:
    """Scraped contests grouped by slate (draft group), newest first."""
    df = DA.scraped_contests(session)
    out = []
    for dg, g in df.groupby("draft_group_id", sort=False):
        g = g.sort_values("entries", ascending=False)
        name = str(g["name"].iloc[0])
        game = name[name.rfind("("):] if "(" in name else ""
        out.append({"draft_group_id": int(dg) if pd.notna(dg) else None, "date": g["date"].max(), "game": game, "showdown": bool(g["is_showdown"].iloc[0]),
                    "contests": [{"id": int(r.contest_id), "name": r.name, "entries": int(r.entries), "fee": float(r.entry_fee),
                                  "has_actual_own": bool(r.has_actual_own)} for r in g.itertuples()]})
    return sorted(out, key=lambda s: s["date"], reverse=True)


def default_selection(slate_list: list[dict]) -> list[int]:
    return [c["id"] for c in slate_list[0]["contests"]] if slate_list else []


def report(session, contest_ids: list[int]) -> dict:
    model = DM.load()
    frames = {cid: f for cid in contest_ids if (f := _frame(session, cid, model)) is not None}
    rep = DA.analyze(session, contest_ids, model, frames=frames)
    rep["charts"] = {s: _scatter(rep["lineups"], s) for s in DA.SOURCES if len(rep["lineups"]) and rep["lineups"][f"pred_{s}"].notna().any()}
    rep["feature_charts"] = [{"key": f["key"], "label": f["label"], "svg": _bars(f["rows"])} for f in rep["features"]]
    rep.pop("frames", None)
    rep["lineups"] = None     # the frame is not for the template
    return rep


def _scatter(df: pd.DataFrame, source: str) -> str:
    """Predicted vs actual copies on log axes, one point per unique lineup (subsampled), the chalk ringed."""
    d = df[df[f"pred_{source}"].notna() & (df[f"pred_{source}"] > 0)]
    if d.empty:
        return ""
    top = d.nlargest(50, "count")
    rest = d.drop(top.index)
    if len(rest) > MAX_POINTS:
        rest = rest.sample(MAX_POINTS, random_state=0)
    pts = []
    for part, kind in ((rest, "cand"), (top, "top")):
        for r in part.itertuples():
            pred = getattr(r, f"pred_{source}")
            pts.append({"x": math.log10(pred), "y": math.log10(r.count), "kind": kind, "shade": 0.5,
                        "title": f"{r.lineup}\nactual {r.count} copies, predicted {pred:.1f} ({r.name[:40]})"})
    lo = min(min(p["x"] for p in pts), min(p["y"] for p in pts))
    hi = max(max(p["x"] for p in pts), max(p["y"] for p in pts))
    fmt = lambda v: f"{10 ** v:g}"   # noqa: E731
    return charts.scatter_svg(pts, xlabel=f"predicted copies (from {'projected' if source == 'proj' else 'actual'} ownership)", ylabel="actual copies",
                              xfmt=fmt, yfmt=fmt, width=520, height=380,
                              iso_lines=[{"x1": lo, "y1": lo, "x2": hi, "y2": hi, "label": "actual = predicted"}],
                              legend=[("cand", "a unique lineup"), ("top", "50 most duplicated"), ("line", "actual = predicted")])


def _bars(rows: list[dict], width: int = 520) -> str:
    """Horizontal residual bars per bucket: projected-ownership residual (solid) and actual-ownership residual (outline)."""
    if not rows:
        return ""
    vals = [abs(r.get("bias_w_proj") or 0) for r in rows] + [abs(r.get("bias_w_actual") or 0) for r in rows if r.get("bias_w_actual") is not None and not np.isnan(r["bias_w_actual"])]
    span = max(max(vals), 0.25)
    row_h, left, mid_w = 22, 150, (width - 170) / 2
    height = 20 + row_h * len(rows)
    mid = left + mid_w
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{width}px;height:auto;font-family:system-ui,sans-serif">',
             f'<line x1="{mid}" y1="6" x2="{mid}" y2="{height - 6}" stroke="{charts.LINE}"/>']
    for i, r in enumerate(rows):
        y = 10 + i * row_h
        parts.append(f'<text x="{left - 6}" y="{y + 14}" font-size="11" text-anchor="end" fill="{charts.INK}">{charts._esc(str(r["bucket"]))}</text>')
        for key, style in (("bias_w_proj", f'fill="{charts.ACCENT}"'), ("bias_w_actual", f'fill="none" stroke="{charts.TOP}" stroke-width="1.5"')):
            v = r.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            w = abs(v) / span * mid_w
            x = mid if v >= 0 else mid - w
            yy = y + 3 if key == "bias_w_proj" else y + 7
            h = 12 if key == "bias_w_proj" else 6
            tip = charts._esc(f"{r['bucket']}: {r['lineups']:,} lineups, {r['copies']:,} copies ({r['share_of_entries']:.1%} of entries); "
                              f"log(actual/pred) {v:+.2f} = x{math.exp(v):.2f}")
            parts.append(f'<rect x="{x:.1f}" y="{yy}" width="{max(w, 0.5):.1f}" height="{h}" {style}><title>{tip}</title></rect>')
        v = r.get("bias_w_proj")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            parts.append(f'<text x="{(mid + 4) if v < 0 else (mid - 4):.1f}" y="{y + 14}" font-size="10" text-anchor="{"start" if v < 0 else "end"}" fill="{charts.MUTED}">x{math.exp(v):.2f}</text>')
    parts.append(f'<text x="{mid}" y="{height - 1}" font-size="9" text-anchor="middle" fill="{charts.MUTED}">under-predicted to the right, over-predicted to the left</text>')
    parts.append("</svg>")
    return "\n".join(parts)
