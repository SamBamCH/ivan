"""
What the pipeline wrote for the loaded slate: the entries file, its contests, the newest choose run.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba import slates as S
from .core import desk


def _entries_path() -> Path:
    return S.entries_csv()                                   # the slate's entries file (slates/<dg>/input/entries.csv)


def _entries_info() -> dict:
    p = _entries_path()
    if not p.exists():
        return dict(path=str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p), exists=False)
    try:
        from ivan.sports.nba.contest.export import read_entries
        e = read_entries(p)
        contests = e.groupby("Contest Name").size().to_dict()
        return dict(path=str(p.relative_to(PROJECT_ROOT)), exists=True, entries=int(len(e)), contests=contests,
                    modified=datetime.fromtimestamp(p.stat().st_mtime))
    except Exception as ex:   # noqa: BLE001
        return dict(path=str(p), exists=True, error=f"{type(ex).__name__}: {ex}")


def _slate_contest_ids() -> set[int]:
    """The contests of the draft group loaded in the desk (historical_contest_information)."""
    dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
    if dg is None:
        return set()
    from ivan.sports.nba.models import HistoricalContest
    return {int(c) for (c,) in desk.session.query(HistoricalContest.contest_id).filter(HistoricalContest.draft_group_id == int(dg)).all()}


def _latest_outputs() -> Optional[dict]:
    """
    The newest choose run that covered a contest of the current slate: the upload CSV, the per-contest
    summary from its choose json, and the exposures recomputed over this slate's entries only.
    """
    mine = _slate_contest_ids()
    for classic in sorted(S.output_dir().glob("chosen_entries.classic.*.csv"), reverse=True):
        stamp = classic.name.split(".")[2]
        js = S.output_dir() / f"chosen_entries.choose.{stamp}.json"
        if not js.exists():
            continue
        try:
            d = json.loads(js.read_text())
        except ValueError:
            continue
        hit = {cid: v for cid, v in d.items() if int(cid) in mine}
        if not hit:
            continue
        out = dict(stamp=stamp, classic=classic.name, when=datetime.strptime(stamp, "%Y%m%d-%H%M%S"), slate_only=len(hit) < len(d))
        try:
            out["contests"] = [dict(name=v.get("contest_name"), entries=v.get("entries"), chosen=v.get("chosen"), roi=v.get("chosen_roi"), cash=v.get("chosen_cash"),
                                    top_roi=v.get("top_quality_roi"), book=(v.get("book") or {}).get("chosen") or {}) for v in hit.values()]
            up = pd.read_csv(classic); up = up[up["Contest ID"].astype(int).isin(mine)]
            out["rows"] = int(len(up))
            lu = S.output_dir() / f"chosen_entries.lineups.{stamp}.csv"
            if lu.exists() and len(up):
                L = pd.read_csv(lu); L = L[L["contest_id"].astype(int).isin(mine)]
                n_entries = L["entry_id"].nunique()
                e = (L.groupby(["player_id", "name", "team", "position", "salary"]).agg(n=("entry_id", "nunique"), projection=("projection", "first"))
                     .reset_index())
                e["exposure"] = e["n"] / max(1, n_entries)
                out["exposures"] = e.sort_values("exposure", ascending=False).head(14)[["name", "team", "position", "salary", "exposure", "projection"]].to_dict("records")
        except Exception as ex:   # noqa: BLE001
            out["error"] = f"{type(ex).__name__}: {ex}"
        return out
    return None
