"""
The post-lock view: the late swap's context, the standings fetch and upload.
"""
from __future__ import annotations

import os
from datetime import datetime
from fastapi import File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from ivan.core.utils import normalize_name
from .core import _active, _busy, _here, _render, desk, LIMITED, LOCK, slate
from .knobs import pipeline_form
from .outputs import _entries_info, _entries_path
from .pipeline import _pipeline_ctx, _submit


def _lateswap_ctx(now: datetime) -> dict:
    """The late swap tab: our entries with each slot's lock state and projection, the lock clock, the newest report."""
    from ivan.sports.nba.contest.export import POSITION_ORDER, read_entries
    from ivan.sports.nba.lateswap import swap as LS
    byid = {int(p.id): p for p in desk.slate.players}
    proj = {}
    for t, df in desk.proj.items():
        for name, r in df.iterrows():
            proj[name] = float(r["fpts_mean"])
    tips = sorted({p.gametime for p in desk.slate.players if p.gametime})
    out = dict(games_started=sum(1 for t in tips if t <= now), games_total=len(tips), next_tip=next((t for t in tips if t > now), None),
               last_tip=max((t for t in tips if t <= now), default=None),
               open_players=sum(1 for p in byid.values() if p.gametime and p.gametime > now), contests=[], report=LS.latest_report(),
               has_display=bool(os.environ.get("DISPLAY")), entries_path=_entries_info().get("path"))
    p = _entries_path()
    names_to_id = LS.name_map(desk.session)
    if p.exists():
        try:
            df = read_entries(p)
        except Exception as ex:   # noqa: BLE001
            out["error"] = f"{type(ex).__name__}: {ex}"; df = None
        if df is not None:
            for (cid, cname), g in df.groupby(["Contest ID", "Contest Name"], sort=False):
                rows = []
                for _, r in g.iterrows():
                    slots = []
                    for slot in POSITION_ORDER:
                        m = str(r[slot]); pid = LS._pid(m)
                        if pid is None and "(" in m:
                            pid = LS._resolve(names_to_id, LS._cell_name(m))
                        pl = byid.get(pid)
                        slots.append(dict(slot=slot, pid=pid, name=pl.name if pl else LS._cell_name(m), team=pl.team if pl else "",
                                          locked=bool(LS._is_locked_cell(m) or (pl and pl.gametime and pl.gametime <= now)),
                                          fpts=proj.get(normalize_name(pl.name)) if pl else None, tip=pl.gametime if pl else None, on_slate=pl is not None))
                    rows.append(dict(entry_id=r["Entry ID"], slots=slots, locked=sum(1 for x in slots if x["locked"]),
                                     fpts=sum((x["fpts"] or 0) for x in slots)))
                from ivan.core import raw as _raw
                art = _raw.latest("nba", "dk/standings", int(cid))
                out["contests"].append(dict(contest_id=int(cid), name=cname, entries=rows,
                                            standings=None if art is None else LS.stamp_time(art.stamp),
                                            standings_current=bool(art is not None and LS.export_is_current(art.stamp, tips, now))))
    rep = out["report"]
    if rep:
        for c in rep.get("contests", []):
            for e in c["entries"]:
                e["delta"] = e["best"]["exp_payout"] - e["original"]["exp_payout"]
            c["gain"] = sum(e["delta"] for e in c["entries"]); c["improved"] = sum(1 for e in c["entries"] if e["improved"])
        if "exposure" not in rep:                                  # reports written before the exposure table
            rep["n_entries"] = sum(len(c["entries"]) for c in rep["contests"]); rep["exposure"] = LS.exposure(rep, byid)
        try:
            rep["when_dt"] = datetime.fromisoformat(rep["when"])
        except (KeyError, ValueError):
            rep["when_dt"] = None
    return out


@slate.post("/lateswap/fetch", response_class=HTMLResponse)
def lateswap_fetch(request: Request):
    """Download every entries contest's standings export now and keep it (scripts/nba/fetch_standings.py as a job): the late
    swap's fetch step, pressed a minute after a tip; the swap then re-runs on the stored exports until the next tip."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        if LIMITED:
            desk.flash.append("no DraftKings login on this desk: upload the standings export instead")
        elif _active():
            desk.flash.append("a job is already running; wait for it or cancel it")
        else:
            _submit("fetch_standings", pipeline_form)
            desk.flash.append("fetching the standings of every contest in the entries file")
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


MAX_STANDINGS = 60 * 1024 * 1024        # a 100k-entry contest's standings csv is ~15 MB; the zip DraftKings hands out is smaller


def _standings_text(name: str, data: bytes) -> str:
    """The standings CSV from what DraftKings' export button gives: a .zip holding one csv, or the csv itself."""
    import io, zipfile
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [m for m in z.infolist() if m.filename.lower().endswith(".csv") and not m.is_dir()]
            if len(members) != 1:
                raise ValueError("the zip should hold exactly one .csv")
            if members[0].file_size > MAX_STANDINGS * 4:
                raise ValueError("the csv inside the zip is too large")
            return z.read(members[0]).decode("utf-8-sig", errors="replace")
    if name.lower().endswith(".csv"):
        return data.decode("utf-8-sig", errors="replace")
    raise ValueError("a .zip or .csv from DraftKings' standings export, please")


@slate.post("/lateswap/standings", response_class=HTMLResponse)
async def standings_upload(request: Request, files: list[UploadFile] = File(...), contest_id: str = Form("")):
    """
    Contest standings exported from DraftKings (the contest page's export, once the contest has locked): the field the late
    swap plays against, on a desk without a DraftKings login. The contest id comes from the file name
    (contest-standings-<id>.zip) or the picker. Kept in the raw store as dk/standings, where `--source raw` finds it.
    """
    import re
    from ivan.core import raw
    from ivan.sports.nba.data_fetch.standings import parse_export
    saved, errors = [], []
    for f in files:
        data = await f.read(MAX_STANDINGS + 1)
        if len(data) > MAX_STANDINGS:
            errors.append(f"{f.filename}: over {MAX_STANDINGS // 2**20} MB"); continue
        try:
            text = _standings_text(f.filename or "", data)
            ent, _ = parse_export(text)
        except Exception as e:   # noqa: BLE001
            errors.append(f"{f.filename}: {type(e).__name__}: {e}"); continue
        m = re.search(r"(\d{6,})", f.filename or "")
        cid = int(m.group(1)) if m else (int(contest_id) if contest_id.strip().isdigit() else None)
        if cid is None:
            errors.append(f"{f.filename}: no contest id in the file name; pick the contest and upload again"); continue
        raw.store("nba", "dk/standings", cid, text, ext="csv", meta={"source": "upload", "file": f.filename})
        saved.append(f"{cid} ({len(ent):,} entries)")
    with LOCK:
        if saved:
            desk.flash.append("standings saved for " + ", ".join(saved))
        for e in errors:
            desk.flash.append("standings upload: " + e)
    return RedirectResponse(_here(request) + "?view=lateswap", status_code=303)
