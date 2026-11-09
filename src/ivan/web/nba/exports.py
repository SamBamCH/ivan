"""
CSV exports and downloads: the pool's projection, the generated lineups, the chooser's files.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from ivan.core.utils import normalize_name
from ivan.sports.nba.models import Lineup
from ivan.sports.nba import slates as S
from ivan.sports.nba.projection.live import override_of
from .core import CONSUMER, desk, LOCK, slate
from ivan.sports.nba.desk import backfill_lineup_slates, lineup_names, lineup_slates


EXPORT_SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")


def _csv_line(values) -> str:
    import csv, io
    buf = io.StringIO(); csv.writer(buf, lineterminator="\n").writerow(values); return buf.getvalue()


def _csv_response(body, filename: str) -> StreamingResponse:
    """
    A CSV download that is never cached on the way: Cloudflare caches a .csv URL by default when the origin sends no
    cache headers, and the brother's desk arrives through the tunnel, so without these every download of the same URL
    came back as the first one for hours. no-store covers browsers and the edge (CDN-Cache-Control is Cloudflare's own).
    """
    return StreamingResponse(body, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                      "Cache-Control": "no-store, private, max-age=0", "CDN-Cache-Control": "no-store",
                                      "Pragma": "no-cache", "Expires": "0"})


@slate.get("/export/projections.csv")
def export_projections():
    """Every player in the current pool with our projection (after the desk's edits) and the public one, one row each."""
    with LOCK:
        desk.maybe_reload()
        if desk.stale:                       # a newer projection is on disk (a pull, a publish): a file has no page state to keep, so take it
            desk.rebuild()
        rows = [list(("name", "dk_id", "team", "opponent", "position", "salary", "status", "starter", "minutes", "solve_minutes", "minutes_p10", "minutes_p90",
                      "fpts", "fpts_sd", "fppm", "public_fpts", "ownership", "pinned", "locked", "in_pool"))]
        # a consumer exports the pool: the laptop's projection as published, with this machine's own edits in it as of its last
        # Publish (the generator reads the same pool, so the file and the lineups agree). The desk's own per-team re-simulation
        # (600 worlds, its own seed) differs from the published 2000-world numbers by a few tenths and is never the file
        # (2026-10-27: 71 of 78 players differed on teams he never touched). The producer exports its desk view, edits included.
        for team in desk.slate.teams():
            proj = desk.proj.get(team); rot = desk.rot.get(team) or {}
            as_published = CONSUMER
            starters = desk.starters_of(team) or set(); locked = set((rot.get("grid").attrs.get("locked", []) if rot.get("grid") is not None else []))
            for p in desk.players(team):
                n = normalize_name(p.name)
                if as_published:
                    mp = p.minutes_proj
                    r = None if mp is None or p.fpts_mean is None else {"minutes": float(mp), "fpts_mean": float(p.fpts_mean), "fpts_std": float(p.fpts_std or 0.0),
                                                                        "fpm": (float(p.fpts_mean) / float(mp)) if mp else None, "minutes_p10": None, "minutes_p90": None}
                else:
                    r = proj.loc[n] if proj is not None and n in proj.index else None
                ov = override_of(desk.overrides, n)
                def f(x, nd=1):
                    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
                rows.append([p.name, p.id, p.team, p.opponent or "", p.position or "", p.salary or "", (p.status or ""), int(n in starters),
                             f(None if r is None else r["minutes"]), f((rot.get("targets") or {}).get(n)),
                             f(None if r is None else r.get("minutes_p10")), f(None if r is None else r.get("minutes_p90")),
                             f(None if r is None else r["fpts_mean"], 2), f(None if r is None else r["fpts_std"], 2), f(None if r is None else r["fpm"], 3),
                             f(getattr(p, "pub_fpts", None), 2), f(p.ownership, 3) if p.ownership not in (None, 0) else "",
                             int(ov.get("minutes") is not None), int(n in locked), int(r is not None and float(r["minutes"]) > 0)])
        proj_at = desk.pool_stamp[0] if desk.pool_stamp and desk.pool_stamp[0] else None      # when the pool's projection was published
        stamp = (proj_at.strftime("%Y%m%d-%H%M") if hasattr(proj_at, "strftime") else str(proj_at or "")[:16].replace(" ", "-").replace(":", "")) or datetime.now().strftime("%Y%m%d-%H%M")
    body = "".join(_csv_line(r) for r in rows)
    return _csv_response(iter([body]), f"nba_projections_{stamp}.csv")


@slate.get("/export/lineups.csv")
def export_lineups(ids: int = 0, draft_group: Optional[int] = None):
    """
    The generated lineups of one slate, one row each, the eight players in DraftKings roster order (PG SG SF PF C G F
    UTIL) and nothing else: names, or DraftKings ids with ?ids=1. The slate is ?draft_group=, else the loaded slate; when
    that slate has no lineups the file is one explanatory line, never another slate's lineups. Names and positions come
    from the pool snapshot written at generation, so the laptop moving the live pool to another slate cannot empty this.
    """
    from types import SimpleNamespace
    from ivan.sports.nba.contest.export import assign_player_order_classic
    from ivan.core.utils import coerce_player_ids
    with LOCK:
        desk.maybe_reload()
        backfill_lineup_slates(desk.session)
        slates = lineup_slates(desk.session)
        current = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
        have = [r["draft_group"] for r in slates if r["draft_group"] is not None]
        dg = draft_group if draft_group is not None else current
        if dg is None or dg not in have:
            # never another slate's lineups under a plausible name (2026-10-27: the server handed out Sunday's 9,844 lineups for
            # Tuesday's slate because nothing had been generated for it); ?draft_group= asks for a specific slate explicitly
            others = ", ".join(str(d) for d in have) if have else "none"
            body = _csv_line([f"no generated lineups for the loaded slate {dg}: run generate first (slates with lineups: {others}; ?draft_group=<id> exports one of them)"])
            return _csv_response(iter([body]), f"nba_lineups_{dg or 'none'}_none.csv")
        lookup = lineup_names(desk.session, int(dg))
        lineups = desk.session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.draft_group_id == int(dg)).all()
        stamp = datetime.now().strftime("%Y%m%d-%H%M")

    def gen():
        yield _csv_line(EXPORT_SLOTS)
        for lid, pids in lineups:
            try:
                pl = [int(x) for x in coerce_player_ids(pids)]
            except (ValueError, TypeError):
                continue
            objs = [SimpleNamespace(id=x, name=lookup.get(x, {}).get("name") or str(x), position=lookup.get(x, {}).get("position")) for x in pl]
            ordered = objs
            if len(objs) == len(EXPORT_SLOTS) and all(o.position for o in objs):
                try:
                    ordered = assign_player_order_classic(objs, prefer_late_in_flex=False)
                except ValueError:
                    ordered = objs                       # an odd eligibility: the stored order
            yield _csv_line([o.id if ids else o.name for o in ordered])
    return _csv_response(gen(), f'nba_lineups_{dg}_{stamp}{"_ids" if ids else ""}.csv')


@slate.get("/download/{name}")
def download(name: str):
    p = S.output_dir() / Path(name).name
    if not p.exists() or not p.name.startswith(("chosen_entries.", "lateswap.")):
        return HTMLResponse("not found", status_code=404)
    return FileResponse(p, filename=p.name, media_type="text/csv")
