"""
The backtest view's "current code" block: a past slate rerun from its tip record by the game simulation
(ivan.sports.nba.gamesim.rerun = scripts/nba/rerun_slate.py), scored layer by layer against the actuals, every player's
drawn distribution of a stat with his actual inside it.

    GET  /<day>/<dg>/rerun?stat=fpts     the block (polls itself while a rerun job of this slate runs)
    POST /<day>/<dg>/rerun/run           queue scripts/nba/rerun_slate.py --slate <dg> -n <sims> as a job
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import Form, Request
from fastapi.responses import HTMLResponse

from ivan.sports.nba.gamesim import rerun as RR
from ivan.sports.nba import tips as T
from .core import _render, runner, slate, slate_jobs

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts" / "nba"
STATS = [("fpts", "fantasy points"), ("minutes", "minutes"), ("pts", "points"), ("reb", "rebounds"), ("ast", "assists"), ("fg3m", "threes"),
         ("stl", "steals"), ("blk", "blocks"), ("tov", "turnovers"), ("fga", "field-goal attempts"), ("fta", "free throws")]
SIMS_MAX = 5000


def rerun_ctx(dg: int, stat: Optional[str] = None) -> dict:
    stat = stat if stat in dict(STATS) else "fpts"
    rec = RR.load_latest(dg)
    jobs = [j for j in slate_jobs(dg) if (j.meta or {}).get("step") == "rerun"]
    active = next((j for j in jobs if j.is_active), None)
    last = jobs[0] if jobs else None
    log_tail = runner.read_log(last, tail_lines=6) if last is not None and (active is not None or last.status == "failed") else ""
    return dict(rr=rec, rr_meta=rec["meta"] if rec else None, rr_players=RR.player_view(rec, stat) if rec else {}, rr_stat=stat, rr_stats=STATS,
                rr_job=active, rr_last=last, rr_log=log_tail, rr_tips=T.tips_of(dg), rr_names=RR.list_reruns(dg), rr_dg=int(dg), rr_sims_max=SIMS_MAX)


@slate.get("/rerun", response_class=HTMLResponse)
def rerun_block(request: Request, stat: str = "fpts"):
    dg = getattr(request.state, "dg", None)
    return _render(request, "_nba_rerun.html", **rerun_ctx(int(dg), stat))


@slate.post("/rerun/run", response_class=HTMLResponse)
def rerun_run(request: Request, sims: int = Form(1000), tip: str = Form(""), stat: str = Form("fpts")):
    dg = int(getattr(request.state, "dg"))
    ctx = rerun_ctx(dg, stat)
    if ctx["rr_job"] is not None:
        return _render(request, "_nba_rerun.html", **ctx)
    n = max(100, min(SIMS_MAX, int(sims)))
    cmd = [sys.executable, str(SCRIPTS / "rerun_slate.py"), "--slate", str(dg), "-n", str(n)] + (["--tip", tip] if tip else [])
    runner.submit("rerun", cmd, meta=dict(step="rerun", slate=dg, n=n))
    return _render(request, "_nba_rerun.html", **rerun_ctx(dg, stat))
