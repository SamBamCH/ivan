# src/ivan/web/app.py
"""
Thin web UI over the NFL pipeline scripts: FastAPI + Jinja + htmx, no build
step, no auth (run it on a LAN or behind Tailscale). One job at a time.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ivan.core.utils import PROJECT_ROOT
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.paths import NFL_DATA_DIR
from ivan.web.jobs import JobRunner
from ivan.web.pipeline_nfl import (CURRENT_ENTRIES, STEP_BY_KEY, STEPS, WEB_INPUT_DIR, WEB_OUTPUT_DIR,
                                   entries_summary, slate_status, slates)

HERE = Path(__file__).parent
engine = init_db("nfl")
runner = JobRunner(log_dir=NFL_DATA_DIR / "jobs", cwd=PROJECT_ROOT)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    WEB_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NFLBase.metadata.create_all(engine)       # working tables on a fresh server; no-op otherwise
    yield


app = FastAPI(title="Ivan", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


# ------------------------------------------------------------------ helpers
def _fmt_time(ts) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%a %H:%M:%S")


def _fmt_dur(sec) -> str:
    if sec is None:
        return ""
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else (f"{sec // 60}m {sec % 60:02d}s" if sec >= 60 else f"{sec}s")


def _fmt_dt(v) -> str:
    if not v:
        return ""
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v)
        except ValueError:
            return v
    return v.strftime("%a %b %-d, %-I:%M %p")


def _fmt_pct(x) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_kb(n) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


templates.env.filters.update(time=_fmt_time, dur=_fmt_dur, dt=_fmt_dt, pct=_fmt_pct, kb=_fmt_kb)


def _with_db(fn):
    session = get_session(engine)
    try:
        return fn(session)
    finally:
        session.close()


def _ctx(request: Request, **extra) -> dict:
    return {"request": request, "steps": STEPS, "active": runner.active(),
            "msg": request.query_params.get("msg"), **extra}


def _redirect(to: str, msg: Optional[str] = None) -> RedirectResponse:
    from urllib.parse import quote
    return RedirectResponse(f"{to}?msg={quote(msg)}" if msg else to, status_code=303)


def _outputs() -> list[dict]:
    files = sorted(WEB_OUTPUT_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    kinds = {"showdown": "DraftKings upload (showdown)", "classic": "DraftKings upload (classic)", "exposures": "Exposures"}
    out = []
    for p in files:
        parts = p.name.split(".")
        kind = parts[1] if len(parts) >= 4 else ""
        out.append({"name": p.name, "kind": kinds.get(kind, kind), "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime, "stamp": parts[2] if len(parts) >= 4 else ""})
    return out


def _job_view(job_id: Optional[str], tail: int) -> dict:
    job = runner.get(job_id) if job_id else (runner.current() or (runner.all()[0] if runner.all() else None))
    if job_id and not job:
        raise HTTPException(404, "no such job")
    progress = None
    if job and job.status == "running":
        step = STEP_BY_KEY.get(job.meta.get("step", ""))
        if step and step.progress:
            try:
                progress = _with_db(step.progress)
            except Exception:
                progress = None
    queued = [j for j in runner.active() if j.status == "queued" and (not job or j.id != job.id)]
    poll = f"/partials/job?tail={tail}" + (f"&id={job.id}" if job_id else "")
    return {"job": job, "progress": progress, "queued": queued,
            "log": runner.read_log(job, tail_lines=tail or None) if job else "", "poll_url": poll}


# ------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _ctx(
        request, status=_with_db(slate_status), entries=entries_summary(),
        outputs=_outputs()[:3], **_job_view(None, 40)))


@app.get("/partials/status", response_class=HTMLResponse)
def partial_status(request: Request):
    return templates.TemplateResponse(request, "_status.html", _ctx(request, status=_with_db(slate_status)))


@app.get("/partials/job", response_class=HTMLResponse)
def partial_job(request: Request, id: Optional[str] = None, tail: int = 40):
    return templates.TemplateResponse(request, "_job.html", _ctx(request, **_job_view(id, tail)))


@app.get("/partials/slates", response_class=HTMLResponse)
def partial_slates(request: Request, refresh: int = 0):
    try:
        items, err = slates(max_age=0 if refresh else 120), None
    except Exception as e:
        items, err = [], f"Couldn't load slates from DraftKings: {e}"
    return templates.TemplateResponse(request, "_slates.html", _ctx(request, slates=items, slates_error=err))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return _redirect("/", "That file was empty.")
    WEB_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (WEB_INPUT_DIR / f"entries.{stamp}.csv").write_bytes(data)
    CURRENT_ENTRIES.write_bytes(data)
    summary = entries_summary()
    if summary and summary["error"]:
        return _redirect("/", summary["error"])
    return _redirect("/", f"Uploaded {file.filename}: {summary['total']} entries across {len(summary['contests'])} contest(s).")


@app.post("/run/{key}")
async def run_step(key: str, request: Request):
    step = STEP_BY_KEY.get(key)
    if not step:
        raise HTTPException(404, "unknown step")
    if step.needs_entries and not CURRENT_ENTRIES.exists():
        return _redirect("/", f"'{step.title}' needs an entries file. Upload one first.")
    form = {k: str(v) for k, v in (await request.form()).items()}
    try:
        cmds = step.build(form)
    except ValueError as e:
        return _redirect("/", str(e))
    ahead = len(runner.active())
    for i, cmd in enumerate(cmds):
        name = step.title if len(cmds) == 1 else f"{step.title} ({i + 1}/{len(cmds)})"
        runner.submit(name, cmd, meta={"step": key, "form": form})
    note = f"Started: {step.title}." if ahead == 0 else f"Queued: {step.title} (behind {ahead} job{'s' if ahead > 1 else ''})."
    return _redirect("/", note)


@app.get("/jobs", response_class=HTMLResponse)
def jobs(request: Request):
    return templates.TemplateResponse(request, "jobs.html", _ctx(request, jobs=runner.all()))


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: str):
    return templates.TemplateResponse(request, "job.html", _ctx(request, **_job_view(job_id, 0)))


@app.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    ok = runner.cancel(job_id)
    return _redirect(f"/jobs/{job_id}", "Cancelled." if ok else "Nothing to cancel.")


@app.get("/exposures", response_class=HTMLResponse)
def exposures(request: Request, file: Optional[str] = None):
    files = [o for o in _outputs() if o["kind"] == "Exposures"]
    if not files:
        return templates.TemplateResponse(request, "exposures.html", _ctx(request, files=[], picked=None, groups=[], siblings=[]))
    picked = next((f for f in files if f["name"] == file), files[0])
    df = pd.read_csv(WEB_OUTPUT_DIR / picked["name"])
    if "roster_slot" in df.columns and "format" in df.columns:
        df["group"] = df["roster_slot"].where(df["format"] == "showdown", df["position"])
    else:
        df["group"] = df["position"]
    order = ["CPT", "FLEX", "QB", "RB", "WR", "TE", "DST"]
    groups = []
    for g in sorted(df["group"].dropna().unique(), key=lambda x: order.index(x) if x in order else 99):
        sub = df[df["group"] == g].sort_values("exposure", ascending=False)
        groups.append({"name": g, "rows": sub.to_dict("records")})
    siblings = [o for o in _outputs() if o["stamp"] == picked["stamp"] and o["kind"] != "Exposures"]
    return templates.TemplateResponse(request, "exposures.html", _ctx(request, files=files, picked=picked, groups=groups, siblings=siblings))


@app.get("/downloads", response_class=HTMLResponse)
def downloads(request: Request):
    return templates.TemplateResponse(request, "downloads.html", _ctx(request, files=_outputs()))


@app.get("/downloads/{name}")
def download(name: str):
    if "/" in name or "\\" in name or not name.endswith(".csv"):
        raise HTTPException(404)
    path = WEB_OUTPUT_DIR / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="text/csv", filename=name)
