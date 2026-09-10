# src/ivan/web/app.py
"""
Thin web UI over the NFL pipeline scripts: FastAPI + Jinja + htmx, no build
step, no auth (run it on a LAN or behind Tailscale). One job at a time.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ivan.core.utils import PROJECT_ROOT
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.paths import NFL_DATA_DIR
from ivan.web.jobs import JobRunner
from ivan.web.pipeline_nfl import (CURRENT_ENTRIES, STEP_BY_KEY, STEPS, WEB_INPUT_DIR, WEB_OUTPUT_DIR, WEIGHT_KEYS,
                                   WEIGHT_LABELS, RESETS, choose_rows, chosen_runs, chosen_view, contest_report,
                                   contest_reports, corr_css, entries_summary, legacy_exposures_view, output_files,
                                   players_frame, report_view, reset_working, slate_status, slates, OUTPUT_KINDS)

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


def _num_or_none(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None          # NaN -> None


def _fmt_money(x, places=2) -> str:
    v = _num_or_none(x)
    return "" if v is None else f"${v:,.{places}f}"


def _fmt_num(x, places=1) -> str:
    v = _num_or_none(x)
    return "" if v is None else f"{v:,.{places}f}"


def _fmt_pct1(x, places=1) -> str:
    """A fraction as a percentage."""
    v = _num_or_none(x)
    return "" if v is None else f"{v * 100:.{places}f}%"


templates.env.filters.update(time=_fmt_time, dur=_fmt_dur, dt=_fmt_dt, pct=_fmt_pct, kb=_fmt_kb,
                             money=_fmt_money, num=_fmt_num, pct1=_fmt_pct1, corr_css=corr_css)


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


_outputs = output_files


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
    entries = entries_summary()
    return templates.TemplateResponse(request, "index.html", _ctx(
        request, status=_with_db(slate_status), entries=entries, weights_rows=choose_rows(entries),
        weight_keys=WEIGHT_KEYS, weight_labels=WEIGHT_LABELS, outputs=_outputs()[:3], **_job_view(None, 40)))


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


@app.post("/reset/{what}")
def reset(what: str):
    if what not in RESETS:
        raise HTTPException(404)
    if runner.active():
        return _redirect("/", "A job is running; wait for it to finish (or cancel it) before clearing.")
    counts = _with_db(lambda s: reset_working(s, what))
    return _redirect("/", "Cleared " + ", ".join(f"{n:,} rows from {t}" for t, n in counts.items()) + ".")


@app.get("/players", response_class=HTMLResponse)
def players(request: Request):
    df = _with_db(players_frame)
    cols = [c for c in ["slot", "name", "team", "opponent", "position", "salary", "status", "fpts_mean", "fpts_std",
                        "fpts_p10", "fpts_p50", "fpts_p90", "ownership_proj"] if c in df.columns]
    return templates.TemplateResponse(request, "players.html", _ctx(request, cols=cols, rows=df[cols].to_dict("records") if len(df) else [],
                                                                      total=len(df), status=_with_db(slate_status)))


@app.get("/players.csv")
def players_csv():
    df = _with_db(players_frame)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="players.{stamp}.csv"'})


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
def exposures(request: Request, run: Optional[str] = None, contest: Optional[str] = None, file: Optional[str] = None):
    runs = [r for r in chosen_runs() if "lineups" in r["files"] or "exposures" in r["files"]]
    if not runs:
        return templates.TemplateResponse(request, "exposures.html", _ctx(request, runs=[], picked=None, view=None))
    picked = next((r for r in runs if r["stamp"] == run or (file and file in r["files"].values())), runs[0])
    if "lineups" in picked["files"]:
        view = chosen_view(WEB_OUTPUT_DIR / picked["files"]["lineups"], (contest or "").strip() or None)
    else:
        view = legacy_exposures_view(WEB_OUTPUT_DIR / picked["files"]["exposures"])
    uploads = [(OUTPUT_KINDS[k], picked["files"][k]) for k in ("showdown", "classic") if k in picked["files"]]
    return templates.TemplateResponse(request, "exposures.html", _ctx(request, runs=runs, picked=picked, view=view, uploads=uploads))


@app.get("/contests", response_class=HTMLResponse)
def contests(request: Request, id: Optional[int] = None):
    rows = _with_db(contest_reports)
    if not rows:
        return templates.TemplateResponse(request, "contests.html", _ctx(request, contests=[], picked=None, report=None))
    picked = next((r for r in rows if r["id"] == id), rows[0])
    report = _with_db(lambda s: contest_report(s, picked["id"])) or {}
    return templates.TemplateResponse(request, "contests.html", _ctx(
        request, contests=rows, picked=picked, report=report, view=report_view(report)))


@app.get("/contests/{contest_id}.json")
def contest_json(contest_id: int):
    report = _with_db(lambda s: contest_report(s, contest_id))
    if report is None:
        raise HTTPException(404, "no report for that contest")
    return JSONResponse(report)


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
