# src/ivan/web/app.py
"""
Thin web UI over the NFL pipeline scripts: FastAPI + Jinja + htmx, no build
step, no auth (run it on a LAN or behind Tailscale). One job at a time.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ivan.core.utils import PROJECT_ROOT
from ivan.database.session import add_missing_columns, get_session, init_db
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.paths import NFL_DATA_DIR
from ivan.web import backtest_view as bv
from ivan.web import validate_view as vv
from ivan.web import dupes_view as dv
from ivan.web import models_view as mv
from ivan.web import plan_view as pv
from ivan.web import backtest_hub as hub
from ivan.web.jobs import JobRunner
from ivan.web.pipeline_nfl import (CURRENT_ENTRIES, STEP_BY_KEY, STEPS, WEB_INPUT_DIR, WEB_OUTPUT_DIR, WEIGHT_KEYS,
                                   WEIGHT_LABELS, RESETS, choose_rows, choose_view, chosen_runs, chosen_view, contest_report,
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
    add_missing_columns(engine, NFLBase.metadata)   # columns added to the models since the tables were created
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
                             money=_fmt_money, num=_fmt_num, pct1=_fmt_pct1, corr_css=corr_css,
                             smoney=bv.money_signed, spct=bv.pct_signed, scss=bv.sign_css)


def _with_db(fn):
    session = get_session(engine)
    try:
        return fn(session)
    finally:
        session.close()


@contextmanager
def _session():
    s = get_session(engine)
    try:
        yield s
    finally:
        s.close()


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
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Browsers ask for this on every page; serve the tab icon instead of logging a 404."""
    return FileResponse(HERE / "static" / "favicon.svg", media_type="image/svg+xml")


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
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
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


@app.get("/choose", response_class=HTMLResponse)
def choose_page(request: Request, run: Optional[str] = None, contest: Optional[str] = None):
    runs = [r for r in chosen_runs() if "choose" in r["files"]]
    if not runs:
        return templates.TemplateResponse(request, "choose.html", _ctx(request, runs=[], picked=None, view=None))
    picked = next((r for r in runs if r["stamp"] == run), runs[0])
    lineups = WEB_OUTPUT_DIR / picked["files"]["lineups"] if "lineups" in picked["files"] else None
    view = choose_view(WEB_OUTPUT_DIR / picked["files"]["choose"], lineups, (contest or "").strip() or None)
    return templates.TemplateResponse(request, "choose.html", _ctx(request, runs=runs, picked=picked, view=view))


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page(request: Request):
    today = datetime.now().date()
    return templates.TemplateResponse(request, "backtest_hub.html", _ctx(
        request, run=None, runs=hub.runs(), default_since=(today - timedelta(days=7)).isoformat(), default_until=today.isoformat(),
        msg=request.query_params.get("msg")))


@app.get("/backtest/open")
def backtest_open(name: str):
    return _redirect(f"/backtest/{name}")


@app.post("/backtest/new")
async def backtest_new(request: Request):
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    try:
        run = hub.new_run(form.get("name", ""), form.get("since", "").strip(), form.get("until", "").strip())
    except ValueError as e:
        return _redirect("/backtest", f"Could not start the run: {e}")
    return _redirect(f"/backtest/{run['name']}")


def _run_jobs(name: str, tail: int = 14) -> list[dict]:
    out = []
    for j in runner.all():
        if (j.meta or {}).get("run") != name:
            continue
        lines = []
        try:
            p = runner.log_path(j)
            if p.is_file():
                lines = [l for l in p.read_text(errors="replace").splitlines() if l.strip()][-tail:]
        except OSError:
            lines = []
        out.append({"id": j.id, "name": j.name, "status": j.status, "stage": (j.meta or {}).get("step", ""),
                    "started": datetime.fromtimestamp(j.started).strftime("%H:%M:%S") if j.started else "",
                    "duration": int(j.duration) if j.duration else None, "returncode": j.returncode, "tail": lines})
    return out[:12]


@app.get("/backtest/{name}", response_class=HTMLResponse)
def backtest_run_page(request: Request, name: str, step: Optional[str] = None):
    run = hub.load_run(name)
    if run is None:
        return _redirect("/backtest", f"No run named {name!r}.")
    step = hub.step_for(run, step)
    ctx = {"run": run, "step": step, "runs": [], "data": None, "slates": [], "models": {}, "view": None, "jobs": _run_jobs(name),
           "msg": request.query_params.get("msg")}
    try:
        with _session() as session:
            if step == "data":
                ctx["data"] = hub.data_check(session, run)
            elif step == "contests":
                ctx["slates"] = hub.contests_view(session, run)
            elif step == "params":
                ctx["models"] = hub.model_version_for(run["since"]) if hasattr(hub, "model_version_for") else hub.model_store.resolve(hub.MODELS_DIR, datetime.fromisoformat(run["since"])).describe()
            elif step == "results":
                ctx["view"] = hub.run_view(run)
                ctx["progress"] = hub.progress_view(run, ctx["jobs"])
                if run.get("status") == "running" and not ctx["progress"]["active"]:
                    run["status"] = "done"
                    hub.save_run(run)
                    ctx["run"] = run
    except ValueError as e:
        return _redirect("/backtest", f"Could not read the run: {e}")
    return templates.TemplateResponse(request, "backtest_hub.html", _ctx(request, **ctx))


@app.post("/backtest/{name}/fetch")
async def backtest_fetch(request: Request, name: str):
    run = hub.load_run(name)
    if run is None:
        return _redirect("/backtest", f"No run named {name!r}.")
    form = await request.form(max_fields=100000, max_files=10)
    keys = [str(k) for k in form.getlist("key")]
    with _session() as session:
        data = hub.data_check(session, run)
    ids = next((it.get("ids") or [] for it in data["needs"] if it["key"] == "scrape"), [])
    jobs = hub.fetch_commands(run, keys, ids, str(form.get("username") or "") or None, bool(form.get("buy_props")))
    for label, cmd in jobs:
        j = runner.submit(f"{name}: {label}", cmd, meta={"step": "fetch", "run": name})
        run.setdefault("jobs", []).append({"id": j.id, "name": label, "at": datetime.now().isoformat(timespec="seconds")})
    run["status"] = "data"
    hub.save_run(run)
    return _redirect(f"/backtest/{name}", f"Started {len(jobs)} fetch job(s) (see Jobs). Reload this page to re-check the window." if jobs else "Nothing ticked.")


@app.post("/backtest/{name}/settle")
async def backtest_settle(request: Request, name: str):
    run = hub.load_run(name)
    if run is None:
        return _redirect("/backtest", f"No run named {name!r}.")
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    j = runner.submit(f"{name}: settle {run['since']}..{run['until']}", hub.settle_command(run, form.get("username") or None, scrape=bool(form.get("scrape"))),
                      meta={"step": "settle", "run": name})
    run.setdefault("jobs", []).append({"id": j.id, "name": "settle", "at": datetime.now().isoformat(timespec="seconds")})
    hub.save_run(run)
    return _redirect(f"/backtest/{name}", "Settling the window (see Jobs); reload to re-check.")


@app.post("/backtest/{name}/continue")
async def backtest_continue(request: Request, name: str):
    run = hub.load_run(name)
    if run is None:
        return _redirect("/backtest", f"No run named {name!r}.")
    run["status"] = "contests"
    hub.save_run(run)
    return _redirect(f"/backtest/{name}?step=contests")


@app.post("/backtest/{name}/contests")
async def backtest_contests(request: Request, name: str):
    run = hub.load_run(name)
    if run is None:
        return _redirect("/backtest", f"No run named {name!r}.")
    form = await request.form(max_fields=100000, max_files=10)
    picks = {}
    for cid in form.getlist("pick"):
        try:
            picks[str(int(cid))] = max(1, int(float(form.get(f"entries_{cid}") or 1)))
        except ValueError:
            picks[str(int(cid))] = 1
    if not picks:
        return _redirect(f"/backtest/{name}?step=contests", "Tick at least one contest.")
    with _session() as session:
        limits = {str(c["contest_id"]): int(c["max_entries"]) for sl in hub.contests_view(session, run, top=10 ** 6) for c in sl["contests"]}
    picks = {cid: min(n, limits.get(cid, n)) for cid, n in picks.items()}
    run["picks"] = picks
    run["status"] = "params"
    hub.save_run(run)
    return _redirect(f"/backtest/{name}?step=params")


def _replay_in_progress(name: str) -> Optional[str]:
    """The id of a queued or running replay / validate job of this run, if any (two at once corrupt the replay databases)."""
    for j in runner.active():
        if (j.meta or {}).get("run") == name and (j.meta or {}).get("step") in ("replay", "validate", "resettle"):
            return j.id
    return None


@app.post("/backtest/{name}/rerun")
async def backtest_rerun(request: Request, name: str):
    """Run the saved parameters again (after a fix, or when the replay found nothing)."""
    run = hub.load_run(name)
    if run is None or not run.get("picks") or not run.get("params"):
        return _redirect(f"/backtest/{name}", "Nothing to rerun: choose contests and parameters first.")
    busy = _replay_in_progress(name)
    if busy:
        return _redirect(f"/backtest/{name}?step=results", f"A replay of this run is still going (job {busy}); cancel it on the Jobs page before starting another.")
    form = {k: str(v) for k, v in (await request.form()).items()}
    resume = form.get("mode") == "resume"
    hub.write_selection(run)
    j1 = runner.submit(f"{name}: {'resume' if resume else 'replay'} {len(run['picks'])} contests x {run['params']['repeats']}",
                       hub.replay_command(run, redo=not resume), meta={"step": "replay", "run": name})
    j2 = runner.submit(f"{name}: assumptions vs actuals", hub.validate_command(run), meta={"step": "validate", "run": name})
    run.setdefault("jobs", []).extend([{"id": j1.id, "name": "replay", "at": datetime.now().isoformat(timespec="seconds")},
                                       {"id": j2.id, "name": "validate", "at": datetime.now().isoformat(timespec="seconds")}])
    run["status"] = "running"
    hub.save_run(run)
    return _redirect(f"/backtest/{name}?step=results", f"Running {name} again (see the progress below).")


@app.post("/backtest/{name}/resettle")
async def backtest_resettle(request: Request, name: str):
    """Re-settle the replayed slates against the live database and re-run the checks; no re-run of the pipeline."""
    run = hub.load_run(name)
    if run is None or not (hub.run_root(name) / "nfl").is_dir():
        return _redirect(f"/backtest/{name}", "Nothing to settle yet.")
    j1 = runner.submit(f"{name}: settle results", hub.resettle_command(run), meta={"step": "resettle", "run": name})
    j2 = runner.submit(f"{name}: assumptions vs actuals", hub.validate_command(run), meta={"step": "validate", "run": name})
    run.setdefault("jobs", []).extend([{"id": j1.id, "name": "resettle", "at": datetime.now().isoformat(timespec="seconds")},
                                       {"id": j2.id, "name": "validate", "at": datetime.now().isoformat(timespec="seconds")}])
    run["status"] = "running"
    hub.save_run(run)
    return _redirect(f"/backtest/{name}?step=results", "Settling the results again against the current database (see the progress below).")


@app.post("/backtest/{name}/run")
async def backtest_run(request: Request, name: str):
    run = hub.load_run(name)
    if run is None or not run.get("picks"):
        return _redirect(f"/backtest/{name}?step=contests", "Choose contests first.")
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    try:
        params = {"repeats": max(1, int(form.get("repeats") or 1)), "workers": max(1, int(form.get("workers") or 1)),
                  "lineups": int(form.get("lineups") or 3000), "sims": int(form.get("sims") or 500), "field_cap": int(form.get("field_cap") or 5000),
                  "seed": int(form.get("seed") or 0),
                  "method": (form.get("method") or "greedy").strip().lower(), "bankroll": float(form.get("bankroll") or 3.0),
                  "cash_pct": float(form.get("cash_pct") if form.get("cash_pct") not in (None, "") else 0.5),
                  "avg_roi_win": float(form.get("avg_roi_win") if form.get("avg_roi_win") not in (None, "") else 0.5),
                  "diversity_lambda": float(form.get("diversity_lambda") if form.get("diversity_lambda") not in (None, "") else 0.1)}
        if params["method"] not in ("greedy", "portfolio"):
            raise ValueError(params["method"])
    except ValueError:
        return _redirect(f"/backtest/{name}?step=params", "The parameters must be numbers (and the chooser greedy or portfolio).")
    busy = _replay_in_progress(name)
    if busy:
        return _redirect(f"/backtest/{name}?step=results", f"A replay of this run is still going (job {busy}); cancel it on the Jobs page before starting another.")
    run["params"] = params
    hub.write_selection(run)
    j1 = runner.submit(f"{name}: replay {len(run['picks'])} contests x {params['repeats']}", hub.replay_command(run), meta={"step": "replay", "run": name})
    j2 = runner.submit(f"{name}: assumptions vs actuals", hub.validate_command(run), meta={"step": "validate", "run": name})
    run.setdefault("jobs", []).extend([{"id": j1.id, "name": "replay", "at": datetime.now().isoformat(timespec="seconds")},
                                       {"id": j2.id, "name": "validate", "at": datetime.now().isoformat(timespec="seconds")}])
    run["status"] = "running"
    hub.save_run(run)
    return _redirect(f"/backtest/{name}?step=results", f"Running {name}: {len(run['picks'])} contests, {params['repeats']} repeat(s) (see Jobs).")


@app.get("/backtest/tax", response_class=HTMLResponse)
def backtest(request: Request, r: Optional[str] = None):
    rows = bv.reports()
    picked = next((x for x in rows if x["id"] == r), rows[0] if rows else None)
    view = bv.report_view(picked["path"]) if picked else None
    return templates.TemplateResponse(request, "backtest.html", _ctx(
        request, reports=rows, view=view, runs=bv.runs_in_progress(),
        dist=bv.dist_rows(view["repeats"]) if view and view.get("repeats") else []))


@app.get("/backtest/file/{report_id:path}/{name}")
def backtest_file(report_id: str, name: str):
    d = bv.report_path(report_id)
    if d is None or name not in bv.FILES or not (d / name).is_file():
        raise HTTPException(404)
    media = {"csv": "text/csv", "json": "application/json", "png": "image/png"}[name.rsplit(".", 1)[1]]
    return FileResponse(d / name, media_type=media, filename=None if name.endswith(".png") else name)


@app.post("/backtest/report")
async def backtest_report(request: Request):
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    root = bv.OUTPUT_DIR / Path(form.get("root", "")).name
    if not root.name.startswith("replay") or not root.is_dir():
        raise HTTPException(404, "unknown replay root")
    since, until = form.get("since", "").strip(), form.get("until", "").strip()
    try:
        datetime.fromisoformat(since); datetime.fromisoformat(until)
        bankroll = float(form.get("bankroll") or 0)
        tax_year = int(form["tax_year"]) if form.get("tax_year") else None
    except (ValueError, KeyError):
        return _redirect("/backtest/tax", "Since / until must be dates (YYYY-MM-DD); bankroll and tax year numbers.")
    cmd = bv.report_command(root, since, until, bankroll, tax_year)
    runner.submit(f"Backtest report: {root.name} {since}..{until}", cmd, meta={"step": "backtest_report", "form": form})
    return _redirect("/backtest/tax", f"Building the report for {root.name} {since}..{until}; it appears here when the job finishes (see Jobs).")


@app.get("/validate", response_class=HTMLResponse)
def validate_page(request: Request, r: Optional[str] = None):
    rows = vv.reports()
    picked = next((x for x in rows if x["id"] == r), rows[0] if rows else None)
    view = vv.report_view(picked["path"]) if picked else None
    today = datetime.now().date()
    return templates.TemplateResponse(request, "validate.html", _ctx(
        request, reports=rows, view=view, replay_roots=vv.replay_roots(),
        default_since=(today.replace(day=1)).isoformat(), default_until=today.isoformat()))


@app.get("/validate/file/{report_id:path}/{name}")
def validate_file(report_id: str, name: str):
    d = vv.report_path(report_id)
    if d is None or name not in vv.FILES or not (d / name).is_file():
        raise HTTPException(404)
    media = {"csv": "text/csv", "json": "application/json"}[name.rsplit(".", 1)[1]]
    return FileResponse(d / name, media_type=media, filename=name)


@app.post("/validate/run")
async def validate_run(request: Request):
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    mode = form.get("mode", "live")
    since, until = form.get("since", "").strip(), form.get("until", "").strip()
    try:
        for v in (since, until):
            if v:
                datetime.fromisoformat(v)
        repeat = int(form.get("repeat") or 0)
    except ValueError:
        return _redirect("/validate", "Since / until must be dates (YYYY-MM-DD); repeat a number.")
    root = None
    if mode.startswith("replay:"):
        root = vv.OUTPUT_DIR / Path(mode.split(":", 1)[1]).name
        if not root.name.startswith("replay") or not (root / "nfl").is_dir():
            raise HTTPException(404, "unknown replay root")
    cmd = vv.validate_command("replay" if root else "live", since, until, root, repeat)
    label = f"replay {root.name} repeat {repeat}" if root else f"live {since}..{until}"
    runner.submit(f"Validation: {label}", cmd, meta={"step": "validate", "form": form})
    return _redirect("/validate", f"Running the validation ({label}); it appears here when the job finishes (see Jobs).")


@app.get("/dupes", response_class=HTMLResponse)
def dupes_page(request: Request, c: Optional[List[int]] = Query(None)):
    """Predicted vs actual lineup duplication over scraped fields (ivan.sports.nfl.dupe_analysis)."""
    slate_list = _with_db(dv.slates) or []
    selected = [int(x) for x in c] if c else dv.default_selection(slate_list)
    for s in slate_list:
        s["selected"] = any(x["id"] in selected for x in s["contests"])
    rep = _with_db(lambda s: dv.report(s, selected)) if selected else {"contests": []}
    return templates.TemplateResponse(request, "dupes.html", _ctx(request, slates=slate_list, selected=selected, rep=rep))


@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request, c: Optional[str] = None, r: Optional[str] = None):
    """One component of the pipeline at a time: its test runs (scoreboard over commits) and the picked run's detail."""
    comp = mv.component(c)
    run_list = mv.runs(comp["key"], lambda jid: (runner.get(jid).status if runner.get(jid) else None))
    picked = next((x for x in run_list if x["name"] == r), run_list[0] if run_list else None)
    if picked is not None:
        picked = {**picked, "report": mv.run_view(picked)}
    today = datetime.now().date()
    return templates.TemplateResponse(request, "models.html", _ctx(
        request, components=mv.COMPONENTS, comp=comp, stage_params=mv.STAGE_PARAMS[comp["through"]], defaults=mv.DEFAULTS,
        board=mv.scoreboard(run_list), view=picked,
        default_since=(today - timedelta(days=7)).isoformat(), default_until=today.isoformat()))


@app.post("/models/run")
async def models_run(request: Request):
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    comp = mv.component(form.get("component"))
    since, until = form.get("since", "").strip(), form.get("until", "").strip()
    back = f"/models?c={comp['key']}"
    try:
        params = {k: int(form.get(k) or mv.DEFAULTS[k]) for k in mv.STAGE_PARAMS[comp["through"]]}
        run = mv.new_run(comp["key"], since, until, params, label=form.get("label", ""))
    except ValueError:
        return _redirect(back, "Since / until must be dates (YYYY-MM-DD); the parameters numbers.")
    busy = next((j for j in runner.active() if (j.meta or {}).get("step") == "replay"), None)
    if busy is not None:                   # two replays at once doubled the memory and froze the machine once
        return _redirect(back, f"A replay is already queued or running ({busy.name}); wait for it or cancel it on the Jobs page.")
    j1 = runner.submit(f"models {comp['key']}: replay through {comp['through']} {since}..{until}", mv.replay_command(run),
                       meta={"step": "replay", "run": f"models/{comp['key']}/{run['name']}"})
    j2 = runner.submit(f"models {comp['key']}: score {run['name']}", mv.validate_command(run),
                       meta={"step": "validate", "run": f"models/{comp['key']}/{run['name']}"})
    run["jobs"] = [{"id": j1.id, "name": "replay"}, {"id": j2.id, "name": "validate"}]
    mv.save_run(run)
    return _redirect(f"{back}&r={run['name']}", f"Running the {comp['label'].lower()} test through the {comp['through']} stage (see Jobs).")


@app.get("/backtest/plan", response_class=HTMLResponse)
def backtest_plan_page(request: Request, since: Optional[str] = None, until: Optional[str] = None, menu: Optional[str] = None,
                       sport: str = "nfl"):
    today = datetime.now().date()
    default_since, default_until = (today - timedelta(days=7)).isoformat(), today.isoformat()
    plan = None
    err = None
    if since and until:
        try:
            with _session() as session:
                plan = pv.plan_view(session, since.strip(), until.strip(), menu or "")
        except (ValueError, json.JSONDecodeError) as e:
            err = f"Could not build the plan: {e}"
    return templates.TemplateResponse(request, "backtest_plan.html", _ctx(
        request, plan=plan, menu_text=(menu if menu else pv.default_menu_text()), default_since=default_since, default_until=default_until,
        msg=err or request.query_params.get("msg")))


@app.post("/backtest/plan/run")
async def backtest_plan_run(request: Request):
    form = {k: str(v) for k, v in (await request.form(max_fields=100000, max_files=10)).items()}
    action = form.get("action", "")
    since, until = form.get("since", "").strip(), form.get("until", "").strip()
    try:
        datetime.fromisoformat(since); datetime.fromisoformat(until)
    except ValueError:
        return _redirect("/backtest/plan", "Since / until must be dates (YYYY-MM-DD).")
    back = f"/backtest/plan?since={since}&until={until}"
    if action == "replay":
        root = pv.replay_root(since, until)
        runner.submit(f"Replay {since}..{until}", pv.replay_command(since, until, root), meta={"step": "replay", "form": form})
        runner.submit(f"Validation: replay {root.name}", pv.validate_replay_command(root, f"replay {since}..{until}"), meta={"step": "validate", "form": form})
        return _redirect(back, f"Replaying the window into {root.name}, then validating it (see Jobs; the report appears under Validation).")
    if action == "fetch_actuals":
        runner.submit("Fetch actuals (schedule, box scores, DST)", pv.fetch_actuals_command(), meta={"step": "fetch_actuals"})
        return _redirect(back, "Fetching box scores (see Jobs). nflverse publishes a day or two after the games.")
    if action == "scrape":
        with _session() as session:
            plan = pv.plan_view(session, since, until, form.get("menu", ""))
        ids = plan["unscraped_ids"]
        if not ids:
            return _redirect(back, "Nothing to scrape: every contest in the pool is already scraped.")
        runner.submit(f"Scrape {len(ids)} contests", pv.scrape_command(ids, form.get("username") or None), meta={"step": "scrape"})
        return _redirect(back, f"Scraping {len(ids)} contests through the saved Chrome profile (see Jobs). A 403 in the log means DraftKings wants a fresh login there.")
    if action == "build_models":
        runner.submit(f"Build models as of {since}", pv.build_models_command(since), meta={"step": "build_models"})
        return _redirect(back, f"Building the as-of {since} model version (see Jobs).")
    if action == "backfill":
        runner.submit(f"Backfill captures {since}..{until}", pv.backfill_command(since, until), meta={"step": "backfill"})
        return _redirect(back, "Backfilling captures from projection snapshots and stored game lines (no props; add --odds on the command line to buy them).")
    return _redirect(back, f"Unknown action {action!r}.")


@app.get("/downloads", response_class=HTMLResponse)
def downloads(request: Request):
    return templates.TemplateResponse(request, "downloads.html", _ctx(request, files=_outputs()))


@app.get("/downloads/{name}")
def download(name: str):
    if "/" in name or "\\" in name or not (name.endswith(".csv") or name.endswith(".json")):
        raise HTTPException(404)
    path = WEB_OUTPUT_DIR / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="text/csv" if name.endswith(".csv") else "application/json", filename=name)
