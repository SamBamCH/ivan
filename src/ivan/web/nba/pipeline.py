"""
The actions bar and the jobs: Refresh (the producer fetches, a consumer pulls the laptop), Snapshot, the chain, generate / sim / choose / field as jobs, the news and autopilot switches, the entries upload, the slate switch.
"""
from __future__ import annotations

import html
import json
import sys
import threading
import time
from datetime import date, datetime, timedelta
from ivan.sports.nba.ingest import restore_from_archive
from ivan.sports.nba import newsfeed as NF
from pathlib import Path
from typing import Optional
from fastapi import File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from ivan.core import feed
from ivan.sports.nba import autopilot
from ivan.core.utils import normalize_name
from ivan.database.session import get_session
from ivan.sports.nba.ingest import after_pull, capture_pool, refresh_pool, IngestError
from ivan.sports.nba.models import Lineup, LineupSimulationResult, NBAPlayer
from ivan.sports.nba.paths import NBA_DATA_DIR
from ivan.web import auth
from .core import CONSUMER, LIMITED, LOCK, MAX_JOBS, NEWS_AUTO_SECONDS, SCRIPTS, _busy, _here, _render, app, desk, engine, runner
from .edits import _apply_typed_pins
from .exposures import _side_ctx
from .knobs import MAX_UPLOAD, PIPELINE_DEFAULTS, _field, clamp_form, pipeline_form, write_desk_config
from .outputs import _entries_info, _entries_path, _latest_outputs
from ivan.sports.nba.desk import lineup_slates


def _start_restore(draft_group: int) -> None:
    """Load a past slate from its archive capture in a background thread (ingest.restore_from_archive): nothing fetched."""
    def run():
        log = []
        sess = get_session(engine)
        try:
            info = restore_from_archive(sess, int(draft_group), log=log.append)
            msg = f"slate {draft_group} restored from its {info['source']} capture of {info['captured_at']:%m-%d %H:%M} ({info['players']} players, {info['odds']} prop rows)"
        except IngestError as e:
            msg = f"restore failed: {e}"
        except Exception as e:   # noqa: BLE001
            msg = f"restore failed, the previous pool is unchanged: {type(e).__name__}: {e}"
        finally:
            sess.close()
        with LOCK:
            desk.rebuild(); desk.busy = None; desk.flash.append(msg)
    desk.busy = f"restoring slate {draft_group} from the archive"
    threading.Thread(target=run, name="ivan-restore", daemon=True).start()


def _start_refresh(what: str, draft_group: int, update_odds: bool, lines: Optional[bool] = None) -> None:
    """Fetch, store, project and write the public projection in a background thread with its own session (one transaction);
    the board shows `what` until it is done, then the desk rebuilds from the new pool. No field build: the sim builds the
    field it needs, the Field + ownership button builds it on demand."""
    def run():
        log = []
        sess = get_session(engine)
        try:
            _, _, news, moved = refresh_pool(sess, int(draft_group), update_odds=update_odds, starters=True, log=log.append, lines=lines)
            desk.changes = dict(at=datetime.now(), teams=moved or {})
            msg = f"{what}: done" + (_news_note(news) if news else "; the news check failed (see the auto box, or scripts/nba/newsfeed.py check)")
        except IngestError as e:
            msg = f"{what} failed: {e}"
        except Exception as e:   # noqa: BLE001 - the transaction rolled back: the old pool stands
            msg = f"{what} failed, the previous pool is unchanged: {type(e).__name__}: {e}"
        finally:
            sess.close()
        with LOCK:
            desk.rebuild(); desk.busy = None; desk.flash.append(msg)
    desk.busy = what
    threading.Thread(target=run, name="ivan-refresh", daemon=True).start()


@app.post("/publish")
def publish(request: Request, pins: str = Form("")):
    """Snapshot: record the projection as it stands (ingest.capture_pool: the capture rows the scorecard scores and the projections
    CSV), the values still typed into a grid applied first. Nothing is projected or pushed: every edit already wrote its team
    into the pool (Desk.reproject), and the server pulls the pool from this laptop when its own Refresh is pressed.
    Terminal: scripts/nba/snapshot.py."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        typed = _apply_typed_pins(pins)
        if typed:
            desk.flash.append(f"{typed} typed value(s) applied first")
        desk.write_local()
        try:
            info = capture_pool(desk.session, log=lambda m: None)
            desk.flash.append(f"snapshot: {info.players} players recorded ({info.captures} capture rows, {len(desk.overrides)} edits)"
                              + (f"; {Path(info.projections_path).name}" if info.projections_path else ""))
            autopilot.touch("publish")
        except IngestError as e:
            desk.flash.append(f"snapshot failed: {e}")
        desk.rebuild()
    return RedirectResponse(_here(request), status_code=303)


def _news_note(r: dict) -> str:
    """'; news: 3 new, 1 on this slate: X · headline' after a refresh read Rotowire (the desk's news clock moves too)."""
    desk.news_checked_at = r["fetched_at"]
    pool = {normalize_name(p.name) for p in desk.slate.players}
    on_slate = [x for x in r["new"] if x.player_name in pool]
    return f"; news: {len(r['new'])} new" + (f", {len(on_slate)} on this slate: " + "; ".join(f"{x.player_raw} · {x.headline}" for x in on_slate[:6]) if on_slate else "")


def _news_check(log) -> dict:
    sess = get_session(engine)
    try:
        r = NF.check(sess, log=log)
        pool = {p.name for p in desk.slate.players}
        on_slate = [x for x in r["new"] if x.player_name in pool]
        desk.news_checked_at = r["fetched_at"]
        return dict(new=len(r["new"]), on_slate=on_slate, total=r["total"])
    finally:
        sess.close()


@app.post("/news/auto")
def news_auto(request: Request, on: str = Form("")):
    """Auto-check every NEWS_AUTO_SECONDS while the desk runs (off by default; nothing else fetches on its own)."""
    with LOCK:
        desk.news_auto = bool(on)
        desk.flash.append("news auto-check " + ("on: every 5 minutes while the desk runs" if desk.news_auto else "off"))
    return RedirectResponse(_here(request), status_code=303)


@app.post("/autopilot")
def autopilot_toggle(request: Request, on: str = Form("")):
    """The autopilot switch (ivan.sports.nba.autopilot; scripts/nba/autopilot.py on|off from a terminal): with it on, the
    capture tick refreshes the loaded slate every 15 minutes, fetches props at the offsets and publishes + pushes the feed,
    standing down while this desk is being worked. Nothing on a consumer."""
    if CONSUMER:
        return RedirectResponse(_here(request), status_code=303)
    with LOCK:
        st = autopilot.set_on(bool(on), by="desk")
        desk.flash.append("autopilot " + ("on: the tick refreshes, fetches props at the offsets and re-projects every 15 minutes while no job runs here; the other desk pulls it on its Refresh"
                                          if st["on"] else "off: the pool moves only when you refresh or edit; the other desk pulls whatever it holds"))
    return RedirectResponse(_here(request), status_code=303)


def _news_loop():
    while True:
        time.sleep(NEWS_AUTO_SECONDS)
        if CONSUMER or not getattr(desk, "news_auto", False):     # a consumer's headlines come with the laptop's pool
            continue
        try:
            with LOCK:
                r = _news_check(lambda m: None)
                if r["on_slate"]:
                    desk.flash.append(f"news: {len(r['on_slate'])} new on this slate: " + "; ".join(f"{x.player_raw} · {x.headline}" for x in r["on_slate"][:6]))
        except Exception:   # noqa: BLE001 - the next tick tries again
            pass


threading.Thread(target=_news_loop, name="ivan-news", daemon=True).start()


@app.post("/reload")
def reload(request: Request):
    """Load what is on disk without fetching anything: the pool another process wrote (scripts/nba/ingest.py or project.py
    in a terminal, a feed pull), the pins file, the lineups. The "new data" badge posts here."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        desk.rebuild()
        desk.flash.append("reloaded from disk")
    return RedirectResponse(_here(request), status_code=303)


@app.post("/refresh")
def refresh(request: Request, what: str = Form("dk")):
    if CONSUMER and what in ("props", "lines"):
        what = "dk"                                              # a consumer has no props button; the laptop's capture loop fetches them
    """dk: pool + statuses + Rotowire starters and news (free). lines: also the game lines, spread and total (~3 credits).
    props: also the player props and the game lines (~12 credits a game). A consumer pulls the feed and reads the news."""
    with LOCK:
        dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
        if CONSUMER:                                             # Refresh from laptop: the producer publishes on request, this desk copies and applies it (feed.pull_ssh)
            try:
                from ivan.sports.nba import changes as CH
                before = CH.snapshot(desk.session)
                changed = feed.pull_from_producer("nba", NBA_DATA_DIR.parent / "feed-state", log=desk.flash.append)
                tables = {k: v for k, v in changed.items() if k != "published"}
                if tables:
                    desk.session.expire_all()                        # the pool was replaced underneath this session
                    r = after_pull(desk.session, dg, log=desk.flash.append)
                    desk.changes = dict(at=datetime.now(), teams=CH.diff(before, CH.snapshot(desk.session)))
                    if r["moved"]:
                        desk.flash.append(f"the laptop moved to slate {r['draft_group']}: your edits of the previous slate are filed, this slate's are back, the previous slate's lineups cleared")
                    if r["teams"]:
                        desk.flash.append(f"the laptop's pool is in (headlines included); your own edits re-projected on top for {', '.join(sorted(r['teams']))}")
                    else:
                        desk.flash.append("the laptop's pool is in, as it stands there (headlines included)")
                else:
                    fs = feed.status("nba")
                    prod = fs.get("producer") or {}
                    desk.flash.append("nothing new: the laptop's pool has not changed since the last pull"
                                      + (f" (autopilot {'on' if prod.get('autopilot') else 'off'} there)" if prod else ""))
                desk.news_checked_at = datetime.now()                # the headlines came with the pull
            except Exception as e:   # noqa: BLE001
                desk.flash.append(f"pull from the laptop failed: {type(e).__name__}: {e}")
        elif dg is None:
            desk.flash.append("no draft group in the pool; pick a slate first")
        elif runner.active():
            desk.flash.append("a job is running; it reads the pool, so refresh after it finishes (or cancel it)")
        elif desk.busy:
            desk.flash.append(f"already {desk.busy}")
        else:
            _start_refresh("refreshing the pool" + (" and the props" if what == "props" else " and the game lines" if what == "lines" else ""), dg,
                           update_odds=(what == "props"), lines=True if what == "lines" else None)
            return RedirectResponse(_here(request), status_code=303)
        desk.rebuild()
    return RedirectResponse(_here(request), status_code=303)


def _pipeline_ctx() -> dict:
    jobs = runner.all()[:8]
    cur = runner.current() or next((j for j in jobs if j.status == "queued"), None)
    log = runner.read_log(cur, tail_lines=14) if cur else (runner.read_log(jobs[0], tail_lines=14) if jobs else "")
    try:
        slates = lineup_slates(desk.session)
    except Exception:   # noqa: BLE001 - a desk without the lineup tables yet
        slates = []
    current = next((p.draft_group_id for p in (desk.slate.players if desk.slate else []) if p.draft_group_id), None)
    from ivan.sports.nba.field_generation import slate_lineups
    n_lineups = slate_lineups(desk.session).count()                    # the loaded slate's (other recent slates' are kept beside them)
    n_sims = desk.session.query(LineupSimulationResult).filter(LineupSimulationResult.lineup_id.in_(slate_lineups(desk.session, Lineup.id))).count()
    projected = desk.session.query(NBAPlayer).filter(NBAPlayer.fpts_mean.isnot(None)).count() > 0
    ent = _entries_info()
    # each step opens when the one before it has run: generate needs a projected pool, sim needs lineups (and entries), choose needs sims
    ready = dict(generate=projected, sim=bool(projected and n_lineups and ent.get("exists")), choose=bool(n_sims and ent.get("exists")),
                 all=bool(projected and ent.get("exists")))
    just_finished = bool(jobs and not runner.active() and jobs[0].finished and (time.time() - jobs[0].finished) < 8)
    now = datetime.now()
    return dict(form=pipeline_form, jobs=jobs, current=cur, log=log, active=bool(runner.active()), entries=ent,
                outputs=_latest_outputs(), lineups_in_db=n_lineups, sims_in_db=n_sims, lineup_slates=slates, current_draft_group=current,
                side=_side_ctx(), ready=ready, projected=projected, just_finished=just_finished,
                fresh=desk.freshness(now), chain=_chain_info(now), news=desk.news_status(now), now=now)


PUBLISH_RED_MIN = 10   # unpublished changes older than this turn the published label red


CHAIN_STEPS = [("snapshot", "snapshot", ("snapshot",)), ("field", "field", ("field",)), ("generate", "lineups", ("generate", "all")),
               ("sim", "sim", ("sim", "all")), ("choose", "picks", ("choose", "all")), ("lateswap", "late swap", ("lateswap",))]


def _chain_info(now: datetime) -> dict:
    """
    The slate's chain of stages and whether each is current: the projection (the pool, written by every edit and refresh;
    its time is the newest fpts_updated_at), then what was built on it: the snapshot (the capture rows the scorecard scores),
    the field + ownership, the lineups, the sim, the picks, the late swap. A stage built before the projection it should
    rest on is stale (amber; red when the projection moved PUBLISH_RED_MIN minutes ago or more and the stage never followed).
    The stages' times are the desk's job runner's (data/nba/jobs) and the capture table's; a script run from a terminal
    shows up through the table only (the snapshot), the rest reads "not run here".
    """
    proj_at = desk.pool_stamp[0] if desk.pool_stamp else None
    if isinstance(proj_at, str):
        try:
            proj_at = datetime.fromisoformat(proj_at)
        except ValueError:
            proj_at = None
    dg = desk.pool_stamp[2] if desk.pool_stamp else None
    jobs = runner.all()
    done = {}
    for j in jobs:
        if j.status == "done" and j.finished and j.name not in done:
            done[j.name] = datetime.fromtimestamp(j.finished)
    snap_at = None
    try:
        from sqlalchemy import text as _t
        v = desk.session.execute(_t("select max(captured_at) from nba_projection_capture where source = 'ours' and draft_group_id = :dg").bindparams(dg=dg)).scalar()
        snap_at = datetime.fromisoformat(str(v)) if v else None
    except Exception:   # noqa: BLE001
        snap_at = None
    stages = []
    worst = ""
    for key, label, names in CHAIN_STEPS:
        at = snap_at if key == "snapshot" else next((done[n] for n in sorted(names, key=lambda n: done.get(n) or datetime.min, reverse=True) if n in done), None)
        stale = bool(at is not None and proj_at is not None and at < proj_at - timedelta(seconds=2))
        level = ""
        if stale:
            level = "red" if (now - proj_at).total_seconds() / 60 >= PUBLISH_RED_MIN else "amber"
        if key == "lateswap" and at is None:
            continue                                            # only shown once a late swap has run
        stages.append(dict(key=key, label=label, at=at, stale=stale, level=level,
                           why=("built before the projection moved" if stale else ("current" if at is not None else "not run on this desk"))))
        if level == "red" or (level == "amber" and worst != "red"):
            worst = level
    return dict(projection_at=proj_at, stages=stages, level=worst,
                behind=[s["label"] for s in stages if s["stale"]], age=None if proj_at is None else (now - proj_at).total_seconds() / 60)


def _contests_ctx() -> dict:
    """The entries file's contests against the loaded slate: each contest's stored draft group (historical_contest_information),
    our entries in it, and whether it belongs to the slate loaded at the top (the sim skips the ones that do not)."""
    ent = _entries_info()
    dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
    rows = []
    if ent.get("exists") and not ent.get("error"):
        try:
            from ivan.sports.nba.lineup_export import read_entries
            from ivan.sports.nba.models import HistoricalContest
            e = read_entries(_entries_path())
            for (cid, name), g in e.groupby(["Contest ID", "Contest Name"]):
                c = desk.session.query(HistoricalContest).filter(HistoricalContest.contest_id == int(cid)).first()
                cdg = None if c is None else c.draft_group_id
                rows.append(dict(contest_id=int(cid), name=str(name).strip(), entries=int(len(g)), fee=None if c is None else c.entry_fee,
                                 field=None if c is None else c.total_entries, draft_group=cdg,
                                 match=("unknown" if cdg is None else ("ok" if dg is not None and int(cdg) == int(dg) else "other"))))
        except Exception as ex:   # noqa: BLE001
            ent = dict(ent, error=f"{type(ex).__name__}: {ex}")
    rows.sort(key=lambda r: (r["match"] != "other", -(r["fee"] or 0) * r["entries"]))
    from ivan.sports.nba.lineup_export import contest_weights
    cw = contest_weights()
    for r in rows:
        r["weights"] = cw.get(str(r["contest_id"]), {})
    return dict(entries=ent, contests=rows, draft_group=dg, other=sum(1 for r in rows if r["match"] == "other"),
                unknown=sum(1 for r in rows if r["match"] == "unknown"))


@app.post("/contest/{contest_id}/weights", response_class=HTMLResponse)
async def contest_weights_set(contest_id: int, request: Request):
    """One contest's chooser knobs from the contests section (= scripts/nba/contest_weights.py set); blank = the panel's value."""
    from ivan.sports.nba.lineup_export import CONTEST_KNOBS, set_contest_weights
    form = await request.form()
    try:
        cur = set_contest_weights(contest_id, {k: form.get(k) for k in CONTEST_KNOBS if k in form})
    except ValueError as e:
        return HTMLResponse(f'<span class="bad">{html.escape(str(e))}</span>')
    return HTMLResponse('<span class="ok">saved</span>' if cur else '<span class="muted">panel</span>')


def _submit(step: str, form: dict) -> list:
    py = sys.executable
    jobs = []
    cfg = write_desk_config(form)
    n_jobs = max(1, min(MAX_JOBS, int(_field(form.get("jobs"), "jobs", int) or MAX_JOBS)))
    if step in ("generate", "all"):
        cmd = [py, str(SCRIPTS / "generate_lineups.py"), str(int(form["lineups"])), str(n_jobs), "-c", str(cfg), "--no-progress"] + (["--fresh"] if str(form.get("fresh", "on")) else [])
        jobs.append(runner.submit("generate", cmd, meta=dict(step="generate")))
    if step in ("sim", "all"):
        jobs.append(runner.submit("sim", [py, str(SCRIPTS / "sim.py"), "--config", str(cfg), "-n", str(int(form["sims"])), "-c", str(int(form["field_cap"]))], meta=dict(step="sim")))
    if step == "lateswap":
        source = "raw" if LIMITED else str(form.get("ls_source") or "auto")     # no DraftKings login here: the uploaded standings
        field = "each" if str(form.get("ls_field") or "shared") == "each" else "shared"
        cmd = [py, str(SCRIPTS / "lateswap.py"), "--source", source, "--field", field,
               "--candidates", str(int(_field(form.get("ls_candidates"), "candidates", int) or 10)),
               "--field-cap", str(int(_field(form.get("ls_field_cap"), "field cap", int) or 5000)),
               "--sims", str(int(_field(form.get("ls_sims"), "sims", int) or 1000))] + (["--live"] if form.get("ls_live") else [])
        jobs.append(runner.submit("lateswap", cmd, meta=dict(step="lateswap")))
    if step == "fetch_standings":
        jobs.append(runner.submit("fetch_standings", [py, str(SCRIPTS / "fetch_standings.py")], meta=dict(step="fetch_standings")))
    if step == "field":
        jobs.append(runner.submit("field", [py, str(SCRIPTS / "field_ownership.py")], meta=dict(step="field")))
    if step in ("choose", "all"):
        ch = json.loads(cfg.read_text())["choose"]
        cmd = [py, str(SCRIPTS / "choose.py"), "--config", str(cfg), "--no-prompt", "--method", str(ch["method"]),
               "--cash-pct", "%g" % ch["cash_pct"], "--avg-roi-win", "%g" % ch["avg_roi_win"], "--diversity-lambda", "%g" % ch["diversity_lambda"], "--bankroll", "%g" % ch["bankroll"]]
        from ivan.sports.nba.lineup_export import CONTEST_WEIGHTS_JSON, contest_weights
        if contest_weights():                                     # the contests section's knobs win over the panel's for their contest
            cmd += ["--weights-json", str(CONTEST_WEIGHTS_JSON)]
        jobs.append(runner.submit("choose", cmd, meta=dict(step="choose")))
    return jobs


@app.get("/pipeline/status", response_class=HTMLResponse)
def pipeline_status(request: Request):
    """Only the job status, log and results: what polls while a job runs, so nobody's typing in the form is overwritten."""
    with LOCK:
        return _render(request, "_nba_pipeline_status.html", **_pipeline_ctx())


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline(request: Request):
    with LOCK:
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


@app.post("/pipeline/{step}", response_class=HTMLResponse)
async def pipeline_run(request: Request, step: str):
    """generate | sim | choose | all | lateswap | field. Generate, sim, late swap and all publish the desk's projections first, so
    the pool is what you see on the grid (2026-10-25: a sim run on a stale pool after a repaint was the late-swap mess); choose
    and field (the public field + ownership, scripts/nba/field_ownership.py) do not."""
    form = await request.form()
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        # only the knobs the request carries move (generate and sim send their own popover's knobs; a button without any,
        # Field + ownership, used to blank every knob); the checkboxes count only when the config form was sent (its cfg field)
        pipeline_form.update({k: str(form.get(k)) for k in PIPELINE_DEFAULTS if k in form and k not in ("fresh", "jam", "ls_live")})
        notes = clamp_form(pipeline_form)
        if notes:
            desk.flash.append("lowered to this desk's limits: " + ", ".join(notes))
        if form.get("cfg"):                                       # the whole knob form (scripts and older pages)
            pipeline_form["jam"] = "on" if form.get("jam") else ""
            pipeline_form["fresh"] = "on" if form.get("fresh") else ""
        elif form.get("cfg_fresh"):                               # generate's knobs (the popover on its button): its checkbox only
            pipeline_form["fresh"] = "on" if form.get("fresh") else ""
        if step == "lateswap":
            pipeline_form["ls_live"] = "on" if form.get("ls_live") else ""
        if step not in ("generate", "sim", "choose", "all", "lateswap", "field"):
            desk.flash.append(f"unknown step {step}")
        elif runner.active():
            desk.flash.append("a job is already running; wait for it or cancel it")
        else:
            ms = desk.mapped_summary()
            if ms["n"] and ms["n_mapped"] < ms["n"] and step != "choose":
                desk.flash.append(f"heads up: {ms['n_mapped']} of {ms['n']} teams mapped and unchanged"
                                  + (f"; changed: {', '.join(ms['changed'])}" if ms["changed"] else "") + (f"; unmapped: {', '.join(ms['unmapped'])}" if ms["unmapped"] else ""))
            if step in ("generate", "sim", "lateswap", "all"):
                # the pool is the projection (every edit wrote its team); a snapshot records what this run was built on
                typed = _apply_typed_pins(str(form.get("pins") or ""))
                desk.write_local()
                try:
                    info = capture_pool(desk.session, log=lambda m: None)
                    desk.flash.append(f"snapshot of {info.players} players ({len(desk.overrides)} edits)" + (f", {typed} typed values applied first" if typed else "") + f"; queued {step}")
                    autopilot.touch("publish")
                except IngestError as e:
                    desk.flash.append(f"snapshot failed: {e}")
            try:
                _submit(step, pipeline_form)
            except (TypeError, ValueError) as e:
                desk.flash.append(f"bad setting: {e}")
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


@app.post("/pipeline/cancel/{job_id}", response_class=HTMLResponse)
def pipeline_cancel(request: Request, job_id: str):
    with LOCK:
        runner.cancel(job_id)
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


@app.post("/entries", response_class=HTMLResponse)
async def entries_upload(request: Request, file: UploadFile = File(...)):
    """The DraftKings entries export, saved as the configured entries file."""
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD or not (file.filename or "").lower().endswith(".csv"):
        with LOCK:
            desk.flash.append("entries upload refused: a .csv under 5 MB, please")
        from fastapi.responses import Response
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    with LOCK:
        p = _entries_path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        cx = _contests_ctx()
        desk.flash.append(f"entries file saved: {file.filename} -> {p.name} ({len(data)} bytes)"
                          + (f"; {cx['other']} contest(s) belong to another slate than the loaded one: the sim skips them" if cx["other"] else ""))
    from fastapi.responses import Response
    return Response(status_code=200, headers={"HX-Refresh": "true"})


@app.post("/slate")
def switch_slate(request: Request, draft_group: int = Form(...), source: str = Form("lobby"), day: str = Form("")):
    """Load another slate: a lobby slate is fetched from DraftKings, an archived one restored from its capture (nothing fetched).
    The pick is remembered (data/nba/active_slate.json)."""
    with LOCK:
        from ivan.sports.nba.paths import ACTIVE_SLATE_JSON
        try:
            ACTIVE_SLATE_JSON.write_text(json.dumps({"draft_group": int(draft_group), "chosen_at": datetime.now().isoformat(timespec="seconds")}))
        except OSError as e:
            desk.flash.append(f"could not save the slate choice ({e}); the capture tick may switch back")
        if runner.active():
            desk.flash.append("a job is running on the current slate; switch after it finishes (or cancel it)")
        elif desk.busy:
            desk.flash.append(f"already {desk.busy}")
        elif source == "archive":
            _start_restore(int(draft_group))
        else:
            _start_refresh(f"loading slate {draft_group}", int(draft_group), update_odds=False)
    if day:
        try:
            return RedirectResponse(f"{auth.root_of(request)}/{date.fromisoformat(day).isoformat()}/{int(draft_group)}/", status_code=303)
        except ValueError:
            pass
    return RedirectResponse(auth.root_of(request) + "/desk/", status_code=303)      # the loaded slate's page once the load is done
