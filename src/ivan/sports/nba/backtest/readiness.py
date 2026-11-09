"""
ivan.sports.nba.backtest.readiness

Whether a past day is ready to backtest, per held slate and per day: the colour of a past date on the calendar and the
picker (ivan.web.frame) and what scripts/nba/readiness.py prints.

    ready     every input of every held slate of the day is on file: the box scores of its games (nba_stats has every team of
              the slate on that date) and, where we played, a final standings export of every contest of ours (the slate's
              results ledger is complete, ivan.sports.nba.backtest.results)
    failed    a fetch failed and the day cannot be backtested until it is fixed: a standings scrape recorded a failure or a
              login needed for one of the slate's contests (slates/<dg>/results/scrape.json, written by `standings.py scrape`
              through standings.record_scrape), the box scores are still missing STALE_HOURS after the slate's lock (the
              nightly runs three times a day; DraftKings keeps standings about two weeks), or the ledger could not be read
    pending   nothing attempted yet, or still inside the window (no colour)

    slate_status(dg, now)   {"state", "why", "box", "money", "scrape"} for one held past slate
    day_status(now)         {iso day: {"state", "why", "slates": [dg, ...]}} for every day with a held past slate; a day is
                            failed when any slate is, ready when every slate is, pending otherwise. Cached for CACHE_SECONDS:
                            the frame asks on every page.

A slate we did not play (no entries file) needs only its box scores. A slate of today or later has no status.
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import text

from ivan.sports.nba import slates as S

STALE_HOURS = 36            # box scores still missing this long after the lock: the nightly is not bringing them
CACHE_SECONDS = 120         # how long day_status stands in this process (the frame asks on every page)
_CACHE: dict = {}
_LOCK = threading.Lock()


def scrape_record_path(dg: int, root: Optional[Path] = None) -> Path:
    return S.paths(int(dg), root).root / "results" / "scrape.json"


def read_scrape_record(dg: int, root: Optional[Path] = None) -> Optional[dict]:
    try:
        return json.loads(scrape_record_path(dg, root).read_text())
    except (OSError, ValueError):
        return None


def _box_scores(dg: int, day: date, root: Optional[Path] = None) -> dict:
    """{"teams": n on the slate, "with_box": n of them in nba_stats on the day, "missing": [teams]} through the slate's own engine
    (its nba_player, the attached history's nba_stats)."""
    eng = S.engine(dg, root)
    with eng.connect() as con:
        teams = {t for (t,) in con.execute(text("select distinct team from nba_player where team is not null")).all()}
        try:
            have = {t for (t,) in con.execute(text("select distinct team_abbreviation from hist.nba_stats where date(game_date) = :d").bindparams(d=day.isoformat())).all()}
        except Exception:   # noqa: BLE001 - a history file without the table (a test)
            have = set()
    missing = sorted(t for t in teams if t not in have)
    return dict(teams=len(teams), with_box=len(teams) - len(missing), missing=missing)


def slate_status(dg: int, now: Optional[datetime] = None, root: Optional[Path] = None) -> dict:
    """One held past slate: the box scores, the money, the scrape record, and the state they add up to."""
    from ivan.sports.nba.backtest import results
    now = now or datetime.now()
    meta = S.read_meta(dg, root)
    day_s = str(meta.get("day") or "")[:10]
    lock = S.meta_time(meta, "lock")
    if not day_s:
        return dict(state="pending", why="the slate's day is not known", box=None, money=None, scrape=None)
    day = date.fromisoformat(day_s)
    out = dict(state="pending", why="", box=None, money=None, scrape=read_scrape_record(dg, root), day=day_s, draft_group=int(dg))
    why = []
    try:
        box = _box_scores(int(dg), day, root)
        out["box"] = box
    except Exception as e:   # noqa: BLE001
        out["box"] = dict(error=f"{type(e).__name__}: {e}"); box = None
    try:
        led = results.slate_ledger(int(dg), now, root=root)
        out["money"] = dict(complete=bool(led.get("complete")), played=bool(led.get("entries_file")) or bool(led.get("contests")),
                            unscored=int((led.get("summary") or {}).get("unscored") or 0), contests=len(led.get("contests") or []))
    except Exception as e:   # noqa: BLE001
        out["money"] = dict(error=f"{type(e).__name__}: {e}")
    failed, waiting = [], []
    # the box scores
    if box is None:
        failed.append(f"box scores could not be read: {out['box']['error']}")
    elif box["teams"] and box["missing"]:
        age_h = ((now - lock).total_seconds() / 3600) if lock else ((now - datetime.combine(day, datetime.min.time())).total_seconds() / 3600 - 24)
        note = f"box scores missing for {len(box['missing'])} of {box['teams']} teams ({', '.join(box['missing'][:6])}{'…' if len(box['missing']) > 6 else ''})"
        (failed if age_h >= STALE_HOURS else waiting).append(note + ("" if age_h >= STALE_HOURS else "; the nightly brings them"))
    elif not box["teams"]:
        waiting.append("no pool stored for the slate")
    # the money
    m = out["money"]
    if m.get("error"):
        failed.append(f"results ledger: {m['error']}")
    elif m.get("played") and not m.get("complete"):
        sc = out["scrape"] or {}
        if sc.get("failed") or sc.get("login_needed"):
            failed.append(f"standings scrape {str(sc.get('at') or '')[:16].replace('T', ' ')}: {len(sc.get('failed') or [])} contest(s) failed"
                          + (", login needed" if sc.get("login_needed") else "") + " (scripts/nba/standings.py scrape)")
        else:
            waiting.append(f"{m['unscored']} contest(s) without a final standings export (scripts/nba/standings.py scrape)")
    if failed:
        out["state"] = "failed"; out["why"] = "; ".join(failed + waiting)
    elif waiting:
        out["state"] = "pending"; out["why"] = "; ".join(waiting)
    else:
        out["state"] = "ready"
        out["why"] = "box scores on file" + ("; every contest scored from a final export" if m.get("played") else "; no entries of ours")
    return out


def day_status(now: Optional[datetime] = None, root: Optional[Path] = None, today: Optional[date] = None, force: bool = False) -> Dict[str, dict]:
    """{iso day: {"state", "why", "slates"}} for every day with a held slate before today. Cached CACHE_SECONDS per process."""
    now = now or datetime.now()
    today = today or now.date()
    key = (str(root) if root is not None else "", today.isoformat())
    with _LOCK:
        c = _CACHE.get(key)
        if c is not None and not force and (now - c[0]).total_seconds() < CACHE_SECONDS:
            return c[1]
    days: Dict[str, list] = {}
    for dg, meta in S.held_meta(root).items():
        d = str(meta.get("day") or "")[:10]
        if not d or date.fromisoformat(d) >= today:
            continue
        try:
            st = slate_status(dg, now, root)
        except Exception as e:   # noqa: BLE001 - one broken slate must not take the page down
            st = dict(state="failed", why=f"{type(e).__name__}: {e}", draft_group=int(dg))
        days.setdefault(d, []).append(st)
    out = {}
    for d, sts in sorted(days.items(), reverse=True):
        states = {s["state"] for s in sts}
        state = "failed" if "failed" in states else ("ready" if states == {"ready"} else "pending")
        why = "; ".join(f"slate {s.get('draft_group')}: {s['why']}" if len(sts) > 1 else s["why"] for s in sts if s.get("why"))
        out[d] = dict(state=state, why=why, slates=[s.get("draft_group") for s in sts])
    with _LOCK:
        _CACHE[key] = (now, out)
    return out


def forget() -> None:
    """Drop the cache (a scrape or a nightly step just changed what is on disk)."""
    with _LOCK:
        _CACHE.clear()
