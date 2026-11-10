"""
ivan.sports.nba.backtest.tips

The record at each game's tip: what the system knew when that game started, written once to the slate's directory and
never rewritten. It is the source of truth every backtest reads for that game: a model re-run on a past slate may use
nothing dated after the tip of the game it is projecting.

    data/nba/slates/<dg>/tips/<YYYYMMDD-HHMM>/
        meta.json          tip, written_at, the teams tipping now, the counts, where each part came from
        players.json       the pool as the desk showed it: id, name, team, opponent, position, salary, status, starter, dk_avg_fpts,
                           minutes_proj, fpts_mean, fpts_std, p_out, minutes_p10/p90, ownership, pub_fpts, rotation_cells (the grid row)
        odds.parquet       the stored props for the slate's players (player, market, sportsbook, side, point, price, last_update): the
                           odds table is replaced per player at every refresh, so this is the only record of the props at the tip
        game_lines.json    spreads and totals for the slate's games as stored
        news.json          the slate's news rows seen by then (nba_slate_news joined to nba_news: player, team, injury, headline, first_seen)
        edits/             a copy of the day's edits as they stood: minutes_overrides.csv, rotations/<TEAM>.json, scenarios/

What is NOT copied, because it is dated already and the rule is the date: the box scores (nba_stats.game_date before the
slate's day), the stints (same), our entries and upload files (the slate's input/ and output/, stamped), the contests
and payouts (the slate's database, written before lock). What is deliberately not here: anything after the tip (DK's
%drafted, the standings, the box score of the game itself): those are the actuals, not the state.

    write_tip(dg, tip, now, players, odds, lines, session)   the archive timer's call (scripts/nba/archive_at_lock.py), once per tip
    tips_of(dg)                                              the tips on file, oldest first
    read_tip(dg, tip)                                        {meta, players, odds, game_lines, news, edits}
    backfill(dg, session)                                    tips from the archive captures (at_tip) for slates archived before this existed:
                                                             players, odds and lines only (no edits or news were kept), said so in meta

The archive tables (archived_slate + blobs, nba_projection_capture source at_tip) are still written for now; the plan
(docs/nba/notes/backtest_redesign_2026-11-05.md) retires the per-capture odds blob once every tip is on disk.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

STAMP = "%Y%m%d-%H%M"
PLAYER_FIELDS = ("id", "name", "team", "opponent", "position", "roster_slot_id", "gametime", "salary", "status", "dk_avg_fpts", "draft_group_id",
                 "starter", "minutes_proj", "fpts_mean", "fpts_std", "p_out", "minutes_p10", "minutes_p90", "ownership", "pub_fpts", "rotation_cells",
                 "p_cut", "fringe_p", "fringe_minutes", "minutes_sd", "rates", "boundary_sd")


def tips_dir(dg: int, root: Optional[Path] = None) -> Path:
    from ivan.sports.nba import slates as S
    return S.paths(int(dg), root).root / "tips"


def tip_dir(dg: int, tip: datetime, root: Optional[Path] = None) -> Path:
    return tips_dir(dg, root) / tip.strftime(STAMP)


def tips_of(dg: int, root: Optional[Path] = None) -> List[datetime]:
    d = tips_dir(dg, root)
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        if p.is_dir() and (p / "meta.json").exists():
            try:
                out.append(datetime.strptime(p.name, STAMP))
            except ValueError:
                continue
    return sorted(out)


def _players_rows(players) -> List[dict]:
    rows = []
    for p in players:
        if isinstance(p, dict):
            r = {k: p.get(k) for k in PLAYER_FIELDS}
        else:
            r = {k: getattr(p, k, None) for k in PLAYER_FIELDS}
        gt = r.get("gametime")
        r["gametime"] = gt.isoformat() if isinstance(gt, datetime) else gt
        rows.append(r)
    return rows


def slate_news(session: Session, dg: int, until: datetime) -> List[dict]:
    """The slate's news rows seen by `until` (nba_slate_news joined to nba_news), newest first."""
    try:
        rows = session.execute(text(
            "select n.id, n.player_name, n.player_raw, n.team, n.position, n.injury, n.injured, n.headline, n.first_seen, s.seen_at "
            "from nba_slate_news s join nba_news n on n.id = s.news_id where s.draft_group_id = :d and s.seen_at <= :t order by n.first_seen desc"
        ).bindparams(d=int(dg), t=until.isoformat(sep=" "))).all()
    except Exception:   # noqa: BLE001 - a database without the news tables
        return []
    return [dict(id=r[0], player=r[1], player_raw=r[2], team=r[3], position=r[4], injury=r[5], injured=bool(r[6]) if r[6] is not None else None,
                 headline=r[7], first_seen=str(r[8]), seen_at=str(r[9])) for r in rows]


def _copy_edits(src: Path, dst: Path) -> dict:
    """Copy the day's edits directory (overrides, rotations, scenarios) as it stands; returns what was copied."""
    out = dict(overrides=False, rotations=0, scenarios=0)
    if not src.is_dir():
        return out
    dst.mkdir(parents=True, exist_ok=True)
    f = src / "minutes_overrides.csv"
    if f.exists():
        shutil.copy2(f, dst / f.name); out["overrides"] = True
    for sub in ("rotations", "scenarios"):
        s = src / sub
        if s.is_dir():
            shutil.copytree(s, dst / sub, dirs_exist_ok=True)
            out[sub] = sum(1 for _ in (dst / sub).rglob("*") if _.is_file())
    return out


def write_tip(dg: int, tip: datetime, now: datetime, players, odds: Optional[pd.DataFrame], lines: Optional[pd.DataFrame],
              session: Optional[Session] = None, root: Optional[Path] = None, source: str = "archive_at_lock", with_edits: bool = True) -> Optional[Path]:
    """Write the tip's record (see the module doc). Returns the directory, or None when it already exists (a tip is written once).
    `with_edits` False leaves the edits out: a backfill cannot know how the day's files stood at the tip."""
    from ivan.sports.nba import slates as S
    d = tip_dir(dg, tip, root)
    if (d / "meta.json").exists():
        return None
    d.mkdir(parents=True, exist_ok=True)
    prow = _players_rows(players)
    (d / "players.json").write_text(json.dumps(prow, default=str))
    n_odds = 0
    if odds is not None and len(odds):
        odds = odds.copy()
        for c in odds.columns:
            if str(odds[c].dtype).startswith("datetime"):
                odds[c] = odds[c].astype(str)
        odds.to_parquet(d / "odds.parquet", index=False); n_odds = int(len(odds))
    n_lines = 0
    if lines is not None and len(lines):
        (d / "game_lines.json").write_text(lines.to_json(orient="records", date_format="iso")); n_lines = int(len(lines))
    news = slate_news(session, dg, now) if session is not None else []
    (d / "news.json").write_text(json.dumps(news, default=str))
    edits = (_copy_edits(S.paths(int(dg), root).edits, d / "edits") if with_edits and S.holds(int(dg), root)
             else dict(overrides=False, rotations=0, scenarios=0, kept=False))
    tipping = sorted({str(p["team"]) for p in prow if p.get("gametime") and str(p["gametime"])[:16] == tip.isoformat(timespec="minutes")})
    meta = dict(draft_group=int(dg), tip=tip.isoformat(timespec="minutes"), written_at=now.isoformat(timespec="seconds"), source=source,
                teams_tipping=tipping, players=len(prow), projected=sum(1 for p in prow if p.get("fpts_mean")), odds_rows=n_odds, line_rows=n_lines,
                news_rows=len(news), edits=edits)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return d


def read_tip(dg: int, tip: datetime, root: Optional[Path] = None) -> Optional[Dict]:
    d = tip_dir(dg, tip, root)
    if not (d / "meta.json").exists():
        return None
    out = dict(meta=json.loads((d / "meta.json").read_text()), players=json.loads((d / "players.json").read_text()),
               odds=pd.read_parquet(d / "odds.parquet") if (d / "odds.parquet").exists() else pd.DataFrame(),
               game_lines=json.loads((d / "game_lines.json").read_text()) if (d / "game_lines.json").exists() else [],
               news=json.loads((d / "news.json").read_text()) if (d / "news.json").exists() else [],
               edits=d / "edits" if (d / "edits").is_dir() else None)
    return out


def tip_for(dg: int, when: datetime, root: Optional[Path] = None) -> Optional[datetime]:
    """The newest tip on file at or before `when` (the state a game that tips at `when` is judged on)."""
    before = [t for t in tips_of(dg, root) if t <= when]
    return before[-1] if before else None


def backfill(dg: int, session: Session, root: Optional[Path] = None, log=print) -> int:
    """Tips from the archive's at_tip captures for a slate archived before tips were written to disk. Returns tips written."""
    from ivan.core import archive
    n = 0
    for cap in archive.captures(session, "nba", source="at_tip"):
        if str(cap.slate_key) != str(int(dg)):
            continue
        tip = cap.lock
        if (tip_dir(dg, tip, root) / "meta.json").exists():
            continue
        blobs = archive.load_blobs(session, cap.id)
        odds = pd.DataFrame(blobs.get("odds") or [])
        lines = pd.DataFrame(blobs.get("game_lines") or [])
        d = write_tip(dg, tip, cap.captured_at, blobs.get("players") or [], odds, lines, session=None, root=root,
                      source=f"backfill from archive capture {cap.id}", with_edits=False)
        if d is not None:
            meta = json.loads((d / "meta.json").read_text())
            meta["note"] = "backfilled from the archive capture: no edits or news were kept at the time"
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            n += 1
            log(f"slate {dg}: tip {tip:%m-%d %H:%M} written from capture {cap.id} ({meta['players']} players, {meta['odds_rows']} prop rows)")
    return n
