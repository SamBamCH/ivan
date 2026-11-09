#!/usr/bin/env python3
"""
scripts/nba/archive_at_lock.py

Keep the record at every lock without touching anything live. At each game's tip of every slate this machine holds
(data/nba/slates/<dg>/; the first tip is the slate lock) this writes, as they stand at that moment:
  - the tip's record on disk, data/nba/slates/<dg>/tips/<YYYYMMDD-HHMM>/ (ivan.sports.nba.backtest.tips): the pool with its
    projections and grid rows, the props, the game lines, the slate's news seen so far and a copy of the day's edits.
    This is what a backtest of that game's models reads: nothing dated after the tip;
  - nba_projection_capture rows (source "at_tip", lock = that tip) from nba_player: the published projections,
    minutes, ownership, status, starters;
  - an archived_slate capture (source "at_tip") of the pool, the stored props and the game lines.
It never fetches DraftKings or the books and never writes nba_player, so it cannot change what the desk or a job
sees. Runs every five minutes from a user timer (deploy/ivan-nba-archive.timer); a tip is archived once, within
ARCHIVE_WINDOW minutes after it (state in data/nba/archive_state.json). The late swap and the backtests read these
to see what changed after lock.

    scripts/nba/archive_at_lock.py            # archive any tip in the window
    scripts/nba/archive_at_lock.py --dry-run  # say what would be archived
    scripts/nba/archive_at_lock.py --now 2026-10-23T20:31   # pretend it is this time
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

import pandas as pd

from ivan.core.archive import archive_slate
from ivan.database.session import init_db
from ivan.sports.nba.data_clean.odds import GAME_LINE_COLS
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.models import NBAPlayer, NBAProjectionCapture, StartingLineups
from ivan.sports.nba.paths import LOGS_DIR, NBA_DATA_DIR
from ivan.sports.nba.projection.market import load_odds

STATE = NBA_DATA_DIR / "archive_state.json"
ARCHIVE_WINDOW = 12            # minutes after a tip during which it is still archived (a missed timer run or a busy desk)
log = logging.getLogger("archive")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"done": []}


def capture_rows(session, players, lock, now) -> int:
    starters = {r.player_name: ("confirmed" if r.confirmed else "expected") for r in session.query(StartingLineups).all()}
    n = 0
    for p in players:
        name = p.name
        session.add(NBAProjectionCapture(
            captured_at=now, draft_group_id=p.draft_group_id, lock=lock, source="at_tip", player_id=p.id, player_name=name,
            team=p.team, opponent=p.opponent, gametime=p.gametime, salary=p.salary, status=p.status,
            starter=starters.get(name, "bench"), minutes_proj=p.minutes_proj, fpts_mean=p.fpts_mean, fpts_std=p.fpts_std,
            minutes_role=p.minutes_role, p_out=p.p_out, minutes_p10=p.minutes_p10, minutes_p90=p.minutes_p90))
        n += 1
    session.commit()
    return n


def archive_tip(session, players, tip, now, dg) -> dict:
    """One tip: the capture rows and the archive (pool + props + lines as stored)."""
    starters = {r.player_name: ("confirmed" if r.confirmed else "expected") for r in session.query(StartingLineups).all()}
    prow = [dict(id=p.id, name=p.name, team=p.team, opponent=p.opponent, position=p.position, roster_slot_id=p.roster_slot_id,
                 gametime=p.gametime.isoformat() if p.gametime else None, salary=p.salary, status=p.status, dk_avg_fpts=p.dk_avg_fpts,
                 draft_group_id=p.draft_group_id, starter=starters.get(p.name, "bench"), minutes_proj=p.minutes_proj,
                 fpts_mean=p.fpts_mean, fpts_std=p.fpts_std, ownership=p.ownership, pub_fpts=p.pub_fpts,
                 rotation_cells=p.rotation_cells, minutes_p10=p.minutes_p10, minutes_p90=p.minutes_p90) for p in players]   # the grid row: the backtest's "our grid" for teams not drawn by hand
    pool = {p.name for p in players}
    odds = load_odds(session)                                     # with each row's game, so a backtest of the record reads only this slate's games
    odds = odds[odds.player.isin(pool)]
    lines = pd.read_sql("select * from game_odds", session.get_bind())
    teams = {p.team for p in players}
    if len(lines) and {"home_abbr", "away_abbr"} <= set(lines.columns):
        lines = lines[lines.home_abbr.isin(teams) | lines.away_abbr.isin(teams)]
        lines = lines[[c for c in GAME_LINE_COLS if c in lines.columns]]
    n = capture_rows(session, players, tip, now)
    cap = archive_slate(session, sport="nba", slate_key=str(dg), lock=tip, mode="classic", players=prow,
                        odds=odds.to_dict("records") if len(odds) else None, game_lines=lines.to_dict("records") if len(lines) else None,
                        source="at_tip", captured_at=now)
    from ivan.sports.nba.backtest import tips
    d = tips.write_tip(dg, tip, now, prow, odds, lines, session=session)      # the record on disk: slates/<dg>/tips/<stamp>/ (players, props, lines, news, edits)
    return dict(capture_rows=n, archive_id=cap.id, odds=int(len(odds)), lines=int(len(lines)), tip_dir=str(d) if d else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", default=None, help="pretend it is this time (ISO)")
    ap.add_argument("--window", type=int, default=ARCHIVE_WINDOW, help="minutes after a tip during which it is archived")
    a = ap.parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS_DIR / "archive.log")])
    now = datetime.fromisoformat(a.now) if a.now else datetime.now()
    ensure_schema(init_db("nba"))
    from ivan.sports.nba import slates as S
    st = load_state(); done = set(st.get("done", []))
    held = S.held()
    if not held:
        log.info("no slate held; nothing to archive"); return 0
    for dg in held:
        with S.using(dg):
            session = S.session(dg)
            try:
                players = session.query(NBAPlayer).all()
                tips = sorted({p.gametime for p in players if p.gametime})
                due = [t for t in tips if 0 <= (now - t).total_seconds() / 60 <= a.window and f"{dg}@{t.isoformat()}" not in done]
                if not due:
                    continue
                for t in due:
                    key = f"{dg}@{t.isoformat()}"
                    if a.dry_run:
                        log.info(f"would archive slate {dg} at tip {t:%a %H:%M}: {len(players)} players, {sum(1 for p in players if p.fpts_mean)} projected"); continue
                    info = archive_tip(session, players, t, now, dg)
                    done.add(key)
                    log.info(f"archived slate {dg} at tip {t:%a %H:%M}: {info['capture_rows']} capture rows, archive {info['archive_id']}, {info['odds']} prop rows, "
                             f"{info['lines']} line rows, tip record {info['tip_dir'] or 'already on disk'}")
            finally:
                session.close()
    if not a.dry_run:
        st["done"] = sorted(done)[-400:]; STATE.write_text(json.dumps(st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
