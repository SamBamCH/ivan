#!/usr/bin/env python3
"""
scripts/nba/feed.py
The NBA data feed between two Ivans (ivan.core.feed, the command line in ivan.core.feed_cli). The consumer pulls; the
producer never pushes.
Producer (the laptop: runs the desk, fetches from DK and the odds API):
    scripts/nba/feed.py publish [--which live|reference|both] [--slate DG] [--quiet]
        write data/feed/nba/*: the live bundle (the pool, props, lines, lineups, headlines, the published edits, the
        projection store), the reference bundle (history, models, captures). The consumer's pull asks for a fresh live
        bundle itself (deploy/feed_serve.sh runs this); the nightly writes both. --slate DG loads that draft group into
        the pool first when the pool holds another (ingest.refresh_pool, the same load as picking the slate in the desk's
        bar; the pick is remembered in data/nba/active_slate.json so the capture tick stays on it): what the consumer's
        bar asks for when a slate the laptop does not hold is picked there (ssh `publish nba DG`).
Consumer (the server):
    scripts/nba/feed.py pull [--no-project]
        what the desk's "Refresh from laptop" does (ivan.core.feed.pull_from_producer): with IVAN_FEED_SSH=<user>@<host>
        in .env, one ssh command has the laptop publish its live bundle, rsync copies its data/feed here (only what
        changed), and the bundles are applied: the feed's tables replaced in this database (edits, lineups and sims here
        untouched), the laptop's pins and grids as the published layer, its projection store mirrored; then only the
        slate's teams this machine edited are re-projected on top (--no-project skips that). The older
        IVAN_FEED_URL (a directory or a feed server, with IVAN_FEED_TOKEN) still works when IVAN_FEED_SSH is unset.
    scripts/nba/feed.py pull --slate DG                           ask the laptop to load slate DG first (what picking a slate the
                                                                  laptop does not hold does in the consumer's bar): both desks move to it
    scripts/nba/feed.py pull --loop 15                            keep pulling every 15 minutes
    scripts/nba/feed.py apply DIR                                 seed this database from bundles copied by hand
"""
from __future__ import annotations

from ivan.core import feed_cli


def before_publish(a) -> None:
    """publish --slate DG: the pool holds that draft group before the bundle is written. Nothing when it already does; else the
    slate is fetched from DraftKings, stored and projected (ingest.refresh_pool, one transaction: the desk's slate pick) and
    the choice is written to data/nba/active_slate.json (the capture tick's slate). The desk process notices the new pool
    on its next request (Desk.maybe_reload)."""
    import json
    from datetime import datetime
    from ivan.database.session import get_session, init_db
    from ivan.sports.nba.ingest import ensure_schema, refresh_pool
    from ivan.sports.nba.models import NBAPlayer
    from ivan.sports.nba.paths import ACTIVE_SLATE_JSON
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    try:
        row = session.query(NBAPlayer.draft_group_id).first()
        held = int(row[0]) if row and row[0] else None
        if held == int(a.slate):
            return
        print(f"loading slate {a.slate} (the pool holds {held})", flush=True)
        refresh_pool(session, int(a.slate), update_odds=False, starters=True, log=print)
        ACTIVE_SLATE_JSON.write_text(json.dumps({"draft_group": int(a.slate), "chosen_at": datetime.now().isoformat(timespec="seconds")}))
    finally:
        session.close()


def before_pull():
    """The slate this machine holds before the pull: when the laptop moved to another, that slate's edits are filed (ingest.after_pull)."""
    from ivan.database.session import get_session, init_db
    from ivan.sports.nba.ingest import ensure_schema
    from ivan.sports.nba.models import NBAPlayer
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    try:
        row = session.query(NBAPlayer.draft_group_id).first()
        return int(row[0]) if row and row[0] else None
    finally:
        session.close()


def after_pull(changed: dict, previous) -> None:
    """The published projection stands as pulled; this machine's own edits of the old slate are filed if the slate moved, and only
    the slate's teams it edited (its own pins or grids) are re-projected on top (ingest.after_pull)."""
    if "live" not in changed:
        return
    from ivan.database.session import get_session, init_db
    from ivan.sports.nba.ingest import after_pull as _after
    session = get_session(init_db("nba"))
    try:
        r = _after(session, previous, log=print)
        if r["teams"]:
            print(f"re-projected {r['players']} players of {len(r['teams'])} team(s) with this machine's own edits on top of the published projection")
        else:
            print("the published projection stands (no edits of this machine's own on this slate)")
    except Exception as e:   # noqa: BLE001
        print(f"re-projection skipped: {type(e).__name__}: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(feed_cli.main("nba", before_pull=before_pull, after_pull=after_pull, doc=__doc__, before_publish=before_publish))
