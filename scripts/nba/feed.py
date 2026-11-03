#!/usr/bin/env python3
"""
scripts/nba/feed.py
The NBA data feed between two Ivans (ivan.core.feed, the command line in ivan.core.feed_cli). The consumer pulls; the
producer never pushes.
Producer (the laptop: runs the desk, fetches from DK and the odds API):
    scripts/nba/feed.py publish [--which live|reference|both] [--slate DG] [--quiet]
        write data/feed/nba/*: the live bundle (the props, lines and headlines every slate shares) with one bundle per held
        slate (data/feed/nba/slates/<dg>/: its pool, starters, market rates, contests, its edits, its projection store, its
        record), and the reference bundle (history, models, captures). The consumer's pull asks for fresh bundles itself
        (deploy/feed_serve.sh runs this); the nightly writes both. --slate DG publishes that slate only, fetching it into
        its own directory first when this machine does not hold it (ingest.refresh_pool): what the consumer's bar asks for
        when a slate the laptop does not hold is picked there (ssh `publish nba DG`).
Consumer (the server):
    scripts/nba/feed.py pull [--no-project]
        what the desk's "Refresh from laptop" and the server's 30-minute timer do (ivan.core.feed.pull_from_producer): with
        IVAN_FEED_SSH=<user>@<host> in .env, one ssh command has the laptop publish, rsync copies its data/feed here (only
        what changed), and every bundle that moved is applied: the shared live tables replaced in the history database,
        each changed slate into its own directory here (its feed tables replaced; this machine's lineups, sims and edits
        of that slate untouched; the laptop's edits as the published layer, its store mirrored, pulled_at noted); then only
        the teams this machine edited on each are re-projected on top (--no-project skips that). The older IVAN_FEED_URL
        (a directory or a feed server, with IVAN_FEED_TOKEN) still works when IVAN_FEED_SSH is unset.
    scripts/nba/feed.py pull --slate DG                           ask the laptop for slate DG (fetched there first when it does not
                                                                  hold it): the slate page's Refresh from laptop
    scripts/nba/feed.py pull --loop 15                            keep pulling every 15 minutes
    scripts/nba/feed.py apply DIR                                 seed this database from bundles copied by hand
"""
from __future__ import annotations

from ivan.core import feed_cli


def before_publish(a) -> None:
    """publish --slate DG: this machine holds that slate before its bundle is written. Nothing when it already does; else the
    slate is fetched from DraftKings into its own directory, stored and projected (ingest.refresh_pool, one transaction: what
    opening the slate on the desk does). The desk makes the slate's desk on its next request."""
    from ivan.sports.nba import slates as S
    from ivan.sports.nba.ingest import refresh_pool
    dg = int(a.slate)
    if S.holds(dg):
        return
    print(f"fetching slate {dg} (not held here yet)", flush=True)
    with S.using(dg):
        session = S.session(dg)
        try:
            refresh_pool(session, dg, update_odds=False, starters=True, log=print, by="feed")
        finally:
            session.close()


def before_pull():
    return None


def after_pull(changed: dict, previous) -> None:
    """The published projection stands as pulled; for every slate the pull brought, this machine's own edits of that slate (its
    pins or grids) are re-projected on top (ingest.after_pull, on the slate's own session)."""
    from ivan.sports.nba import slates as S
    from ivan.sports.nba.ingest import after_pull as _after
    for dg, got in (changed.get("slates") or {}).items():
        if not got.get("tables") and not got.get("mirror"):
            continue
        with S.using(int(dg)):
            session = S.session(int(dg))
            try:
                r = _after(session, log=print)
                if r["teams"]:
                    print(f"slate {dg}: re-projected {r['players']} players of {len(r['teams'])} team(s) with this machine's own edits on top of the published projection")
                else:
                    print(f"slate {dg}: the published projection stands (no edits of this machine's own)")
            except Exception as e:   # noqa: BLE001
                print(f"slate {dg}: re-projection skipped: {type(e).__name__}: {e}")
            finally:
                session.close()


if __name__ == "__main__":
    raise SystemExit(feed_cli.main("nba", before_pull=before_pull, after_pull=after_pull, doc=__doc__, before_publish=before_publish))
