"""
ivan.sports.nba.changes

What a pool refresh moved, per team: statuses (Q -> OUT), the Rotowire lineup (in / out of the five, confirmed),
players added to or dropped from the pool, projections that moved PROJ_MOVE fantasy points or more, and the new
headlines about the slate's players. ingest.run_ingest takes the snapshot before and returns the diff
(out["changes"]); the desk keeps the last one and shows it on each team card ("since the last refresh"), ingest.py
prints it.

    snapshot(session)                 the pool's statuses, starters and projections now
    diff(before, after, news=None)    {team: [{kind, player, text}]}

The props' freshness after news (2026-11-07): a book's line priced the role the player had when it was posted, so once his
DK status or his starter flag moves (Rotowire's five, or the desk's hand call over it) the lines posted before that are set
aside until the books post again (the next props fetch replaces the game's rows):

    status_state(session)                       player -> {status, starter, at}
    mark_status_changes(session, before, now)   after the pool was replaced: nba_player.status_changed_at = now where the
                                                state moved since `before`, the earlier stamp carried for the rest
    touch_status(session, name, now)            a hand call on the desk moves one player now
    status_cutoffs(session)                     player -> status_changed_at, what market.fresh_rows drops rows before
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from ivan.sports.nba.names import normalize_name

PROJ_MOVE = 3.0


def snapshot(session: Session) -> dict:
    from ivan.sports.nba.models import NBAPlayer, StartingLineups
    players = {}
    for p in session.query(NBAPlayer).all():
        players[normalize_name(p.name)] = dict(raw=p.name, team=p.team, status=(p.status or "").upper().replace("NONE", ""),
                                               fpts=None if p.fpts_mean is None else float(p.fpts_mean))
    starters, confirmed = {}, {}
    for r in session.query(StartingLineups).all():
        starters.setdefault(r.team, set()).add(normalize_name(r.player_name))
        confirmed[r.team] = bool(r.confirmed)
    return dict(at=datetime.now(), players=players, starters=starters, confirmed=confirmed,
                draft_group=next((p.draft_group_id for p in session.query(NBAPlayer).limit(1)), None))


def _status(raw) -> str:
    return (raw or "").upper().replace("NONE", "")


def status_state(session: Session) -> dict:
    """player -> {status: the DK status (blank = active), starter: in the five (Rotowire's lineup, the desk's hand call over it),
    at: when either last moved (nba_player.status_changed_at)}. The overrides need the slate in context; without one the
    hand calls are not read."""
    from ivan.sports.nba.models import NBAPlayer, StartingLineups
    five = {normalize_name(r.player_name) for r in session.query(StartingLineups).all()}
    try:
        from ivan.sports.nba.projection.live import read_overrides_full
        ov = read_overrides_full()
    except Exception:   # noqa: BLE001 - no slate in context, no file
        ov = {}
    out = {}
    for p in session.query(NBAPlayer).all():
        n = normalize_name(p.name)
        call = (ov.get(n) or {}).get("starter")
        out[n] = dict(status=_status(p.status), starter=(float(call) > 0) if call is not None else (n in five), at=p.status_changed_at)
    return out


def mark_status_changes(session: Session, before: Optional[dict], now: Optional[datetime] = None) -> List[str]:
    """After the pool was replaced (the rows are new, their stamps blank): every player whose status or starter flag differs from
    `before` (status_state taken before the store) is stamped `now`; the rest carry the stamp they had. A player new to the pool
    carries none. Returns the players stamped, flushed (the caller's transaction commits)."""
    from ivan.sports.nba.models import NBAPlayer
    now = now or datetime.now()
    after = status_state(session)
    marked: List[str] = []
    for p in session.query(NBAPlayer).all():
        n = normalize_name(p.name)
        b = (before or {}).get(n); a = after.get(n)
        if b is None or a is None:
            continue
        if a["status"] != b["status"] or bool(a["starter"]) != bool(b["starter"]):
            p.status_changed_at = now; marked.append(n)
        else:
            p.status_changed_at = b.get("at")
    session.flush()
    return marked


def touch_status(session: Session, name: str, now: Optional[datetime] = None) -> bool:
    """A hand call on the desk (a starter toggled) moves the player's state now: his props posted before this are set aside."""
    from ivan.sports.nba.models import NBAPlayer
    key = normalize_name(name); hit = False
    for p in session.query(NBAPlayer).all():
        if normalize_name(p.name) == key:
            p.status_changed_at = now or datetime.now(); hit = True
    session.flush()
    return hit


def status_cutoffs(session: Session) -> Dict[str, datetime]:
    """player -> when his status or starter flag last moved, for the players it did (market.fresh_rows' cutoffs)."""
    from ivan.sports.nba.models import NBAPlayer
    return {normalize_name(p.name): p.status_changed_at for p in session.query(NBAPlayer).all() if p.status_changed_at is not None}


def diff(before: dict, after: dict, news: Optional[Iterable] = None) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    if not before or not after or before.get("draft_group") != after.get("draft_group"):
        return out                                            # another slate: everything is new, nothing "changed"

    def add(team, kind, player, text):
        out.setdefault(team or "", []).append(dict(kind=kind, player=player, text=text))

    bp, ap = before["players"], after["players"]
    for n, a in ap.items():
        b = bp.get(n)
        if b is None:
            add(a["team"], "pool", n, f"{a['raw']} added to the pool")
            continue
        if b["status"] != a["status"]:
            add(a["team"], "status", n, f"{a['raw']}: {b['status'] or 'active'} → {a['status'] or 'active'}")
        if b["fpts"] is not None and a["fpts"] is not None and abs(a["fpts"] - b["fpts"]) >= PROJ_MOVE:
            add(a["team"], "projection", n, f"{a['raw']}: projection {b['fpts']:.1f} → {a['fpts']:.1f}")
    for n, b in bp.items():
        if n not in ap:
            add(b["team"], "pool", n, f"{b['raw']} left the pool")
    for team in set(before["starters"]) | set(after["starters"]):
        s0, s1 = before["starters"].get(team, set()), after["starters"].get(team, set())
        raw = lambda n: (ap.get(n) or bp.get(n) or {}).get("raw", n)
        for n in sorted(s1 - s0):
            add(team, "lineup", n, f"{raw(n)} into the starting five")
        for n in sorted(s0 - s1):
            add(team, "lineup", n, f"{raw(n)} out of the starting five")
        if after["confirmed"].get(team) and not before["confirmed"].get(team):
            add(team, "lineup", None, "lineup confirmed")
    for x in news or []:
        n = getattr(x, "player_name", None)
        if n in ap:
            add(ap[n]["team"], "news", n, f"{getattr(x, 'player_raw', n)}: {getattr(x, 'headline', '')}")
    return out
