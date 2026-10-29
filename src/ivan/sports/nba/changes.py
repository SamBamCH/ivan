"""
ivan.sports.nba.changes

What a pool refresh moved, per team: statuses (Q -> OUT), the Rotowire lineup (in / out of the five, confirmed),
players added to or dropped from the pool, projections that moved PROJ_MOVE fantasy points or more, and the new
headlines about the slate's players. ingest.run_ingest takes the snapshot before and returns the diff
(out["changes"]); the desk keeps the last one and shows it on each team card ("since the last refresh"), ingest.py
prints it.

    snapshot(session)                 the pool's statuses, starters and projections now
    diff(before, after, news=None)    {team: [{kind, player, text}]}
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name

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
