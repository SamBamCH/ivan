"""
The lineup pool's slates: which draft group each generated lineup belongs to, and the lineups' player names for the export.
"""
from __future__ import annotations

from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.models import Lineup, LineupPool, NBAPlayer


def lineup_slates(session) -> list[dict]:
    """The slates with generated lineups, newest first: draft group, count, the pool snapshot's lock when known."""
    rows = session.execute(__import__("sqlalchemy").text("select draft_group_id, count(*) from lineup group by draft_group_id")).all()
    locks = {int(r[0]): r[1] for r in session.execute(__import__("sqlalchemy").text("select draft_group_id, min(lock) from lineup_pool group by draft_group_id")).all()}
    out = [dict(draft_group=None if dg is None else int(dg), n=int(n), lock=locks.get(int(dg)) if dg is not None else None) for dg, n in rows]
    return sorted(out, key=lambda r: (r["draft_group"] is None, -(r["draft_group"] or 0)))


def backfill_lineup_slates(session, log=lambda m: None) -> int:
    """
    Lineups generated before the slate was stamped: give each the newest slate whose pool held all its players, from the
    pool snapshots, the live pool, or the projection captures (the server has those for every slate it published).
    """
    from ivan.core.utils import coerce_player_ids
    from ivan.sports.nba.models import NBAProjectionCapture
    null_rows = session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.draft_group_id.is_(None)).all()
    if not null_rows:
        return 0
    pools: dict[int, set] = {}
    for dg, pid in session.query(LineupPool.draft_group_id, LineupPool.player_id).all():
        pools.setdefault(int(dg), set()).add(int(pid))
    for p in session.query(NBAPlayer.draft_group_id, NBAPlayer.id).all():
        if p[0] is not None:
            pools.setdefault(int(p[0]), set()).add(int(p[1]))
    for dg, pid in session.query(NBAProjectionCapture.draft_group_id, NBAProjectionCapture.player_id).filter(
            NBAProjectionCapture.draft_group_id.isnot(None), NBAProjectionCapture.player_id.isnot(None)).distinct().all():
        pools.setdefault(int(dg), set()).add(int(pid))
    order = sorted(pools, reverse=True)
    n = 0
    for lid, pids in null_rows:
        try:
            ids = {int(x) for x in coerce_player_ids(pids)}
        except (ValueError, TypeError):
            continue
        dg = next((d for d in order if ids <= pools[d]), None)
        if dg is not None:
            session.query(Lineup).filter(Lineup.id == lid).update({"draft_group_id": dg}, synchronize_session=False); n += 1
    session.commit()
    if n:
        log(f"{n} older lineups given their slate")
    return n


def lineup_names(session, draft_group: int) -> dict[int, dict]:
    """player id -> {name, position} for a slate: the pool snapshot, then the live pool, then the projection captures (names only)."""
    from ivan.sports.nba.models import NBAProjectionCapture
    out: dict[int, dict] = {}
    for r in session.query(LineupPool).filter(LineupPool.draft_group_id == draft_group).all():
        out[int(r.player_id)] = dict(name=r.name, position=r.position)
    for p in session.query(NBAPlayer).filter(NBAPlayer.draft_group_id == draft_group).all():
        out.setdefault(int(p.id), dict(name=p.name, position=p.position))
    for pid, name in session.query(NBAProjectionCapture.player_id, NBAProjectionCapture.player_name).filter(
            NBAProjectionCapture.draft_group_id == draft_group, NBAProjectionCapture.player_id.isnot(None)).distinct().all():
        out.setdefault(int(pid), dict(name=name, position=None))
    # positions for players without a snapshot: the archived pool of that slate (the laptop keeps the blobs), then the same
    # name's position in any pool we know (a player's DK positions do not change between slates)
    missing = [pid for pid, v in out.items() if not v.get("position")]
    if missing:
        try:
            from ivan.core import archive
            caps = [c for c in archive.captures(session, "nba") if str(c.slate_key) == str(draft_group)]
            if caps:
                for r in archive.load_blobs(session, max(caps, key=lambda c: c.captured_at).id).get("players", []):
                    if r.get("id") is not None and r.get("position"):
                        out.setdefault(int(r["id"]), dict(name=r.get("name"), position=None))
                        out[int(r["id"])]["position"] = out[int(r["id"])]["position"] or r["position"]
        except Exception:   # noqa: BLE001 - no archive here
            pass
        by_name: dict[str, str] = {}
        for name, pos in session.query(LineupPool.name, LineupPool.position).all():
            if pos:
                by_name.setdefault(normalize_name(name), pos)
        for name, pos in session.query(NBAPlayer.name, NBAPlayer.position).all():
            if pos:
                by_name.setdefault(normalize_name(name), pos)
        for pid, v in out.items():
            if not v.get("position") and v.get("name"):
                v["position"] = by_name.get(normalize_name(v["name"]))
    return out
