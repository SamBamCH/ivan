#!/usr/bin/env python3
"""
scripts/nba/notes.py

Player notes on the loaded slate (ivan.sports.nba.notes; the desk's note button on tonight's grid does the same).
A note belongs to its slate and stands until cleared; the slate's scorecard lists the notes that stood at lock.

    scripts/nba/notes.py list                                  the loaded slate's notes
    scripts/nba/notes.py list --draft-group 155667 --all       a slate's notes, cleared ones too
    scripts/nba/notes.py set "Ausar Thompson" "ankle, coach said limited"
    scripts/nba/notes.py clear "Ausar Thompson"
"""
import argparse

from ivan.core.utils import normalize_name
from ivan.database.session import get_session, init_db
from ivan.sports.nba import notes as N
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.models import NBAPlayer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list"); ls.add_argument("--draft-group", type=int); ls.add_argument("--all", action="store_true")
    st = sub.add_parser("set"); st.add_argument("player"); st.add_argument("text")
    cl = sub.add_parser("clear"); cl.add_argument("player")
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    pool = session.query(NBAPlayer).all()
    dg = getattr(a, "draft_group", None) or next((p.draft_group_id for p in pool if p.draft_group_id), None)
    if dg is None:
        raise SystemExit("no slate loaded; pass --draft-group")
    if a.cmd == "list":
        rows = N.at(session, dg, None) if a.all else list(N.active(session, dg).values())
        for r in rows:
            print(f"  {r.team or '':3} {r.player_raw:26} {r.text}" + (f"   [cleared {r.cleared_at:%m-%d %H:%M}]" if r.cleared_at else f"   [{r.updated_at:%m-%d %H:%M}]"))
        print(f"{len(rows)} note(s) on slate {dg}")
        return 0
    by = {normalize_name(p.name): p for p in pool}
    key = normalize_name(a.player)
    p = by.get(key)
    if p is None:
        raise SystemExit(f"{a.player} is not on the loaded slate")
    if a.cmd == "set":
        lock = min((x.gametime for x in pool if x.gametime), default=None)
        N.set_note(session, dg, lock.date().isoformat() if lock else None, key, a.text, team=p.team, raw=p.name)
        print(f"note on {p.name}: {a.text}")
    else:
        print(f"cleared {N.clear(session, dg, key)} note(s) on {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
