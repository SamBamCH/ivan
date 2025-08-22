#!/usr/bin/env python3
"""
Calculate player exposures across all saved lineups and include roster slot.

- Connects via init_db('nfl') / get_session(engine)
- Reads every row from ivan.sports.nfl.models.Lineup
- Counts, for each player_id, how many DISTINCT lineups include them
- Joins to ivan.sports.nfl.models.NFLPlayer to get (name, roster_slot_id)
- Prints exposure table sorted by descending exposure %

Exposure = (# of lineups containing player) / (total # of lineups)
"""

from collections import Counter
import json
from typing import List, Tuple

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, Lineup

# 511 = CPT, 512 = FLEX (fallback to raw id if unrecognized)
ROSTER_SLOT_MAP = {
    511: "CPT",
    512: "FLEX",
}

def _coerce_player_ids(val) -> List[int]:
    """Handle JSON column variations (already-parsed list vs stored as str)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, list):
                return [int(x) for x in data]
        except Exception:
            return []
    return []

def main() -> None:
    # 1) connect
    engine = init_db("nfl")
    session = get_session(engine)

    # 2) pull all lineups
    rows: List[Tuple] = session.query(Lineup.id, Lineup.player_ids_list).all()
    total_lineups = len(rows)
    if total_lineups == 0:
        print("No lineups found.")
        return

    # 3) count appearances per player (unique per lineup)
    counts = Counter()
    for _, plist in rows:
        ids = set(_coerce_player_ids(plist))
        for pid in ids:
            counts[pid] += 1

    # 4) fetch names + roster slots for all players observed
    all_ids = list(counts.keys())
    id_to_info = {
        pid: (name, roster_slot_id)
        for pid, name, roster_slot_id in (
            session.query(NFLPlayer.id, NFLPlayer.name, NFLPlayer.roster_slot_id)
                   .filter(NFLPlayer.id.in_(all_ids))
                   .all()
        )
    }

    # 5) assemble, sort by exposure desc (then by count desc, then by id)
    results = []
    for pid, c in counts.items():
        exposure = c / total_lineups
        name, slot_id = id_to_info.get(pid, (f"Unknown ({pid})", None))
        slot_label = ROSTER_SLOT_MAP.get(slot_id, str(slot_id) if slot_id is not None else "—")
        results.append((pid, name, slot_label, c, exposure))

    results.sort(key=lambda r: (r[4], r[3], -r[0]), reverse=True)

    # 6) print
    print(f"Total lineups: {total_lineups}")
    print(f"{'Player ID':>9}  {'Name':<32} {'Slot':<6} {'#Lineups':>8} {'Exposure':>9}")
    print("-" * 72)
    for pid, name, slot_label, count, exp in results:
        print(f"{pid:>9}  {name:<32.32} {slot_label:<6} {count:>8} {exp*100:8.2f}%")

if __name__ == "__main__":
    main()
