#!/usr/bin/env python
import json
import os

from ivan.database.session import init_db, get_session
from ivan.sports.lol.lineup_generation import generate_and_store_lol_field_lineups
from ivan.core.utils import PROJECT_ROOT

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'lol.json')

def deduplicate_lineups():
    """
    Remove any saved Lineup records that share the same set of player IDs,
    regardless of order, keeping only the first occurrence.
    """
    engine  = init_db("mlb")
    session = get_session(engine)

    seen_keys  = set()
    duplicates = []

    for lu in session.query(Lineup).all():
        # FIX: call the method to get the list, or directly access the JSON
        ids = lu.player_ids()               # ← note the () 
        # ids = lu.player_ids_list          # ← alternative: iterate the JSON column
        key = tuple(sorted(ids))

        if key in seen_keys:
            duplicates.append(lu)
        else:
            seen_keys.add(key)

    if duplicates:
        for dup in duplicates:
            session.delete(dup)
        session.commit()

    print(f"Deduplicator: removed {len(duplicates)} duplicate lineups, kept {len(seen_keys)} unique.")


def main():
     # 1) init db & session
    engine  = init_db("lol")
    session = get_session(engine)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    generate_and_store_lol_field_lineups(session, config, num_lineups=10000)


if __name__ == "__main__":
    main()