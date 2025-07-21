#!/usr/bin/env python
import json
import importlib
from pathlib import Path

from ivan.core.utils       import project_root, show_player_exposures
from ivan.database.session import init_db, get_session

def main():
    # — load config & determine sport —
    cfg_path = project_root() / 'config' / 'lol.json'
    sport    = cfg_path.stem           # "lol"
    cfg      = json.load(open(cfg_path, encoding='utf-8'))

    # — init DB & session —
    engine  = init_db(sport, cfg.get('database_url'))
    session = get_session(engine)

    # — dynamic import of Lineup class —
    models    = importlib.import_module(f"ivan.sports.{sport}.models")
    LineupCls = getattr(models, "Lineup")

    # 1) query all lineups
    all_lineups = session.query(LineupCls).all()

    # 2) extract each lineup’s player-ID list via your helper
    lineup_id_lists = [lu.player_ids() for lu in all_lineups]

    # 3) show exposures
    show_player_exposures(session, sport, lineup_id_lists)

if __name__ == '__main__':
    main()
