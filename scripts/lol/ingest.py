import os
import csv
import json
from ivan.core.utils import PROJECT_ROOT, normalize_name
from ivan.database.session import init_db, get_session, Base
from ivan.sports.lol.models import LOLPlayer

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'lol.json')

# load config & DB
cfg = json.load(open(CONFIG_PATH, encoding='utf-8'))
engine = init_db('lol', cfg.get('database_url'))
Base.metadata.create_all(engine)
sess = get_session(engine)

pid_fp = cfg['paths']['ids']
missing_gameinfo = []
missing_projections = []
initialized_count = 0

with open(pid_fp, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        player_id = row.get('ID')
        name      = normalize_name(row.get('Name', ''))
        team      = row.get('TeamAbbrev')
        raw_gi    = row.get('Game Info', '')
        gametime  = raw_gi  # we’re just storing the full string

        # ——— extract opponent from "AWAY@HOME ..." ———
        opponent = None
        if raw_gi:
            matchup = raw_gi.split(' ')[0]
            if '@' in matchup:
                away_team, home_team = matchup.split('@', 1)
                if team == away_team:
                    opponent = home_team
                elif team == home_team:
                    opponent = away_team
        if opponent is None:
            missing_gameinfo.append((player_id, raw_gi))

        # ——— position & roster slot ———
        game_position   = row.get('Position')
        roster_position = row.get('Roster Position')

        # ——— salary (track blanks or parse errors) ———
        salary_str = (row.get('Salary') or '').strip()
        try:
            salary = int(salary_str)
        except (TypeError, ValueError):
            salary = None
            missing_projections.append((player_id, salary_str))

        # ——— build & merge model instance ———
        p = LOLPlayer(
            id              = player_id,
            name            = name,
            team            = team,
            opponent        = opponent,
            gametime        = gametime,
            salary          = salary,
            game_position   = game_position,
            roster_position = roster_position
        )
        sess.merge(p)
        initialized_count += 1

# commit & summary
sess.commit()
print(f"Initialized {initialized_count} players")
if missing_gameinfo:
    print(f"⚠️ Missing opponent for {len(missing_gameinfo)} rows:", missing_gameinfo)
if missing_projections:
    print(f"⚠️ Bad/missing salary for {len(missing_projections)} rows:", missing_projections)
