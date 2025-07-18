import os
import csv
import json
from collections import defaultdict
from datetime import datetime

from ivan.core.utils            import PROJECT_ROOT, normalize_name
from ivan.database.session      import init_db, get_session, Base
from ivan.sports.lol.models     import LOLPlayer

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'lol.json')

# ——— load config & DB ———
cfg    = json.load(open(CONFIG_PATH, encoding='utf-8'))
engine = init_db('lol', cfg.get('database_url'))
Base.metadata.create_all(engine)
sess   = get_session(engine)

# ——— load odds mapping ———
# expect cfg['paths']['odds'] → e.g. "data/lol_odds.json"
odds_fp = cfg['paths'].get('odds')
team_odds = {}
missing_odds_teams = set()
if odds_fp:
    odds_path = os.path.join(PROJECT_ROOT, odds_fp)
    try:
        with open(odds_path, encoding='utf-8') as f:
            team_odds = json.load(f)
        # normalize any keys if needed, e.g. uppercase
        team_odds = { k.upper(): v for k,v in team_odds.items() }
    except Exception as e:
        print(f"❌ Failed to load odds JSON at {odds_path}: {e}")

# ——— prepare to import players ———
pid_fp             = cfg['paths']['ids']
missing_gameinfo   = []
missing_salary     = []
initialized_count  = 0
batch_ts           = datetime.utcnow()

with open(pid_fp, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        player_id = row.get('ID')
        name      = normalize_name(row.get('Name', ''))
        team      = row.get('TeamAbbrev', '').upper()
        raw_gi    = row.get('Game Info', '')
        gametime  = raw_gi  # storing full string for now

        # — extract opponent from "AWAY@HOME ..." —
        opponent = None
        if raw_gi:
            matchup = raw_gi.split(' ')[0]
            if '@' in matchup:
                away, home = matchup.split('@',1)
                if team == away:   opponent = home
                elif team == home: opponent = away
        if opponent is None:
            missing_gameinfo.append((player_id, raw_gi))

        # — parse salary —
        salary_str = (row.get('Salary') or '').strip()
        try:
            salary = int(salary_str.replace(',', ''))
        except (TypeError, ValueError):
            salary = None
            missing_salary.append((player_id, salary_str))

        # — lookup odds for this team —
        odds = team_odds.get(team)
        if odds is None:
            missing_odds_teams.add(team)

        # — build & merge model instance —
        p = LOLPlayer(
            id               = player_id,
            name             = name,
            team             = team,
            opponent         = opponent,
            gametime         = gametime,
            salary           = salary,
            game_position    = row.get('Position'),
            roster_position  = row.get('Roster Position'),
            odds             = odds,   # new column on your model
            created_at       = batch_ts
        )
        sess.merge(p)
        initialized_count += 1

# — commit & summary —
sess.commit()
print(f"Initialized {initialized_count} players")

if missing_gameinfo:
    print(f"⚠️ Missing opponent for {len(missing_gameinfo)} rows:", missing_gameinfo)

if missing_salary:
    print(f"⚠️ Bad/missing salary for {len(missing_salary)} rows:", missing_salary)

if missing_odds_teams:
    print(f"⚠️ No odds found for teams: {sorted(missing_odds_teams)} (check your odds JSON)")
