from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer, HistoricalContest
from ivan.sports.common.draftkings import ingest_players, get_upcoming_contest_ids
from ivan.sports.nba.refactor.historical.data_fetch import fetch_starting_lineups
from ivan.sports.nba.refactor.historical.data_clean import parse_sd
from ivan.sports.nba.refactor.cli import view_lineups
from ivan.core.utils import normalize_name
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="ingest and update player odds")
    p.add_argument("-p", action='store_true', help='preflock flag. Determines if we fetch payload from DK or not.')
    return p.parse_args()

args = parse_args()
#build session
engine = init_db('nba')
session = get_session(engine)

#get upcoming contest ids
filtered = get_upcoming_contest_ids(sport="NBA", days_ahead=1, entry_filter=100)
for c in filtered:
    cid = str(c["id"])
    existing = session.query(HistoricalContest).filter_by(contest_id=cid).one_or_none()

    contest_date    = parse_sd(c.get("sd"))
    entry_fee       = float(c.get("a"))
    total_entries   = int(c.get("m"))

    if not existing:
        hc = HistoricalContest(
            contest_id    = cid,
            draft_group_id= c.get("dg"),
            name          = c.get("n"),
            date          = contest_date,
            entry_fee     = entry_fee,
            contest_type  = c.get("gameType"),
            total_entries = total_entries,
            multi_entry   = c.get("mec")
        )
        session.add(hc)

session.commit() 
print(f'Saved {len(filtered)} upcoming contests to the database.')
#'refresh' player pool
if args.p:
    session.query(NBAPlayer).delete(synchronize_session=False)
    players = ingest_players('nba')
    for player in players:
        dbp = NBAPlayer(
            id=player.dk_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            position=player.position,
            roster_slot_id=player.roster_slot_id,
            gametime=player.gametime,
            salary=player.salary,
            status=player.status,
        )
        session.merge(dbp)
    
    session.commit()

#ingest projections
csv_path = Path("data/nba/input/stok.csv")
df = pd.read_csv(csv_path)
# Update database players from CSV
for _, row in df.iterrows():
    name = normalize_name(row['Name'])
    player = session.query(NBAPlayer).filter(
        NBAPlayer.name.ilike(name)
    ).first()
    if player:
        player.fpts_mean = row['Projection']
        player.fpts_std = row['Std Dev']
        player.ownership = row['Ownership%']
        session.merge(player)
    else:
        print(f'no player found matching {name}')

# Commit all changes to database
session.commit()
print("✓ All player projections and ownership values upserted successfully")

# Remove players with no fpts projection
session.query(NBAPlayer).filter(NBAPlayer.fpts_mean.is_(None)).delete(synchronize_session=False)
session.commit()

#fetch starting lineups
fetch_starting_lineups(session)

#view recap of slate
view_lineups(session)