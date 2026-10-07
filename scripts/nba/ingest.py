from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer, HistoricalContest
from ivan.sports.common.draftkings import ingest_players, get_upcoming_contest_ids
from ivan.sports.nba.historical.data_fetch import fetch_starting_lineups
from ivan.sports.nba.historical.data_clean import parse_sd
from ivan.sports.nba.cli import view_lineups
from ivan.sports.nba.models import NBAOdds, NBABase
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.core.config import require_env
from ivan.core.utils import normalize_name
import argparse
from pathlib import Path
import pandas as pd
from ivan.sports.nba.paths import PROJECTIONS_CSV

def parse_args():
    p = argparse.ArgumentParser(description="ingest and update player odds")
    p.add_argument("-p", action='store_true', help='preflock flag. Determines if we fetch payload from DK or not.')
    p.add_argument("--update-odds", action='store_true',
                   help='fetch player props from the-odds-api for the teams on the slate, store them, and '
                        'derive per-minute market rates (projection/market.py) for project.py')
    p.add_argument("--no-projections-csv", action='store_true', help='skip stok.csv (use project.py instead)')
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

#player props -> market rates
if args.update_odds:
    from ivan.database.session import add_missing_columns
    from ivan.sports.nba.projection.market import NBA_MARKETS, market_means
    from ivan.sports.nba.projection.live import LiveSlate, basis_minutes_for_market, store_market_rates, read_overrides
    add_missing_columns(engine, NBABase.metadata)
    teams = {p.team for p in session.query(NBAPlayer).all()}
    raw = get_odds("basketball_nba", require_env("ODDS_API_KEY"), list(teams), markets=NBA_MARKETS)
    odds_df = parse_odds(raw)
    if odds_df is None or odds_df.empty:
        raise SystemExit("no NBA player props came back; the market rates were left as they were")
    session.query(NBAOdds).delete(synchronize_session=False)
    session.bulk_insert_mappings(NBAOdds, odds_df.to_dict("records"))
    session.commit()
    means = market_means(odds_df)
    slate = LiveSlate(session, use_market=False)
    basis = basis_minutes_for_market(slate, read_overrides(), set())
    n = store_market_rates(session, means, basis)
    print(f"✓ {len(odds_df)} prop rows from {odds_df.sportsbook.nunique()} books; "
          f"market rates for {means.player.nunique()} players ({n} player-stats)")

#ingest projections
if args.no_projections_csv or not PROJECTIONS_CSV.exists():
    print("no stok.csv projections ingested; run scripts/nba/project.py")
    fetch_starting_lineups(session)
    view_lineups(session)
    raise SystemExit(0)
csv_path = PROJECTIONS_CSV
df = pd.read_csv(csv_path)
# Update database players from CSV
for _, row in df.iterrows():
    name = normalize_name(row['Player'])
    player = session.query(NBAPlayer).filter(
        NBAPlayer.name.ilike(name)
    ).first()
    if player:
        player.fpts_mean = row['Projection']
        player.fpts_std = row['Std Dev']
        player.ownership = row['Ownership %']
        session.merge(player)
    else:
        print(f'no player found matching {name}')

# Commit all changes to database

# Step 1: Collect player names from stok.csv
existing_players = set(df['Player'].apply(normalize_name).tolist())

# Step 2: Query all players in the database
all_players = session.query(NBAPlayer).all()

# Step 3: Remove players not in the CSV
for player in all_players:
    if player.name not in existing_players:
        session.delete(player)

# Commit changes after deleting players not in the CSV
session.commit()
print(f"✓ Removed players not in stok.csv")

session.commit()
print("✓ All player projections and ownership values upserted successfully")

# Remove players with no fpts projection
session.query(NBAPlayer).filter(NBAPlayer.fpts_mean.is_(None)).delete(synchronize_session=False)
session.commit()

#fetch starting lineups
fetch_starting_lineups(session)

#view recap of slate
view_lineups(session)
