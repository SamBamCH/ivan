from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLOdds, NFLPlayer
from ivan.sports.common.draftkings import ingest_players
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.sports.nfl.refactor.player import fit_fpts_distributions
import argparse
from sqlalchemy import select, func

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    p = argparse.ArgumentParser(description="ingest and update player odds")
    p.add_argument("-p", action='store_true', help='Prelock flag, determines if we need to fetch DK payload or not.')
    p.add_argument("--update-odds", type=str2bool, nargs='?', const=True, default=True, help="Update player odds")
    return p.parse_args()

#create session
args = parse_args()
engine = init_db('nfl')
session = get_session(engine)

#clear old data
if args.p:
    if args.update_odds:
        #clear both player and odds
        session.query(NFLOdds).delete(synchronize_session=False)
        session.query(NFLPlayer).delete(synchronize_session=False)
    else:
        #clear player but not odds
        session.query(NFLPlayer).delete(synchronize_session=False)

else:
    if args.update_odds:
        #clear odds but not player
        session.query(NFLOdds).delete(synchronize_session=False)
    else:
        raise ValueError('-p and --update-odds both false, exiting...')
        
#ingest players from dk
if args.p:
    print('fetching players')
    players = ingest_players('nfl', dedupe_by_name=False)
    for player in players:
        dbp = NFLPlayer(
            id=player.dk_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            position=player.position,
            roster_slot_id=player.roster_slot_id,
            gametime=player.gametime,
            salary=player.salary,
            status=player.status
        )
        session.merge(dbp)

    session.commit()

if args.update_odds:
    #get odds
    api_key = '' #type THE-ODDS-API api key here
    relevant_teams = set(session.scalars(select(NFLPlayer.team).distinct()))
    odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams))
    #parse and upsert odds
    df_odds = parse_odds(odds)
    records = df_odds.to_dict(orient='records')
    session.bulk_insert_mappings(NFLOdds, records)
    session.commit()

if args.p:
    #remove out players
    session.query(NFLPlayer).filter(func.upper(NFLPlayer.status).in_(['OUT', 'IR'])).delete(synchronize_session=False)
    # #remove players with no odds (check by name, not id, since we have 2 rows per player)
    # players_with_odds = session.query(NFLOdds.player).distinct().all()
    # players_with_odds_set = {p[0] for p in players_with_odds}
    # # Delete both roster_slot_id rows if player name has no odds
    # session.query(NFLPlayer).filter(~NFLPlayer.name.in_(players_with_odds_set)).delete(synchronize_session=False)
    session.commit()

#fit fpts for each player
fit_fpts_distributions(session)

#run team simulations to capture team dependence? overright the original FPTS values if we want to. Worried that this will bias the simulations too much, since this aspect will already be captured in the simulation. 

#visualize slate
