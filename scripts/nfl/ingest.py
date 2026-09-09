from ivan.core.config import require_env
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLOdds, NFLPlayer
from ivan.sports.common.draftkings import ingest_players
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.sports.nfl.refactor.player import fit_fpts_distributions
import argparse
import logging
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
    p.add_argument("--draft-group", type=int, default=None, help="DK draft group id of the slate to ingest; skips the interactive slate prompt")
    p.add_argument("--no-ownership", action='store_true', help="Skip ownership projection after fitting distributions")
    p.add_argument("--no-snapshot", action='store_true', help="Skip the projection snapshot used for model training")
    return p.parse_args()

#create session
args = parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")
engine = init_db('nfl')
session = get_session(engine)

#Fetch everything first; only touch the database once both fetches succeeded, so a
#failed DK or odds call (expired key, outage) leaves the previous slate intact.
players = None
if args.p:
    print('fetching players')
    players = ingest_players('nfl', dedupe_by_name=False, draft_group_id=args.draft_group)
    if not players:
        raise SystemExit('DraftKings returned no players for that slate; nothing changed.')

records = None
if args.update_odds:
    api_key = require_env("ODDS_API_KEY")
    if players is not None:
        relevant_teams = {p.team for p in players if p.team}
    else:
        relevant_teams = set(session.scalars(select(NFLPlayer.team).distinct()))
    odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams))
    df_odds = parse_odds(odds)
    if df_odds is None or len(df_odds) == 0:
        raise SystemExit('the-odds-api returned no odds (see the error above). '
                         'Check ODDS_API_KEY and your plan at the-odds-api.com; nothing changed.')
    records = df_odds.to_dict(orient='records')
elif not args.p:
    raise ValueError('-p and --update-odds both false, exiting...')

#now write: clear what we are replacing, insert, commit once
if players is not None:
    session.query(NFLPlayer).delete(synchronize_session=False)
    for player in players:
        session.merge(NFLPlayer(
            id=player.dk_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            position=player.position,
            roster_slot_id=player.roster_slot_id,
            gametime=player.gametime,
            salary=player.salary,
            status=player.status
        ))
if records is not None:
    session.query(NFLOdds).delete(synchronize_session=False)
    session.bulk_insert_mappings(NFLOdds, records)
session.commit()

if args.p:
    #remove out players
    session.query(NFLPlayer).filter(func.upper(NFLPlayer.status).in_(['OUT', 'IR'])).delete(synchronize_session=False)
    session.commit()

#fit fpts for each player
fit_fpts_distributions(session)

#project ownership from field sampling (needs only the fitted distributions), then snapshot
if not args.no_ownership:
    from ivan.core.config import load_config
    from ivan.sports.nfl.ownership import project_and_snapshot
    project_and_snapshot(session, load_config("nfl"), snapshot=not args.no_snapshot)

#run team simulations to capture team dependence? overright the original FPTS values if we want to. Worried that this will bias the simulations too much, since this aspect will already be captured in the simulation. 

#visualize slate
