from ivan.core.config import require_env
from ivan.database.session import init_db, get_session, add_missing_columns
from ivan.sports.nfl.models import NFLOdds, NFLPlayer, NFLGameOdds
from ivan.sports.common.draftkings import ingest_players
from ivan.sports.common.odds import get_odds, parse_odds, get_game_odds, parse_game_odds
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
from ivan.sports.nfl.models import NFLBase
for added in add_missing_columns(engine, NFLBase.metadata):
    print(f'schema: added column {added}')
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
    odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams), raw_sport="nfl")
    df_odds = parse_odds(odds)
    if df_odds is None or len(df_odds) == 0:
        raise SystemExit('the-odds-api returned no odds (see the error above). '
                         'Check ODDS_API_KEY and your plan at the-odds-api.com; nothing changed.')
    records = df_odds.to_dict(orient='records')
    # Game lines (spread, total, moneyline) for every upcoming game: the DST
    # projection and the per-game diagnostics read them from game_odds.
    game_odds_json, _ = get_game_odds("americanfootball_nfl", api_key, markets="h2h,spreads,totals", raw_sport="nfl")
    df_game_odds = parse_game_odds(game_odds_json) if game_odds_json else None
    if df_game_odds is None or df_game_odds.empty:
        print('the-odds-api returned no game lines; DST projections will fall back to salary.')
        df_game_odds = None
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
if args.update_odds and df_game_odds is not None:
    df_game_odds = df_game_odds.drop_duplicates(subset=["game_id", "bookmaker", "market", "outcome_name", "point", "last_update"])
    session.query(NFLGameOdds).filter(NFLGameOdds.game_id.in_(df_game_odds['game_id'].unique().tolist())).delete(synchronize_session=False)
    session.bulk_insert_mappings(NFLGameOdds, df_game_odds.to_dict(orient="records"))
    print(f"game lines: {df_game_odds['game_id'].nunique()} games, {len(df_game_odds)} rows")
session.commit()

# Archive the raw inputs of this capture (pool, props, lines) so the slate can
# be replayed later with whatever projection code is current. Never modelled.
from ivan.core.archive import ensure_tables
from ivan.sports.nfl.archive_capture import archive_ingest_capture
ensure_tables(engine)
from ivan.core import raw as _raw
print(f"raw artifacts indexed: {_raw.sync_index(session, 'nfl')} new")
cap = archive_ingest_capture(session, players, records, df_game_odds if args.update_odds else None)
if cap is not None:
    print(f"archived capture #{cap.id}: slate {cap.slate_key} ({cap.mode}, lock {cap.lock:%Y-%m-%d %H:%M}) "
          f"{cap.n_players} players, {cap.n_odds} odds rows, {cap.n_lines} line rows")

if args.p:
    #remove out players
    session.query(NFLPlayer).filter(func.upper(NFLPlayer.status).in_(['OUT', 'IR'])).delete(synchronize_session=False)
    session.commit()

#fit fpts for each player
from ivan.core.config import load_config
_fit_info = fit_fpts_distributions(session, load_config("nfl")) or {}

#project ownership from field sampling (needs only the fitted distributions), then snapshot
if not args.no_ownership:
    from ivan.core.config import load_config
    from ivan.sports.nfl.ownership import project_and_snapshot
    project_and_snapshot(session, load_config("nfl"), snapshot=not args.no_snapshot)

# write the projections down for the validation layer (ivan.sports.nfl.run_artifacts)
try:
    from ivan.core import artifacts
    from ivan.sports.nfl.paths import OUTPUT_DIR
    from ivan.sports.nfl.run_artifacts import write_projections
    _proj_path = artifacts.projections_path(OUTPUT_DIR)
    print(f"projections artifact: {write_projections(session, _proj_path)} rows -> {_proj_path}")
    from ivan.sports.nfl.run_artifacts import write_environment
    print(f"game environment artifact: {write_environment(session, _proj_path)} team rows")
    from ivan.sports.nfl.run_artifacts import write_usage
    print(f"usage artifact: {write_usage(_fit_info.get('usage'), _proj_path, _fit_info.get('unpriced_volume', ''))} team-stat rows")
except Exception as _e:
    print(f"WARNING: could not write the projections artifact: {type(_e).__name__}: {_e}")

#run team simulations to capture team dependence? overright the original FPTS values if we want to. Worried that this will bias the simulations too much, since this aspect will already be captured in the simulation. 

#visualize slate
