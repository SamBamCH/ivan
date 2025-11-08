from ivan.database.session import init_db, get_session
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.sports.nba.models import NBAPlayer, NBAOdds
from ivan.sports.nba.fpts_calculations import fit_fpts_distributions
from ivan.sports.nba.ownership import project_ownership
from sqlalchemy import select
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s %(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

engine = init_db('nba')
session = get_session(engine)

logger.info("[update_odds][1] connected to database")

api_key = '58699e4fe47856c9182db84884772536'
relevant_teams = set(session.scalars(select(NBAPlayer.team).distinct()))
markets = (
        "player_points,player_rebounds,player_assists,player_threes,"
        "player_blocks,player_steals,player_turnovers"
    )

logger.debug(f"Teams to fetch odds for: {sorted(relevant_teams)} ({len(relevant_teams) // 2} games)")

odds = get_odds("basketball_nba", api_key, list(relevant_teams), markets=markets)
df_odds = parse_odds(odds)
logger.info(f"[update_odds][2] ✓ Parsed {len(df_odds)} odds rows from API response")

# This prevents stale odds from accumulating and allows clean re-runs
if not df_odds.empty:
    deleted = session.query(NBAOdds).delete(synchronize_session=False)
    session.commit()
    if deleted > 0:
        logger.info(f"[update_odds][2] Cleared {deleted} existing player odds rows")

records = df_odds.to_dict(orient='records')
session.bulk_insert_mappings(NBAOdds, records)
session.commit()
logger.info(f"[update_odds][3] ✓ Inserted {len(records)} rows into nba_odds table")

logger.info("[update_odds][4] Starting FPTS distribution calculation")
logger.info("[update_odds][4] Using role-based marginals and correlations (starters vs backups)")
logger.info("[update_odds][4] Player role detected via median minutes to avoid injury misclassification")

fit_fpts_distributions(session, n_samples=10000)

project_ownership(session)

logger.info("[update_odds][5] ✓ FPTS distributions calculated and stored")