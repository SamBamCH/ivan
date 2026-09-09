from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer, ProjectionSnapshot, HistoricalContest, ContestPayout, ClassicContestMetadata
from ivan.sports.nba.historical.data_fetch import fetch_boxscores
from ivan.sports.nba.historical.data_clean import parse_sd, upsert_payouts, normalize_player_names, clean_old_diagnostics
from ivan.sports.nba.historical.data_calculation import build_field_distribution_model
from ivan.sports.common.draftkings import get_upcoming_contest_ids, get_upcoming_payouts, build_requests_session_from_chrome, download_contest_dataframe
from ivan.sports.nba.historical.contest_processing import process_classic_contest

from sqlalchemy import text, distinct
import logging
from pathlib import Path
import pickle
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#connect to DB
engine  = init_db('nba')
session = get_session(engine)

#CLEAR OLD DATA----------------------
#clear NBAPlayer -> ProjectionSnapshot.
players = session.query(NBAPlayer).all()
for player in players:
    snapshot = ProjectionSnapshot(
        id=player.id,
        snapshot_date=player.gametime,
        player_name=player.name,
        team=player.team,
        opponent=player.opponent,
        position=player.position,
        roster_slot_id=player.roster_slot_id,
        gametime=player.gametime,
        salary=player.salary,
        status=player.status,
        fpts_mean=player.fpts_mean,
        fpts_std=player.fpts_std,
        fpts_updated_at=player.fpts_updated_at,
        ownership=player.ownership
    )
    # use merge to perform an upsert (update existing snapshot with same PK or insert new)
    session.merge(snapshot)
session.commit()
logging.info(f"Created or updated {len(players)} projection snapshots")
# session.query(NBAPlayer).delete()
session.commit()
session.expunge_all()
        #TODO: run diagnostics to make sure this looks correct when upserting? 
# clear Lineups and LineupSimulationResults
session.execute(text("DELETE FROM lineup_simulation_results"))
session.commit()
session.execute(text("DELETE FROM lineup"))
session.commit()

clean_old_diagnostics(days_to_keep=1)
#DATA_RETRIEVAL------------------------
#fetch updated stats
normalize_player_names(session)
fetch_boxscores(session=session) #NOTE: will hit rate limit issues if fetching more than like 50 games. I just stop the script and run it again in like 30 min when that happens. Once we get all the stats, this will be a non-issue.
#get_contest_ids

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
logging.info(f'Saved {len(filtered)} upcoming contests to the database.')
#get_payouts
all_contests = session.query(HistoricalContest.contest_id).all()
all_contests = {row.contest_id for row in all_contests}

existing = session.query(distinct(ContestPayout.contest_id)).all()
existing = {row[0] for row in existing if row[0] is not None}

missing = sorted(all_contests - existing)

# only attempt payouts for contests within the last 3 days (or in the future)
three_days_ago = datetime.utcnow() - timedelta(days=3)
filtered_missing = []
for cid in missing:
    hc = session.query(HistoricalContest).filter_by(contest_id=cid).one_or_none()
    if not hc or hc.date is None:
        continue
    if hc.date >= three_days_ago:
        filtered_missing.append(cid)

missing = filtered_missing

for idx, cid in enumerate(missing, start=1):
    payload = get_upcoming_payouts(cid)
    if payload is not None:
        upsert_payouts(session, cid, payload)

logging.info(f'Saved {len(missing)} upcoming contest payouts to the database.')
#get_contest_data
to_process = (
    session.query(HistoricalContest)
    .filter(
        HistoricalContest.date >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
        HistoricalContest.date < datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
        HistoricalContest.total_entries >= 1000
    )
    .all()
)
chrome_session = build_requests_session_from_chrome(
    user_data_dir="/home/sam-pratt/ivan/chrome-selenium/dk-profile",
    profile_directory='Default',
    headless=False,
)

for hc in to_process:
    try:
        df = download_contest_dataframe(chrome_session, hc.contest_id)
        process_classic_contest(session, hc, df)
    except Exception as e:
        continue

    #TODO: add a verification for projectionsnapshot. Skip upsertion to DB if not processed correctly. need to keep the database clean for fitting out model later. 

#BUILD_MODELS-------------------
#field distribution model
field_distribution_model = build_field_distribution_model(session)
out_path = Path("data/nba/models/field_distribution_model.pkl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(field_distribution_model, f)
#TODO: add diagnostics tests for field sampling to ensure accuracy, should be housed in tests/nba/...
