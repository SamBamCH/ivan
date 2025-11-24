from ivan.database.session import get_session, init_db
from ivan.sports.nba.models import HistoricalContest, ClassicContestMetadata
from ivan.sports.common.draftkings import download_contest_dataframe, build_requests_session_from_chrome
from ivan.sports.nba.historical_contest import process_classic_contest
from datetime import datetime, timedelta
# connect to DB

def main(
    days_back: int = 14
):

    engine = init_db('nba')
    session = get_session(engine)

    now = datetime.utcnow()
    yesterday = now - timedelta(days=2)
    cutoff = now - timedelta(days=days_back)

    #fetch completed contests
    processed = session.query(ClassicContestMetadata.contest_id)

    #fetch contest_ids from DB that we want to include in the model

    to_process = (
        session.query(HistoricalContest)
        .filter(
            HistoricalContest.date >= cutoff,
            HistoricalContest.date <= yesterday,
            ~HistoricalContest.contest_id.in_(processed),
            HistoricalContest.total_entries >= 5000
        )
        .order_by(HistoricalContest.date.desc())
        .all()
    )

    sess = build_requests_session_from_chrome(
        user_data_dir="/home/sam-pratt/ivan/chrome-selenium/dk-profile",
        profile_directory='Default',
        headless=False,
    )

    print(f'{len(to_process)} contest IDs pulled from Database to fetch.')

    for hc in to_process:
        try:
            df = download_contest_dataframe(sess, hc.contest_id)
            success = process_classic_contest(session, hc, df)
        except Exception as e:
            continue


if __name__ == "__main__":
    main()

