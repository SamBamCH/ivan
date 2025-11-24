from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import HistoricalContest
from ivan.sports.nba.refactor.pre_lock.field_sampling import sample_field
from ivan.sports.nba.refactor.cli import plot_field_sampling_diagnostics

def simulate_one_contest(
        contest_id,
        contest_name,
        field_cap,
        num_sims
): 
    engine = init_db('nba')
    session = get_session(engine)

    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    capped_field_size = min(field_cap, contest.total_entries)
    contest_params = {
        "entry_fee": float(getattr(contest, "entry_fee", 4.0)),
        "total_entries": int(getattr(contest, "total_entries", 5000)),
        "multi_entry": int(getattr(contest, "multi_entry")),
    }

    counts_by_lineup = sample_field(
        session,
        field_size=capped_field_size,
        contest_params=contest_params
    )

    plot_field_sampling_diagnostics(session, counts_by_lineup, contest_params)
    
