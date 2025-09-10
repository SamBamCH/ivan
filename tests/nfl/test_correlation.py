from pathlib import Path
from sqlalchemy.orm import Session
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.correlated_sampling_validation import run_end_to_end_and_validate

engine = init_db("nfl")
session: Session = get_session(engine)
run_end_to_end_and_validate(session, model_dir=Path("data/nfl/models"), n_samples=10000, hist_max_rows=1000)
