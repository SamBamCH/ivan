import json
from sqlalchemy.orm import Session

def run_contest(
        contest_id: int,
        num_sims: int,
        config_path: str,
        session: Session,
        sport: str
) -> None:
    # 1) load config
    cfg = json.load(open(config_path, encoding="utf-8"))

    # 2) Query all players from DB

    # 3) Sample lineups
    
    # 4) Load payouts

    # 5) Run sims

    # 6) Write results to DB

