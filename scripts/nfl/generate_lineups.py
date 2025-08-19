from pathlib import Path
from ivan.sports.nfl.correlated_samples import generate_correlated_samples
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.generate_lineups_logic import generate_and_store_field_lineups 



engine  = init_db('nfl')
session = get_session(engine)

samples = generate_correlated_samples(session)  #TODO: add a num_samples key to config? 

created = generate_and_store_field_lineups(
    session,
    samples,
    config={
        "lineup": {
            "min_salary": 0,
            "max_salary": 50000,
            "qb_stack_min": 1,
            "max_players_per_team": 4,
            "min_fpts_threshold": None,
        },
        "verbose": False
    },
    num_lineups=10,
    batch_size=10,
)
print(f"created {created} NFL lineups")