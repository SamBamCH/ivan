from ivan.database.session import init_db, get_session
from ivan.sports.nfl.correlated_samples import generate_correlated_samples

engine = init_db('nfl')
session = get_session(engine)

samples = generate_correlated_samples(session)

print(samples)