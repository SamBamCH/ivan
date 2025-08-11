from ivan.sports.nfl.correlated_samples import generate_correlated_samples
from ivan.database.session import get_session, init_db



engine = init_db('nfl')
session = get_session(engine)

samples = generate_correlated_samples(session)