from ivan.sports.nfl.correlated_samples import SamplingConfig, generate_correlated_samples
from ivan.database.session import init_db, get_session

engine = init_db('nfl')
session = get_session(engine)

cfg = SamplingConfig(
    n_samples=500,   # plenty for variety
    alpha=0.95,
    rho_game=0.60,
    jitter=0.01,
    print_sanity=True
)
samples = generate_correlated_samples(session, cfg=cfg)