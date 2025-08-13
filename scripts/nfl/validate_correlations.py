from pathlib import Path
from sqlalchemy.orm import Session
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.correlated_samples import SamplingConfig, generate_correlated_samples
from ivan.sports.nfl.correlated_sampling_validation import (
    validate_marginals, validate_quantiles, validate_dependence
)

engine = init_db('nfl')
session = get_session(engine)

cfg = SamplingConfig(n_samples=5000, alpha=0.85, seed=42)
samples = generate_correlated_samples(session, model_dir='data/nfl/models', cfg=cfg)

# 1) PIT uniformity
ks_results = validate_marginals(session, samples, 'data/nfl/models')
bad = [(pid, ks, p) for pid, ks, p in ks_results if p < 0.01]
print(f"PIT uniformity: {len(bad)} players failing @ p<0.01 out of {len(ks_results)}")

# 2) Quantile MAE vs theoretical
q_mae = validate_quantiles(session, samples, 'data/nfl/models')
print("Median player MAE:", sorted(m for _, m in q_mae)[len(q_mae)//2])

# 3) Dependence vs vine
dep = validate_dependence(session, samples, 'data/nfl/models', alpha=cfg.alpha)
for team, d in dep.items():
    emp = d["emp_corr"]; exp = d["expected_corr"]
    # report average absolute error on off-diagonals where expected available
    mask = (exp != 0)
    if mask.sum() > 0:
        err = np.mean(np.abs(emp[mask] - exp[mask]))
        print(f"{team}: mean |emp-exp| (Spearman) = {err:.3f}")
