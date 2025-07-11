import os
import sys
import json
import pickle
from ivan.database.session import init_db, get_session, Base
from ivan.core.utils import project_root

# Ensure the ivan package is on the path
repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src')
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ivan.sports.mlb.model_calculations import build_nb_model, build_team_copulas, build_slate_bin_regressors


def main():
    # --- 1) load config & DB session
    cfg_path = project_root() / 'config' / 'mlb.json'
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    engine = init_db('mlb', cfg.get('database_url'))
    Base.metadata.create_all(engine)
    session = get_session(engine)

    # --- 2) build models
    print("Building NB model by OPS…")
    nb_model, df_nb = build_nb_model(session)

    print("Building team copulas…")
    team_copulas, team_scalers, team_matrices = build_team_copulas(session)

    # --- 3) dump to disk
    outdir = project_root() / 'data' / 'mlb' / 'models'
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open(outdir / 'team_copulas.pkl', 'wb') as f:
        pickle.dump(team_copulas, f)
    with open(outdir / 'team_scalers.pkl', 'wb') as f:
        pickle.dump(team_scalers, f)

    # --- 4) build bin-based slate models
    print("Fitting fixed-bin slate regressors…")
    bin_models = build_slate_bin_regressors(session)

    # --- persist bin regressors ---
    with open(outdir / 'bin_regressors.pkl', 'wb') as f:
        pickle.dump(bin_models, f)

    print("✅ Bin regressors saved to data/mlb/models/bin_regressors.pkl")


if __name__ == '__main__':
    main()