import json
import pickle
from pathlib import Path

from ivan.database.session import init_db, get_session, Base
from ivan.core.utils        import project_root
from ivan.sports.mlb.model_calculations import (
    build_nb_model,
    build_team_copulas,
    build_slate_bin_regressors,
)

def build_models(
    db_key: str = "mlb",
    config_path: Path | None = None,
    out_dir: Path | None = None,
):
    """
    1) Reads <db_key>.json from config/
    2) Spins up engine + session
    3) Runs NB model, team copulas, slate-bin regressors
    4) Pickles each to disk under out_dir (default data/<db_key>/models/)
    """
    # — locate config file —
    if config_path is None:
        config_path = project_root() / "config" / f"{db_key}.json"
    cfg = json.load(open(config_path, encoding="utf-8"))

    # — init DB & session —
    engine = init_db(db_key, cfg.get("database_url"))
    Base.metadata.create_all(engine)
    session = get_session(engine)

    # — build each model —
    print("▶ Building NB model by OPS…")
    nb_model, df_nb = build_nb_model(session)

    print("▶ Building team copulas…")
    team_copulas, team_scalers, team_matrices = build_team_copulas(session)

    print("▶ Fitting fixed-bin slate regressors…")
    bin_models = build_slate_bin_regressors(session)

    # — determine output directory —
    if out_dir is None:
        out_dir = project_root() / "data" / db_key / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # — dump everything —
    with open(out_dir / "nb_model.pkl",      "wb") as f: pickle.dump(nb_model, f)
    with open(out_dir / "team_copulas.pkl",  "wb") as f: pickle.dump(team_copulas, f)
    with open(out_dir / "team_scalers.pkl",  "wb") as f: pickle.dump(team_scalers, f)
    with open(out_dir / "bin_regressors.pkl","wb") as f: pickle.dump(bin_models, f)

    print(f"✅ All models saved to {out_dir}")
