"""
Build NFL Classic field distribution models.

Trains Random Forest regressors to predict projection and ownership
distribution shapes based on contest parameters.

Usage:
    python -m ivan.sports.nfl.build_models
"""

import json
import pickle
from pathlib import Path

from ivan.database.session import init_db, get_session, Base
from ivan.core.utils import project_root
from ivan.sports.nfl.model_calculations import build_slate_bin_regressors


def build_slate_models(
    db_key: str = "nfl",
    config_path: Path | None = None,
    out_dir: Path | None = None,
):
    """
    1) Reads nfl.json config
    2) Spins up engine + session
    3) Runs slate bin regressors
    4) Pickles to disk under out_dir (default data/nfl/models/)
    """
    # — locate config file —
    if config_path is None:
        config_path = project_root() / "config" / f"{db_key}.json"
    cfg = json.load(open(config_path, encoding="utf-8"))

    # — init DB & session —
    engine = init_db(db_key, cfg.get("database_url"))
    Base.metadata.create_all(engine)
    session = get_session(engine)

    # — build models —
    print("▶ Building classic field distribution regressors…")
    bin_models = build_slate_bin_regressors(session)

    # — determine output directory —
    if out_dir is None:
        out_dir = project_root() / "data" / db_key / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # — dump models —
    with open(out_dir / "classic_proj_model.pkl", "wb") as f:
        pickle.dump(bin_models['proj_model'], f)
    with open(out_dir / "classic_own_model.pkl", "wb") as f:
        pickle.dump(bin_models['own_model'], f)
    with open(out_dir / "classic_scaler.pkl", "wb") as f:
        pickle.dump(bin_models['scaler'], f)
    with open(out_dir / "classic_bin_edges.pkl", "wb") as f:
        pickle.dump(bin_models['bin_edges'], f)

    print(f"✅ All models saved to {out_dir}")
