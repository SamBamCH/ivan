#!/usr/bin/env python3
"""
scripts/nfl/build_models.py

Build marginals + copulas for NFL player and team data.
"""
import warnings, numpy as _np
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
_np.seterr(all="ignore")
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from ivan.core import model_store

import joblib
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
from scipy.stats import kstest, anderson_ksamp
from sqlalchemy.orm import Session

from ivan.sports.nfl.paths import MODELS_DIR
from ivan.sports.nfl.player_level_calculations import build_models as build_player_models
from ivan.sports.nfl.build_team10_vine import build_team10_vine
from ivan.sports.nfl.calibrate_game_factors import calibrate_game_factors
from ivan.sports.nfl.dst_scoring import build_dst_game_table
from ivan.sports.nfl.refactor.dst import fit_dst_odds_model, describe_dst_odds_model
from ivan.sports.nfl.refactor.historical.data_fetch import seasons_to_fetch
from ivan.sports.nfl.team_level_calculations   import build_team_models, _compute_fpts
from ivan.sports.nfl.build_team_positional_table import build_team_positional_table
from ivan.sports.nfl.slate_models import build_slate_models
from ivan.sports.nfl.diagnostics              import (
    plot_marginal_fit,
    test_marginal_ks,
    plot_copula_corr,
    compare_empirical_vs_model_scatter,
    estimate_tail_dependence
)
from ivan.sports.nfl.models import NFLStat
from ivan.database.session import init_db, get_session

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url',    default=None)
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--cutoff', default=None,
                        help="build an as-of version from data before this date (YYYY-MM-DD) into MODELS_DIR/asof/<cutoff>: "
                             "DST params, classic field models, the dupe model and the sd calibration (the legacy marginals/vine are skipped)")
    args = parser.parse_args()

    if args.cutoff:
        return build_asof(date.fromisoformat(args.cutoff), args.db_url)
    model_dir = Path(args.model_dir or MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("→ Building player‐level models")
    build_player_models(args.db_url, model_dir) 

    _years = seasons_to_fetch()   # every season in nfl_stats through the one in progress
    nrows = build_dst_game_table(args.db_url, years=_years)
    print(f"   wrote {nrows} rows to nfl_dst_game")

    # DST mean from the game lines, shape from history (writes dst_odds_params.json)
    engine = init_db("nfl", args.db_url)
    params = fit_dst_odds_model(get_session(engine), model_dir)
    print("\n".join("   " + line for line in describe_dst_odds_model(params).splitlines()))

    # new: fixed-width team-week positional fpts table
    n_team_rows = build_team_positional_table(args.db_url)
    print(f"   wrote {n_team_rows} rows to nfl_team_week_positional_fpts") 

    # === fit HGV-20 artifacts === 
    print("→ Building team‐level models")
    try: 
        import pyvinecopulib
    except ImportError:
        raise SystemExit("pyvinecopulib is required for NFL team models; please install it and try again")
    print("→ Fitting per-team 10D vine (QB..DST)")
    build_team10_vine(model_dir, args.db_url)     # writes team10_vine.pkl

    build_slate_models()
    model_store.write_manifest(model_dir, None, {"dst_odds_params": {"n": params.get("n")}, "team10_vine": {}, "classic_models": {}, "player_models": {}})


def build_asof(cutoff: date, db_url: str | None = None) -> int:
    """
    The models the replay of a slate at `cutoff` may use: trained on contests that finished and games played
    strictly before it, written to MODELS_DIR/asof/<cutoff>/ with a manifest (ivan.core.model_store).
    """
    import subprocess, sys
    from ivan.sports.nfl.actuals import week_for_date
    out = model_store.asof_dir(MODELS_DIR, cutoff)
    out.mkdir(parents=True, exist_ok=True)
    engine = init_db("nfl", db_url)
    session = get_session(engine)
    before = week_for_date(session, cutoff)
    built = {}
    print(f"→ as-of {cutoff}: history before season/week {before} -> {out}")
    params = fit_dst_odds_model(session, out, before=before)
    built["dst_odds_params"] = {"n": params.get("n"), "before": list(before) if before else None}
    print("\n".join("   " + line for line in describe_dst_odds_model(params).splitlines()))
    build_slate_models(out_dir=out, cutoff=datetime.combine(cutoff, datetime.min.time()))
    built["classic_models"] = {"cutoff": cutoff.isoformat()}
    script = Path(__file__).resolve().parent
    cmd = [sys.executable, str(script / "build_dupe_model.py"), "--cutoff", cutoff.isoformat(), "--model-dir", str(out)] + (["--db-url", db_url] if db_url else [])
    print("→ " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    built["dupe_model"] = {"cutoff": cutoff.isoformat(), "ok": rc == 0}
    cmd = [sys.executable, str(script / "backtest_projections.py"), "--until", (cutoff - timedelta(days=1)).isoformat(),
           "--write-calibration", "--model-dir", str(out)] + (["--db-url", db_url] if db_url else [])
    print("→ " + " ".join(cmd))
    rc2 = subprocess.run(cmd).returncode
    built["sd_calibration"] = {"until": (cutoff - timedelta(days=1)).isoformat(), "ok": rc2 == 0}
    cmd = [sys.executable, str(script / "build_ownership_calibration.py"), "--cutoff", cutoff.isoformat(), "--model-dir", str(out)] + (["--db-url", db_url] if db_url else [])
    print("→ " + " ".join(cmd))
    rc3 = subprocess.run(cmd).returncode
    built["ownership_calibration"] = {"cutoff": cutoff.isoformat(), "ok": rc3 == 0}
    cmd = [sys.executable, str(script / "build_ownership_model.py"), "build", "--cutoff", cutoff.isoformat(), "--model-dir", str(out)] + (["--db-url", db_url] if db_url else [])
    print("→ " + " ".join(cmd))
    rc4 = subprocess.run(cmd).returncode
    built["ownership_model"] = {"cutoff": cutoff.isoformat(), "ok": rc4 == 0}
    cmd = [sys.executable, str(script / "build_dupe_model.py"), "v2", "build", "--cutoff", cutoff.isoformat(), "--model-dir", str(out)] + (["--db-url", db_url] if db_url else [])
    print("→ " + " ".join(cmd))
    rc5 = subprocess.run(cmd).returncode
    built["dupe_model_v2"] = {"cutoff": cutoff.isoformat(), "ok": rc5 == 0}
    m = model_store.write_manifest(out, cutoff, built, notes=["legacy player marginals / team vine not built for as-of versions (the pipeline does not read them)"])
    print(f"✅ as-of models written to {out} ({', '.join(sorted(m['built']))})")
    return 0 if rc == 0 and rc2 == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main() or 0)
