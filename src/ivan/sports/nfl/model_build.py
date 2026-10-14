# src/ivan/sports/nfl/model_build.py
"""
Every model the NFL pipeline reads from the model directory, one function each,
and build() to run them in order and write the manifest (ivan.core.model_store):

    dst_table              nfl_dst_game, DST points per team-game (a DB table; live only)
    dst                    dst_odds_params.json      DST mean from the lines, shape from history
    classic                classic_*.pkl             classic field projection / ownership shapes
    dupe                   dupe_model_v2.json        copies = N x k x prod(own)^beta on real fields
    sd                     sd_calibration.json       predictive spread factor per position
    ownership_calibration  ownership_calibration.json  actual ~ a + b x projected, per slot/position
    ownership_model        ownership_model.pkl       learned showdown ownership

A live build (cutoff None) writes the live directory from everything the database
holds. An as-of build (cutoff given) writes MODELS_DIR/asof/<cutoff>/ from contests
that finished and games played strictly before the cutoff, which is what a replay
of a slate at that date may use. scripts/nfl/build_models.py is argparse around build();
the per-model scripts call the matching function.
"""
from __future__ import annotations

import pickle
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

from ivan.core import model_store
from ivan.sports.nfl.paths import MODELS_DIR

Log = Callable[[str], None]
STEPS = ("dst_table", "dst", "classic", "dupe", "sd", "ownership_calibration", "ownership_model")
SD_SINCE = "2025-09-01"          # first season with projection snapshots
MIN_OWNERSHIP_CONTESTS = 20      # below this the learned ownership model is not fitted


# ------------------------------------------------------------------------------ one model each
def build_dst_table(session, log: Log = print) -> dict:
    """Rebuild nfl_dst_game for every season in nfl_stats through the one in progress."""
    from ivan.sports.nfl.dst_scoring import build_dst_game_table
    from ivan.sports.nfl.refactor.historical.data_fetch import seasons_to_fetch
    years = seasons_to_fetch()
    n = build_dst_game_table(None, years=years, session=session)
    log(f"   nfl_dst_game: {n} rows over seasons {years[0]}..{years[-1]}" if years else f"   nfl_dst_game: {n} rows")
    return {"rows": int(n), "seasons": list(years)}


def build_dst_params(session, out_dir: Path, before=None, log: Log = print) -> dict:
    """The two-part DST model (refactor.dst) on team-games before `before` = (season, week), or all."""
    from ivan.sports.nfl.refactor.dst import describe_dst_odds_model, fit_dst_odds_model
    params = fit_dst_odds_model(session, out_dir, before=before)
    log("\n".join("   " + line for line in describe_dst_odds_model(params).splitlines()))
    return {"n": params.get("n"), "before": list(before) if before else None}


def build_classic_field_models(session, out_dir: Path, cutoff: Optional[datetime] = None, log: Log = print) -> dict:
    """Classic field distribution regressors (model_calculations) -> classic_{proj_model,own_model,scaler,bin_edges}.pkl."""
    from ivan.sports.nfl.model_calculations import build_slate_bin_regressors
    models = build_slate_bin_regressors(session, cutoff)
    for name in ("proj_model", "own_model", "scaler", "bin_edges"):
        with open(out_dir / f"classic_{name}.pkl", "wb") as f:
            pickle.dump(models[name], f)
    log(f"   classic field models -> {out_dir}")
    return {"cutoff": cutoff.date().isoformat() if cutoff else None}


def build_dupe_model(session, out_dir: Path, cutoff: Optional[str] = None, log: Log = print):
    """The product-rule duplication model (dupe_model) on the fields the settle job kept; returns the fitted model."""
    from ivan.sports.nfl import dupe_model as DM
    df = DM.training_frame(session, cutoff=cutoff)
    m = DM.fit(df, cutoff=cutoff)
    m.cv = {k: {"contests": v["contests"], "folds": v["folds"]} for k, v in DM.backtest(df).items()} if len(df) else {}
    path = m.save(out_dir)
    log(f"   dupe model -> {path}: {m.describe()}")
    return m


def build_sd_calibration(session, out_dir: Path, since: str = SD_SINCE, until: Optional[str] = None,
                         positions: Iterable[str] = None, log: Log = print) -> dict:
    """
    Score the projection snapshots in [since, until] (backtest_projections) and write each
    position's predictive-sd factor (refactor.spread); positions with too few scored rows keep
    their old factor.
    """
    from ivan.sports.nfl import backtest_projections as BP
    from ivan.sports.nfl.refactor.spread import write_sd_calibration
    positions = list(positions or BP.POSITIONS)
    until = until or date.today().isoformat()
    scored = BP.scored_projections(session, since, until, positions)
    factors, counts, skipped = BP.sd_calibration_factors(scored, positions)
    for pos, n in skipped.items():
        log(f"   sd calibration: {pos} has {n} scored rows (need {BP.MIN_ROWS}); factor unchanged")
    cal = write_sd_calibration(factors, counts, (since, until), model_dir=out_dir)
    log("   sd calibration: " + " ".join(f"{p} x{f:.2f} (n={counts[p]})" for p, f in sorted(factors.items()))
        + f"; all factors now {cal['factors']}")
    return {"since": since, "until": until, "factors": cal["factors"], "n": counts, "skipped": skipped}


def build_ownership_calibration(session, out_dir: Path, cutoff: Optional[str] = None, min_n: int = 30, log: Log = print) -> dict:
    """Per-slot/position lines from projected to actual ownership (ownership_model.fit_calibration) -> ownership_calibration.json."""
    from ivan.sports.nfl import ownership_model as OM
    cal = OM.fit_calibration(session, cutoff, min_n)
    path = OM.save_calibration(cal, out_dir)
    log(f"   ownership calibration from {cal['contests']} contests ({cal['rows']} rows), {len(cal['maps'])} lines -> {path}")
    return {"cutoff": cutoff, "contests": cal["contests"], "rows": cal["rows"], "lines": len(cal["maps"])}


def build_ownership_model(session, out_dir: Path, cutoff: Optional[str] = None, log: Log = print) -> dict:
    """The learned showdown ownership model (ownership_model.fit) -> ownership_model.pkl; skipped on too few contests."""
    from ivan.sports.nfl import ownership_model as OM
    df = OM.training_frame(session, cutoff)
    n = int(df.contest_id.nunique()) if len(df) else 0
    if n < MIN_OWNERSHIP_CONTESTS:
        log(f"   ownership model: only {n} contests before the cutoff; not built")
        return {"cutoff": cutoff, "contests": n, "built": False}
    m = OM.fit(df, cutoff=cutoff)
    path = m.save(out_dir)
    log(f"   ownership model on {m.n_contests} contests ({m.n_rows} rows) -> {path}")
    return {"cutoff": cutoff, "contests": m.n_contests, "rows": m.n_rows, "built": True}


# ------------------------------------------------------------------------------ all of them
def build(session, cutoff: Optional[date] = None, out_dir: Optional[Path] = None, only: Optional[Iterable[str]] = None,
          sd_since: str = SD_SINCE, log: Log = print) -> dict:
    """
    Run the steps (all of STEPS, or `only`) and write the manifest. A failing step is
    recorded as ok=False with its error and the others still run; the manifest's
    "failed" list says which. Returns the manifest.
    """
    from ivan.sports.nfl.actuals import week_for_date
    steps = [s for s in STEPS if only is None or s in set(only)]
    unknown = set(only or ()) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown model step(s) {sorted(unknown)}; choose from {STEPS}")
    if cutoff is not None:
        out = Path(out_dir) if out_dir else model_store.asof_dir(MODELS_DIR, cutoff)
        before = week_for_date(session, cutoff)
        cut_dt = datetime.combine(cutoff, datetime.min.time())
        cut_iso = cutoff.isoformat()
        sd_until = (cutoff - timedelta(days=1)).isoformat()
        steps = [s for s in steps if s != "dst_table"]        # the table is all of history; a replay cuts it by the clock
        log(f"-> models as of {cutoff}: history before season/week {before} -> {out}")
    else:
        out = Path(out_dir) if out_dir else MODELS_DIR
        before, cut_dt, cut_iso, sd_until = None, None, None, None
        log(f"-> live models -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    runners = {
        "dst_table": lambda: build_dst_table(session, log),
        "dst": lambda: build_dst_params(session, out, before, log),
        "classic": lambda: build_classic_field_models(session, out, cut_dt, log),
        "dupe": lambda: {"cutoff": cut_iso, **_dupe_details(build_dupe_model(session, out, cut_iso, log))},
        "sd": lambda: build_sd_calibration(session, out, sd_since, sd_until, log=log),
        "ownership_calibration": lambda: build_ownership_calibration(session, out, cut_iso, log=log),
        "ownership_model": lambda: build_ownership_model(session, out, cut_iso, log),
    }
    built, failed = {}, []
    for step in steps:
        log(f"-> {step}")
        try:
            built[step] = {"ok": True, **runners[step]()}
        except Exception as e:   # noqa: BLE001 - one model failing must not lose the others
            failed.append(step)
            built[step] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            log(f"   {step} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    notes = [f"failed: {', '.join(failed)}"] if failed else None
    m = model_store.write_manifest(out, cutoff, built, notes=notes)
    m["failed"] = failed
    log(f"{'as-of' if cutoff else 'live'} models written to {out} ({', '.join(steps)})" + (f"; FAILED {failed}" if failed else ""))
    return m


def _dupe_details(m) -> dict:
    return {"fitted_on": getattr(m, "fitted_on", None), "contests": getattr(m, "contests", None)}
