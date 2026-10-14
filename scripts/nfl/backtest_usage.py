#!/usr/bin/env python3
"""
Projection-only replay: fit every archived slate's projections under two
unpriced-volume models and score both against the actual DraftKings points.

    scripts/nfl/backtest_usage.py --since 2025-11-01 --until 2025-11-30 [--modes salary,environment] [--out DIR]

The same restore the full replay does (clock frozen at lock - 1 minute,
history limited to earlier weeks), but only the projection step, so twelve
showdown slates take minutes instead of hours. Nothing is generated,
simulated or chosen.

Writes usage_ab.csv (one row per player, slate and mode) and a validation
report (summary.json with reports.players for the second mode and
reports.players_salary for the first, plus reports.usage_ab) under
data/nfl/output/validation/usage_<stamp>/, so the Validation page shows it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ivan.core import archive, clock, replay as rp, validation as V   # noqa: E402
from ivan.core import seed as seedmod                                 # noqa: E402
from ivan.core.config import load_config                              # noqa: E402
from ivan.database.session import add_missing_columns, get_session, init_db   # noqa: E402
from ivan.sports.nfl import actuals                                   # noqa: E402
from ivan.sports.nfl.models import NFLBase                            # noqa: E402
from ivan.sports.nfl.paths import OUTPUT_DIR                          # noqa: E402
from ivan.sports.nfl.replay_adapter import NFLReplayAdapter           # noqa: E402
from ivan.sports.nfl.run_artifacts import projections_frame           # noqa: E402
from ivan.sports.nfl.validate import SLOT_LABEL                       # noqa: E402

DB_DIR = PROJECT_ROOT / "data"


def fit_slate(session, adapter, cap, mode: str, cfg: dict, log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restore one archived slate and fit its projections under `mode`; returns the projections frame with actuals."""
    rp.clear_working_tables(session, adapter.working_tables)
    blobs = archive.load_blobs(session, cap.id)
    with open(log_path, "a") as fh, contextlib.redirect_stdout(fh):
        print(f"==== {cap.slate_key} {mode} ====")
        adapter.restore(session, blobs, print)
        from ivan.sports.nfl.refactor.player import fit_fpts_distributions
        with rp._env_var("IVAN_UNPRICED_VOLUME", mode):
            info = fit_fpts_distributions(session, cfg) or {}
    df = projections_frame(session)
    usage_rows = pd.DataFrame(info.get("usage") or [])
    if not usage_rows.empty:
        usage_rows["coverage"] = usage_rows.fixed / usage_rows.total.replace(0, np.nan)
        usage_rows["slate"] = rp.slate_dir_name(cap); usage_rows["mode"] = mode
    df = df[df.roster_slot_id != 511].copy()                       # captain rows copy the FLEX fit
    sw = actuals.week_for_date(session, cap.lock)
    table = actuals.actual_points(session, *sw) if sw else {}
    df["actual_fpts"] = [actuals.lookup_points(table, str(n), str(t), str(p)) for n, t, p in zip(df.name, df.team, df.position)]
    df["slot"] = [SLOT_LABEL.get(int(r), str(r)) for r in df.roster_slot_id]
    df["mode"] = mode
    df["slate"] = rp.slate_dir_name(cap)
    df["season"], df["week"] = (sw if sw else (None, None))
    return df, usage_rows


def report_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"fpts_mean": "proj_mean", "fpts_std": "proj_sd", "ownership_proj": "own_proj"}).copy()
    for q in V.Q_COLS:
        src = "fpts_p" + q[1:]
        if src in out.columns:
            out[q] = out[src]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True); ap.add_argument("--until", required=True)
    ap.add_argument("--modes", default="salary,environment")
    ap.add_argument("--replay-db", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--slates", default=None, help="comma-separated slate keys")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    since = datetime.fromisoformat(a.since); until = datetime.fromisoformat(a.until) + timedelta(days=1) - timedelta(seconds=1)
    live_db = DB_DIR / "nfl" / "nfl_history.sqlite"
    replay_db = Path(a.replay_db) if a.replay_db else DB_DIR / "nfl" / "replay" / "nfl_replay.usage.sqlite"
    adapter = NFLReplayAdapter(a.config)
    if not replay_db.exists():
        rp.make_replay_db(live_db, replay_db, adapter.working_tables)
    engine = init_db("nfl", f"sqlite:///{replay_db}")
    archive.ensure_tables(engine)
    added = add_missing_columns(engine, NFLBase.metadata)
    if added:
        print(f"replay database: added columns {added}")
    session = get_session(engine)
    cfg = load_config("nfl", adapter.config_path)
    caps = archive.slates(session, "nfl", since, until)
    if a.slates:
        keep = {s.strip() for s in a.slates.split(",")}
        caps = [c for c in caps if c.slate_key in keep]
    if not caps:
        print("no archived slates in the window"); return 1
    out = Path(a.out) if a.out else OUTPUT_DIR / "validation" / f"usage_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "fit.log"
    print(f"{len(caps)} archived slate(s), modes {modes} -> {out}")
    frames = []
    usage_frames = []
    for cap in caps:
        as_of = cap.lock - timedelta(minutes=1)
        for mode in modes:
            t0 = datetime.now()
            with clock.frozen(as_of), rp._env_var(seedmod.ENV, str(a.seed)):
                df, urows = fit_slate(session, adapter, cap, mode, cfg, log_path)
            frames.append(df)
            if not urows.empty:
                usage_frames.append(urows)
            vs = df.volume_source.value_counts().to_dict()
            print(f"  {rp.slate_dir_name(cap)} {mode:11s} {len(df):3d} players, actuals {int(df.actual_fpts.notna().sum()):3d}, "
                  f"sources {vs}  ({(datetime.now() - t0).seconds}s)")
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(out / "usage_ab.csv", index=False)

    reports = {}
    for mode in modes:
        rep = V.player_report(report_frame(allf[allf["mode"] == mode]))
        rep.pop("frame", None)
        reports["players" if mode == modes[-1] else f"players_{mode}"] = rep
    ab = compare(allf, modes)
    reports["usage_ab"] = ab
    if usage_frames:
        uf = pd.concat(usage_frames, ignore_index=True)
        uf.to_csv(out / "usage.csv", index=False)
        rep = V.usage_report(uf[uf["mode"] == modes[-1]])
        rep.pop("frame", None)
        reports["usage"] = rep
    checks = V.build_checks(players=reports["players"], usage=reports.get("usage"))
    s = {"title": a.title or f"unpriced volume A/B {' vs '.join(modes)} {a.since}..{a.until}", "created_at": datetime.now().isoformat(timespec="seconds"),
         "since": a.since, "until": a.until, "contests": 0, "coverage": [], "checks": checks, "check_counts": V.summarise_checks(checks),
         "reports": reports}
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print_summary(ab, modes)
    print(f"wrote {out}/")
    return 0


def compare(allf: pd.DataFrame, modes: list[str]) -> dict:
    """Error of each mode by volume source / position / salary tier, and the paired comparison on the players the modes disagree on."""
    df = allf[allf.actual_fpts.notna()].copy()
    df["err"] = df.actual_fpts - df.fpts_mean
    df["salary_tier"] = "all"
    try:
        base = df[df["mode"] == modes[0]]
        tiers = pd.qcut(base.salary.rank(method="first"), 4, labels=["Q1 cheap", "Q2", "Q3", "Q4 expensive"]).astype(str)
        tier_of = dict(zip(zip(base.slate, base.player_id), tiers))
        df["salary_tier"] = [tier_of.get((s, p), "all") for s, p in zip(df.slate, df.player_id)]
    except ValueError:
        pass

    def block(g: pd.DataFrame) -> dict:
        e = g.err.values.astype(float)
        return {"n": int(len(g)), "bias": round(float(e.mean()), 3) if len(g) else None, "mae": round(float(np.abs(e).mean()), 3) if len(g) else None,
                "proj_mean": round(float(g.fpts_mean.mean()), 3) if len(g) else None, "actual_mean": round(float(g.actual_fpts.mean()), 3) if len(g) else None}

    out = {"modes": modes, "by_mode": {}, "paired": {}}
    for mode in modes:
        m = df[df["mode"] == mode]
        out["by_mode"][mode] = {"overall": block(m),
                                "by_volume_source": {str(k): block(g) for k, g in m.groupby("volume_source", dropna=False)},
                                "by_position": {str(k): block(g) for k, g in m.groupby("position")},
                                "by_salary_tier": {str(k): block(g) for k, g in m.groupby("salary_tier")},
                                "unpriced_by_position": {str(k): block(g) for k, g in m[m.volume_source != "market"].groupby("position")}}
    if len(modes) == 2:
        a_, b_ = modes
        key = ["slate", "player_id"]
        pa = df[df["mode"] == a_].set_index(key); pb = df[df["mode"] == b_].set_index(key)
        both = pa.join(pb[["fpts_mean", "fpts_std", "volume_source", "err"]], rsuffix="_b", how="inner")
        diff = both[(both.volume_source != both.volume_source_b) | ((both.fpts_mean - both.fpts_mean_b).abs() > 0.05)]
        ea, eb = np.abs(diff.err.values), np.abs(diff.err_b.values)
        out["paired"] = {"n_common": int(len(both)), "n_changed": int(len(diff)),
                         "mae_a": round(float(ea.mean()), 3) if len(diff) else None, "mae_b": round(float(eb.mean()), 3) if len(diff) else None,
                         "bias_a": round(float(diff.err.mean()), 3) if len(diff) else None, "bias_b": round(float(diff.err_b.mean()), 3) if len(diff) else None,
                         "b_closer_share": round(float((eb < ea).mean()), 3) if len(diff) else None,
                         "proj_mean_a": round(float(diff.fpts_mean.mean()), 3) if len(diff) else None,
                         "proj_mean_b": round(float(diff.fpts_mean_b.mean()), 3) if len(diff) else None,
                         "actual_mean": round(float(diff.actual_fpts.mean()), 3) if len(diff) else None,
                         "changed_by_position": {str(k): {"n": int(len(g)), "mae_a": round(float(g.err.abs().mean()), 3), "mae_b": round(float(g.err_b.abs().mean()), 3),
                                                          "bias_a": round(float(g.err.mean()), 3), "bias_b": round(float(g.err_b.mean()), 3)}
                                                 for k, g in diff.groupby("position")},
                         "rows": diff.reset_index()[["slate", "name", "team", "position", "salary", "fpts_mean", "fpts_mean_b", "actual_fpts", "volume_source", "volume_source_b"]]
                                 .rename(columns={"fpts_mean": f"proj_{a_}", "fpts_mean_b": f"proj_{b_}"}).round(2).to_dict("records")}
    return out


def print_summary(ab: dict, modes: list[str]) -> None:
    print("\nprojected vs actual DK points (players with actuals)")
    for mode in modes:
        bm = ab["by_mode"][mode]
        o = bm["overall"]
        print(f"  {mode:11s} n {o['n']:4d}  bias {o['bias']:+.2f}  MAE {o['mae']:.2f}")
        for k, v in bm["by_volume_source"].items():
            print(f"      {k:12s} n {v['n']:4d}  bias {v['bias']:+.2f}  MAE {v['mae']:.2f}  proj {v['proj_mean']:.2f} actual {v['actual_mean']:.2f}")
        for k, v in bm["unpriced_by_position"].items():
            print(f"      unpriced {k:4s} n {v['n']:4d}  bias {v['bias']:+.2f}  MAE {v['mae']:.2f}  proj {v['proj_mean']:.2f} actual {v['actual_mean']:.2f}")
    p = ab.get("paired") or {}
    if p:
        a_, b_ = modes
        print(f"\npaired on the {p['n_changed']} of {p['n_common']} players the modes disagree on:")
        print(f"  {a_:11s} MAE {p['mae_a']:.2f} bias {p['bias_a']:+.2f} proj {p['proj_mean_a']:.2f} | {b_:11s} MAE {p['mae_b']:.2f} bias {p['bias_b']:+.2f} proj {p['proj_mean_b']:.2f} | actual {p['actual_mean']:.2f}; {b_} closer on {p['b_closer_share']:.0%}")
        for k, v in p["changed_by_position"].items():
            print(f"    {k:4s} n {v['n']:3d}  {a_} MAE {v['mae_a']:.2f} bias {v['bias_a']:+.2f} | {b_} MAE {v['mae_b']:.2f} bias {v['bias_b']:+.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
