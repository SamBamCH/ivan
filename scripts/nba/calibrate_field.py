#!/usr/bin/env python3
"""
scripts/nba/calibrate_field.py

Calibrate the lineup generator to a real field. For a derived contest
(scripts/nba/standings.py derive), the real field is described by two
curves measured with OUR projections at lock: the ownership of the most-used
players and the distribution of the lineups' projected totals below the
mean-optimal lineup. Candidate pools are then generated in process from the
slate's capture (the same solver the generate step uses, independent draws
around the lock projections) over a grid of randomness x floor, and each is
scored on how closely it reproduces those curves.

    scripts/nba/calibrate_field.py --contest-id 196456057
    scripts/nba/calibrate_field.py --contest-id 196456057 --randomness 0.15,0.25,0.35,0.5 --floors 0.85,0.9,0.95 -n 400

The score per setting is the mean absolute ownership gap over the top-40
players (points of ownership) plus the gap in the median and 90th-percentile
projection shortfall (points), reported side by side; the best setting is the
one to put in config generate.randomness / generate.min_fpts_frac.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from ivan.core import archive
from ivan.core.config import load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nba import standings as S
from ivan.sports.nba.models import HistoricalContest
from ivan.sports.nba.pre_lock.lineup_generation import FieldPlayer, generate_single_classic_lineup
from ivan.sports.nba.utils import _coerce_positions


def field_curves(lu: pd.DataFrame, side: pd.DataFrame, opt: float, top_n: int = 40) -> dict:
    n = int(lu.copies.sum())
    own = Counter()
    for r in lu.itertuples():
        for j in range(1, 9):
            own[int(getattr(r, f"p{j}"))] += int(r.copies)
    own_pct = {pid: 100.0 * c / n for pid, c in own.items()}
    gaps = np.repeat((opt - lu.proj_sum.values), lu.copies.values) if "proj_sum" in lu else np.array([])
    return dict(n=n, own=own_pct, top=[pid for pid, _ in own.most_common(top_n)],
                gap_p50=float(np.median(gaps)) if gaps.size else np.nan, gap_p90=float(np.percentile(gaps, 90)) if gaps.size else np.nan,
                within10=float((gaps <= 0.10 * opt).mean()) if gaps.size else np.nan)


def generate_pool(players: list, proj: pd.DataFrame, cfg: dict, n: int, randomness: float, floor_frac: float, rng: np.random.Generator) -> list:
    means = {p.id: float(proj.fpts_mean.get(p.id, 0.0) or 0.0) for p in players}
    sds = {p.id: float(proj.fpts_std.get(p.id, 0.0) or 0.0) for p in players}
    opt_ids = generate_single_classic_lineup(players=players, base_scores=means, config=cfg, max_salary_per_team=cfg.get("max_salary_per_team", 25000))
    opt = sum(means[i] for i in opt_ids)
    out, seen, attempts = [], set(), 0
    while len(out) < n and attempts < 6 * n:
        attempts += 1
        scores = {pid: max(0.0, means[pid] + randomness * sds[pid] * rng.standard_normal()) * (1 + rng.uniform(-0.05, 0.05)) for pid in means}
        ids = generate_single_classic_lineup(players=players, base_scores=scores, config=cfg, min_fpts=floor_frac * opt, max_salary_per_team=cfg.get("max_salary_per_team", 25000))
        if not ids:
            continue
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key); out.append(key)
    return out, opt


def pool_curves(pool: list, means: dict, opt: float, top: list) -> dict:
    n = len(pool)
    own = Counter(i for l in pool for i in l)
    own_pct = {pid: 100.0 * c / n for pid, c in own.items()}
    gaps = np.array([opt - sum(means[i] for i in l) for l in pool])
    return dict(n=n, own=own_pct, gap_p50=float(np.median(gaps)), gap_p90=float(np.percentile(gaps, 90)), within10=float((gaps <= 0.10 * opt).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contest-id", type=int, required=True)
    ap.add_argument("--randomness", default="0.15,0.25,0.35,0.5")
    ap.add_argument("--floors", default="0.85,0.9")
    ap.add_argument("-n", type=int, default=300, help="lineups per setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--derived-dir", default=None, help="where the derived contests live (default data/nba/derived/contests)")
    a = ap.parse_args()
    session = get_session(init_db("nba"))
    lu, side, summary = S.load_field(a.contest_id, a.derived_dir) if a.derived_dir else S.load_field(a.contest_id)
    if lu is None:
        print(f"no derived field for {a.contest_id}: run scripts/nba/standings.py derive"); return 1
    contest = session.get(HistoricalContest, a.contest_id)
    cap, cap_players = S.capture_for_contest(session, contest)
    proj = S.projection_at_lock(session, int(contest.draft_group_id))
    cfg = load_config("nba")
    players = [FieldPlayer(id=int(p["id"]), name=p["name"], salary=int(p["salary"]), team=p.get("team") or "", opponent=p.get("opponent"),
                           position=tuple(_coerce_positions(p.get("position") or [])), fpts_mean=float(proj.fpts_mean.get(int(p["id"]), 0.0) or 0.0),
                           game_id=None, gametime=None)
               for p in cap_players if int(p["id"]) in proj.index and (proj.fpts_mean.get(int(p["id"])) or 0) > 15]
    means = {p.id: p.fpts_mean for p in players}
    names = {int(p["id"]): p["name"] for p in cap_players}
    opt_ids = generate_single_classic_lineup(players=players, base_scores=means, config=cfg, max_salary_per_team=cfg.get("max_salary_per_team", 25000))
    opt = sum(means[i] for i in opt_ids)
    real = field_curves(lu, side, opt)
    print(f"{contest.name} (#{a.contest_id}): {real['n']:,} entries, {len(lu):,} unique lineups; our lock projection's optimal {opt:.1f}")
    print(f"  real field: median {real['gap_p50']:.1f} below the optimal, p90 {real['gap_p90']:.1f}, {100 * real['within10']:.0f}% within 10%")
    print("  real top ownership: " + ", ".join(f"{names.get(pid, pid).split()[-1]} {real['own'][pid]:.0f}%" for pid in real["top"][:8]))
    rng = np.random.default_rng(a.seed)
    rows = []
    for rnd in [float(x) for x in a.randomness.split(",")]:
        for fl in [float(x) for x in a.floors.split(",")]:
            pool, _ = generate_pool(players, proj, cfg, a.n, rnd, fl, rng)
            if not pool:
                continue
            pc = pool_curves(pool, means, opt, real["top"])
            own_gap = float(np.mean([abs(pc["own"].get(pid, 0.0) - real["own"][pid]) for pid in real["top"]]))
            rows.append(dict(randomness=rnd, floor=fl, lineups=len(pool), gap_p50=round(pc["gap_p50"], 1), gap_p90=round(pc["gap_p90"], 1), within10=round(pc["within10"], 2),
                             own_gap_top40=round(own_gap, 1), top5=" ".join(f"{names.get(pid, pid).split()[-1]}:{pc['own'].get(pid, 0):.0f}" for pid in real["top"][:5]),
                             score=round(own_gap + abs(pc["gap_p50"] - real["gap_p50"]) + 0.5 * abs(pc["gap_p90"] - real["gap_p90"]), 1)))
            print(f"  randomness {rnd:.2f} floor {fl:.2f}: median gap {pc['gap_p50']:.1f} p90 {pc['gap_p90']:.1f} within10 {100 * pc['within10']:.0f}% | ownership gap {own_gap:.1f} pts | score {rows[-1]['score']}")
    df = pd.DataFrame(rows).sort_values("score")
    pd.set_option("display.width", 200)
    print("\nranked (lower score = closer to the real field):")
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nbest: randomness {best.randomness:.2f}, floor {best.floor:.2f} -> config generate.randomness / generate.min_fpts_frac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
