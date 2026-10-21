#!/usr/bin/env python3
"""
scripts/nba/validate.py

The minutes / per-minute model against what happened, and what to check before lock
(ivan.sports.nba.validation).

    scripts/nba/validate.py scorecard                      every slate with box scores and no scorecard yet
    scripts/nba/validate.py scorecard --draft-group 154575 [--force]
    scripts/nba/validate.py scorecard --date 2026-10-20    every slate that locked that day
    scripts/nba/validate.py history                        the headline numbers per scored slate
    scripts/nba/validate.py review [--team BOS] [--all]    the current pool: minutes provenance and flags per team
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.database.session import get_session, init_db
from ivan.sports.nba import validation as V
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, read_overrides


def cmd_scorecard(session, a) -> int:
    if a.draft_group:
        dgs = [int(a.draft_group)]
    elif a.date:
        rows = session.execute(text("select distinct draft_group_id from nba_projection_capture where source='ours' and date(lock) = :d"), {"d": a.date}).all()
        dgs = [int(r[0]) for r in rows]
    else:
        dgs = V.pending_draft_groups(session)
        if not dgs:
            print("nothing pending: every slate with box scores has a scorecard (see `history`)")
            return 0
    for dg in dgs:
        card = V.build_scorecard(session, dg)
        if card is None:
            print(f"slate {dg}: no captures"); continue
        if card.rows.empty:
            print(V.format_scorecard(card)); continue
        if not a.force and card.key in V.scored_keys():
            print(f"slate {dg} ({card.day}): already scored; --force to redo")
        else:
            path = V.save_scorecard(card)
            print(f"wrote {path}")
        print(V.format_scorecard(card, top=a.top))
    return 0


def cmd_history(session, a) -> int:
    h = V.scorecard_history()
    if h.empty:
        print("no scorecards yet"); return 0
    pd.set_option("display.width", 200)
    print(h.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    rot = h.dropna(subset=["fpts_mae"])
    if len(rot) > 1:
        print(f"\n{len(rot)} slates: minutes bias {rot.min_bias.mean():+.2f} MAE {rot.min_mae.mean():.2f} | fpts bias {rot.fpts_bias.mean():+.2f} MAE {rot.fpts_mae.mean():.2f} "
              f"(DK avg {rot.dk_mae.mean():.2f}) | bench minutes bias {rot.bench_min_bias.mean():+.2f} | starters {rot.starter_min_bias.mean():+.2f} | DNP {int(rot.dnp.sum())}")
    return 0


def cmd_review(session, a) -> int:
    slate = LiveSlate(session)
    if not slate.players:
        print("no players in nba_player"); return 1
    overrides = read_overrides()
    params = ProjectionParams(n_sims=0)
    proj = {t: slate.project(t, overrides, set(), params, rng=np.random.default_rng(0)) for t in slate.teams()}
    cards = V.review_slate(session, slate, proj, overrides, teams=[a.team.upper()] if a.team else None)
    for c in cards:
        if a.all or a.team or c["score"] >= a.min_score:
            print(V.format_review(c)); print()
    if not a.team:
        print("alerts by team: " + ", ".join(f"{c['team']} {c['score']}" for c in cards))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scorecard"); s.add_argument("--draft-group", type=int, default=None); s.add_argument("--date", default=None)
    s.add_argument("--force", action="store_true"); s.add_argument("--top", type=int, default=V.TOP_N)
    sub.add_parser("history")
    r = sub.add_parser("review"); r.add_argument("--team", default=None); r.add_argument("--all", action="store_true", help="every team, not only the flagged ones")
    r.add_argument("--min-score", type=int, default=4, help="print teams with at least this alert score (default 4)")
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    return {"scorecard": cmd_scorecard, "history": cmd_history, "review": cmd_review}[a.cmd](session, a)


if __name__ == "__main__":
    raise SystemExit(main())
