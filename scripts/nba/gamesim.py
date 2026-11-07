#!/usr/bin/env python3
"""
scripts/nba/gamesim.py

The NBA game simulation's fitting data and checks (ivan.sports.nba.gamesim; plan docs/nba/notes/game_sim_plan_2026-11-07.md).

    scripts/nba/gamesim.py events [--since 2026-10-01] [--rebuild] [--max-games 50]
        parse every game with stints and a stored feed into data/nba/derived/gamesim/events.parquet (incremental)
    scripts/nba/gamesim.py check [--game-ids 0022500001 ...]
        the parse against the box score: per-stat exact share and gap, counted vs estimated possessions, both teams' counts
    scripts/nba/gamesim.py exits [--since 2025-10-01] [--team BOS] [--csv data/nba/diagnostics/gamesim/exits.csv]
        who inherits a mid-game exit: the heir rules scored (held-out half), the flow matrix, a team's own exits
        (docs/nba/notes/midgame_exits_2026-11-07.md)
    scripts/nba/gamesim.py fit-script [--no-write]
        layer A: pace regression, segment shape, per-possession and segment noise, the lead's pull-back, the pace-efficiency
        coupling and the endgame tie bump from the lined games -> data/nba/models/gamesim/script.json
    scripts/nba/gamesim.py check-script [-n 2000]
        layer A against the lined games: PIT coverage of the final total and margin, pace error, overtime and garbage-time rates
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ivan.database.session import get_session, init_db
from ivan.sports.nba.gamesim import fouls as F
from ivan.sports.nba.gamesim import possessions as X
from ivan.sports.nba.gamesim import script as S
from ivan.sports.nba.paths import DIAG_DIR


def cmd_events(args) -> None:
    s = get_session(init_db("nba"))
    X.build(s, since=args.since, rebuild=args.rebuild, max_games=args.max_games, log=print)


def cmd_check(args) -> None:
    s = get_session(init_db("nba"))
    ev = X.load_events()
    if args.game_ids:
        ev = ev[ev.game_id.isin(args.game_ids)]
    if ev.empty:
        print("no events stored for those games: run `events` first"); return
    ids = ",".join(repr(str(g)) for g in ev.game_id.unique())
    box = pd.read_sql(text(f"select * from nba_stats where game_id in ({ids})"), s.get_bind())
    r = X.reconcile(ev, box)
    print(r["players"].round(3).to_string(index=False))
    for k, v in r["summary"].items():
        print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")
    tg = r["teams"]
    print(f"team-games with the two teams' possessions more than 3 apart: {int((tg.poss - tg.poss_opp).abs().gt(3).sum())} of {len(tg)}")


def cmd_exits(args) -> None:
    s = get_session(init_db("nba"))
    r = F.run(s, since=args.since)
    pd.set_option("display.width", 200)
    print(f"{r['summary']['exits']} exits ({r['summary']['foul_outs']} foul-outs), {r['summary']['owed_mean']:.1f} minutes owed on average, by group {r['summary']['by_group']}")
    print("\nflow fitted on the first half (out group -> share of owed to each heir group):")
    for og, d in r["flow_fitted"].items():
        print(f"  {og:>6}: " + ", ".join(f"{hg} {v:.2f}" for hg, v in d.items()))
    print("\nrules scored on the second half:")
    print(r["scores_heldout"].round(3).to_string(index=False))
    print("\nwhere the owed minutes went (all exits):")
    fl = r["flow"]; print(fl[fl.kind == "all"].round(3).to_string(index=False))
    if args.team:
        ex = r["exits"]; ex = ex[ex.team == args.team]
        print(f"\n{args.team}: {len(ex)} exits"); print(ex[["game_date", "player", "group", "minute", "kind", "owed", "margin"]].to_string(index=False))
        print(F.midgame_shares(r["exits"], r["heirs"], team=args.team).round(2).head(30).to_string(index=False))
    if args.csv:
        p = Path(args.csv); p.parent.mkdir(parents=True, exist_ok=True)
        r["exits"].to_csv(p, index=False); r["heirs"].to_csv(p.with_name(p.stem + "_heirs.csv"), index=False)
        print(f"\nwritten {p} and {p.with_name(p.stem + '_heirs.csv')}")


def _fmt(d: dict) -> str:
    return ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


def cmd_fit_script(args) -> None:
    s = get_session(init_db("nba"))
    seg = X.team_segments(X.load_events()); lines = S.load_lines(s, seg)
    p = S.fit(seg, lines, write=not args.no_write)
    print(f"{p['fit']['games']} lined games {p['fit']['lined_from']}..{p['fit']['lined_to']}")
    print(_fmt({k: v for k, v in p.items() if k not in ("segment_shape", "fit")}))
    print("segment shape: " + " ".join(f"{x:.3f}" for x in p["segment_shape"]))
    print("written " + str(S.MODEL_JSON) if not args.no_write else "not written")


def cmd_check_script(args) -> None:
    s = get_session(init_db("nba"))
    seg = X.team_segments(X.load_events()); lines = S.load_lines(s, seg)
    r = S.check(seg, lines, S.load_params(), n=args.n, stints=F.load_stints(s))
    print(_fmt(r["summary"]))
    g = r["games"]
    for col in ("pit_total", "pit_margin", "pit_poss"):
        h, _ = __import__("numpy").histogram(g[col], bins=5, range=(0, 1))
        print(f"{col:>11} PIT by fifth: {h.tolist()}   (flat = calibrated)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events"); e.add_argument("--since"); e.add_argument("--rebuild", action="store_true"); e.add_argument("--max-games", type=int); e.set_defaults(fn=cmd_events)
    c = sub.add_parser("check"); c.add_argument("--game-ids", nargs="*"); c.set_defaults(fn=cmd_check)
    x = sub.add_parser("exits"); x.add_argument("--since"); x.add_argument("--team"); x.add_argument("--csv", default=str(DIAG_DIR / "gamesim" / "exits.csv")); x.set_defaults(fn=cmd_exits)
    f = sub.add_parser("fit-script"); f.add_argument("--no-write", action="store_true"); f.set_defaults(fn=cmd_fit_script)
    k = sub.add_parser("check-script"); k.add_argument("-n", type=int, default=2000); k.set_defaults(fn=cmd_check_script)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
