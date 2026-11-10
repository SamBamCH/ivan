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
    scripts/nba/gamesim.py check-rotation [--start 2026-01-10 --end 2026-01-31] [-n 300] [--jitter 1.6]
        layer B with the real margin path plugged in: the new draw (fouls, heirs by flow, the path's garbage time) against the
        current sampler on the same proposed grids: minutes MAE, bias and 10-90 coverage by role, by final margin, for the
        players who really reached five fouls
    scripts/nba/gamesim.py fit-cuts [--no-write]
        the soft pool's table (projection/cuts.py -> data/nba/models/cuts.json): per role and own recent cut share, how often the
        rotation does not hold for a player (under 8 minutes), how often that is a DNP, the token minutes; the fringe profile
    scripts/nba/gamesim.py check-usage [--start 2026-01-10 --end 2026-02-28]
        layer C with the truth plugged in: the shares over the real fives and possessions against the per-minute rate x true
        minutes, per stat (attempts, threes, trips, turnovers, assists, rebounds, steals, blocks), overall / starters / bench /
        nights a 25-minute teammate sat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ivan.database.session import get_session, init_db
from ivan.sports.nba.gamesim import fouls as F
from ivan.sports.nba.gamesim import possessions as X
from ivan.sports.nba.gamesim import rotation as R
from ivan.sports.nba.gamesim import script as S
from ivan.sports.nba.gamesim import usage as U
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
    ev = X.load_events(); seg = X.team_segments(ev); lines = S.load_lines(s, seg)
    p = S.fit(seg, lines, write=not args.no_write)
    print(f"{p['fit']['games']} lined games {p['fit']['lined_from']}..{p['fit']['lined_to']}")
    print(_fmt({k: v for k, v in p.items() if k not in ("segment_shape", "fit")}))
    print("segment shape: " + " ".join(f"{x:.3f}" for x in p["segment_shape"]))
    if not args.no_write:
        t = S.write_teams(seg, ev, p)
        print(f"written {S.MODEL_JSON} and {S.TEAMS_JSON} ({len(t['paces'])} team paces; league oreb share {t['oreb_share']:.3f}, assisted {t['ast_share']:.3f}, "
              f"steals per turnover {t['p_stl']:.3f}, blocks per miss {t['p_blk']:.3f}, live FT misses {t['ft_miss_live']:.2f})")
    else:
        print("not written")


def cmd_check_script(args) -> None:
    s = get_session(init_db("nba"))
    seg = X.team_segments(X.load_events()); lines = S.load_lines(s, seg)
    r = S.check(seg, lines, S.load_params(), n=args.n, stints=F.load_stints(s))
    print(_fmt(r["summary"]))
    g = r["games"]
    for col in ("pit_total", "pit_margin", "pit_poss"):
        h, _ = __import__("numpy").histogram(g[col], bins=5, range=(0, 1))
        print(f"{col:>11} PIT by fifth: {h.tolist()}   (flat = calibrated)")


def cmd_check_usage(args) -> None:
    from datetime import date
    from ivan.sports.nba.projection.groups import group_of, position_map
    s = get_session(init_db("nba"))
    ev = X.load_events()
    box = pd.read_sql(text("select game_id, player_name, team_abbreviation, position, minutes from nba_stats where substr(game_id,1,3) in ('002','004')"), s.get_bind())
    groups = {n: group_of(p) for n, p in position_map(s).items()}
    r = U.check(ev, box, groups, date.fromisoformat(args.start), date.fromisoformat(args.end), log=print)
    print(_fmt(r["summary"]))
    pd.set_option("display.width", 220)
    print(r["table"].round(3).to_string(index=False))


def cmd_check_rotation(args) -> None:
    from datetime import date
    from ivan.sports.nba.projection.groups import group_of, position_map
    s = get_session(init_db("nba"))
    groups = {n: group_of(p) for n, p in position_map(s).items()}
    r = R.check(s, date.fromisoformat(args.start), date.fromisoformat(args.end), n=args.n, groups=groups, jitter_scale=args.jitter, log=print)
    print(_fmt(r["summary"]))
    pd.set_option("display.width", 220)
    for key in ("by_role", "by_margin", "five_fouls"):
        print(); print(r[key].round(3).to_string(index=False))


def cmd_fit_cuts(args) -> None:
    from ivan.sports.nba.projection import cuts as CUTS
    from ivan.sports.nba.projection.history import load_box_scores
    s = get_session(init_db("nba"))
    box = load_box_scores(s)
    t = CUTS.fit(box)
    ft = CUTS.fringe_from_tips(s, box)
    if ft:
        t["fringe"] = ft
        print(f"fringe from the tip records: {ft['_source']}; plays {ft['_p_play_all']:.2f} of nights, {ft['_mean_all']} min when he does")
    print(f"{t['rows']} player-games through {t['fitted_on']}; token minutes when cut but used {t['token_minutes']}")
    print(f"{'role':8s} " + " ".join(f"{b:>14s}" for b, _, _ in CUTS.BINS) + "   P(cut) [P(DNP | cut)] (n)")
    for role, _, _ in CUTS.ROLES:
        print(f"{role:8s} " + " ".join(f"{t['p_cut'][role][b]:.2f} [{t['p_dnp_given_cut'][role][b]:.2f}] {t['n'][role][b]:>5d}" for b, _, _ in CUTS.BINS))
    print("fringe (planned zero): " + "; ".join(f"{b}: plays {f['p_play']:.2f}, {f['mean']:.1f} ± {f['sd']:.1f} min (n {f.get('n', '?')}{', pooled' if f.get('pooled') else ''})" for b, f in t["fringe"].items() if not b.startswith("_")))
    if not args.no_write:
        print("written " + str(CUTS.save_table(t)))


def cmd_draw_world(args) -> None:
    import time
    from ivan.sports.nba import slates
    from ivan.sports.nba.gamesim import world as W
    slates.use(args.slate); s = slates.session(args.slate)
    t = time.time()
    w = W.World.draw(s, n=args.n, seed=args.seed, log=print)
    print(f"{len(w.names)} players, {len(w.games)} games, {args.n} worlds in {time.time() - t:.1f} s")
    print(_fmt(W.moments(w)))
    f = w.frame()
    pool = pd.read_sql(text("select id pid, fpts_mean pool_fpts, fpts_std pool_sd, minutes_proj from nba_player"), s.get_bind())
    m = f.merge(pool, on="pid", how="left")
    print(f"fantasy mean vs the pool: bias {(m.fpts_mean - m.pool_fpts).mean():+.2f}, MAE {(m.fpts_mean - m.pool_fpts).abs().mean():.2f}, corr {m[['fpts_mean', 'pool_fpts']].corr().iloc[0, 1]:.3f}; "
          f"sd {m.fpts_std.mean():.2f} vs {m.pool_sd.mean():.2f}; minutes {m.minutes.mean():.2f} vs {m.minutes_proj.mean():.2f}")
    pd.set_option("display.width", 220)
    cols = ["player", "team", "minutes_proj", "minutes", "pool_fpts", "fpts_mean", "fpts_std", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fga", "fta", "fouls"]
    print(m.sort_values("pool_fpts", ascending=False).head(args.top)[cols].round(2).to_string(index=False))


def cmd_calibrate_world(args) -> None:
    from ivan.sports.nba import slates
    from ivan.sports.nba.gamesim import world as W
    slates.use(args.slate); s = slates.session(args.slate)
    params = S.load_params()
    real = dict(sd_total=params["real_sd_total"], sd_margin=params["real_sd_margin"], p_ot=params["p_ot"])
    games = W.slate_inputs(s)
    p = W.calibrate(s, games, params, real, n=args.n, rounds=args.rounds, log=print)
    S.MODEL_JSON.write_text(json.dumps(p, indent=1))
    print("written " + str(S.MODEL_JSON) + ": " + _fmt({k: v for k, v in p.items() if k.startswith("world_")}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events"); e.add_argument("--since"); e.add_argument("--rebuild", action="store_true"); e.add_argument("--max-games", type=int); e.set_defaults(fn=cmd_events)
    c = sub.add_parser("check"); c.add_argument("--game-ids", nargs="*"); c.set_defaults(fn=cmd_check)
    x = sub.add_parser("exits"); x.add_argument("--since"); x.add_argument("--team"); x.add_argument("--csv", default=str(DIAG_DIR / "gamesim" / "exits.csv")); x.set_defaults(fn=cmd_exits)
    f = sub.add_parser("fit-script"); f.add_argument("--no-write", action="store_true"); f.set_defaults(fn=cmd_fit_script)
    k = sub.add_parser("check-script"); k.add_argument("-n", type=int, default=2000); k.set_defaults(fn=cmd_check_script)
    u = sub.add_parser("check-usage"); u.add_argument("--start", default="2026-01-10"); u.add_argument("--end", default="2026-02-28"); u.set_defaults(fn=cmd_check_usage)
    b = sub.add_parser("check-rotation"); b.add_argument("--start", default="2026-01-10"); b.add_argument("--end", default="2026-01-31"); b.add_argument("-n", type=int, default=300)
    b.add_argument("--jitter", type=float, default=None); b.set_defaults(fn=cmd_check_rotation)
    fc = sub.add_parser("fit-cuts"); fc.add_argument("--no-write", action="store_true"); fc.set_defaults(fn=cmd_fit_cuts)
    d = sub.add_parser("draw-world"); d.add_argument("--slate", type=int, required=True); d.add_argument("-n", type=int, default=1000); d.add_argument("--seed", type=int, default=None)
    d.add_argument("--top", type=int, default=15); d.set_defaults(fn=cmd_draw_world)
    cw = sub.add_parser("calibrate-world"); cw.add_argument("--slate", type=int, required=True); cw.add_argument("-n", type=int, default=400); cw.add_argument("--rounds", type=int, default=2)
    cw.set_defaults(fn=cmd_calibrate_world)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
