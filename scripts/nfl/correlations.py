#!/usr/bin/env python3
"""
scripts/nfl/correlations.py

The working slate's correlations game by game (ivan.sports.nfl.correlation_check): the alerts, then every slot pair
that moves a lineup with the league's Spearman, the team's own history (count, standard error, tonight's QB only), what
the sim is given, and the target and sampled Pearson of the players in those slots (the newest stored sim report of a
contest on the game). The desk's correlation section is the same call.

    correlations.py                          # every game
    correlations.py --game CAR-GB            # one game
    correlations.py --pair GB QB WR1         # the team-games behind one pair (one line per game)
    correlations.py --pair GB QB DST --opp   # GB's QB against its opponent's DST in each game
    correlations.py --json
"""
from __future__ import annotations

import argparse
import json

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import correlation_check as cc


def _n(v, w=6, p=2) -> str:
    return f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", help="AWAY-HOME or HOME-AWAY, as the desk names it (CAR-GB)")
    ap.add_argument("--pair", nargs=3, metavar=("TEAM", "SLOT_A", "SLOT_B"), help="the history behind one pair")
    ap.add_argument("--opp", action="store_true", help="with --pair: slot B is the opponent's")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    session = get_session(init_db("nfl"))
    try:
        if a.pair:
            team, sa, sb = a.pair[0].upper(), a.pair[1].upper(), a.pair[2].upper()
            h = cc.pair_history(session, team, sa, sb, opponent="opp" if a.opp else None)
            if a.json:
                print(json.dumps(h, indent=1, default=str))
                return 0
            print(f"{team} {sa} x {'opp ' if a.opp else ''}{sb}: rho {_n(h['rho'])} over {h['n']} games (se {_n(h['se'])}) · "
                  f"with {h['qb']} {_n(h['rho_qb'])} over {h['n_qb']} · last {h['n_recent']} {_n(h['rho_recent'])}")
            for p in h["points"]:
                print(f"  {p['season']} wk{p['week']:<3d} vs {p['opp']:4s} {p['x']:6.1f} {p['y']:6.1f}  {p['qb']}{' *' if p['same_qb'] else ''}")
            return 0
        out = cc.slate_check(session)
        games = [g for g in out["games"] if not a.game or set(a.game.upper().split("-")) == set(g["teams"])]
        if a.json:
            print(json.dumps({**out, "games": games}, indent=1, default=str))
            return 0
        print(f"method {out['method']} · team matrices shrunk {out['shrink']:g} games toward the league · Spearman unless marked")
        for g in games:
            ln = g["lines"]
            line = f"{ln['fav']} {ln['spread']:g} · {ln['total']:g}" if ln and ln["fav"] else (f"PK · {ln['total']:g}" if ln else "no line")
            print(f"\n{g['away']} @ {g['home']}  {line}  [{g['level']}]")
            for t, h in g["history"].items():
                print(f"  {t}: {h['games']} games{'' if h['own'] else ' (league matrix)'}, QB {h['qb']} {h['qb_games']}g")
            if g["sampled"]:
                s = g["sampled"]
                print(f"  sampled: {s['contest']} · {s['n_samples']} draws · largest error {_n(s['max_abs_error'], 5, 3)} (noise ~{_n(s['se'], 5, 3)})")
            for x in g["alerts"]:
                print(f"  {x['level'].upper():5s} {x['text']}")
            print(f"  {'pair':24s} {'league':>6s} {'own':>6s} {'n':>4s} {'se':>5s} {'QB now':>6s} {'n':>3s} {'used':>6s} | {'target':>6s} {'drawn':>6s}  players")
            for p in g["pairs"]:
                name = f"{p['a_team']} {p['a']} x {p['b_team'] + ' ' if p['kind'] == 'cross' else ''}{p['b']}"
                who = f"{p['players'][0]} x {p['players'][1]}" if p["players"] else ""
                print(f"  {name:24s} {_n(p['league'])} {_n(p['own'])} {p['n']:4d} {_n(p['se'], 5)} {_n(p['own_qb'])} {p['n_qb']:3d} {_n(p['used'])} | "
                      f"{_n(p['target'])} {_n(p['sampled'])}  {who}{'  <- ' + ', '.join(p['flags']) if p['flags'] else ''}")
        if out["independent"]:
            print("\nsampled independently: " + "; ".join(f"{x['name']} ({x['team']}, {x['reason']})" for x in out["independent"]))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
