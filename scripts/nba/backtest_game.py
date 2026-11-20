#!/usr/bin/env python3
"""
scripts/nba/backtest_game.py

One game of a past slate (ivan.sports.nba.backtest.game): what the desk's backtest view shows at each depth. The game is
drawn by the game simulation on this checkout from the slate's tip record (data/nba/slates/<dg>/tips/), history cut at
that day; the draw and the edits are kept under data/nba/slates/<dg>/backtest/<AWAY-HOME>/.

    scripts/nba/backtest_game.py --slate 156831                         the slate's games: the line against the final, the projection's errors
    scripts/nba/backtest_game.py --slate 156831 --game MIL-MIA           the game as on file (drawn first when no draw is on file)
    scripts/nba/backtest_game.py --slate 156831 --game MIA --simulate [--worlds 2000]
                                                                        draw it again with the stored edits (the page's Simulate game)
    scripts/nba/backtest_game.py --slate 156831 --game MIA --set "bam adebayo=36, nikola jovic=0" [--start "pelle larsson"] [--bench NAME]
                                                [--spread -4.5] [--total 231]
                                                                        add edits and draw (minutes typed for a player; 0 takes him out; the home spread, the total)
    scripts/nba/backtest_game.py --slate 156831 --game MIA --actual     every player at the minutes he played, the five who started (Actual minutes)
    scripts/nba/backtest_game.py --slate 156831 --game MIA --reset      drop the edits and the what-if (Reset)
    scripts/nba/backtest_game.py --slate 156831 --game MIA --player "bam adebayo"
                                                                        his drawn distributions against what he did
"""
import argparse

from ivan.database.session import get_session, init_db
from ivan.sports.nba import slates as S
from ivan.sports.nba.backtest import game as G
from ivan.sports.nba.backtest import slate as SB


def f(x, spec=".1f", none="   —") -> str:
    return none if x is None else format(x, spec)


def show_games(dg: int, session) -> None:
    if G.slate_games(dg):
        rows = G.overview(session, dg)
        print(f"slate {dg}: {len(rows)} games (what was published at each game's tip against the box scores)")
    else:                                                   # no tip records: the projection at lock, from the slate's scorecard
        _, ss = S.open(dg)
        with S.using(dg):
            bt = SB.slate_backtest(ss, dg)
        ss.close()
        rows = G.overview(session, dg, bt["players"], bt["day"])
        print(f"slate {dg} {bt['day']}: {len(rows)} games (the projection at lock against the box scores; no tip records, so no game can be opened)")
    for g in rows:
        fin = g["final"]
        line = f"O/U {f(g['total'], 'g')} {g['home']} {f(g['spread_home'], '+g')}" if g["total"] is not None else "no line"
        res = f"final {g['away']} {fin['away']:.0f} {g['home']} {fin['home']:.0f}{' OT' if fin['overtime'] else ''}" if fin else "no box score"
        t = " · ".join(f"{x['team']} {x['fpts_proj']:.0f}->{f(x['fpts_act'], '.0f')}" for x in g["teams"])
        print(f"  {g['key']:8} {g['tip']:%H:%M}  {line:22} {res:30} fpts {t}  min MAE {f(g['min_mae'])}  fpts MAE {f(g['fpts_mae'])}  bias {f(g['fpts_bias'], '+.1f')}"
              + ("  drawn" if g["drawn"] else "") + (" + what-if" if g["what_if"] else ""))
        for m in g["misses"]:
            print(f"           {m['name']} {m['delta']:+.1f}")


def show_game(v: dict) -> None:
    g, fin = v["game"], v["final"]
    print(f"{g['away']} @ {g['home']} · {g['tip']:%a %m-%d %H:%M} · tip record {g['record']:%H:%M} · line {g['home']} {f(g['spread_home'], '+g')}, O/U {f(g['total'], 'g')}"
          + (f" · final {g['away']} {fin['away']:.0f}, {g['home']} {fin['home']:.0f}{' (OT)' if fin['overtime'] else ''}" if fin else " · no box score"))
    m = v["shown"]
    if m:
        w = m["game"]
        print(f"  {v['which']}: {m['n']} worlds, commit {m['commit']}{'*' if m.get('dirty') else ''}, rotation {m['rotation']} · drawn total {w['mean_total']:.1f}, home margin {w['mean_margin']:+.1f}, "
              f"{g['home']} wins {100 * w['p_home']:.0f}%, by 20+ {100 * w['blowout']:.0f}%, overtime {100 * w['p_ot']:.1f}%")
    if v["edits"]:
        print(f"  edits: {v['edits']}")
    for t in v["teams"]:
        tt = t["totals"]
        print(f"  {t['team']}  fpts published {tt['fpts_pub']:.0f} · drawn {tt['fpts_drawn']:.0f} · actual {tt['fpts_act']:.0f}   minutes MAE drawn {f(t['min_mae'])} (published {f(t['min_mae_pub'])})"
              f"   fpts MAE drawn {f(t['fpts_mae'])} (published {f(t['fpts_mae_pub'])})")
        print(f"    {'player':24} {'min pub':>7} {'set':>5} {'drawn':>6} {'act':>5}   {'fpts pub':>8} {'base':>6} {'what-if':>7} {'act':>6} {'pct':>4}   " + " ".join(f"{lab:>9}" for _, lab in G.STATS))
        for r in t["rows"]:
            stats = " ".join(f"{f(s['drawn'], '4.1f', '   —')}>{f(s['act'], '<4g', '—   ')}" for s in r["stats"])
            print(f"    {('*' if r['starter'] else ' ') + r['name'][:23]:24} {f(r['min_pub'], '7.1f', '      —')} {f(r['min_set'], '5g', '     ')} {f(r['min_drawn'], '6.1f', '     —')} {f(r['min_act'], '5.1f', '    —')}"
                  f"   {f(r['fpts_pub'], '8.1f', '       —')} {f(r['fpts_base'], '6.1f', '     —')} {f(r['fpts_what_if'], '7.1f', '       ')} {f(r['fpts_act'], '6.1f', '     —')}"
                  f" {f(None if r['pct'] is None else 100 * r['pct'], '4.0f', '    ')}   {stats}")
        if t["outside"]:
            print("    played, not in the pool at the tip: " + ", ".join(f"{o['name']} {o['minutes']:.0f} min" for o in t["outside"]))


def show_player(p: dict) -> None:
    print(f"{p['name']} ({p['team']} {p['pos']}, ${p['salary']}) · {'the what-if' if p['what_if'] else 'the base draw'} · {p['n']} worlds" if p["drawn"] else f"{p['name']}: no draw on file for him")
    for c in p["charts"]:
        ch = c["chart"]
        if not ch:
            continue
        print(f"  {c['label']:15} drawn mean {ch['mean']:6.1f}  10-50-90 {ch['p10']:>3} · {ch['p50']:>3} · {ch['p90']:>3}"
              + (f"  published {c['published']:.1f}" if c["published"] is not None else "") + (f"  actual {c['actual']:g} (percentile {100 * ch['pct']:.0f})" if c["actual"] is not None else "  no box score")
              + (f"  unedited mean {ch['other_mean']:.1f}" if ch["other_mean"] is not None else ""))
        top = max(b["a"] for b in ch["bins"]) or 1.0
        for b in ch["bins"]:
            print(f"    {b['label']:>7} {'#' * int(round(40 * b['a'] / top)):40} {100 * b['a']:5.1f}%{'  <- actual' if b['hit'] else ''}")


def pairs(text: str) -> dict:
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.rsplit("=", 1)
            out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate", type=int, required=True, help="the draft group")
    ap.add_argument("--game", help="AWAY-HOME, or one of its teams")
    ap.add_argument("--simulate", action="store_true"); ap.add_argument("--worlds", type=int, default=None)
    ap.add_argument("--set", default="", help='minutes typed: "name=min, name=min"'); ap.add_argument("--start", action="append", default=[]); ap.add_argument("--bench", action="append", default=[])
    ap.add_argument("--spread", type=float, default=None, help="the home team's spread"); ap.add_argument("--total", type=float, default=None)
    ap.add_argument("--actual", action="store_true"); ap.add_argument("--reset", action="store_true"); ap.add_argument("--player")
    a = ap.parse_args()
    session = get_session(init_db("nba"))
    if not a.game:
        show_games(a.slate, session); return 0
    g = G.find_game(a.slate, a.game)
    if g is None:
        print(f"slate {a.slate} has no game {a.game} on its tip records: " + ", ".join(x["key"] for x in G.slate_games(a.slate))); return 1
    key = g["key"]
    if a.reset:
        G.reset(a.slate, key)
    edits = G.load_edits(a.slate, key)
    changed = bool(a.set or a.start or a.bench or a.spread is not None or a.total is not None or a.actual)
    if a.actual:
        edits = dict(G.actual_edits(session, a.slate, key), **{k: edits[k] for k in ("spread_home", "total") if k in edits})
    if a.set:
        edits.setdefault("minutes", {}).update(pairs(a.set))
    for n in a.start:
        edits.setdefault("starters", {})[n] = True
    for n in a.bench:
        edits.setdefault("starters", {})[n] = False
    if a.spread is not None:
        edits["spread_home"] = a.spread
    if a.total is not None:
        edits["total"] = a.total
    if changed or a.simulate or G.load_draw(a.slate, key) is None:
        G.simulate(session, a.slate, key, n=a.worlds, edits=edits, log=print)
    if a.player:
        show_player(G.player_view(session, a.slate, key, a.player))
    else:
        show_game(G.view(session, a.slate, key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
