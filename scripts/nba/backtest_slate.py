#!/usr/bin/env python3
"""
scripts/nba/backtest_slate.py

A past slate, read only (ivan.sports.nba.slate_backtest): what the desk's backtest view shows. Our entries' results
(rescored from the box scores), the chooser's claim beside them, each contest, and the players with the biggest misses
against our projection at lock. Nothing is loaded into the desk or deleted.

    scripts/nba/backtest_slate.py --draft-group 155794
    scripts/nba/backtest_slate.py --date 2026-10-28          every slate of that day with entries of ours
    scripts/nba/backtest_slate.py --draft-group 155794 --players 20
"""
import argparse
from datetime import date

from sqlalchemy import text

from ivan.database.session import get_session, init_db
from ivan.sports.nba import slate_backtest as SB
from ivan.sports.nba.ingest import ensure_schema


def show(b: dict, n_players: int) -> None:
    sm = b["summary"]
    roi = "—" if sm["roi"] is None else f"{100 * sm['roi']:+.0f}%"
    claim = "—" if sm["claimed_roi"] is None else f"{100 * sm['claimed_roi']:+.0f}%"
    print(f"slate {b['draft_group']} {b['day']} lock {b['lock']:%H:%M}" if b["lock"] else f"slate {b['draft_group']} {b['day']}")
    print(f"  net ${sm['net']:.2f} on ${sm['fees']:.2f} in fees ({roi}); the sim claimed {claim}; {sm['cashed']}/{sm['entries']} entries cashed"
          + ("" if sm["final"] else "  [no box scores yet: DK's export points]"))
    for r in b["contests"]:
        c = r.get("claim") or {}
        print(f"  {r['name'][:40]:40} ${r['entry_fee']:<6g} x{r.get('found', 0):<3} field {r['field']:<6} best {r['best_rank'] or '-':<6} cashed {r['cashed']:<3}"
              f" net ${r['net']:8.2f}  claim {'' if c.get('roi') is None else f'{100 * c['roi']:+.0f}%'}")
    rows = [dict(p, team=t) for t, ps in b["players"].items() for p in ps if p["fpts_act"] is not None and p["fpts_proj"] is not None]
    rows.sort(key=lambda p: -abs(p["fpts_act"] - p["fpts_proj"]))
    if rows:
        print(f"  biggest misses (fpts scored − projected):")
        for p in rows[:n_players]:
            print(f"    {p['team']:3} {p['name']:26} min {p['min_proj'] or 0:4.0f}->{p['min_act'] or 0:4.0f}  fpts {p['fpts_proj']:5.1f}->{p['fpts_act']:5.1f}"
                  f"  {p['fpts_act'] - p['fpts_proj']:+6.1f}" + (f"  ours {100 * p['exposure']:.0f}%" if p.get("exposure") else "") + (f"  note: {p['note']}" if p.get("note") else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft-group", type=int); ap.add_argument("--date"); ap.add_argument("--players", type=int, default=10)
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    if a.draft_group:
        dgs = [a.draft_group]
    elif a.date:
        d = date.fromisoformat(a.date)
        dgs = [int(x) for (x,) in session.execute(text("select distinct draft_group_id from historical_contest_information where date(date) = :d").bindparams(d=d.isoformat())).all()]
    else:
        ap.error("--draft-group or --date")
    for dg in dgs:
        b = SB.slate_backtest(session, dg)
        if a.date and not b["contests"]:
            continue
        show(b, a.players)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
