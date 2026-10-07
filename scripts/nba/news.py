"""
Late news in one command: name the player who is out, see what the model
does with his team, pin any minutes you know better, write the overrides and
the projections.

  python scripts/nba/news.py "Giannis Antetokounmpo"
  python scripts/nba/news.py "Giannis Antetokounmpo" --accept       # no prompts
  python scripts/nba/news.py "Giannis Antetokounmpo" --conditional  # use "without him" history for rates

For each teammate the table shows the model's minutes without him, his
history, and, when the team has played without him before, how his minutes
and production per minute went in those games. You are then asked for
minutes player by player; Enter keeps the model's number, a number pins it,
0 scratches him. The remaining team minutes re-allocate around what you pin.

Rates stay forward looking: a player priced by the books keeps his market
per-minute rate (set when the props were fetched), everyone else keeps his
history rate times the team usage bump. --conditional instead scales each
teammate's rate toward what he did per minute in the games without the
absent player, weighted by how many such games exist. Nothing here guesses a
new role from the past; the minutes you type are the forward-looking input.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from ivan.core.utils import normalize_name
from ivan.database.session import init_db, get_session, add_missing_columns
from ivan.sports.nba.models import NBABase
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.history import without_player
from ivan.sports.nba.projection.live import LiveSlate, read_overrides, write_overrides

COND_PRIOR_GAMES = 3.0   # games without him before the conditional rate carries half the weight


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("player", nargs="+", help="player(s) ruled out")
    ap.add_argument("--accept", action="store_true", help="take the model's minutes without prompting")
    ap.add_argument("--conditional", action="store_true", help="scale teammates' rates toward their 'without him' history")
    ap.add_argument("--keep-overrides", action="store_true", help="keep existing pins in minutes_overrides.csv for other teams")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    console = Console()
    engine = init_db("nba")
    add_missing_columns(engine, NBABase.metadata)
    session = get_session(engine)
    slate = LiveSlate(session)
    by_name = {normalize_name(p.name): p for p in slate.players}
    outs = {normalize_name(n) for n in args.player}
    unknown = [n for n in outs if n not in by_name]
    if unknown:
        raise SystemExit(f"not on the slate: {unknown}. Slate names: {sorted(by_name)[:10]}...")

    overrides = read_overrides() if args.keep_overrides else {}
    for n in outs:
        overrides[n] = 0.0
    teams = sorted({by_name[n].team for n in outs})
    today = datetime.now().date()
    rng = np.random.default_rng(0)

    for team in teams:
        absent = [n for n in outs if by_name[n].team == team]
        before = slate.project(team, {k: v for k, v in overrides.items() if k not in outs}, set(), rng=rng)
        cond = pd.concat([without_player(slate.box, team, a, before=today + pd.Timedelta(days=1)) for a in absent]) if absent else pd.DataFrame()
        if len(absent) > 1 and len(cond):
            cond = cond.groupby(level=0).agg({"g_with": "min", "min_with": "mean", "fpm_with": "mean",
                                              "g_without": "min", "min_without": "mean", "fpm_without": "mean"})
        rate_mults = {}
        if args.conditional and len(cond):
            for name, r in cond.iterrows():
                if r["g_with"] >= 3 and r["fpm_with"] > 0:
                    w = r["g_without"] / (r["g_without"] + COND_PRIOR_GAMES)
                    rate_mults[name] = float(1 + w * (r["fpm_without"] / r["fpm_with"] - 1))
        after = slate.project(team, overrides, set(), rate_mults=rate_mults, rng=rng)

        tbl = Table(title=f"{team} without {', '.join(by_name[a].name for a in absent)}")
        for col in ["player", "sal", "was", "now", "hist", "w/o: g", "w/o: min", "w/o: fpm", "fpm now", "fpts", "src"]:
            tbl.add_column(col, justify="left" if col in ("player", "src") else "right")
        order = after.sort_values("minutes", ascending=False).index
        for name in order:
            r = after.loc[name]; b = before.loc[name] if name in before.index else None
            c = cond.loc[name] if name in cond.index else None
            tbl.add_row(by_name[name].name, str(by_name[name].salary),
                        "" if b is None else f"{b['minutes']:.1f}", f"{r['minutes']:.1f}", f"{r['baseline']:.1f}",
                        "" if c is None else f"{int(c['g_without'])}",
                        "" if c is None else f"{c['min_without']:.1f}",
                        "" if c is None or not np.isfinite(c['fpm_with']) else f"{c['fpm_without']:.2f} vs {c['fpm_with']:.2f}",
                        f"{r['fpm']:.2f}", f"{r['fpts_mean']:.1f}",
                        ("mkt" if r.get("priced") else "hist") + (f" x{rate_mults[name]:.2f}" if name in rate_mults else ""))
        console.print(tbl)

        if not args.accept and sys.stdin.isatty():
            console.print("[cyan]Minutes per player: Enter keeps the model, a number pins, 0 scratches, q stops asking.[/cyan]")
            for name in order:
                ans = console.input(f"  {by_name[name].name:28s} [{after.loc[name, 'minutes']:.1f}] ").strip()
                if ans.lower() == "q":
                    break
                if ans:
                    try:
                        overrides[name] = float(ans)
                    except ValueError:
                        console.print("[red]  not a number, skipped[/red]")
            after = slate.project(team, overrides, set(), rate_mults=rate_mults, rng=rng)
            console.print("[bold]after your pins:[/bold]")
            for name, r in after.sort_values("minutes", ascending=False).iterrows():
                flag = " (pinned)" if r["pinned"] else ""
                console.print(f"  {by_name[name].name:28s} {r['minutes']:5.1f} min  {r['fpts_mean']:5.1f} fpts{flag}")

        if not args.dry_run:
            slate.write(team, after)

    if args.dry_run:
        console.print("[dim]dry run, nothing written[/dim]")
        return
    session.commit()
    path = write_overrides(overrides)
    console.print(f"[green]projections written for {', '.join(teams)}; pins saved to {path}[/green]")
    console.print("[dim]next: python scripts/nba/sim.py --fresh (regenerate lineups with the new numbers)[/dim]")


if __name__ == "__main__":
    main()
