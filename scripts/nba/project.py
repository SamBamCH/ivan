"""
Project the current NBA player pool: minutes first, then per-minute rates
(market rates where the props price a player, box score history elsewhere),
then fantasy points by Monte Carlo. Writes fpts_mean, fpts_std and
minutes_proj on nba_player.

Availability  DK status OUT -> not playing. Everyone else on the team plays.
Starters      starting_lineups rows (Rotowire, fetched by ingest.py) mark the
              confirmed five; a team with no rows gets starters=unknown.
Late news     the day's edits/minutes_overrides.csv (columns player, minutes
              (0 = scratch), fppm, min_low, min_high, starter, fpts, pub and a
              total per stat: points, rebounds, assists, fg3m, steals, blocks,
              turnovers; the desk's pin boxes write the same file) pins minutes
              and stats; the rest of the team is re-allocated around the pins.
              scripts/nba/news.py builds a scratch for you.

  python scripts/nba/project.py                 # project and write
  python scripts/nba/project.py --dry-run       # print only
  python scripts/nba/project.py --team MIL      # one team
  python scripts/nba/project.py --out "Giannis Antetokounmpo"   # ad-hoc scratch
  python scripts/nba/project.py --no-market     # history rates only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, read_overrides
from ivan.sports.nba import slates as S


def team_table(slate: LiveSlate, team: str, proj, console: Console, title_extra: str = ""):
    tp = {normalize_name(p.name): p for p in slate.players if p.team == team}
    team_starters = slate.starters.get(team)
    tbl = Table(title=f"{team}  (team fpts/game {slate.team_fpts.get(team, slate.hist.league_team_fpts):.0f}"
                      f"{'' if team_starters else ', starters unknown'}){title_extra}")
    for col in ["player", "sal", "st", "hist", "min", "fpts", "sd", "val", "src", "note"]:
        tbl.add_column(col, justify="left" if col in ("player", "note", "src") else "right")
    for name, r in proj.sort_values("minutes", ascending=False).iterrows():
        p = tp[name]
        note = []
        if r["pinned"]: note.append("pinned")
        if r["games"] == 0: note.append("no history")
        elif r["games"] < 5: note.append(f"{int(r['games'])}g")
        if abs(r["usage_mult"] - 1) > 0.02: note.append(f"usage x{r['usage_mult']:.2f}")
        st = "" if team_starters is None else ("S" if name in team_starters else "")
        tbl.add_row(p.name, str(p.salary), st, f"{r['baseline']:.1f}", f"{r['minutes']:.1f}",
                    f"{r['fpts_mean']:.1f}", f"{r['fpts_std']:.1f}", f"{1000 * r['fpts_mean'] / max(p.salary, 1):.2f}",
                    "mkt" if r.get("priced") else "hist", ", ".join(note))
    outs = [p.name for n, p in tp.items() if n not in proj.index]
    if outs:
        tbl.caption = "out: " + ", ".join(outs)
    console.print(tbl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--team", help="only this team (DK abbreviation)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--overrides", default=None, help="minutes CSV (default data/nba/input/minutes_overrides.csv if present)")
    ap.add_argument("--out", action="append", default=[], help="treat this player as OUT (repeatable)")
    ap.add_argument("--no-starters", action="store_true", help="ignore starting_lineups")
    ap.add_argument("--no-market", action="store_true", help="ignore stored market rates")
    ap.add_argument("--sims", type=int, default=None, help="Monte Carlo draws (default config/nba.json projection.n_sims, the number every path projects at)")
    args = ap.parse_args()

    console = Console()
    dg, session = S.open(args.slate)
    slate = LiveSlate(session, use_starters=not args.no_starters, use_market=not args.no_market)
    if not slate.players:
        raise SystemExit("no players in nba_player; run ingest.py -p first")
    from ivan.sports.nba.ingest import project_team
    from ivan.sports.nba.projection.live import read_overrides_full
    overrides = read_overrides(Path(args.overrides)) if args.overrides else read_overrides_full()
    forced_out = {normalize_name(n) for n in args.out}
    params = ProjectionParams.from_config() if args.sims is None else ProjectionParams.from_config(n_sims=args.sims)
    rng = np.random.default_rng(0)
    adhoc = bool(forced_out or args.dry_run or args.overrides or args.no_starters or args.no_market)   # a look, not the projection

    written = 0
    teams = [args.team] if args.team else slate.teams()
    for team in teams:
        if adhoc:
            proj = slate.project(team, overrides, forced_out, params, rng=rng)
            team_table(slate, team, proj, console)
            if not args.dry_run:
                written += slate.write(team, proj)
        else:
            # the desk's own path (ingest.project_team): the team's pool rows and its projection store record, so the desk shows it
            rec = project_team(session, slate, team, overrides, params)
            team_table(slate, team, rec["frame"], console)
            written += int(len(rec["frame"]))
    pool = {normalize_name(p.name) for p in slate.players}
    missing = sorted(set(overrides) - pool)
    if missing:
        console.print(f"[yellow]overrides not on the slate: {missing}[/yellow]")
    if args.dry_run:
        console.print("[dim]dry run, nothing written[/dim]")
    else:
        session.commit()
        console.print(f"[green]wrote projections for {written} players"
                      f"{' (market rates for ' + str(len(slate.market)) + ')' if slate.market else ''}[/green]")


if __name__ == "__main__":
    main()
