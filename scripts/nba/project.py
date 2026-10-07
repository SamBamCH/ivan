"""
Project the current NBA player pool from box score history: minutes first,
then per-minute rates, then fantasy points by Monte Carlo. Writes fpts_mean,
fpts_std and minutes_proj on nba_player.

Availability  DK status OUT -> not playing. Everyone else on the team plays.
Starters      starting_lineups rows (Rotowire, fetched by ingest.py) mark the
              confirmed five; a team with no rows gets starters=unknown.
Late news     --overrides data/nba/input/minutes_overrides.csv (default when
              the file exists), columns: player,minutes. A pinned player keeps
              those minutes exactly; put 0 for a late scratch. The rest of his
              team is re-allocated around the pins. Names are matched after
              normalisation, so "Giannis Antetokounmpo" is fine.

  python scripts/nba/project.py                 # project and write
  python scripts/nba/project.py --dry-run       # print only
  python scripts/nba/project.py --team MIL      # one team
  python scripts/nba/project.py --out "Giannis Antetokounmpo"   # ad-hoc scratch
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from ivan.core.utils import normalize_name
from ivan.database.session import init_db, get_session, add_missing_columns
from ivan.sports.nba.models import NBAPlayer, StartingLineups, NBABase
from ivan.sports.nba.paths import MINUTES_OVERRIDES_CSV
from ivan.sports.nba.projection.history import load_box_scores, build_history
from ivan.sports.nba.projection.engine import project_team, RosterPlayer, ProjectionParams

OUT_STATUSES = {"OUT", "O", "IR", "SUSP"}


def read_overrides(path: Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    if "player" not in cols or "minutes" not in cols:
        raise SystemExit(f"{path}: need columns player,minutes (got {list(df.columns)})")
    return {normalize_name(str(r[cols["player"]])): float(r[cols["minutes"]]) for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", help="only this team (DK abbreviation)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--overrides", default=None, help=f"minutes CSV (default {MINUTES_OVERRIDES_CSV} if present)")
    ap.add_argument("--out", action="append", default=[], help="treat this player as OUT (repeatable)")
    ap.add_argument("--no-starters", action="store_true", help="ignore starting_lineups")
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()

    console = Console()
    engine = init_db("nba")
    add_missing_columns(engine, NBABase.metadata)
    session = get_session(engine)

    overrides = read_overrides(Path(args.overrides) if args.overrides else MINUTES_OVERRIDES_CSV)
    forced_out = {normalize_name(n) for n in args.out}

    box = load_box_scores(session)
    hist = build_history(box)
    today = datetime.now().date()
    state = hist.before(today + pd.Timedelta(days=1))   # everything we have
    team_fpts = hist.team_before(today + pd.Timedelta(days=1))

    starters: dict[str, set[str]] = {}
    if not args.no_starters:
        for r in session.query(StartingLineups).all():
            if getattr(r, "confirmed", 1):
                starters.setdefault(r.team, set()).add(normalize_name(r.player_name))

    players = session.query(NBAPlayer).all()
    if args.team:
        players = [p for p in players if p.team == args.team]
    if not players:
        raise SystemExit("no players in nba_player; run ingest.py -p first")

    params = ProjectionParams(n_sims=args.sims)
    rng = np.random.default_rng(0)
    written = 0
    unmatched_overrides = set(overrides)
    for team in sorted({p.team for p in players}):
        tp = [p for p in players if p.team == team]
        team_starters = starters.get(team)
        roster = []
        for p in tp:
            name = normalize_name(p.name)
            status = (p.status or "").upper()
            out = status in OUT_STATUSES or name in forced_out
            pin = overrides.get(name)
            if pin is not None:
                unmatched_overrides.discard(name)
                if pin <= 0:
                    out = True
            starter = None if team_starters is None else (name in team_starters)
            roster.append(RosterPlayer(name, not out, starter, p.salary, None if out else pin))
        proj = project_team(roster, state, hist, team_fpts.get(team), params, rng=rng)

        tbl = Table(title=f"{team}  (team fpts/game {team_fpts.get(team, hist.league_team_fpts):.0f}"
                          f"{'' if team_starters else ', starters unknown'})", show_lines=False)
        for col in ["player", "sal", "st", "hist min", "min", "fpts", "sd", "val", "note"]:
            tbl.add_column(col, justify="right" if col not in ("player", "note") else "left")
        by_name = {normalize_name(p.name): p for p in tp}
        for name, r in proj.sort_values("minutes", ascending=False).iterrows():
            p = by_name[name]
            note = []
            if r["pinned"]: note.append("pinned")
            if r["games"] == 0: note.append("no history")
            elif r["games"] < 5: note.append(f"{int(r['games'])} games")
            if abs(r["usage_mult"] - 1) > 0.02: note.append(f"usage x{r['usage_mult']:.2f}")
            st = "" if team_starters is None else ("S" if name in team_starters else "")
            tbl.add_row(p.name, str(p.salary), st, f"{r['baseline']:.1f}", f"{r['minutes']:.1f}",
                        f"{r['fpts_mean']:.1f}", f"{r['fpts_std']:.1f}",
                        f"{1000 * r['fpts_mean'] / max(p.salary, 1):.2f}", ", ".join(note))
            if not args.dry_run:
                p.fpts_mean = float(r["fpts_mean"]); p.fpts_std = float(r["fpts_std"])
                p.minutes_proj = float(r["minutes"]); p.fpts_updated_at = datetime.now()
                written += 1
        outs = [p.name for p in tp if normalize_name(p.name) not in proj.index]
        if outs:
            tbl.caption = "out: " + ", ".join(outs)
        console.print(tbl)
        if not args.dry_run:
            # players ruled out carry no projection
            for p in tp:
                if normalize_name(p.name) not in proj.index:
                    p.fpts_mean = None; p.fpts_std = None; p.minutes_proj = 0.0

    if unmatched_overrides:
        console.print(f"[yellow]overrides not on the slate: {sorted(unmatched_overrides)}[/yellow]")
    if args.dry_run:
        console.print("[dim]dry run, nothing written[/dim]")
    else:
        session.commit()
        console.print(f"[green]wrote projections for {written} players[/green]")


if __name__ == "__main__":
    main()
