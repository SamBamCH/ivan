#!/usr/bin/env python3
"""
scripts/nba/mapping.py — the team mappings a player now out or a changed five makes void (ivan.sports.nba.slate.mapping): what every
refresh and every re-projection on the desk does on their own; from a terminal to see or force it.

    venv/bin/python scripts/nba/mapping.py --slate DG            # list the stale mappings (team: who is out) and void them
    venv/bin/python scripts/nba/mapping.py --slate DG --dry-run  # list only
"""
from __future__ import annotations

import argparse

from ivan.sports.nba.slate import mapping as MAP
from ivan.sports.nba import slates as S
from ivan.sports.nba.ingest import void_stale_mappings
from ivan.sports.nba.models import NBAPlayer
from ivan.sports.nba.names import normalize_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dg, session = S.open(a.slate)
    if a.dry_run:
        from ivan.sports.nba.projection.live import read_overrides_full
        local = read_overrides_full(layered=False) if S.overrides_csv().exists() else {}
        by_team: dict = {}
        for name, team, status in session.query(NBAPlayer.name, NBAPlayer.team, NBAPlayer.status).all():
            by_team.setdefault(team, {})[normalize_name(name)] = status or ""
        stale = {}
        for t, st in sorted(by_team.items()):
            if not t:
                continue
            outs = MAP.stale_outs(t, st, local)
            lineup = MAP.stale_starters(t, MAP.live_starters(session, t, st.keys(), local), local)
            if outs or lineup:
                stale[t] = (outs, lineup)
                print(f"{t}: " + (f"{', '.join(outs)} out since it was mapped" if outs else "")
                      + (" and " if outs and lineup else "") + (f"the five changed (-{', '.join(lineup['benched'])} +{', '.join(lineup['promoted'])})" if lineup else "") + ": void")
        print(f"{len(stale)} stale mapping(s) on slate {dg}")
        return 0
    voided = void_stale_mappings(session, log=print)
    if voided:
        from ivan.sports.nba.ingest import project_team
        from ivan.sports.nba.projection.live import LiveSlate, read_overrides_full
        from ivan.sports.nba.projection.engine import ProjectionParams
        slate = LiveSlate(session); ov = read_overrides_full()
        for v in voided:
            project_team(session, slate, v["team"], ov, ProjectionParams.from_config())
        session.commit()
    print(f"{len(voided)} mapping(s) voided on slate {dg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
