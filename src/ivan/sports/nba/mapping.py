"""
A team's mapping is void once a player it counted on is out.

The mapping (the painted grid with its locked rows and the minutes pins, projection.rotation.save_drawn) is made against a pool:
the players available tonight, recorded in the grid's fingerprint. The minutes model initialises that pool itself (the last
game's rotation minus tonight's outs, plus returners) and fills it well; what it cannot know is which rows the desk held by
hand for a pool that no longer exists. So when a player the mapping had available is ruled out (DK's status, or a scratch
call on the desk), the mapping is void: the grid and the team's positive minutes pins go, the model's own rotation for the new
pool is what the card shows, and the team reads "unmapped" until it is accepted (Hold all) or painted and held again. The
voided mapping is kept as a scenario assuming the player in ("Johnson in"), so a reversed call puts it straight back
(scenarios pick themselves when their assumptions hold).

    void_stale(session, local_overrides, log)   every team of the pool whose local grid has a new out: voids it; returns the reports
    stale_outs(team, statuses, overrides)       the newly out players of one team ([] when the mapping stands)

Runs inside every refresh (ingest.run_ingest, before the projection) and before the desk re-projects a team (Desk.reproject).
A consumer's published grid (the laptop's) is never voided here: the laptop voids its own and the pull brings the result.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional

from ivan.sports.nba.names import normalize_name
from ivan.sports.nba import scenarios as SC
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.live import OUT_STATUSES, override_of

Log = Callable[[str], None]
log_ = logging.getLogger(__name__)


def _out(status: Optional[str], ov: dict) -> bool:
    st = (status or "").upper()
    return st in OUT_STATUSES or (ov.get("minutes") is not None and float(ov["minutes"]) <= 0)


def stale_outs(team: str, statuses: Dict[str, str], overrides: dict) -> List[str]:
    """The players the team's local mapping had available who are out now (DK status or a scratch pin); [] when there is no local
    grid, no fingerprint, or nobody new is out. `statuses`: {normalized name: DK status} for the team's players."""
    g = ROT.load_drawn(team)
    if g is None or g.attrs.get("source") != "local":
        return []
    fp = g.attrs.get("fingerprint") or {}
    if not fp.get("available"):
        return []
    return sorted(n for n in fp["available"] if n in statuses and _out(statuses[n], override_of(overrides, n)))


def void_mapping(team: str, outs: Iterable[str], names: Iterable[str], local_overrides: dict, log: Log = print) -> dict:
    """Void the team's mapping for `outs`: keep it as a scenario assuming them in (their own rows left out of it), drop the grid and
    the team's positive minutes pins (a scratch pin, minutes 0, is the call itself and stays). Mutates `local_overrides`; the
    caller writes it and re-projects. Returns {team, out, scenario, pins}."""
    outs = sorted(set(outs)); names = set(names)
    short = lambda n: n.split()[-1].title()   # noqa: E731
    team_ov = {n: dict(ov) for n, ov in local_overrides.items() if n in names and n not in outs}
    label = " + ".join(short(n) for n in outs) + " in"
    slug = None
    try:
        sc = SC.save_scenario(team, label, team_ov, assumes={n: "in" for n in outs},
                              note=f"the mapping before {', '.join(short(n) for n in outs)} was ruled out; kept by the desk when it voided the grid")
        slug = sc["slug"]
        SC.set_active(team, None)                       # saved, not live: its assumption does not hold right now
    except Exception as e:   # noqa: BLE001 - the keepsake must not stop the voiding
        log(f"{team}: the old mapping could not be kept as a scenario: {type(e).__name__}: {e}")
    ROT.clear_drawn(team)
    pins = 0
    for n in list(local_overrides):
        if n in names and n not in outs and local_overrides[n].get("minutes") is not None and float(local_overrides[n]["minutes"]) > 0:
            rec = {k: v for k, v in local_overrides[n].items() if k != "minutes" and v is not None}
            if rec:
                local_overrides[n] = rec
            else:
                local_overrides.pop(n)
            pins += 1
    log(f"{team}: {', '.join(short(n) for n in outs)} now out: the mapping is void (grid cleared, {pins} minutes pin(s) lifted"
        + (f"; kept as scenario '{label}'" if slug else "") + "); the model's rotation stands until the team is mapped again")
    return dict(team=team, out=outs, scenario=slug, pins=pins, label=label)


def void_stale(session, local_overrides: dict, log: Log = print, teams: Optional[Iterable[str]] = None) -> List[dict]:
    """Every team in the pool (or `teams`) whose local mapping has a new out: void it (void_mapping). Mutates `local_overrides`;
    the caller writes it and re-projects the teams named in the reports."""
    from ivan.sports.nba.models import NBAPlayer
    rows = session.query(NBAPlayer.name, NBAPlayer.team, NBAPlayer.status).all()
    by_team: Dict[str, Dict[str, str]] = {}
    for name, team, status in rows:
        if team:
            by_team.setdefault(team, {})[normalize_name(name)] = status or ""
    out = []
    for team in sorted(teams) if teams is not None else sorted(by_team):
        statuses = by_team.get(team) or {}
        outs = stale_outs(team, statuses, local_overrides)
        if outs:
            out.append(void_mapping(team, outs, statuses.keys(), local_overrides, log))
    return out
