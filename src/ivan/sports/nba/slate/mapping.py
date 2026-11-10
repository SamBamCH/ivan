"""
A team's mapping is void once a player it counted on is out, or the five starters it was drawn for change.

The mapping (the painted grid with its locked rows and the minutes pins, projection.rotation.save_drawn) is made against a pool:
the players available tonight and the five starters, recorded in the grid's fingerprint. The minutes model initialises that
pool itself (the last game's rotation minus tonight's outs, plus returners, the starters opening) and fills it well; what it
cannot know is which rows the desk held by hand for a pool that no longer exists. So when a player the mapping had available
is ruled out (DK's status, or a scratch call on the desk), or the lineup comes in differently from the five it was drawn for
(Rotowire's lineup with the hand starter calls; only a full five against a full five counts, a lineup not posted yet is no
change), the mapping is void: the grid and the team's positive minutes pins go, the model's own rotation for the new pool and
the new five is what the card shows, and the team reads "unmapped" until it is accepted (Hold all) or painted and held again.
A lineup that comes in as mapped changes nothing. The voided mapping is kept as a scenario assuming what it assumed (the
player in: "Johnson in"; the old starter starting: "Hauser starting"), so a reversed call or lineup puts it straight back
(scenarios pick themselves when their assumptions hold).

    void_stale(session, local_overrides, log)        every team of the pool whose local grid has a new out or a new five: voids it; the reports
    stale_outs(team, statuses, overrides)            the newly out players of one team ([] when the mapping stands)
    stale_starters(team, starters, overrides)        the lineup change of one team ({old, new, benched, promoted}, or None)
    void_mapping(team, outs, names, overrides, log, lineup)   void one team's mapping for the outs and/or the lineup change

Runs inside every refresh (ingest.run_ingest, before the projection) and before the desk re-projects a team (Desk.reproject).
A consumer's published grid (the laptop's) is never voided here: the laptop voids its own and the pull brings the result.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional

from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.slate import scenarios as SC
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


def stale_starters(team: str, starters: Optional[Iterable[str]], overrides: dict) -> Optional[dict]:
    """The lineup change that voids the team's local mapping: {old, new, benched, promoted} when the grid's fingerprint recorded five
    starters and the five now (`starters`: Rotowire's lineup with the hand calls, Desk.starters_of) are a full five that differs;
    None when there is no local grid or fingerprint, fewer than five are known on either side (a lineup not posted yet is no
    change), or the five came in as mapped."""
    g = ROT.load_drawn(team)
    if g is None or g.attrs.get("source") != "local":
        return None
    fp = g.attrs.get("fingerprint") or {}
    old = {normalize_name(n) for n in (fp.get("starters") or [])}
    new = {normalize_name(n) for n in (starters or [])}
    if len(old) < 5 or len(new) < 5 or old == new:
        return None
    return dict(old=sorted(old), new=sorted(new), benched=sorted(old - new), promoted=sorted(new - old))


def change_label(outs: Iterable[str], lineup: Optional[dict]) -> str:
    """What voided the mapping, as the scenario's name: "Johnson in", "Hauser starting", "Johnson in · Hauser starting"."""
    short = lambda n: n.split()[-1].title()   # noqa: E731
    parts = []
    outs = sorted(set(outs))
    if outs:
        parts.append(" + ".join(short(n) for n in outs) + " in")
    if lineup:
        benched = [n for n in lineup["benched"] if n not in outs]        # a starter ruled out is covered by "in"
        if benched:
            parts.append(" + ".join(short(n) for n in benched) + " starting")
    return " · ".join(parts) or "before the change"


def void_mapping(team: str, outs: Iterable[str], names: Iterable[str], local_overrides: dict, log: Log = print, lineup: Optional[dict] = None) -> dict:
    """Void the team's mapping for `outs` and/or the `lineup` change (stale_starters): keep it as a scenario assuming what it assumed (the
    outs in, their own rows left out of it; the benched starters starting, the promoted ones on the bench), drop the grid and the
    team's positive minutes pins (a scratch pin, minutes 0, is the call itself and stays). Mutates `local_overrides`; the caller
    writes it and re-projects. Returns {team, out, lineup, scenario, pins, label}."""
    outs = sorted(set(outs)); names = set(names)
    short = lambda n: n.split()[-1].title()   # noqa: E731
    team_ov = {n: dict(ov) for n, ov in local_overrides.items() if n in names and n not in outs}
    label = change_label(outs, lineup)
    assumes = {n: "in" for n in outs}
    if lineup:
        assumes.update({n: "starts" for n in lineup["benched"] if n not in outs})
        assumes.update({n: "bench" for n in lineup["promoted"]})
    what = []
    if outs:
        what.append(f"{', '.join(short(n) for n in outs)} was ruled out")
    if lineup:
        what.append("the lineup came in differently (" + ", ".join([f"-{short(n)}" for n in lineup["benched"]] + [f"+{short(n)}" for n in lineup["promoted"]]) + ")")
    slug = None
    try:
        sc = SC.save_scenario(team, label, team_ov, assumes=assumes,
                              note=f"the mapping before {' and '.join(what)}; kept by the desk when it voided the grid")
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
    log(f"{team}: {' and '.join(what)}: the mapping is void (grid cleared, {pins} minutes pin(s) lifted"
        + (f"; kept as scenario '{label}'" if slug else "") + "); the model's rotation stands until the team is mapped again")
    return dict(team=team, out=outs, lineup=lineup, scenario=slug, pins=pins, label=label, why=" and ".join(what))


def live_starters(session, team: str, names: Iterable[str], overrides: dict) -> Optional[set]:
    """The team's five now: Rotowire's lineup (starting_lineups) with the hand starter calls applied, as Desk.starters_of reads them;
    None when there is no lineup and no call."""
    from ivan.sports.nba.models import StartingLineups
    base = {normalize_name(r.player_name) for r in session.query(StartingLineups).filter(StartingLineups.team == team).all()}
    calls = {n: override_of(overrides, n).get("starter") for n in names}
    calls = {n: v for n, v in calls.items() if v is not None}
    if not base and not calls:
        return None
    out = set(base)
    for n, v in calls.items():
        (out.add if float(v) > 0 else out.discard)(n)
    return out


def void_stale(session, local_overrides: dict, log: Log = print, teams: Optional[Iterable[str]] = None) -> List[dict]:
    """Every team in the pool (or `teams`) whose local mapping has a new out or whose five changed: void it (void_mapping). Mutates
    `local_overrides`; the caller writes it and re-projects the teams named in the reports."""
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
        lineup = stale_starters(team, live_starters(session, team, statuses.keys(), local_overrides), local_overrides)
        if outs or lineup:
            out.append(void_mapping(team, outs, statuses.keys(), local_overrides, log, lineup=lineup))
    return out
