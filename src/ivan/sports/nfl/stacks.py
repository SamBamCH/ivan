# src/ivan/sports/nfl/stacks.py
"""
Describe a lineup's construction in DFS terms: team split, captain side,
QB stacks with their bring-backs, RB/DST pairs. Labels follow
the DFS convention ("4-2", "QB+2 with opp 1"); the web UI's Exposures page uses them.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

PASS_CATCHERS = {"WR", "TE"}
STACK_MATES = {"WR", "TE", "RB"}


def short(name: str | None) -> str:
    """Last word of a player name ("jaxon smithnjigba" -> "smithnjigba"; "patriots" stays)."""
    return (name or "").split()[-1] if name else "?"


def describe_lineup(players: Iterable[dict]) -> dict:
    """
    players: dicts with name, team, opponent, position, slot (CPT/FLEX/QB/RB/WR/TE/DST).
    Returns {"teams": "SEA 4 · NE 2", "stacks": ["4-2, CPT on the majority side", "QB darnold +2 ...", ...]}.
    """
    players = [p for p in players if p]
    teams = Counter(p.get("team") for p in players if p.get("team"))
    teams_text = " · ".join(f"{t} {n}" for t, n in teams.most_common())
    stacks: list[str] = []

    # showdown: team split and which side the captain is on
    cpt = next((p for p in players if p.get("slot") == "CPT"), None)
    if cpt is not None and teams:
        counts = sorted(teams.values(), reverse=True)
        split = "-".join(str(c) for c in counts) if len(counts) > 1 else f"{counts[0]}-0"
        top = counts[0]
        cpt_n = teams.get(cpt.get("team"), 0)
        if len(counts) > 1 and counts[0] == counts[1]:
            side = "even split"
        elif cpt_n == top:
            side = "CPT on the majority side"
        else:
            side = "CPT on the minority side"
        stacks.append(f"{split}, {side}")

    # QB stacks: same-team pass catchers (RB noted separately) and bring-backs from the opponent
    qbs = [p for p in players if (p.get("position") or "").upper() == "QB"]
    for qb in qbs:
        team, opp = qb.get("team"), qb.get("opponent")
        mates = [p for p in players if p is not qb and p.get("team") == team and (p.get("position") or "").upper() in STACK_MATES]
        catchers = [p for p in mates if (p.get("position") or "").upper() in PASS_CATCHERS]
        backs = [p for p in mates if (p.get("position") or "").upper() == "RB"]
        opp_players = [p for p in players if opp and p.get("team") == opp and (p.get("position") or "").upper() in STACK_MATES]
        text = f"QB {short(qb.get('name'))} +{len(catchers)}"
        if catchers:
            text += " (" + ", ".join(short(p.get("name")) for p in catchers) + ")"
        if backs:
            text += " + RB " + ", ".join(short(p.get("name")) for p in backs)
        text += f" with opp {len(opp_players)}"
        if opp_players:
            text += " (" + ", ".join(short(p.get("name")) for p in opp_players) + ")"
        stacks.append(text)
    if not qbs:
        stacks.append("no QB")
    if len(qbs) > 1:
        stacks.append(f"{len(qbs)} QBs")

    # RB and DST from the same team (game-script stack); a DST facing the lineup's own QB is a conflict
    dsts = [p for p in players if (p.get("position") or "").upper() == "DST"]
    for d in dsts:
        rbs = [p for p in players if (p.get("position") or "").upper() == "RB" and p.get("team") == d.get("team")]
        if rbs:
            stacks.append("RB/DST " + ", ".join(short(p.get("name")) for p in rbs) + f" + {short(d.get('name'))}")
        for qb in qbs:
            if qb.get("team") == d.get("opponent"):
                stacks.append(f"DST {short(d.get('name'))} vs own QB {short(qb.get('name'))}")

    return {"teams": teams_text, "stacks": stacks}
