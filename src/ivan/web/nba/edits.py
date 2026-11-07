"""
Edits typed on a team page applied to the Desk: one row's pins, a grid's typed values, the values a page sends along before a run.
"""
from __future__ import annotations

import json
from typing import Optional
from ivan.core.utils import normalize_name
from ivan.sports.nba.projection.live import STAT_KEYS, override_of
from ivan.sports.nba.projection import rotation as ROT
from .core import desk


def _num(raw: str, label: str) -> Optional[float]:
    raw = (raw or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        desk.flash.append(f"'{raw}' is not a number for {label}")
        return None


def _apply_pin_rows(team: str, rows: list) -> tuple[list, int, int]:
    """Apply typed grid values for one team ([{player, minutes, fppm, fpts, pub, points, rebounds, assists, fg3m, steals, blocks,
    turnovers}], blank clears that pin) to this machine's own
    layer. A row whose minutes changed loses its lock so the skeleton can move it. Returns (players whose minutes changed, public
    numbers changed, values set). The caller re-projects."""
    changed_min, changed_pub, n_set = [], 0, 0
    for r in rows:
        key = normalize_name(str(r.get("player") or ""))
        if not key:
            continue
        old = override_of(desk.overrides, key)
        vals = {}
        for k in ("minutes", "fppm", "fpts", "pub", *STAT_KEYS):
            if k in r:
                try:
                    vals[k] = _num(r.get(k), k)
                except ValueError as e:
                    desk.flash.append(str(e)); continue
        if not vals:
            continue
        if "minutes" in vals and vals["minutes"] != old.get("minutes"):
            changed_min.append(key)
        if "pub" in vals and vals["pub"] != old.get("pub"):
            changed_pub += 1
        if any(vals.get(k) != old.get(k) for k in vals):
            n_set += 1
        desk.set_override(key, **vals)
    if changed_min:
        g = ROT.load_drawn(team)
        if g is not None:
            kept = [n for n in g.attrs.get("locked", []) if n not in changed_min]
            if len(kept) != len(g.attrs.get("locked", [])):
                ROT.save_drawn(team, g, locked=kept, fingerprint=g.attrs.get("fingerprint"))
    return changed_min, changed_pub, n_set


def _apply_typed_pins(pins: str) -> int:
    """
    Values typed into the grids but not yet refigured, sent along with Publish / Generate / Sim as JSON {team: [rows]} (the page
    collects every dirty input): applied first, so a number typed and then published is what gets published (2026-10-27: an
    fpts typed on the consumer's desk vanished on Publish because the page reload threw the unsent input away). Returns the
    number of values set.
    """
    if not pins:
        return 0
    try:
        by_team = json.loads(pins)
    except ValueError:
        desk.flash.append("typed values not applied: the page sent them in a form the desk could not read")
        return 0
    total = 0
    for team, rows in (by_team or {}).items():
        _, _, n = _apply_pin_rows(str(team), rows or [])
        total += n
    return total
