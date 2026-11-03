"""
A team's scenarios: its edits saved under an assumption, matched against the facts.
"""
from __future__ import annotations

import json
from ivan.sports.nba import scenarios as SC
from typing import Optional
from ivan.core.utils import normalize_name
from ivan.sports.nba.projection import rotation as ROT
from .state import QUESTION_STATUSES
from ivan.sports.nba import slates as S


class ScenarioMixin:
    def team_overrides(self, team: str) -> dict:
        names = {normalize_name(p.name) for p in self.players(team)}
        return {n: dict(ov) for n, ov in self.local_overrides.items() if n in names}

    def raw_facts(self, team: str, calls: Optional[dict] = None) -> dict:
        """The facts a scenario is matched against: DK statuses and Rotowire's starters, plus the calls typed into the late-news form
        (not the pins: an active scenario's own scratch pin must not keep it matching after DK says otherwise)."""
        from ivan.sports.nba.projection.live import OUT_STATUSES
        out, avail, status_now = set(), set(), {}
        for p in self.players(team):
            n = normalize_name(p.name); status_now[n] = (p.status or "").upper()
            (out if status_now[n] in OUT_STATUSES else avail).add(n)
        starters = set(self.slate.starters.get(team) or set())
        # the calls typed into the late-news form stay facts until DK's own status for the player moves (then DK wins)
        kept = self.__dict__.setdefault("_calls", {}).get(team, {})
        merged = {n: c for n, (c, st0) in kept.items() if status_now.get(n) == st0}
        merged.update(calls or {})
        for n, call in merged.items():
            if call == "out":
                out.add(n); avail.discard(n); starters.discard(n)
            elif call in ("in", "back"):
                avail.add(n); out.discard(n)
            elif call == "starts":
                avail.add(n); out.discard(n); starters.add(n)
            elif call == "bench":
                starters.discard(n)
        return dict(out=out, available=avail, starters=starters)

    def note_call(self, team: str, player: str, call: str) -> None:
        """Remember a typed late-news call as a fact for the scenario pick, with DK's status at the time (see raw_facts)."""
        st = next(((p.status or "").upper() for p in self.players(team) if normalize_name(p.name) == player), "")
        self.__dict__.setdefault("_calls", {}).setdefault(team, {})[player] = (call, st)

    def scenario_list(self, team: str) -> list:
        out = []
        for sc in SC.list_scenarios(team):
            a = sc.get("assumes") or {}
            sc["assumes_text"] = ", ".join(f"{n.split()[-1]} {c}" for n, c in a.items()) or "no assumption (never picked by itself)"
            sc["holds"] = bool(a) and SC.holds(a, self.raw_facts(team))
            out.append(sc)
        return out

    def save_scenario(self, team: str, name: str) -> dict:
        statuses = {normalize_name(p.name): (p.status or "").upper() for p in self.players(team)}
        q = [n for n, st in statuses.items() if st in QUESTION_STATUSES]
        grid = None
        gp = ROT.drawn_path(team)
        if gp.exists():
            try:
                grid = json.loads(gp.read_text())
            except (OSError, ValueError):
                grid = None
        assumes = SC.assumptions(self.team_overrides(team), statuses, q, grid)
        return SC.save_scenario(team, name, self.team_overrides(team), assumes)

    def activate_scenario(self, team: str, slug: str) -> Optional[dict]:
        new = SC.activate(team, slug, self.local_overrides, [normalize_name(p.name) for p in self.players(team)])
        if new is None:
            return None
        self.local_overrides = new
        self.write_local()
        self.reproject(team)
        return SC.load_scenario(team, slug)

    def auto_scenarios(self, calls: Optional[dict] = None) -> list:
        """
        When the facts of a team with scenarios change (a refresh moved a status or the lineup, a late-news call), the one
        scenario whose assumptions all hold is activated. Returns the messages to flash. Two matches or none: said, not acted on.
        """
        msgs = []
        seen = self.__dict__.setdefault("_facts_seen", {})
        if not S.scenarios_dir().is_dir():
            return msgs
        teams = {p.team for p in self.slate.players if p.team}
        for d in S.scenarios_dir().iterdir():
            team = d.name
            if not d.is_dir() or team not in teams:
                continue
            scs = SC.list_scenarios(team)
            if not scs:
                continue
            facts = self.raw_facts(team, (calls or {}).get(team))
            key = (frozenset(facts["out"]), frozenset(facts["available"]), frozenset(facts["starters"]))
            if seen.get(team) == key:
                continue
            seen[team] = key
            matches = [sc for sc in scs if sc.get("assumes") and SC.holds(sc["assumes"], facts)]
            act = SC.active_slug(team)
            if len(matches) == 1:
                if matches[0]["slug"] != act:
                    self.activate_scenario(team, matches[0]["slug"])
                    msgs.append(f"{team}: the news matches scenario '{matches[0]['name']}' ({', '.join(f'{n.split()[-1]} {c}' for n, c in matches[0]['assumes'].items())}): activated")
            elif len(matches) > 1:
                msgs.append(f"{team}: {len(matches)} scenarios match the news ({', '.join(m['name'] for m in matches)}): pick one")
            elif act:
                cur = SC.load_scenario(team, act)
                if cur and cur.get("assumes") and not SC.holds(cur["assumes"], facts):
                    msgs.append(f"{team}: the active scenario '{cur['name']}' no longer matches the news; no saved scenario does")
        return msgs
