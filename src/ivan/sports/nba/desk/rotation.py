"""
The rotation grid the team page shows: tonight's grid from the projection store, a played game's from the stints, the mapping fingerprint.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Optional
import numpy as np
import pandas as pd
from ivan.core.utils import normalize_name
from ivan.sports.nba.projection.live import override_of
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.groups import group_of as SK_group
from .state import ADD_BACK_MINUTES, FOUL_SHORT_PF, FOUL_SHORT_SHARE, ROTATION_MIN, short_name


class RotationMixin:
    SCORE_STEP = 3   # minutes per cell of the score strip above a past game's grid

    def score_strip(self, team: str, game_id: str) -> Optional[dict]:
        """
        The game's running score above its grid (pbp.game_score, the stored play-by-play; no fetch): the team's and the
        opponent's score at the end of every SCORE_STEP minutes, the margin, and the exact score per minute for the hover.
        None when the feed is not on disk.
        """
        from ivan.sports.nba import pbp as PBP
        try:
            sc = PBP.game_score(str(game_id))
        except Exception:   # noqa: BLE001 - a missing or odd feed: no strip
            sc = None
        if not sc or team not in sc:
            return None
        opp = next(t for t in sc["final"] if t != team)
        step = self.SCORE_STEP
        cells = []
        for k in range(48 // step):
            m = (k + 1) * step - 1
            us, them = sc[team][m], sc[opp][m]
            cells.append(dict(start=k * step, us=us, them=them, margin=us - them,
                              tip=" · ".join(f"{i + 1}': {sc[team][i]}-{sc[opp][i]}" for i in range(k * step, m + 1))))
        return dict(team=team, opp=opp, step=step, cells=cells, final=(sc[team][-1], sc[opp][-1]), overtime=sc["overtime"],
                    home=sc["home"] == team)

    def rotation(self, team: str, source: str = "auto", game_id: Optional[str] = None, overlay: bool = False) -> dict:
        """The rotation grid (_rotation) with each row's rookie mark (rookie_of): the same on tonight's grid and a past game's.
        `overlay` (a past game only): what we projected for that game over what was played (overlay_ours)."""
        rot = self._rotation(team, source=source, game_id=game_id)
        if overlay and rot.get("kind") == "last":
            self.overlay_ours(team, rot)
        notes = self.notes() if rot.get("kind") != "last" else {}
        for r in rot.get("rows") or []:
            r["note"] = notes[r["key"]].text if r["key"] in notes else None
            rk = self.rookie_of(r["key"])
            r["rookie"] = rk is not None
            r["rookie_games"] = rk["games"] if rk else None
        return rot

    def _rotation(self, team: str, source: str = "auto", game_id: Optional[str] = None) -> dict:
        """
        The team's rotation grid for the card.
          source "auto"    the projection's grid: the coach's slots filled at the solve's minutes, the desk's locked rows held
                           (ivan.sports.nba.projection.skeleton). Out players show their usual stretches faintly (the vacated
                           slots), players the schedule left off the floor are listed under the grid with an add button.
          source "propose" the same grid with every lock lifted (a preview, nothing saved)
          source "last"    one of this season's games (`game_id`, else the latest) as played, fouls marked on the minute, the
                           starters who had not started the game before flagged (new_starter)
        """
        from ivan.sports.nba import pbp as PBP
        from ivan.sports.nba.projection.live import OUT_STATUSES
        from ivan.sports.nba.projection import skeleton as SK
        players = self.players(team)
        names = {normalize_name(p.name): p for p in players}
        is_out = {n: (p.status or "").upper() in OUT_STATUSES or (override_of(self.overrides, n).get("minutes") is not None and override_of(self.overrides, n).get("minutes") <= 0)
                  for n, p in names.items()}
        dropped = {n for n, p in names.items() if (p.status or "").upper() not in OUT_STATUSES and override_of(self.overrides, n).get("minutes") is not None
                   and override_of(self.overrides, n).get("minutes") <= 0}
        starters = self.starters_of(team) or set()
        today = datetime.now().date(); season_start = date(today.year if today.month >= 10 else today.year - 1, 10, 1)
        games = [dict(g, season="this") for g in ROT.team_games(self.session, team, since=season_start)]
        last_start = date(season_start.year - 1, 10, 1)              # last season's games stay selectable (a returning player's habits, a roster that
        games += [dict(g, season="last") for g in ROT.team_games(self.session, team, before=season_start, since=last_start, limit=100)]   # has barely played)
        rot = self.rot.get(team) or {}
        curve = rot.get("curve")
        # the dropdown: date, opponent, the final from the team's side and how many of the five had not started the game
        # before
        try:
            scores = ROT.game_scores(self.session, [g["game_id"] for g in games])
            new_starters = ROT.starter_changes(self.session, team, games)
        except Exception:   # noqa: BLE001 - labels only; the list stands without them
            scores, new_starters = {}, {}
        for i, g in enumerate(games):
            try:
                d = date.fromisoformat(str(g["game_date"])[:10])
                day = f"{d:%a %m-%d}" if g.get("season") != "last" else f"{d:%m-%d-%y}"
            except ValueError:
                day = str(g.get("game_date", ""))
            sc = scores.get(str(g["game_id"])) or {}
            us, them = sc.get(team), (sc.get(g["opp"]) if g.get("opp") else next((v for t, v in sc.items() if t != team), None))
            g["final"] = f"{'W' if us > them else 'L' if us < them else 'T'} {us}-{them}" if us is not None and them is not None else ""
            g["new_starters"] = new_starters.get(str(g["game_id"]))
            g["short"] = " · ".join(x for x in [day + (f" vs {g['opp']}" if g.get("opp") else ""), g["final"],
                                                 f"{g['new_starters']} new starter{'s' if g['new_starters'] != 1 else ''}" if g["new_starters"] is not None else ""] if x)
            g["long"] = " · ".join(x for x in [str(g.get("game_date", "")), f"vs {g['opp']}" if g.get("opp") else "", g["final"],
                                                f"{g['new_starters']} of the five had not started the game before" if g["new_starters"] is not None else "",
                                                ("the last game" if i == 0 and g.get("season") == "this" else "")] if x)
        proj = self.proj.get(team)
        targets = rot.get("targets") or {}
        base = dict(team=team, games=games, kind="none", drawn=False, locked=[], rows=[], counts=[], benched=[], game_id=None, last_game=None,
                    has_history=curve is not None and len(curve) > 0, curve_games=int(curve.games.max()) if curve is not None and len(curve) else 0,
                    fouls=False, owners=[])

        if source == "last":
            last = ROT.last_game_grid(self.session, team, since=season_start, game_id=game_id)      # the latest game unless one is picked
            if last is None:
                return base
            try:
                fresh = ROT.new_starters(self.session, team, last["game_id"], last["game_date"])    # started this game, not the one before
            except Exception:   # noqa: BLE001
                fresh = set()
            grid = last["grid"]
            fouls: dict = {}
            try:
                f = PBP.game_fouls(self.session, last["game_id"], team)
                for r in f[f.personal == 1].itertuples():
                    fouls.setdefault(str(r.player_name), []).append((int(r.minute), None if r.foul_no is None or pd.isna(r.foul_no) else int(r.foul_no)))
            except Exception:   # noqa: BLE001
                pass
            rows = []
            try:
                pairs = PBP.game_subs(self.session, last["game_id"], team).to_dict("records")
            except Exception:   # noqa: BLE001
                pairs = []
            # hover text on the cells where a player came on or went off: the recorded substitution, the game clock, how many changed at once
            def clock(t):
                per = min(4, int(t // 720) + 1) if t < 2880 else 4 + int((t - 2880) // 300) + 1
                left = (per * 720 - t) if t < 2880 else (2880 + (per - 4) * 300 - t)
                return f"{'Q' + str(per) if per <= 4 else 'OT' + str(per - 4)} {int(left // 60)}:{int(left % 60):02d}"
            at_once: dict = {}
            for pr in pairs:
                at_once[round(float(pr["elapsed"]), 1)] = at_once.get(round(float(pr["elapsed"]), 1), 0) + 1
            tips: dict = {}
            for pr in pairs:
                t = float(pr["elapsed"]); m = int(np.floor(t / 60.0 + 0.5)); k = at_once[round(t, 1)]
                note = "" if k == 1 else f" · {k} changed at once, the feed's pairing is its typing order"
                if 1 <= m <= 47:
                    tips.setdefault(pr["player_in"], {})[m] = f"in at {clock(t)} for {pr['player_out']}{note}"
                    tips.setdefault(pr["player_out"], {})[m - 1] = f"out at {clock(t)}, {pr['player_in']} in{note}"
            spots, owners = ROT.spots_of(grid, starters=None, groups={n: SK_group(names[n].position) for n in names}, positions={n: names[n].position for n in names},
                                         pairs=pairs)
            for n in grid.index:
                cells = [bool(x) for x in grid.loc[n].values]
                marks: list = [None] * 48
                for m, no in fouls.get(n, []):
                    marks[m] = no if no is not None else "T"
                hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
                mins = int(grid.loc[n].sum()); pf = len(fouls.get(n, []))
                cell_tips = [None] * 48
                for m, txt in tips.get(n, {}).items():
                    cell_tips[m] = txt
                for m, no in fouls.get(n, []):
                    cell_tips[m] = (f"foul {no}" if no is not None else "technical") + (f" · {cell_tips[m]}" if cell_tips[m] else "")
                rows.append(dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, status=((names[n].status or '').upper() if n in names else ''),
                                 starter=bool(cells[0]), new_starter=bool(cells[0]) and n in fresh, out=bool(is_out.get(n, False)), gone=n not in names, vacated=False, locked=False, minutes=mins, target=None,
                                 hist=hist, cells=cells, marks=marks, pf=pf, spots=[int(x) for x in spots.loc[n].values], tips=cell_tips,
                                 group=SK_group(names[n].position) if n in names else "wing", rank=ROT._spot_rank(names[n].position if n in names else None, None),
                                 foul_short=bool(pf >= FOUL_SHORT_PF and hist and mins < FOUL_SHORT_SHARE * hist), heir_of=""))
            return {**base, "kind": "last", "rows": rows, "counts": ROT.column_counts(grid), "game_id": last["game_id"], "last_game": last["game_date"], "fouls": True,
                    "score": self.score_strip(team, last["game_id"]), "owners": owners, "games": games}

        grid = rot.get("grid")
        if grid is None or proj is None:
            return base
        locked = list(grid.attrs.get("locked", []))
        if source == "propose":
            avail = [n for n in grid.index]
            groups = {n: SK_group(names[n].position) for n in names}
            grid = SK.build_grid(curve, targets, avail, starters=[n for n in avail if n in starters], groups=groups, locks=None, vacated=rot.get("vacated"),
                                 shares=rot.get("shares"))
            locked = []
        inherited: dict = {}
        for owner, heirs in (grid.attrs.get("inherited") or {}).items():
            for h, share in heirs.items():
                if share >= 0.2:
                    inherited.setdefault(h, []).append(f"{owner.replace(' (old slot)', '')} {share:.0%}")
        rows = []
        shares = rot.get("shares") or {}
        spots, owners = ROT.spots_of(grid, starters=starters, groups={n: SK_group(names[n].position) for n in names}, positions={n: names[n].position for n in names},
                                     shares=shares)
        for n in grid.index:
            hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
            mins = int(grid.loc[n].sum())
            if mins <= 0 and n not in locked:
                continue
            rows.append(dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, status=((names[n].status or '').upper() if n in names else ''),
                             starter=n in starters, out=False, gone=False, vacated=False, locked=n in locked, minutes=mins,
                             target=None if n not in targets else round(float(targets[n]), 1),
                             hist=hist, cells=[bool(x) for x in grid.loc[n].values], marks=None, pf=None, foul_short=False,
                             spots=[int(x) for x in spots.loc[n].values], group=SK_group(names[n].position) if n in names else "wing",
                             rank=ROT._spot_rank(names[n].position if n in names else None, None),
                             heir_of=", ".join(inherited.get(n, []))))
        # the vacated slots: out players with a curve, their usual stretches shown faintly; a player dropped by hand is marked so he can be restored
        for n in names:
            if is_out.get(n) and curve is not None and n in curve.index and float(curve.loc[n, "minutes"]) >= 3:
                cells = [bool(float(curve.loc[n, m]) >= 0.5) for m in range(48)]
                rows.append(dict(name=n, key=n, pos=names[n].position, salary=names[n].salary, status=(names[n].status or '').upper(), starter=n in starters, out=True, gone=False, vacated=True,
                                 locked=False, minutes=int(round(float(curve.loc[n, "minutes"]))), target=None, hist=float(curve.loc[n, "minutes"]),
                                 cells=cells, marks=None, pf=None, foul_short=False, heir_of="", spots=[-1] * 48, dropped=n in dropped,
                                 group=SK_group(names[n].position), rank=ROT._spot_rank(names[n].position, None)))
        # one order for every panel (tonight, the unlocked preview, the game below): the saved projection's minutes,
        # starters not promoted, vacated slots last; so the rows line up whatever is shown
        rows = self.order_rows(rows, owners)
        self.mark_starter_rule(rows)
        lanes = self.lanes_of(rows, owners)
        # off the floor tonight: available players the schedule gave no minutes (added back by hand)
        benched = []
        for n, p in names.items():
            if any(r["key"] == n for r in rows) or (is_out.get(n) and n not in dropped):
                continue
            hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
            benched.append(dict(name=n, key=n, pos=p.position, salary=p.salary, hist=hist, status=(p.status or "").upper(), dropped=n in dropped,
                                add_minutes=round(max(ADD_BACK_MINUTES, hist or 0.0))))
        benched.sort(key=lambda r: -(r["salary"] or 0))
        # they sit in the grid as blank rows (paint one to give him minutes; a dropped player shows DROPPED with his + to restore)
        out_rows = [r for r in rows if r.get("vacated") or r.get("out")]
        rows = [r for r in rows if not (r.get("vacated") or r.get("out"))]
        for b in benched:
            rows.append(dict(name=b["name"], key=b["key"], pos=b["pos"] or "", salary=b["salary"], status=b["status"], starter=False, out=False, gone=False,
                             vacated=False, locked=False, minutes=0, target=None, hist=b["hist"], cells=[False] * 48, marks=[None] * 48, pf=0,
                             spots=[-1] * 48, tips=[None] * 48, group=SK_group(b["pos"] or ""), rank=ROT._spot_rank(b["pos"], None), foul_short=False,
                             heir_of="", dropped=b["dropped"], bench=True))
        rows += out_rows
        avail_keys = {r["key"] for r in rows}
        shares_json = json.dumps({o: {i: round(v, 3) for i, v in hv.items() if i in avail_keys} for o, hv in shares.items() if o in avail_keys})
        n_pins = sum(1 for n in names if override_of(self.overrides, n).get("minutes") is not None)
        return {**base, "kind": "model" if source != "propose" else "propose", "rows": rows, "counts": ROT.column_counts(grid), "drawn": bool(locked), "pins": n_pins,
                "locked": locked, "benched": benched, "owners": owners, "lanes": lanes, "shares_json": shares_json}

    @staticmethod
    def mark_starter_rule(rows: list[dict]) -> list[dict]:
        """sep=True on the last row of the leading block of starters: the dark line under the starters. A starter listed lower
        (a game's starter who is out tonight, or one the model plays less than a bench player) stays bold below the line."""
        last = None
        for i, r in enumerate(rows):
            if r.get("starter") and not r.get("vacated"):
                last = i
            else:
                break
        if last is None:
            last = max((i for i, r in enumerate(rows) if r.get("starter") and not r.get("vacated")), default=None)
        for i, r in enumerate(rows):
            r["sep"] = (i == last)
        return rows

    def row_order(self, team: str) -> dict[str, int]:
        """player -> rank from the saved projection's grid: minutes descending, salary as the tie-break; starters are not promoted."""
        rot = self.rot.get(team) or {}
        grid = rot.get("grid")
        if grid is None:
            return {}
        sal = {normalize_name(p.name): (p.salary or 0) for p in self.players(team)}
        mins = {str(n): int(grid.loc[n].sum()) for n in grid.index}
        names = sorted(mins, key=lambda n: (-mins[n], -sal.get(n, 0), n))
        return {n: i for i, n in enumerate(names)}

    def overlay_ours(self, team: str, rot: dict) -> dict:
        """
        Our grid for the game's day over the played grid (slate_backtest.grid_for_day: the grid drawn on the desk that day, else the
        archive's grid rows at that day's lock): each row gets `ours` (our 48 cells) and `cmp` per minute ("both" on the floor in
        both, "act" played but not in ours, "ours" in ours but sat, ""), players we projected who did not play are added as
        DNP rows, and rot["overlay"] says where ours came from and the share of the 240 player-minutes on the floor in both
        (None when we have no grid for that day). What the backtest view draws under each team, on the team page.
        """
        from ivan.sports.nba import slate_backtest as SB
        day = rot.get("last_game")
        try:
            d = date.fromisoformat(str(day)[:10])
        except (TypeError, ValueError):
            rot["overlay"] = dict(found=False, source="the game's day is not known", agreement=None); return rot
        ours, src = SB.grid_for_day(self.session, team, d)
        if ours is None:
            rot["overlay"] = dict(found=False, source=src, agreement=None); return rot
        both = 0; seen = set()
        for r in rot.get("rows") or []:
            n = r["key"]; seen.add(n)
            o = [bool(x) for x in ours.loc[n].values] if n in ours.index else [False] * 48
            r["ours"] = o; r["min_ours"] = sum(o)
            r["cmp"] = [("both" if a and b else "act" if a else "ours" if b else "") for a, b in zip(r["cells"], o)]
            both += sum(1 for a, b in zip(r["cells"], o) if a and b)
        names = {normalize_name(p.name): p for p in self.players(team)}
        for n in ours.index:
            if n in seen or int(ours.loc[n].sum()) <= 0:
                continue
            o = [bool(x) for x in ours.loc[n].values]
            p = names.get(n)
            rot["rows"].append(dict(name=n, key=n, pos=p.position if p else "", salary=p.salary if p else None, status=(p.status or "").upper() if p else "",
                                    starter=False, new_starter=False, out=False, gone=p is None, vacated=False, locked=False, minutes=0, target=None, hist=None,
                                    cells=[False] * 48, marks=[None] * 48, pf=0, spots=[-1] * 48, tips=[None] * 48, group="wing", rank=9.0, foul_short=False,
                                    heir_of="", ours=o, min_ours=sum(o), cmp=["ours" if b else "" for b in o], dnp=True))
        rot["overlay"] = dict(found=True, source=src, agreement=both / 240.0)
        return rot

    def game_grid(self, team: str, game_id: Optional[str] = None, overlay: bool = False) -> dict:
        """A played game's grid for the panel under tonight's: rows in the same order as tonight's panel, then the rest by minutes.
        `overlay`: our grid for that day over it (overlay_ours)."""
        rot = self.rotation(team, source="last", game_id=game_id, overlay=overlay)
        # the game's starters first (struck through when out tonight: those are the minutes to give away), then the bench
        rot["rows"] = self.mark_starter_rule(self.order_rows(rot.get("rows") or [], rot.get("owners") or []))
        rot["lanes"] = self.lanes_of(rot["rows"], rot.get("owners") or [])
        return rot

    @staticmethod
    def order_rows(rows: list[dict], owners: list[str]) -> list[dict]:
        """
        Starters in spot order 1-5 (guards to bigs, the lanes' order). The bench in spot order too when every bench player
        fills one spot all game; when any of them fills several, the bench is by minutes instead. Vacated slots last.
        """
        spot_of_owner = {o: k for k, o in enumerate(owners)}
        def spots_filled(r):
            return sorted({sp for sp, on in zip(r.get("spots") or [], r.get("cells") or []) if on and sp >= 0})
        starters = [r for r in rows if r.get("starter") and not r.get("vacated")]
        bench = [r for r in rows if not r.get("starter") and not r.get("vacated")]
        vacated = [r for r in rows if r.get("vacated")]
        starters.sort(key=lambda r: (spot_of_owner.get(r["key"], (spots_filled(r) or [9])[0]), -r["minutes"]))
        single = all(len(spots_filled(r)) <= 1 for r in bench)
        if single:
            bench.sort(key=lambda r: ((spots_filled(r) or [9])[0], -r["minutes"], -(r.get("salary") or 0)))
        else:
            bench.sort(key=lambda r: (-r["minutes"], -(r.get("salary") or 0)))
        vacated.sort(key=lambda r: -r["minutes"])
        return starters + bench + vacated

    @staticmethod
    def lanes_of(rows: list[dict], owners: list[str]) -> list[dict]:
        """
        The five spots traced minute by minute: for each spot, the stretches (start, length, player) of who held it, so a line
        (the centre's, say) reads as one band with the holder's name on each stretch; a minute nobody holds is a gap.
        """
        lanes = []
        for k, owner in enumerate(owners):
            holder = [None] * 48
            for r in rows:
                if r.get("vacated"):
                    continue
                for m, sp in enumerate(r.get("spots") or []):
                    if sp == k and r["cells"][m]:
                        holder[m] = r["key"]
            segs = []; start = 0
            for m in range(1, 49):
                if m == 48 or holder[m] != holder[start]:
                    name = holder[start]
                    segs.append(dict(start=start, length=m - start, key=name, short=short_name(name) if name else ""))
                    start = m
            lanes.append(dict(spot=k, owner=owner, segments=segs))
        return lanes

    def apply_rotation(self, team: str, players: list[str], grid: list[list[int]], locked: Optional[list[str]] = None, note: str = "") -> int:
        """
        Save the painted grid: the locked rows are held exactly by the skeleton and their row totals become the players'
        minutes (a row painted empty scratches him); a row that was locked and is now unlocked loses its minutes pin; the
        unlocked rows are re-filled around the locks by the next projection. Out players' rows are dropped.
        """
        from ivan.sports.nba.projection.live import OUT_STATUSES
        status = {normalize_name(p.name): (p.status or "").upper() for p in self.players(team)}
        keep = [i for i, n in enumerate(players) if n and len(grid[i]) == ROT.MINUTES]       # rows without a key (a popup's table) are not the grid's
        players = [players[i] for i in keep]; grid = [grid[i] for i in keep]
        g = pd.DataFrame(np.array(grid, dtype=bool).reshape(len(players), ROT.MINUTES), index=[normalize_name(n) for n in players], columns=list(range(ROT.MINUTES)))
        g = g.loc[[n for n in g.index if status.get(n, "") not in OUT_STATUSES]]
        lock_names = [normalize_name(n) for n in (locked if locked is not None else list(g.index))]
        lock_names = [n for n in lock_names if n in g.index]
        before = ROT.load_drawn(team)
        was_locked = set(before.attrs.get("locked", list(before.index))) if before is not None else set()
        for n in was_locked - set(lock_names):
            self.set_override(n, minutes=None)
        for n in lock_names:
            self.set_override(n, minutes=float(g.loc[n].sum()))
        # the fingerprint is taken after the pins: a row locked empty scratches its player, and that is part of the mapping
        ROT.save_drawn(team, g, note, locked=lock_names, fingerprint=self.fingerprint(team))
        self.reproject(team)
        return len(lock_names)

    def fingerprint(self, team: str) -> dict:
        """What a mapping is made against: who is available, who is out, each player's status, the five starters (Rotowire or hand calls)."""
        from ivan.sports.nba.projection.live import OUT_STATUSES
        avail, out, status = [], [], {}
        for p in self.players(team):
            n = normalize_name(p.name); st = (p.status or "").upper()
            ov = override_of(self.overrides, n)
            (out if (st in OUT_STATUSES or (ov.get("minutes") is not None and ov.get("minutes") <= 0)) else avail).append(n)
            if st and st != "NONE":
                status[n] = st
        return dict(available=sorted(avail), out=sorted(out), status=status, starters=sorted(self.starters_of(team) or []))

    def mapped_state(self, team: str) -> dict:
        """
        The team's mapping: unmapped (no held grid: nothing locked, or rows with minutes left to the model), mapped (every row
        with minutes locked and the inputs as they were when it was saved) or changed (held, but the availability, a status
        or the starters moved since: the differences listed). A refresh re-renders the card, so a change shows at once.
        """
        g = ROT.load_drawn(team)
        if g is None:
            return dict(state="unmapped", diff=[], saved_at=None)
        locked = set(g.attrs.get("locked", [])); rows = [str(n) for n in g.index if g.loc[n].sum() > 0]
        free = [n for n in rows if n not in locked]
        if not rows or free:
            return dict(state="unmapped", diff=([f"{len(free)} row(s) left to the model"] if rows else []), saved_at=g.attrs.get("saved_at"))
        fp = g.attrs.get("fingerprint")
        if not fp:
            return dict(state="changed", diff=["saved before fingerprints: map it again to track changes"], saved_at=g.attrs.get("saved_at"))
        now = self.fingerprint(team); diff = []
        short = lambda n: n.split()[-1]   # noqa: E731
        for n in sorted(set(fp["available"]) - set(now["available"])):
            diff.append(f"{short(n)} now OUT" if n in now["out"] else f"{short(n)} gone")
        for n in sorted(set(now["available"]) - set(fp["available"])):
            diff.append(f"{short(n)} back" if n in fp["out"] else f"{short(n)} new")
        if set(fp["starters"]) != set(now["starters"]):
            diff.append("starters: " + ", ".join([f"+{short(n)}" for n in sorted(set(now["starters"]) - set(fp["starters"]))] +
                                                 [f"-{short(n)}" for n in sorted(set(fp["starters"]) - set(now["starters"]))]))
        for n, st in now["status"].items():
            if n in now["available"] and n in fp["available"] and (fp.get("status") or {}).get(n) != st:
                diff.append(f"{short(n)} {st}")
        return dict(state="changed" if diff else "mapped", diff=diff, saved_at=g.attrs.get("saved_at"))

    def mapped_summary(self) -> dict:
        teams = sorted({p.team for p in self.slate.players if p.team})
        states = {t: self.mapped_state(t) for t in teams}
        return dict(n=len(teams), n_mapped=sum(1 for v in states.values() if v["state"] == "mapped"),
                    changed=[t for t, v in states.items() if v["state"] == "changed"], unmapped=[t for t, v in states.items() if v["state"] == "unmapped"])

    def clear_rotation(self, team: str) -> int:
        """Clear locks: the drawn grid goes, and so does every typed minutes pin on the team (locked rows' pins and hand-typed ones alike),
        so the grid is the model's again. Rates, bands, starter calls and the sim / public overrides stay. Returns the pins lifted."""
        ROT.clear_drawn(team)
        n = 0
        for p in self.players(team):
            key = normalize_name(p.name)
            if override_of(self.overrides, key).get("minutes") is not None:
                self.set_override(key, minutes=None); n += 1
        self.reproject(team)
        return n

    def add_back(self, team: str, player: str) -> float:
        """A player the rotation left off the floor, back in by hand: a minutes pin at his history (at least ADD_BACK_MINUTES)."""
        n = normalize_name(player)
        curve = (self.rot.get(team) or {}).get("curve")
        hist = float(curve.loc[n, "minutes"]) if curve is not None and n in curve.index else 0.0
        mins = float(max(ADD_BACK_MINUTES, round(hist)))
        self.set_override(n, minutes=mins)
        self.reproject(team)
        return mins
