"""
The rotation grid the team page shows: tonight's heatmap from the World's draw in the projection store, a played game's from the stints, the mapping fingerprint.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional
import numpy as np
import pandas as pd
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.projection.live import override_of
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.groups import group_of as SK_group
from .state import ADD_BACK_MINUTES, FOUL_SHORT_PF, FOUL_SHORT_SHARE, short_name

UNDER_MIN = 8.0            # a night under this many minutes is a cut night (the minutes hover's share, DNPs included)
NO_DRAW = dict(cover=None, min_med=None, min_p10=None, min_p90=None, plays=None, under=None)


class RotationMixin:
    SCORE_STEP = 3   # minutes per cell of the score strip above a past game's grid

    def score_strip(self, team: str, game_id: str) -> Optional[dict]:
        """
        The game's running score above its grid (pbp.game_score, the stored play-by-play; no fetch): the team's and the
        opponent's score at the end of every SCORE_STEP minutes, the margin, and the exact score per minute for the hover.
        None when the feed is not on disk.
        """
        from ivan.sports.nba.data_clean import pbp as PBP
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

    def game_context(self, team: str) -> dict:
        """
        {game_id: its context} for the team's played games (ivan.sports.nba.projection.rotation.game_context; terminal:
        scripts/nba/game_context.py): not close (a blowout), a regular absent, a regular who left early. What the
        played-game picker and the recent-games hovers shade, so the games that are no picture of the usual rotation are
        told apart at a glance. Cached until the box scores grow.
        """
        cache = self.__dict__.setdefault("_game_ctx", {})
        key = (team, getattr(self, "_season_key", None))
        if key not in cache:
            try:
                cache[key] = ROT.game_context(self.session, self.slate.box, team)
            except Exception:   # noqa: BLE001 - shading only: the lists stand without it
                cache[key] = {}
        return cache[key]

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
          source "auto"    tonight: one row per player with minutes, each cell the share of the World's worlds he is on the
                           floor in that minute (`cover`, 48 shares from the team's stored draw: the same worlds every
                           number on the card is the mean of), with his drawn minutes beside it (min_med, min_p10, min_p90,
                           plays, under). Nothing is painted: the rows follow the minutes typed in the min column. A record
                           written before the coverage was stored has none (`has_cover` False): the plan's grid shows
                           instead until the team is projected again. Out players show their usual stretches faintly
                           (the vacated slots); players the plan left off the floor are blank rows, faintly shaded where
                           the World plays them anyway.
          source "last"    one of this season's games (`game_id`, else the latest) as played, fouls marked on the minute, the
                           starters who had not started the game before flagged (new_starter). A row's minutes are what he
                           played (his stints in seconds, overtime included: the box score's number and the minutes hover's),
                           not the cells counted; `ot` is the overtime part, which has no column
        """
        from ivan.sports.nba.data_clean import pbp as PBP
        from ivan.sports.nba.projection.live import OUT_STATUSES
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
        # the dropdown: date, opponent, the final from the team's side, how many of the five had not started the game
        # before, and what makes the game a poor picture of the usual rotation (game_context: a blowout, a regular absent,
        # a player who left early), which also shades its line
        try:
            scores = ROT.game_scores(self.session, [g["game_id"] for g in games])
            new_starters = ROT.starter_changes(self.session, team, games)
        except Exception:   # noqa: BLE001 - labels only; the list stands without them
            scores, new_starters = {}, {}
        ctx = self.game_context(team)
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
            c = ctx.get(str(g["game_id"])) or {}
            g["shade"], g["why"] = c.get("shade", ""), c.get("why", "")
            g["short"] = " · ".join(x for x in [day + (f" vs {g['opp']}" if g.get("opp") else ""), g["final"],
                                                 f"{g['new_starters']} new starter{'s' if g['new_starters'] != 1 else ''}" if g["new_starters"] is not None else "",
                                                 c.get("tag", "")] if x)
            g["long"] = " · ".join(x for x in [str(g.get("game_date", "")), f"vs {g['opp']}" if g.get("opp") else "", g["final"],
                                                f"{g['new_starters']} of the five had not started the game before" if g["new_starters"] is not None else "",
                                                c.get("why", ""), ("the last game" if i == 0 and g.get("season") == "this" else "")] if x)
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
                played = (last.get("minutes") or {}).get(n); ot = float((last.get("overtime") or {}).get(n) or 0.0)
                mins = int(round(played)) if played is not None else int(grid.loc[n].sum()); pf = len(fouls.get(n, []))   # what he played (the box score's number), not the cells counted
                cell_tips = [None] * 48
                for m, txt in tips.get(n, {}).items():
                    cell_tips[m] = txt
                for m, no in fouls.get(n, []):
                    cell_tips[m] = (f"foul {no}" if no is not None else "technical") + (f" · {cell_tips[m]}" if cell_tips[m] else "")
                rows.append(dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, status=((names[n].status or '').upper() if n in names else ''),
                                 starter=bool(cells[0]), new_starter=bool(cells[0]) and n in fresh, out=bool(is_out.get(n, False)), gone=n not in names, vacated=False, locked=False, minutes=mins, target=None,
                                 hist=hist, cells=cells, marks=marks, pf=pf, spots=[int(x) for x in spots.loc[n].values], tips=cell_tips, played=played, ot=ot,
                                 group=SK_group(names[n].position) if n in names else "wing", rank=ROT._spot_rank(names[n].position if n in names else None, None),
                                 foul_short=bool(pf >= FOUL_SHORT_PF and hist and mins < FOUL_SHORT_SHARE * hist), heir_of=""))
            shown = ctx.get(str(last["game_id"])) or {}
            return {**base, "kind": "last", "rows": rows, "counts": ROT.column_counts(grid), "game_id": last["game_id"], "last_game": last["game_date"], "fouls": True,
                    "shade": shown.get("shade", ""), "why": shown.get("why", ""),
                    "score": self.score_strip(team, last["game_id"]), "owners": owners, "games": games}

        grid = rot.get("grid")
        if grid is None or proj is None:
            return base
        drawn = self.drawn_of(team)
        has_cover = bool(drawn)
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
            if mins <= 0:
                continue
            rows.append(dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, status=((names[n].status or '').upper() if n in names else ''),
                             starter=n in starters, out=False, gone=False, vacated=False, locked=False, minutes=mins, **(drawn.get(n) or NO_DRAW),
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
        # one order for both panels (tonight and the game below): the saved projection's minutes,
        # starters not promoted, vacated slots last; so the rows line up whatever is shown
        rows = self.order_rows(rows, owners)
        self.mark_starter_rule(rows)
        # off the floor tonight: available players the schedule gave no minutes (added back by hand)
        benched = []
        for n, p in names.items():
            if any(r["key"] == n for r in rows) or (is_out.get(n) and n not in dropped):
                continue
            hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
            benched.append(dict(name=n, key=n, pos=p.position, salary=p.salary, hist=hist, status=(p.status or "").upper(), dropped=n in dropped,
                                add_minutes=round(max(ADD_BACK_MINUTES, hist or 0.0))))
        benched.sort(key=lambda r: -(r["salary"] or 0))
        # they sit in the grid as blank rows (type minutes to play him; a dropped player shows DROPPED with his + to restore)
        out_rows = [r for r in rows if r.get("vacated") or r.get("out")]
        rows = [r for r in rows if not (r.get("vacated") or r.get("out"))]
        for b in benched:
            rows.append(dict(name=b["name"], key=b["key"], pos=b["pos"] or "", salary=b["salary"], status=b["status"], starter=False, out=False, gone=False,
                             vacated=False, locked=False, minutes=0, target=None, hist=b["hist"], cells=[False] * 48, marks=[None] * 48, pf=0,
                             spots=[-1] * 48, tips=[None] * 48, group=SK_group(b["pos"] or ""), rank=ROT._spot_rank(b["pos"], None), foul_short=False,
                             heir_of="", dropped=b["dropped"], bench=True, **(drawn.get(b["key"]) or NO_DRAW)))
        rows += out_rows
        n_pins = sum(1 for n in names if override_of(self.overrides, n).get("minutes") is not None)
        # the foot: with the draw, the mean number on the floor each minute (five, whatever the rows show); without it the plan grid's count
        counts = [round(sum(r["cover"][m] for r in rows if r.get("cover")), 1) for m in range(48)] if has_cover else ROT.column_counts(grid)
        return {**base, "kind": "model", "rows": rows, "counts": counts, "pins": n_pins, "benched": benched, "owners": owners, "has_cover": has_cover}

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
        Our grid for the game's day over the played grid (backtest.slate.grid_for_day: the grid drawn on the desk that day, else the
        archive's grid rows at that day's lock): each row gets `ours` (our 48 cells) and `cmp` per minute ("both" on the floor in
        both, "act" played but not in ours, "ours" in ours but sat, ""), players we projected who did not play are added as
        DNP rows, and rot["overlay"] says where ours came from and the share of the 240 player-minutes on the floor in both
        (None when we have no grid for that day). What the backtest view draws under each team, on the team page.
        """
        from ivan.sports.nba.backtest import slate as SB
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

    def drawn_of(self, team: str) -> dict:
        """
        {player: his night as the World drew it} from the team's stored record (projection.store, the draw every number on
        the card is the mean of): cover (48 shares: the worlds he is on the floor in each minute), min_med, min_p10,
        min_p90, plays (the share of worlds he plays at all) and under (the share under UNDER_MIN minutes, DNPs included;
        None when the record has no minutes distribution). {} when the record carries no coverage (written before it was
        stored, or the game not drawn yet).
        """
        proj = self.proj.get(team)
        if proj is None or "cover" not in proj.columns:
            return {}
        num = lambda v: None if v is None or pd.isna(v) else float(v)   # noqa: E731
        out = {}
        for n, r in proj.iterrows():
            c = r.get("cover")
            if not isinstance(c, list) or len(c) != ROT.MINUTES:
                continue
            d = r.get("minutes_dist"); pz = num(r.get("p_zero"))
            out[str(n)] = dict(cover=[float(x) for x in c], min_med=num(r.get("minutes_med")), min_p10=num(r.get("minutes_p10")), min_p90=num(r.get("minutes_p90")),
                               plays=None if pz is None else 1.0 - pz, under=float(sum(d[:int(UNDER_MIN)])) if isinstance(d, list) and d else None)
        return out

    def map_team(self, team: str) -> int:
        """
        Map the team (the card's Map button; scripts/nba/mapping.py --map TEAM): every player with minutes is pinned at the
        minutes he shows, so a later refresh moves nobody, and the pool it was looked at against (who is available, the
        statuses, the five starters: fingerprint) is saved beside the plan's grid (projection.rotation.save_drawn). The card
        reads "mapped" until any of that moves, and a player ruled out or a changed five voids it (slate.mapping). Returns
        the number of players held.
        """
        grid = (self.rot.get(team) or {}).get("grid")
        if grid is None:
            raise ValueError(f"{team} has no projection to map")
        held = [str(n) for n in grid.index if int(grid.loc[n].sum()) > 0]
        for n in held:
            if override_of(self.overrides, n).get("minutes") is None:
                self.set_override(n, minutes=float(grid.loc[n].sum()))
        # the fingerprint is taken after the pins: they are part of what was mapped
        ROT.save_drawn(team, grid.loc[held], locked=held, fingerprint=self.fingerprint(team))
        self.reproject(team)
        return len(held)

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
        The team's mapping: unmapped (never mapped, or a record from the painted grid that left rows to the model), mapped (Map
        was pressed and the inputs are as they were then) or changed (mapped, but the availability, a status or the starters
        moved since: the differences listed). A refresh re-renders the card, so a change shows at once.
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
        """Clear pins: the mapping goes, and so does every minutes pin on the team (the ones Map set and the hand-typed ones alike),
        so the minutes are the model's again. Rates, bands, starter calls and the sim / public overrides stay. Returns the pins lifted."""
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
