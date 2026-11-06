"""
A team's rows and card: every player's minutes, rates, flags and inputs; the card's alerts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.projection.live import override_of
from ivan.sports.nba.projection.history import STATS
from ivan.sports.nba.validation import market_points, recent_minutes, review_team
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.book_minutes import book_minutes
from .state import MARKET_GAP, MINUTES_DEV, NEAR_TIP_MIN, ROTATION_MIN
from ivan.sports.nba import slates as S


class RowsMixin:
    def considerations(self, team: str) -> list:
        """What is unusual about the team's game tonight (ivan.sports.nba.considerations; scripts/nba/considerations.py)."""
        from ivan.sports.nba import considerations as CN
        ps = self.players(team)
        opp = next((p.opponent for p in ps if p.opponent), None)
        try:
            return CN.team_considerations(team, ps, self.slate.box, self.slate.state, self.starters_of(team),
                                          confirmed=self.slate.starters_confirmed.get(team, False), spread=self.slate.spread(team),
                                          opponent=opp, opp_players=self.players(opp) if opp else [],
                                          is_rookie=lambda n: self.rookie_of(n) is not None, tip=self.tip(team))
        except Exception as e:   # noqa: BLE001 - a convenience: never break the card
            return [dict(level="info", kind="error", text=f"special considerations failed: {type(e).__name__}: {e}")]

    def starters_of(self, team: str) -> Optional[set]:
        """Rotowire's lineup with the hand starter overrides applied; None when there is no lineup and no override."""
        base = self.slate.starters.get(team)
        ov = {normalize_name(p.name): override_of(self.overrides, normalize_name(p.name)).get("starter") for p in self.players(team)}
        ov = {n: v for n, v in ov.items() if v is not None}
        if base is None and not ov:
            return None
        out = set(base or ())
        for n, v in ov.items():
            (out.add if float(v) > 0 else out.discard)(n)
        return out

    def priced_total(self, team: str, name: str) -> Optional[float]:
        """A priced player's projected total as it stands (his book total): kept when only his minutes change."""
        ov = override_of(self.overrides, name)
        if ov.get("minutes") is not None and ov.get("fppm") is not None:
            return float(ov["minutes"]) * float(ov["fppm"])
        r = self.proj.get(team)
        if r is None or name not in r.index or not bool(r.loc[name].get("priced")):
            return None
        return float(r.loc[name, "fpts_mean"])

    def review(self, team: str) -> dict:
        """The rotation review (ivan.sports.nba.validation.review_team): provenance of every player's minutes and the flags."""
        if getattr(self, "_recent", None) is None or self._recent_stamp != self.pool_stamp:
            self._recent = recent_minutes(self.slate.box); self._recent_stamp = self.pool_stamp
        return review_team(self.slate, team, self.proj[team], self._recent, market_points(self.session), self.pins())

    def team_rows(self, team: str) -> list[dict]:
        proj, hist = self.proj[team], self.hist_only[team]
        review_flags = {r["name"]: r["flags"] for r in self.review(team)["rows"]}
        state = self.slate.state
        starters = self.starters_of(team)
        confirmed = self.slate.starters_confirmed.get(team, False)
        mk = self.market_means()
        rows = []
        for p in self.players(team):
            name = normalize_name(p.name)
            s = state.loc[name] if name in state.index else None
            r = proj.loc[name] if name in proj.index else None
            h = hist.loc[name] if name in hist.index else None
            status = (p.status or "").upper()
            out = r is None
            nomin = out and name in ((self.rot.get(team) or {}).get("zero_minutes") or [])
            starter = None if starters is None else (name in starters)
            flags = []
            hist_min = float(s["min_ewm"]) * float(s.get("play_ewm", 1.0)) if s is not None else None
            if nomin:
                flags.append("no minutes in the rotation")
            elif out and status != "OUT":
                flags.append("scratched")
            if status == "OUT" and hist_min and hist_min >= ROTATION_MIN:
                flags.append(f"out, frees {hist_min:.0f} min")
            if status in ("Q", "GTD", "D") and hist_min and hist_min >= ROTATION_MIN:
                flags.append("questionable")
            if starter and s is not None and float(s["start_ewm"]) < 0.5:
                flags.append("new starter")
            if starter is False and s is not None and float(s["start_ewm"]) > 0.6 and not out:
                flags.append("benched")
            if r is not None and hist_min is not None and abs(float(r["minutes"]) - hist_min) >= MINUTES_DEV:
                flags.append(f"{float(r['minutes']) - hist_min:+.0f} min vs history")
            if r is not None and s is None:
                flags.append("no history")
            market = mk.get(name)
            if market and h is not None and s is not None and h["fpts_mean"] > 0:
                gap = market / float(h["fpts_mean"]) - 1
                if abs(gap) >= MARKET_GAP:
                    flags.append(f"market {gap:+.0%} vs history")
            if r is not None and not r.get("priced") and float(r["minutes"]) >= 20:
                flags.append("unpriced")
            if r is not None and r.get("minutes_p90") is not None and not pd.isna(r.get("minutes_p90")) and float(r["minutes_p90"]) > 48:
                flags.append(f"impossible band: p90 {float(r['minutes_p90']):.0f} min")
            if r is not None and float(r["fpts_mean"]) >= 8 and float(r["fpts_std"]) > 1.2 * float(r["fpts_mean"]):
                flags.append(f"distribution suspect: sd {float(r['fpts_std']):.0f} on a mean of {float(r['fpts_mean']):.0f}")
            flags += [f for f in review_flags.get(name, []) if not f.startswith(("questionable", "pinned"))]
            # the flags as cell colours: which column each one belongs to, and how loud
            cells = {"name": [], "min": [], "hist": [], "fpm": []}
            for f in flags:
                if f.startswith(("out", "scratched", "questionable")):
                    cells["name"].append(f)
                elif f.startswith(("new team", "no history")):
                    cells["hist"].append(f)
                elif f.startswith("market"):
                    cells["fpm"].append(f)
                else:
                    cells["min"].append(f)
            rotation = r is not None and float(r["minutes"]) >= ROTATION_MIN      # the deep bench's quirks are not worth a colour
            loud = {k: ("bad" if any(x.startswith(("dnp-prone", "deep bench", "out", "scratched", "impossible band", "distribution suspect")) for x in v)
                        else "warn" if v and (rotation or k == "name") else "") for k, v in cells.items()}
            sa = self.season.get(name, {})
            cur, last_season = sa.get("season"), sa.get("last")
            fpts = None if r is None else float(r["fpts_mean"])
            ov = override_of(self.overrides, name)
            mk = self.market_rows.get(name, {})
            stats = [] if r is None else [dict(stat=st, proj=float(r[f"proj_{st}"]), line=mk.get(st, {}).get("line"), mean=mk.get(st, {}).get("mean"),
                                              books=mk.get(st, {}).get("books"), odds=mk.get(st, {}).get("odds"), priced=st in mk) for st in STATS]
            bm = book_minutes({st: v.get("mean") for st, v in self.market_rows.get(name, {}).items()}, s, self.slate.hist.league_rates) if not out else None
            rows.append(dict(id=p.id, name=p.name, key=name, salary=p.salary, pos=p.position, status=status if status != "NONE" else "", nomin=nomin,
                             book_min=None if bm is None else round(bm["minutes"], 1), book_src=None if bm is None else bm["source"],
                             book_by=None if bm is None else bm["by_stat"],
                             starter=("S" if starter else ("" if starter is not None else "?")),
                             confirmed=confirmed, out=out,
                             hist_min=hist_min, minutes=None if r is None else float(r["minutes"]),
                             min_p10=None if r is None or pd.isna(r.get("minutes_p10", np.nan)) else float(r["minutes_p10"]),
                             min_p90=None if r is None or pd.isna(r.get("minutes_p90", np.nan)) else float(r["minutes_p90"]),
                             pinned=ov.get("minutes") is not None, pin=ov.get("minutes"), starter_pin=ov.get("starter"),
                             fppm_pin=ov.get("fppm"), low_pin=ov.get("min_low"), high_pin=ov.get("min_high"), stats=stats,
                             fpts_pin=ov.get("fpts"), pub_pin=ov.get("pub"), fpts_model=None if r is None else float(r.get("fpts_model", r["fpts_mean"])),
                             layer=self._layer_of(name), local=dict(self.local_overrides.get(name, {})), published_pins=self._published_of(name),
                             fpts=fpts, sd=None if r is None else float(r["fpts_std"]),
                             fpm=None if r is None else float(r["fpm"]),
                             season_games=cur["games"] if cur else 0, season_min=cur["minutes"] if cur else None, season_fpm=cur["fpm"] if cur else None,
                             last_games=last_season["games"] if last_season else 0, last_min=last_season["minutes"] if last_season else None,
                             last_fpm=last_season["fpm"] if last_season else None,
                             value=None if fpts is None or not p.salary else fpts / (p.salary / 1000.0),
                             own=None if p.ownership in (None, 0) else float(p.ownership), pub=getattr(p, "pub_fpts", None),
                             hist_fpts=None if h is None else float(h["fpts_mean"]), market=market,
                             dk_avg=p.dk_avg_fpts, priced=bool(r is not None and r.get("priced")), recent=self.recent.get(name, []),
                             team=p.team, opp=p.opponent,
                             usage=None if r is None else float(r["usage_mult"]), flags=flags,
                             cells={k: " · ".join(v) for k, v in cells.items()}, loud=loud))
        # starters first, then the players who are out, then the rest of the bench; minutes descending within each group
        rows.sort(key=lambda x: (0 if (x["starter"] == "S" and not x["out"]) else (1 if x["out"] else 2), -(x["minutes"] or 0), -(x["salary"] or 0)))
        return rows

    def team_card(self, team: str, now: datetime) -> dict:
        rows = self.team_rows(team)
        tip = self.tip(team)
        mins_to_tip = (tip - now).total_seconds() / 60 if tip else None
        starters = self.starters_of(team)
        confirmed = self.slate.starters_confirmed.get(team, False)
        alerts = []
        for r in rows:
            for f in r["flags"]:
                if f.startswith("out, frees") or f == "scratched":
                    alerts.append(("bad", f"{r['name']}: {f}"))
                elif f == "no minutes in the rotation":
                    if (r["hist_min"] or 0) >= ROTATION_MIN:
                        alerts.append(("warn", f"{r['name']}: {f} ({r['hist_min']:.0f} min history)"))
                elif f in ("new starter", "benched") or f.startswith("market"):
                    alerts.append(("warn", f"{r['name']}: {f}"))
                elif f == "questionable":
                    alerts.append(("warn", f"{r['name']}: questionable ({r['hist_min']:.0f} min)"))
                elif f.endswith("vs history") and r["minutes"] and r["minutes"] >= 12:
                    alerts.append(("info", f"{r['name']}: {f}"))
                elif f.startswith(("dnp-prone", "deep bench", "impossible band", "distribution suspect")):
                    alerts.append(("bad", f"{r['name']}: {f}"))
                elif f.startswith(("new team", "above recent", "market implies", "no history")) and r["minutes"] and r["minutes"] >= 10:
                    alerts.append(("warn", f"{r['name']}: {f}"))
        if starters is None:
            alerts.append(("warn" if (mins_to_tip is not None and mins_to_tip <= NEAR_TIP_MIN) else "info", "no lineup from Rotowire"))
        elif not confirmed and mins_to_tip is not None and mins_to_tip <= NEAR_TIP_MIN:
            alerts.append(("warn", "lineup still expected, not confirmed"))
        avail = [r for r in rows if not r["out"]]
        base_sum = sum((r["hist_min"] or 0) for r in avail)
        if abs(base_sum - 241) >= 25:
            alerts.append(("info", f"history minutes of the available roster sum to {base_sum:.0f} (team gets 241)"))
        rv = self.review(team)
        alerts += [(k, t) for k, t in rv["alerts"] if t.startswith("rotation depth")]
        unpriced = [r for r in avail if not r["priced"] and (r["minutes"] or 0) >= 20]
        if unpriced:
            alerts.append(("info", f"{len(unpriced)} rotation players unpriced"))
        score = sum({"bad": 10, "warn": 3, "info": 1}[k] for k, _ in alerts)
        team_fpts = float(self.proj[team]["fpts_mean"].sum()) if len(self.proj[team]) else 0.0
        assigned = sum((r["minutes"] or 0) for r in avail)
        to_place = (241.0 - assigned) if any(r["pinned"] for r in rows) else 0.0
        if abs(to_place) < 1.5:
            to_place = 0.0
        return dict(team=team, opp=next((p.opponent for p in self.players(team)), ""), tip=tip, mins_to_tip=mins_to_tip,
                    assigned=assigned, to_place=to_place, rotation_drawn=ROT.drawn_path(team).exists(), rotation=self.rotation(team), mapped=self.mapped_state(team),
                    questions=self.questions(team, now), scenarios=self.scenario_list(team),
                    published=dict(grid=ROT.drawn_source(team), pins=sum(1 for r in rows if "published" in r.get("layer", {}).values()),
                                   own=sum(1 for r in rows if "local" in r.get("layer", {}).values()), consumer=S.published_pins() is not None or (S.is_consumer() and S.published_rotations_dir().is_dir())),
                    last_game=self.game_grid(team), considerations=self.considerations(team), notes=self.team_notes(team),
                    changes=((getattr(self, "changes", None) or {}).get("teams") or {}).get(team, []), changes_at=(getattr(self, "changes", None) or {}).get("at"),
                    locked=bool(mins_to_tip is not None and mins_to_tip < 0), alerts=alerts, score=score,
                    starters="confirmed" if confirmed else ("expected" if starters else "none"),
                    n_out=sum(1 for r in rows if r["out"]), n_pinned=sum(1 for r in rows if r["pinned"]),
                    team_fpts=team_fpts, trailing=self.slate.team_fpts.get(team), rows=rows, info={r["key"]: r for r in rows},
                    n_book=sum(1 for r in rows if r.get("book_min") is not None))
