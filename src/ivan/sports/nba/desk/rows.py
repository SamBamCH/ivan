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
from ivan.sports.nba.backtest.validation import market_points, recent_minutes, review_team

STAT_ORDER = ("points", "rebounds", "assists", "fg3m", "steals", "blocks", "turnovers")      # the grid's stat columns, left to right
STAT_LABELS = {"points": "pts", "rebounds": "reb", "assists": "ast", "fg3m": "3pm", "steals": "stl", "blocks": "blk", "turnovers": "tov"}
ALT_SPAN = 3.0                                                                               # alternates shown in the hover: within this many of the line, per book
STAT_GAP = 0.15                                                                              # a stat cell is amber when our projection sits this far (share) from the books' mean, at least 1


CHART_BINS = 16            # the stat chart's bars: a stat whose range is wider is binned (points in twos or threes)
CHART_MIN_GAMES = 5        # fewer games than this in the season and last season's are added to the history


def stat_chart(history: list, shares: Optional[list], line: Optional[float] = None) -> dict:
    """
    The stat hover's chart: the player's own games and our drawn distribution over the same bins. `history` = the stat in each
    game he played, `shares` = the World's share of worlds at each whole value (projection.worlds.stat_shares; None when the
    record has none). Whole values, binned in equal steps when the range passes CHART_BINS. Returns {bins: [{lo, hi, label,
    hist, ours}] (shares of games / of worlds), step, ymax, n_hist, hist_mean, ours_mean, line_at (the books' line as a
    position in bins from the left edge, None when off the chart), has_ours}.
    """
    h = [int(round(float(x))) for x in history if x is not None and not pd.isna(x)]
    p = [float(x) for x in (shares or [])]
    tail = 0.0
    while len(p) > 1 and tail + p[-1] < 0.01:                 # the last 1% of the worlds: a tail the eye cannot see that only widens the bins
        tail += p.pop()
    top = max([max(h) if h else 0, len(p) - 1, int(np.ceil(line)) if line is not None else 0, 1])
    step = int(np.ceil((top + 1) / CHART_BINS))
    n_bins = int(np.ceil((top + 1) / step))
    bins = []
    for k in range(n_bins):
        lo, hi = k * step, (k + 1) * step - 1
        bins.append(dict(lo=lo, hi=hi, label=str(lo) if step == 1 else f"{lo}-{hi}",
                         hist=(sum(1 for x in h if lo <= x <= hi) / len(h)) if h else 0.0, ours=float(sum(p[lo:hi + 1]))))
    ymax = max([b["hist"] for b in bins] + [b["ours"] for b in bins] + [0.05])
    tot = sum(p)
    return dict(bins=bins, step=step, ymax=ymax, n_hist=len(h), hist_mean=(sum(h) / len(h)) if h else None,
                ours_mean=(sum(i * x for i, x in enumerate(p)) / tot) if tot > 0 else None, has_ours=tot > 0,
                line_at=None if line is None else (float(line) + 0.5) / step)


def _split_props(rows: list, line) -> tuple[list, list, int]:
    """The hover's two parts from every (book, line) on file: the main lines by line, lowest first (one row each: posted, book, line,
    over, under; a row posted before the player's status moved is kept and flagged `stale`) and the alternates condensed to one row per
    book, the lines within ALT_SPAN of the consensus line (the median main line when there is none) with their over price, and how
    many more the book posts; stale alternates are left out. An alternate market prices overs only. Third: the stale count."""
    main = sorted((b for b in rows if not b["alt"]), key=lambda b: (float(b["line"]) if b.get("line") is not None else 0.0, str(b["book"])))
    alts = [b for b in rows if b["alt"] and not b.get("stale")]
    n_stale = sum(1 for b in rows if b.get("stale"))
    if not alts:
        return main, [], n_stale
    centre = float(line) if line is not None and not pd.isna(line) else (float(np.median([b["line"] for b in main])) if main else float(np.median([b["line"] for b in alts])))
    out = []
    for book in sorted({b["book"] for b in alts}):
        mine = sorted((b for b in alts if b["book"] == book), key=lambda b: b["line"])
        near = [b for b in mine if abs(b["line"] - centre) <= ALT_SPAN]
        out.append(dict(book=book, text=" · ".join(f"{b['line']:g} {b['over']}" for b in near), more=len(mine) - len(near), date=max((b["date"] or "") for b in mine)))
    return main, out, n_stale
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.book_minutes import book_minutes
from .state import MARKET_GAP, MINUTES_DEV, NEAR_TIP_MIN, ROTATION_MIN
from ivan.sports.nba import slates as S


class RowsMixin:
    def considerations(self, team: str) -> list:
        """What is unusual about the team's game tonight (ivan.sports.nba.slate.considerations; scripts/nba/considerations.py)."""
        from ivan.sports.nba.slate import considerations as CN
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
        """The rotation review (ivan.sports.nba.backtest.validation.review_team): provenance of every player's minutes and the flags."""
        if getattr(self, "_recent", None) is None or self._recent_stamp != self.pool_stamp:
            self._recent = recent_minutes(self.slate.box); self._recent_stamp = self.pool_stamp
        return review_team(self.slate, team, self.proj[team], self._recent, market_points(self.session), self.pins())

    def recent_shaded(self, name: str) -> list[dict]:
        """The player's recent games (Desk.recent) with each game's context from his team's side that night (game_context): margin,
        shade ("g-blow" a blowout, "g-inj" a regular absent or a rotation player gone early, his own absence not counted) and why."""
        out = []
        for g in self.recent.get(name, []):
            c = self.game_context(g["team"]).get(g.get("game_id")) or {}
            others = [n for n in (c.get("absent") or []) if normalize_name(n) != normalize_name(name)]
            inj = bool(others or [n for n in (c.get("left_early") or []) if normalize_name(n) != normalize_name(name)])
            shade = " ".join(x for x, on in (("g-blow", c.get("not_close")), ("g-inj", inj)) if on)
            out.append(dict(g, margin=c.get("margin"), shade=shade, why=c.get("why", "") if shade else ""))
        return out

    def team_rows(self, team: str) -> list[dict]:
        proj, hist = self.proj[team], self.hist_only[team]
        review_flags = {r["name"]: r["flags"] for r in self.review(team)["rows"]}
        state = self.slate.state
        starters = self.starters_of(team)
        confirmed = self.slate.starters_confirmed.get(team, False)
        mk = self.market_means()
        sib = self.sibling_changes()
        elsewhere = {c["key"]: c for c in (sib["changes"] if sib else [])}
        rows = []
        for p in self.players(team):
            name = normalize_name(p.name)
            s = state.loc[name] if name in state.index else None
            r = proj.loc[name] if name in proj.index else None
            fringe = bool(r is not None and r.get("fringe", False))
            fr_p = None if not fringe else float(r.get("fringe_p") or 0.0); fr_min = None if not fringe else float(r.get("fringe_minutes") or 0.0)
            r = None if fringe else r                                   # a fringe row is "no minutes" on the card; its chance rides along
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
            if name in elsewhere and (elsewhere[name]["down"] or (hist_min and hist_min >= ROTATION_MIN)):
                flags.append(f"{elsewhere[name]['there']} on the {sib['label']} slate's pool, refreshed {sib['refreshed_at']:%H:%M} (here {elsewhere[name]['here']}): refresh this slate")
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
            if r is not None and not pd.isna(r.get("p_cut", np.nan)) and float(r["p_cut"]) >= 0.3 and float(r["minutes"]) >= 10:
                flags.append(f"cut-prone: {float(r['p_cut']):.0%} of nights under 8 min")
            if r is not None and r.get("minutes_p90") is not None and not pd.isna(r.get("minutes_p90")) and float(r["minutes_p90"]) > 48:
                flags.append(f"impossible band: p90 {float(r['minutes_p90']):.0f} min")
            if r is not None and not pd.isna(r.get("fpts_std", np.nan)) and float(r["fpts_mean"]) >= 8 and float(r["fpts_std"]) > 1.2 * float(r["fpts_mean"]):
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
            fpts = None if r is None else float(r["fpts_mean"])                       # the World's mean (a hand fpts pin over it)
            sd = None if r is None or pd.isna(r.get("fpts_std", np.nan)) else float(r["fpts_std"])
            fm = None if r is None else r.get("fpts_model", np.nan)
            fpts_model = None if r is None else (float(fm) if fm is not None and not pd.isna(fm) else fpts)
            plan = None if r is None else float(r["minutes"]) * float(r["fpm"])         # the mapping's total: minutes x fantasy points per minute
            q = None if r is None else r.get("quantiles")
            q = q if isinstance(q, list) and len(q) == 101 else None
            min_sim = None if r is None or pd.isna(r.get("minutes_sim", np.nan)) else float(r["minutes_sim"])
            if fringe and name in proj.index and not pd.isna(proj.loc[name].get("minutes_sim", np.nan)):
                min_sim = float(proj.loc[name]["minutes_sim"])              # what the World drew for him across his play-worlds
            p_cut = None if r is None or pd.isna(r.get("p_cut", np.nan)) else float(r["p_cut"])
            p_zero = None if r is None or pd.isna(r.get("p_zero", np.nan)) else float(r["p_zero"])
            opt = self.optimal_of(p.id)
            ov = override_of(self.overrides, name)
            mk = self.market_rows.get(name, {})
            stats = []
            for st in STAT_ORDER:                                   # one entry per stat column, out or not (the cells must line up)
                m = mk.get(st, {}); pv = None if r is None else float(r[f"proj_{st}"]); mean = m.get("mean")
                sv = None if r is None or pd.isna(r.get(f"sim_{st}", np.nan)) else float(r[f"sim_{st}"])
                stats.append(dict(stat=st, label=STAT_LABELS[st], proj=pv, sim=sv, line=m.get("line"), mean=mean, books=m.get("books"), odds=m.get("odds"),
                                  priced=st in mk, basis=m.get("basis"),
                                  pinned=st in (list(r.get("stat_pinned") or []) if r is not None else []),
                                  gap=bool(r is not None and st in mk and mean and pv is not None and abs(pv - float(mean)) >= max(STAT_GAP * float(mean), 1.0))))
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
                             fpts_pin=ov.get("fpts"), pub_pin=ov.get("pub"), fpts_model=fpts_model, fpts_plan=plan,
                             p10=None if q is None else q[10], p50=None if q is None else q[50], p90=None if q is None else q[90], min_sim=min_sim,
                             opt=opt.get("share"), opt_mean=bool(opt.get("in_mean")), p_cut=p_cut, p_zero=p_zero, fringe=fringe, fringe_p=fr_p, fringe_minutes=fr_min,
                             minutes_sd=None if r is None or r.get("minutes_sd") is None or pd.isna(r.get("minutes_sd")) else float(r["minutes_sd"]),
                             layer=self._layer_of(name), local=dict(self.local_overrides.get(name, {})), published_pins=self._published_of(name),
                             fpts=fpts, sd=sd,
                             fpm=None if r is None else float(r["fpm"]),
                             season_games=cur["games"] if cur else 0, season_min=cur["minutes"] if cur else None, season_fpm=cur["fpm"] if cur else None,
                             last_games=last_season["games"] if last_season else 0, last_min=last_season["minutes"] if last_season else None,
                             last_fpm=last_season["fpm"] if last_season else None,
                             value=None if fpts is None or not p.salary else fpts / (p.salary / 1000.0),
                             own=None if p.ownership in (None, 0) else float(p.ownership), pub=getattr(p, "pub_fpts", None),
                             hist_fpts=None if h is None else float(h["fpts_mean"]), market=market,
                             dk_avg=p.dk_avg_fpts, priced=bool(r is not None and r.get("priced")), recent=self.recent_shaded(name),
                             team=p.team, opp=p.opponent,
                             usage=None if r is None else float(r["usage_mult"]), flags=flags,
                             cells={k: " · ".join(v) for k, v in cells.items()}, loud=loud))
        # starters first, then the players who are out, then the rest of the bench; minutes descending within each group
        rows.sort(key=lambda x: (0 if (x["starter"] == "S" and not x["out"]) else (1 if x["out"] else 2), -(x["minutes"] or 0), -(x["salary"] or 0)))
        return rows

    def stat_hover(self, team: str, name: str, stat: str) -> dict:
        """The stat cell's hover (fetched when the cell is hovered): a chart of the player's own games against our drawn distribution
        (stat_chart: his games played this season, last season's added when there are under CHART_MIN_GAMES; ours from the World's
        record), the books' line on it; beside it ours, the books' mean over the minutes it was priced at, the consensus line and
        prices, each book's main line (set-aside lines flagged stale), how many lines are set aside and when he was last downgraded."""
        m = self.market_rows.get(name, {}).get(stat, {})
        r = self.proj.get(team)
        pv = None if r is None or name not in r.index else float(r.loc[name, f"proj_{stat}"])
        p = next((x for x in self.players(team) if normalize_name(x.name) == name), None)
        main, alts, n_stale = _split_props(self.prop_rows_for(name, stat), m.get("line"))
        box = self.slate.box
        mine = box[(box.player_name == name) & box.played]
        today = datetime.now().date(); start = datetime(today.year if today.month >= 10 else today.year - 1, 10, 1).date()
        season = mine[mine.game_date >= start]
        span = "this season"
        if len(season) < CHART_MIN_GAMES and len(mine) > len(season):
            season = mine[mine.game_date >= datetime(start.year - 1, 10, 1).date()]; span = "this season and last"
        dist = None if r is None or name not in r.index or "stat_dist" not in r.columns else r.loc[name, "stat_dist"]
        chart = stat_chart(list(season[stat]) if stat in season.columns else [], dist.get(stat) if isinstance(dist, dict) else None, m.get("line"))
        chart.update(span=span, hist_min=float(season.minutes.mean()) if len(season) else None,
                     ours_min=None if r is None or name not in r.index else float(r.loc[name, "minutes"]))
        return dict(stat=stat, label=STAT_LABELS.get(stat, stat), proj=pv, mean=m.get("mean"), basis=m.get("basis"), line=m.get("line"), odds=m.get("odds"), chart=chart,
                    rows=main, alts=alts, stale=n_stale,
                    status_at=None if p is None or p.status_changed_at is None else p.status_changed_at.strftime("%H:%M"))

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
                elif " slate's pool, refreshed " in f:                      # a status a fresher sibling slate knows: bad when it rules him out
                    alerts.append(("bad" if f.split(" on the ")[0] in ("OUT", "O", "IR", "SUSP") else "warn", f"{r['name']}: {f}"))
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
        plan_fpts = float((self.proj[team]["minutes"] * self.proj[team]["fpm"]).sum()) if len(self.proj[team]) else 0.0
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
                    team_fpts=team_fpts, plan_fpts=plan_fpts, trailing=self.slate.team_fpts.get(team), rows=rows, info={r["key"]: r for r in rows},
                    world=self.world.get(team) or {}, optimal=self.optimal_status(),
                    stat_totals={st: sum((s["proj"] or 0.0) for x in rows if not x["out"] for s in x["stats"] if s["stat"] == st) for st in STAT_ORDER},
                    n_book=sum(1 for r in rows if r.get("book_min") is not None))
