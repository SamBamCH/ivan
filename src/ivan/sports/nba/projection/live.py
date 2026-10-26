"""
The live slate: nba_player rows + starters + market rates + overrides in,
projections written back. Shared by project.py and news.py.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBAPlayer, StartingLineups, NBAMarketRate, NBABase
from ivan.sports.nba.paths import MINUTES_OVERRIDES_CSV
from .history import load_box_scores, build_history, HistoryState
from .engine import project_team, RosterPlayer, ProjectionParams

OUT_STATUSES = {"OUT", "O", "IR", "SUSP"}


OVERRIDE_KEYS = ("minutes", "fppm", "min_low", "min_high", "starter")      # starter: 1 / 0 overrides the Rotowire lineup for the player


def read_overrides_full(path: Path | None = None) -> dict[str, dict]:
    """player -> {minutes, fppm, min_low, min_high, starter} (any of them None): the hand overrides file (columns player, minutes[, fppm, min_low, min_high, starter])."""
    path = Path(path) if path else MINUTES_OVERRIDES_CSV
    if not path.exists():
        return {}
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    if "player" not in cols:
        raise SystemExit(f"{path}: need a player column (got {list(df.columns)})")
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        rec = {}
        for k in OVERRIDE_KEYS:
            v = r[cols[k]] if k in cols else None
            rec[k] = None if v is None or pd.isna(v) else float(v)
        if any(v is not None for v in rec.values()):
            out[normalize_name(str(r[cols["player"]]))] = rec
    return out


def read_overrides(path: Path | None = None) -> dict[str, float]:
    """player -> pinned minutes only (the older callers' view of the same file)."""
    return {k: v["minutes"] for k, v in read_overrides_full(path).items() if v.get("minutes") is not None}


def write_overrides(overrides: dict, path: Path | None = None) -> Path:
    """Write the overrides file. Values are a minutes pin (float) or a dict with any of minutes, fppm, min_low, min_high; floats keep the player's other columns from the file."""
    path = Path(path) if path else MINUTES_OVERRIDES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_overrides_full(path) if path.exists() else {}
    rows = []
    for name, v in overrides.items():
        rec = dict(old.get(name, {}))
        if isinstance(v, dict):
            rec.update({k: v.get(k) for k in OVERRIDE_KEYS})
        else:
            rec["minutes"] = None if v is None else float(v)
        if any(rec.get(k) is not None for k in OVERRIDE_KEYS):
            rows.append({"player": name, **{k: rec.get(k) for k in OVERRIDE_KEYS}})
    pd.DataFrame(rows, columns=["player", *OVERRIDE_KEYS]).to_csv(path, index=False)
    return path


def override_of(overrides: dict, name: str) -> dict:
    """One player's override as a full dict, whether the mapping holds floats (minutes) or dicts."""
    v = overrides.get(name)
    if v is None:
        return {}
    return dict(v) if isinstance(v, dict) else {"minutes": float(v)}


class LiveSlate:
    def __init__(self, session: Session, use_starters: bool = True, use_market: bool = True,
                 history: tuple[pd.DataFrame, HistoryState] | None = None):
        """history: (box, HistoryState) built earlier, to skip the box-score load and the history walk (a few seconds)."""
        self.session = session
        NBABase.metadata.create_all(session.get_bind(), tables=[NBAMarketRate.__table__])
        if history is not None:
            self.box, self.hist = history
        else:
            self.box = load_box_scores(session)
            self.hist = build_history(self.box)
        horizon = datetime.now().date() + pd.Timedelta(days=1)
        self.state = self.hist.before(horizon)
        self.team_fpts = self.hist.team_before(horizon)
        self.starters: dict[str, set[str]] = {}
        self.starters_confirmed: dict[str, bool] = {}
        self.starters_fetched_at = None
        if use_starters:
            for r in session.query(StartingLineups).all():
                # expected lineups count as the best guess; `starters_confirmed` says whether Rotowire confirmed them
                self.starters.setdefault(r.team, set()).add(normalize_name(r.player_name))
                self.starters_confirmed[r.team] = bool(r.confirmed)
                self.starters_fetched_at = r.fetched_at or self.starters_fetched_at
        self.market: dict[str, dict[str, float]] = {}
        self.market_means: dict[str, dict[str, float]] = {}
        if use_market:
            for r in session.query(NBAMarketRate).all():
                self.market.setdefault(r.player_name, {})[r.stat] = float(r.rate)
                self.market_means.setdefault(r.player_name, {})[r.stat] = float(r.market_mean)
        self.players = session.query(NBAPlayer).all()
        # games this season per player and per (player, team): the salary blend fades as they arrive
        season_start = date(horizon.year if horizon.month >= 10 else horizon.year - 1, 10, 1)
        this = self.box[(self.box.game_date >= season_start) & self.box.played]
        self.games_season = this.groupby("player_name").size().to_dict()
        self.games_team = this.groupby(["player_name", "team"]).size().to_dict()

    def teams(self) -> list[str]:
        return sorted({p.team for p in self.players})

    def roster(self, team: str, overrides: dict[str, float], forced_out: set[str],
               rate_mults: dict[str, float] | None = None) -> list[RosterPlayer]:
        team_starters = self.starters.get(team)
        roster = []
        for p in self.players:
            if p.team != team:
                continue
            name = normalize_name(p.name)
            status = (p.status or "").upper()
            out = status in OUT_STATUSES or name in forced_out
            ov = override_of(overrides, name)
            pin = ov.get("minutes")
            if pin is None and ov.get("min_low") is not None and ov.get("min_high") is not None:
                pin = 0.5 * (float(ov["min_low"]) + float(ov["min_high"]))      # a band alone says where his minutes are: its middle
            if pin is not None and pin <= 0:
                out = True
            starter = None if team_starters is None else (name in team_starters)
            if ov.get("starter") is not None:
                starter = bool(float(ov["starter"]) > 0)         # the desk's hand call beats Rotowire
            st = self.state.loc[name] if name in self.state.index else None
            roster.append(RosterPlayer(name, not out, starter, p.salary, None if out else pin,
                                       market_rates=self.market.get(name), market_means=self.market_means.get(name), position=p.position,
                                       rate_mult=(rate_mults or {}).get(name, 1.0), status=status or None,
                                       fpm_override=ov.get("fppm"), minutes_low=ov.get("min_low"), minutes_high=ov.get("min_high"),
                                       new_team=bool(st is not None and str(st["team"]) != str(p.team)),
                                       games_on_team=int(self.games_team.get((name, p.team), 0)), games_this_season=int(self.games_season.get(name, 0))))
        return roster

    def rotation_for(self, team: str, overrides: dict, forced_out: set[str]) -> dict | None:
        """
        The skeleton's inputs for project_team: every roster player's stint curve (absent players' curves are the
        vacated slots), the desk's locked rows, the boundary spreads. The grid itself is built inside project_team from
        the solve's minutes and comes back in this dict under "grid". None when nobody is available.
        """
        from .rotation import boundary_spreads, curves, load_locks
        from ivan.sports.nba.pbp import sub_shares
        roster = self.roster(team, overrides, forced_out)
        avail = [p.name for p in roster if p.available]
        if not avail:
            return None
        names = [p.name for p in roster]
        curve = curves(self.session, players=names, n_games=10)
        locks = load_locks(team)
        if locks is not None:
            locks = locks.loc[[n for n in locks.index if n in avail]]
            if not len(locks):
                locks = None
        try:
            shares = sub_shares(self.session, players=names)          # the coach's habit: who comes in for whom (recorded SUB events)
        except Exception:   # noqa: BLE001 - no substitution table yet
            shares = {}
        from .groups import rotation_depth
        try:
            from ivan.core.config import load_config
            block = (load_config("nba") or {}).get("projection") or {}
        except Exception:   # noqa: BLE001
            block = {}
        depth = None
        if block.get("depth_cut", True):
            depth = rotation_depth(self.box, team) + int(block.get("depth_extra", 0))
        pool, pool_info = None, None
        if str(block.get("pool", "depth_cut")) == "last_game":
            # the pool rule: the last ordinary game this group played together, minus the outs, plus the returners
            from .rotation import last_game_pool
            from datetime import date as _date
            today = _date.today(); season_start = _date(today.year if today.month >= 10 else today.year - 1, 10, 1)
            usual = {} if curve.empty else {str(n): float(curve.loc[n, "minutes"]) for n in curve.index}
            try:
                pool_info = last_game_pool(self.session, team, avail, usual, since=season_start)
            except Exception:   # noqa: BLE001 - no stints yet: the depth cut stands
                pool_info = None
            if pool_info is not None:
                pool, depth = pool_info["pool"], None
        return {"skeleton": True, "curve": curve if not curve.empty else None, "locks": locks,
                "vacated": [p.name for p in roster if not p.available and (curve.empty or p.name in curve.index)],
                "spreads": boundary_spreads(self.session, avail), "drawn": locks is not None, "shares": shares,
                "depth": depth, "pool": pool, "pool_info": pool_info}

    def group_targets(self, team: str) -> dict:
        """The team's trailing minutes by position group for the solve (groups.group_targets), from the box scores."""
        from .groups import group_targets, position_map
        if getattr(self, "_posmap", None) is None:
            try:
                self._posmap = position_map(self.session)
            except Exception:   # noqa: BLE001
                self._posmap = {}
        try:
            return group_targets(self.box, team, datetime.now().date(), self._posmap)
        except Exception:   # noqa: BLE001 - no targets is a plain solve
            return {}

    def spread(self, team: str) -> float | None:
        """The team's own consensus spread from game_odds (negative = favoured); None when unpriced."""
        if getattr(self, "_lines", None) is None:
            from .lines import GameLines
            try:
                self._lines = GameLines.load(self.session)
            except Exception:   # noqa: BLE001 - no lines table yet
                self._lines = GameLines(pd.DataFrame())
        return self._lines.team_spread(team)

    def project(self, team: str, overrides: dict[str, float], forced_out: set[str],
                params: ProjectionParams | None = None, rate_mults: dict[str, float] | None = None,
                rng: np.random.Generator | None = None) -> pd.DataFrame:
        params = params or ProjectionParams.from_config()
        if params.minutes_model in ("team", "rotation") and params.spread is None:
            params = replace(params, spread=self.spread(team))
        if params.alloc == "solve" and params.group_targets is None:
            params = replace(params, group_targets=self.group_targets(team))
        if params.minutes_model == "rotation" and params.rotation is None:
            params = replace(params, rotation=self.rotation_for(team, overrides, forced_out))
        if params.rotation is not None:
            self._last_rotation = {**getattr(self, "_last_rotation", {}), team: params.rotation}
        return project_team(self.roster(team, overrides, forced_out, rate_mults), self.state, self.hist,
                            self.team_fpts.get(team), params, rng=rng)

    def write(self, team: str, proj: pd.DataFrame) -> int:
        """Store a team's projection on nba_player; players not in proj (out) get None."""
        n = 0
        for p in self.players:
            if p.team != team:
                continue
            name = normalize_name(p.name)
            if name in proj.index:
                r = proj.loc[name]
                p.fpts_mean = float(r["fpts_mean"]); p.fpts_std = float(r["fpts_std"])
                p.minutes_proj = float(r["minutes"]); p.fpts_updated_at = datetime.now(); n += 1
                p.fpts_quantiles = r["quantiles"] if "quantiles" in r and isinstance(r["quantiles"], list) else None
                p.minutes_role = r.get("role"); p.p_out = None if r.get("p_out") is None else float(r["p_out"])
                p.rates = {k: float(v) for k, v in r["rates"].items()} if isinstance(r.get("rates"), dict) else None
                p.minutes_p10 = None if pd.isna(r.get("minutes_p10", np.nan)) else float(r["minutes_p10"])
                p.minutes_p90 = None if pd.isna(r.get("minutes_p90", np.nan)) else float(r["minutes_p90"])
                rot = getattr(self, "_last_rotation", {}).get(team)
                p.rotation_cells = [int(x) for x in rot["grid"].loc[name].values] if rot and name in rot["grid"].index else None
                p.boundary_sd = float(rot["spreads"][name]) if rot and name in (rot.get("spreads") or {}) else None
            else:
                p.fpts_mean = None; p.fpts_std = None; p.minutes_proj = 0.0; p.fpts_quantiles = None
                p.minutes_role = None; p.p_out = None; p.rates = None; p.minutes_p10 = None; p.minutes_p90 = None
                p.rotation_cells = None; p.boundary_sd = None
        return n


def basis_minutes_for_market(slate: LiveSlate, overrides: dict[str, float], forced_out: set[str]) -> dict[str, float]:
    """Projected minutes under today's availability: the denominator for market rates."""
    out = {}
    for team in slate.teams():
        proj = project_team(slate.roster(team, overrides, forced_out), slate.state, slate.hist,
                            slate.team_fpts.get(team), ProjectionParams(market_weight=0.0), simulate=False)
        out.update(proj["minutes"].to_dict())
    return out


def store_market_rates(session: Session, means: pd.DataFrame, basis: dict[str, float]) -> int:
    from .market import market_rates
    mr = market_rates(means, basis)
    session.query(NBAMarketRate).delete(synchronize_session=False)
    now = datetime.now(); n = 0
    by_key = {(r.player, r.stat): r for r in means.itertuples(index=False)}
    for player, rates in mr.rates.items():
        for stat, rate in rates.items():
            m = by_key[(player, stat)]
            session.add(NBAMarketRate(player_name=player, stat=stat, market_mean=float(m.mean), rate=float(rate),
                                      basis_minutes=mr.basis_minutes[player], books=int(m.books),
                                      line=None if pd.isna(m.line) else float(m.line), fetched_at=now))
            n += 1
    session.commit()
    return n
