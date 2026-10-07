"""
The live slate: nba_player rows + starters + market rates + overrides in,
projections written back. Shared by project.py and news.py.
"""
from __future__ import annotations

from datetime import datetime
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


def read_overrides(path: Path | None = None) -> dict[str, float]:
    path = Path(path) if path else MINUTES_OVERRIDES_CSV
    if not path.exists():
        return {}
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    if "player" not in cols or "minutes" not in cols:
        raise SystemExit(f"{path}: need columns player,minutes (got {list(df.columns)})")
    return {normalize_name(str(r[cols["player"]])): float(r[cols["minutes"]]) for _, r in df.iterrows()}


def write_overrides(overrides: dict[str, float], path: Path | None = None) -> Path:
    path = Path(path) if path else MINUTES_OVERRIDES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"player": list(overrides), "minutes": list(overrides.values())}).to_csv(path, index=False)
    return path


class LiveSlate:
    def __init__(self, session: Session, use_starters: bool = True, use_market: bool = True):
        self.session = session
        NBABase.metadata.create_all(session.get_bind(), tables=[NBAMarketRate.__table__])
        self.box = load_box_scores(session)
        self.hist: HistoryState = build_history(self.box)
        horizon = datetime.now().date() + pd.Timedelta(days=1)
        self.state = self.hist.before(horizon)
        self.team_fpts = self.hist.team_before(horizon)
        self.starters: dict[str, set[str]] = {}
        if use_starters:
            for r in session.query(StartingLineups).all():
                if getattr(r, "confirmed", 1):
                    self.starters.setdefault(r.team, set()).add(normalize_name(r.player_name))
        self.market: dict[str, dict[str, float]] = {}
        if use_market:
            for r in session.query(NBAMarketRate).all():
                self.market.setdefault(r.player_name, {})[r.stat] = float(r.rate)
        self.players = session.query(NBAPlayer).all()

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
            pin = overrides.get(name)
            if pin is not None and pin <= 0:
                out = True
            starter = None if team_starters is None else (name in team_starters)
            roster.append(RosterPlayer(name, not out, starter, p.salary, None if out else pin,
                                       market_rates=self.market.get(name),
                                       rate_mult=(rate_mults or {}).get(name, 1.0)))
        return roster

    def project(self, team: str, overrides: dict[str, float], forced_out: set[str],
                params: ProjectionParams | None = None, rate_mults: dict[str, float] | None = None,
                rng: np.random.Generator | None = None) -> pd.DataFrame:
        return project_team(self.roster(team, overrides, forced_out, rate_mults), self.state, self.hist,
                            self.team_fpts.get(team), params or ProjectionParams(), rng=rng)

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
            else:
                p.fpts_mean = None; p.fpts_std = None; p.minutes_proj = 0.0; p.fpts_quantiles = None
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
