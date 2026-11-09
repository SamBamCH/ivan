# src/ivan/sports/nba/projection/record.py
"""
The record of a projection: one nba_projection_capture row per player (the rows the scorecard scores, with DraftKings' own
FPPG beside them as a yardstick) and the projections CSV artifact (ivan.core.artifacts).

    record_projection             the capture rows and the CSV for a set of team frames (project_slate's fresh frames, or the
                                  store's for a snapshot)
    record_external_projections   a third-party CSV (Stokastic) kept as capture rows under its own source
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.models import (NBAPlayer, NBAMarketRate, NBAProjectionCapture)
from ivan.sports.nba import slates as S
from ivan.sports.nba.projection.live import LiveSlate

Log = Callable[[str], None]


@dataclass
class ProjectInfo:
    players: int = 0
    priced: int = 0
    projections_path: Optional[str] = None
    captures: int = 0


def starter_label(slate: LiveSlate, name: str, team: str) -> str:
    st = slate.starters.get(team)
    if st is None:
        return "unknown"
    if name not in st:
        return "bench"
    return "confirmed" if slate.starters_confirmed.get(team) else "expected"


def history_context(slate: LiveSlate, name: str) -> dict:
    """Where a player's minutes baseline came from: his trailing state (team of the last game, games, minutes, start and play rates)."""
    st = slate.state
    if name not in st.index:
        return dict(hist_team=None, hist_games=0, hist_min=None, start_ewm=None, play_ewm=None)
    s = st.loc[name]
    return dict(hist_team=s["team"], hist_games=int(s["games"]), hist_min=float(s["min_ewm"]) * float(s["play_ewm"]),
                start_ewm=float(s["start_ewm"]), play_ewm=float(s["play_ewm"]))


def market_fpts_by_player(session: Session) -> dict[str, float]:
    """The books' implied fantasy points per priced player: the market means at DK weights (turnovers, never priced, left out)."""
    from ivan.sports.nba.scoring import DK_WEIGHTS
    out: dict[str, float] = {}
    for r in session.query(NBAMarketRate).all():
        out[r.player_name] = out.get(r.player_name, 0.0) + DK_WEIGHTS.get(r.stat, 0.0) * float(r.market_mean)
    return out


def record_projection(session: Session, slate: LiveSlate, frames: list, now: datetime, *, source: str = "ours", write_capture: bool = True,
                      write_csv: bool = True, write_ownership: bool = False, minutes_model: Optional[str] = None, log: Log = print,
                      info: Optional[ProjectInfo] = None) -> ProjectInfo:
    """The capture rows (nba_projection_capture, one per player, plus DK's own FPPG as a yardstick) and the projections CSV
    artifact for a set of team frames. Shared by project_slate (fresh frames) and capture_pool (the store's)."""
    info = info or ProjectInfo()
    allp = pd.concat(frames)
    info.players = int(len(allp)); info.priced = int(allp["priced"].sum()) if "priced" in allp else 0
    by_name = {normalize_name(p.name): p for p in slate.players}
    market_fpts = market_fpts_by_player(session)
    rows = []
    for name, r in allp.iterrows():
        p = by_name.get(name)
        if p is None:
            continue                                        # a stored frame from before a roster change: the player is gone
        rows.append(dict(player_id=p.id, player_name=name, team=p.team, opponent=p.opponent, gametime=p.gametime,
                         salary=p.salary, status=p.status, starter=starter_label(slate, name, p.team),
                         minutes_proj=float(r["minutes"]), fpts_mean=float(r["fpts_mean"]), fpts_std=None if pd.isna(r.get("fpts_std", np.nan)) else float(r["fpts_std"]),
                         priced=bool(r.get("priced", False)), pinned=bool(r.get("pinned", False)),
                         baseline_min=float(r["baseline"]), fpm=float(r["fpm"]), usage_mult=float(r["usage_mult"]),
                         market_fpts=market_fpts.get(name), minutes_model=minutes_model, minutes_role=r.get("role"),
                         p_out=None if r.get("p_out") is None else float(r["p_out"]),
                         minutes_p10=None if pd.isna(r.get("minutes_p10", np.nan)) else float(r["minutes_p10"]),
                         minutes_p90=None if pd.isna(r.get("minutes_p90", np.nan)) else float(r["minutes_p90"]),
                         **history_context(slate, name)))
    for p in slate.players:   # players ruled out: a row with no projection keeps the record of the scratch
        if normalize_name(p.name) not in allp.index:
            rows.append(dict(player_id=p.id, player_name=normalize_name(p.name), team=p.team, opponent=p.opponent,
                             gametime=p.gametime, salary=p.salary, status=p.status, starter="out",
                             minutes_proj=0.0, fpts_mean=None, fpts_std=None, priced=False, pinned=False,
                             baseline_min=None, fpm=None, usage_mult=None, market_fpts=market_fpts.get(normalize_name(p.name)),
                             **history_context(slate, normalize_name(p.name))))
    dg = next((p.draft_group_id for p in slate.players if p.draft_group_id), None)
    lock = min((p.gametime for p in slate.players if p.gametime), default=None)
    if write_ownership:
        from ivan.sports.nba.ingest import refresh_ownership       # the pipeline's (it imports this module)
        refresh_ownership(session, log=log)
    if write_capture:
        for r in rows:
            session.add(NBAProjectionCapture(captured_at=now, draft_group_id=dg, lock=lock, source=source, **r))
        # DK's own FPPG as a free yardstick, once per capture
        for p in slate.players:
            if p.dk_avg_fpts is not None:
                session.add(NBAProjectionCapture(captured_at=now, draft_group_id=dg, lock=lock, source="dk_avg",
                                                 player_id=p.id, player_name=normalize_name(p.name), team=p.team,
                                                 opponent=p.opponent, gametime=p.gametime, salary=p.salary,
                                                 status=p.status, fpts_mean=float(p.dk_avg_fpts)))
        session.commit()
        info.captures = len(rows)
    if write_csv:
        try:
            from ivan.core import artifacts
            path = artifacts.projections_path(S.output_dir(), now)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, index=False)
            info.projections_path = str(path)
        except Exception as e:   # noqa: BLE001
            log(f"WARNING: projections artifact not written: {type(e).__name__}: {e}")
    log(f"projected {info.players} players ({info.priced} priced); {info.captures} capture rows"
        f"{'; ' + info.projections_path if info.projections_path else ''}")
    return info


def record_external_projections(session: Session, csv_path: Path, source: str = "stokastic", log: Log = print) -> int:
    """Keep a third-party projection CSV (Player, Projection, Std Dev, Ownership %) as capture rows; never touches ours."""
    df = pd.read_csv(csv_path)
    by_name = {normalize_name(p.name): p for p in session.query(NBAPlayer).all()}
    now = datetime.now(); n = 0
    dg = next((p.draft_group_id for p in by_name.values() if p.draft_group_id), None)
    lock = min((p.gametime for p in by_name.values() if p.gametime), default=None)
    for _, r in df.iterrows():
        name = normalize_name(str(r.get("Player", "")))
        p = by_name.get(name)
        session.add(NBAProjectionCapture(captured_at=now, draft_group_id=dg, lock=lock, source=source,
                                         player_id=p.id if p else None, player_name=name, team=p.team if p else None,
                                         opponent=p.opponent if p else None, gametime=p.gametime if p else None,
                                         salary=p.salary if p else None, status=p.status if p else None,
                                         fpts_mean=float(r.get("Projection")) if pd.notna(r.get("Projection")) else None,
                                         fpts_std=float(r.get("Std Dev")) if pd.notna(r.get("Std Dev")) else None))
        n += 1
    session.commit()
    log(f"recorded {n} {source} projections ({sum(1 for _, r in df.iterrows() if normalize_name(str(r.get('Player',''))) in by_name)} matched the pool)")
    return n
