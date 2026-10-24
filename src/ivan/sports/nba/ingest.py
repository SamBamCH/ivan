"""
NBA slate ingest: fetch, store, project. Mirrors ivan.sports.nfl.ingest.

fetch_slate   DraftKings draftables for a draft group (raw dk/draftables), the
              Rotowire lineups page (raw rotowire/lineups), and optionally the
              player props for the slate's teams (raw odds/events, odds/props).
store_slate   nba_player (whole pool, DK statuses, DK's own FPPG), starting_lineups
              with fetched_at, odds + nba_market_rate, and an archive capture.
project_slate the minutes-anchored projection (projection/live.py) on the pool,
              a projections CSV artifact, and one nba_projection_capture row per
              player so accuracy can be read by offset before lock.

Nothing here prompts; scripts/nba/ingest.py and capture.py pass the draft group.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.core.archive import archive_slate, ensure_tables as ensure_archive_tables
from ivan.core.config import require_env
from ivan.core.utils import PROJECT_ROOT, normalize_name
from ivan.database.session import add_missing_columns
from ivan.sports.common.draftkings import (get_players_for_slate, _derive_opponent_for_player,
                                           _parse_gametime_to_local, list_slates)
from ivan.sports.common.odds import get_game_odds, get_odds, parse_game_odds, parse_odds
from ivan.sports.nba.historical.data_fetch import fetch_rotowire_html, parse_rotowire, store_starting_lineups
from ivan.sports.nba.models import (NBABase, NBAGameOdds, NBAPlayer, NBAOdds, NBAMarketRate, NBAProjectionCapture, StartingLineups)
from ivan.sports.nba.paths import INPUT_DIR, MINUTES_OVERRIDES_CSV, OUTPUT_DIR
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, basis_minutes_for_market, read_overrides_full, store_market_rates
from ivan.sports.nba.projection.market import NBA_MARKETS, market_means

Log = Callable[[str], None]
ODDS_SPORT = "basketball_nba"


class IngestError(RuntimeError):
    pass


def ensure_schema(engine) -> None:
    NBABase.metadata.create_all(engine)
    add_missing_columns(engine, NBABase.metadata)
    ensure_archive_tables(engine)


# ------------------------------------------------------------------------------ slates
def classic_slates(day: Optional[datetime] = None) -> list[dict]:
    """Today's (or `day`'s) classic NBA draft groups from the DK lobby: id, start, name, contests."""
    out = []
    for s in list_slates("NBA"):
        if str(s.get("game_type", "")).lower() != "classic":
            continue
        # DK's start is a display string ("Tue 2:00PM"); the lock comes from the draftables' game times
        out.append({"id": int(s["id"]), "start": s.get("start"), "name": s.get("name"), "contests": s.get("contests")})
    return out


# ------------------------------------------------------------------------------ fetch
@dataclass
class SlateFetch:
    draft_group_id: int
    players: list[dict] = field(default_factory=list)      # rows for nba_player
    lock: Optional[datetime] = None
    starters: Optional[pd.DataFrame] = None
    odds: Optional[pd.DataFrame] = None
    game_lines: Optional[pd.DataFrame] = None               # parse_game_odds rows for the slate's games
    teams: set = field(default_factory=set)
    fetched_at: datetime = field(default_factory=datetime.now)


def _fppg(d: dict) -> Optional[float]:
    for a in d.get("draftStatAttributes") or []:
        try:
            if int(a.get("id", -1)) == 90:
                return float(a.get("value"))
        except (TypeError, ValueError):
            continue
    for a in d.get("draftStatAttributes") or []:
        try:
            return float(a.get("value"))
        except (TypeError, ValueError):
            continue
    return None


def pool_from_draftables(draftables: list[dict], draft_group_id: int) -> tuple[list[dict], set]:
    """One row per player (first roster slot), DK status and FPPG kept."""
    rows, seen, teams = [], set(), set()
    for d in draftables:
        name = normalize_name(d.get("displayName") or "")
        if not name or name in seen:
            continue
        if d.get("draftableId") is None or d.get("salary") is None or d.get("position") is None:
            continue
        seen.add(name)
        team = d.get("teamAbbreviation"); comp = d.get("competition") or {}
        opp = _derive_opponent_for_player(team, comp)
        teams.update(t for t in (team, opp) if t)
        gt = _parse_gametime_to_local(comp.get("startTime"))
        if gt is not None and getattr(gt, "tzinfo", None) is not None:
            gt = gt.replace(tzinfo=None)          # naive local time, like every other datetime in the DB
        rows.append(dict(id=int(d["draftableId"]), name=name, team=team, opponent=opp, position=d["position"],
                         roster_slot_id=d.get("rosterSlotId"), gametime=gt,
                         salary=int(d["salary"]), status=d.get("status"), dk_avg_fpts=_fppg(d),
                         draft_group_id=int(draft_group_id)))
    return rows, teams


def fetch_slate(session: Session, draft_group_id: int, *, update_odds: bool, starters: bool = True,
                api_key: Optional[str] = None, log: Log = print) -> SlateFetch:
    out = SlateFetch(draft_group_id=int(draft_group_id))
    log(f"fetching DK draftables for draft group {draft_group_id}")
    draftables = get_players_for_slate(int(draft_group_id), raw_sport="nba")
    out.players, out.teams = pool_from_draftables(draftables, draft_group_id)
    if not out.players:
        raise IngestError("DraftKings returned no players for that draft group; nothing changed.")
    times = [p["gametime"] for p in out.players if p.get("gametime")]
    out.lock = min(times) if times else None
    if starters:
        try:
            html = fetch_rotowire_html()
            raw.store("nba", "rotowire/lineups", datetime.now().strftime("%Y-%m-%d"), html, ext="html")
            out.starters = parse_rotowire(html)
            log(f"rotowire: {len(out.starters)} starters listed, "
                f"{int(out.starters.confirmed.sum()) if len(out.starters) else 0} confirmed")
        except Exception as e:   # noqa: BLE001 - starters are optional
            log(f"WARNING: rotowire fetch failed: {type(e).__name__}: {e}")
    if update_odds:
        key = api_key or require_env("ODDS_API_KEY")
        odds = get_odds(ODDS_SPORT, key, sorted(out.teams), markets=NBA_MARKETS, raw_sport="nba")
        df = parse_odds(odds)
        if df is None or len(df) == 0:
            raise IngestError("the-odds-api returned no NBA props; check ODDS_API_KEY and the plan. Nothing changed.")
        out.odds = df
        log(f"props: {len(df)} rows, {df.player.nunique()} players, {df.sportsbook.nunique()} books")
    return out


def fetch_game_lines(api_key: Optional[str] = None, log: Log = print) -> Optional[pd.DataFrame]:
    """Moneyline, spread and total for every upcoming NBA game (raw odds/lines; ~3 credits). None when the API returns nothing."""
    key = api_key or require_env("ODDS_API_KEY")
    payload, _ = get_game_odds(ODDS_SPORT, key, markets="h2h,spreads,totals", raw_sport="nba")
    df = parse_game_odds(payload) if payload else None
    if df is None or df.empty:
        log("WARNING: the-odds-api returned no NBA game lines")
        return None
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert("America/Chicago").dt.tz_localize(None)
    log(f"game lines: {df.game_id.nunique()} games, {len(df)} rows, {df.bookmaker.nunique()} books")
    return df


GAME_LINE_COLS = ["game_id", "sport_key", "commence_time", "home_team", "away_team", "home_abbr", "away_abbr", "bookmaker", "market",
                  "last_update", "outcome_name", "side", "point", "price"]


def store_game_lines(session: Session, lines: pd.DataFrame, fetched_at: Optional[datetime] = None) -> int:
    """Replace the stored lines of every game in `lines`. Returns rows stored."""
    if lines is None or lines.empty:
        return 0
    games = lines["game_id"].unique().tolist()
    session.query(NBAGameOdds).filter(NBAGameOdds.game_id.in_(games)).delete(synchronize_session=False)
    rows = lines[GAME_LINE_COLS].to_dict("records")
    now = fetched_at or datetime.now()
    for r in rows:
        r["fetched_at"] = now
        r["price"] = None if pd.isna(r.get("price")) else int(r["price"])
        r["point"] = None if pd.isna(r.get("point")) else float(r["point"])
        r["commence_time"] = None if pd.isna(r.get("commence_time")) else pd.Timestamp(r["commence_time"]).to_pydatetime()
    session.bulk_insert_mappings(NBAGameOdds, rows)
    session.commit()
    return len(rows)


# ------------------------------------------------------------------------------ store
@dataclass
class StoreInfo:
    players: int = 0
    starters: int = 0
    odds: int = 0
    market_rates: int = 0
    game_lines: int = 0
    capture_id: Optional[int] = None


@contextmanager
def one_transaction(session: Session):
    """
    Everything inside commits once, at the end: the pool is replaced, projected and its public side refreshed before any
    other connection sees a row of it (SQLite WAL readers keep the old pool until the commit). The functions inside call
    session.commit() as they always did; here that becomes a flush. On an error everything rolls back and the old pool stays.
    """
    real_commit = session.commit
    session.commit = session.flush            # type: ignore[method-assign]
    try:
        yield
        session.commit = real_commit          # type: ignore[method-assign]
        real_commit()
    except BaseException:
        session.commit = real_commit          # type: ignore[method-assign]
        session.rollback()
        raise


def retire_edits(old_draft_group: int, log: Log = print) -> None:
    """
    Edits are per slate: when the pool moves to another slate the minute / rate / starter pins and the drawn rotation grids
    are filed under data/nba/input/edits/<old draft group>/ and the live files start empty (a pin on a name would otherwise
    follow him to every later slate; a drawn grid from one night stretched a player's draw the next, 2026-10-23).
    """
    import shutil
    from ivan.sports.nba.projection.rotation import ROTATIONS_DIR
    keep = INPUT_DIR / "edits" / str(old_draft_group)
    moved = 0
    if MINUTES_OVERRIDES_CSV.exists():
        keep.mkdir(parents=True, exist_ok=True)
        shutil.move(str(MINUTES_OVERRIDES_CSV), str(keep / MINUTES_OVERRIDES_CSV.name)); moved += 1
    if ROTATIONS_DIR.exists():
        for f in ROTATIONS_DIR.glob("*.json"):
            keep.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(keep / f.name)); moved += 1
    if moved:
        log(f"slate changed: {moved} edit file(s) of slate {old_draft_group} filed under {keep.relative_to(PROJECT_ROOT)}; the new slate starts clean")


def refresh_pool(session: Session, draft_group_id: int, *, update_odds: bool = False, starters: bool = True, n_sims: int = 1000,
                 log: Log = print) -> tuple:
    """
    The desk's Refresh and slate switch: fetch DraftKings (and the props with update_odds), then replace the pool, project it
    and refresh its public side in ONE transaction. Nothing downstream can see a half-written or unprojected pool
    (2026-10-23: a sim started during a rewrite found no point guard). Returns (StoreInfo, ProjectInfo).
    """
    fetched = fetch_slate(session, int(draft_group_id), update_odds=update_odds, starters=starters, log=log)
    current = session.query(NBAPlayer.draft_group_id).first()
    if current and current[0] and int(current[0]) != int(draft_group_id):
        retire_edits(int(current[0]), log=log)                 # a new slate starts clean: pins and drawn rotations belong to the old one
    with one_transaction(session):
        info = store_slate(session, fetched, log=log)
        pinfo = project_slate(session, n_sims=n_sims, log=log)
    return info, pinfo


def store_slate(session: Session, fetched: SlateFetch, *, log: Log = print) -> StoreInfo:
    info = StoreInfo()
    # the pool: replace in full (one slate at a time in nba_player)
    session.query(NBAPlayer).delete(synchronize_session=False)
    for row in fetched.players:
        session.add(NBAPlayer(**row))
    session.commit()
    info.players = len(fetched.players)
    if fetched.starters is not None:
        info.starters = store_starting_lineups(session, fetched.starters, fetched_at=fetched.fetched_at)
    if fetched.odds is not None:
        # refresh the players just fetched; other players keep their latest lines (a fetch for one
        # slate's teams must not strip the market rates of another slate's games)
        players = [str(x) for x in fetched.odds.player.dropna().unique()]
        for i in range(0, len(players), 500):
            session.query(NBAOdds).filter(NBAOdds.player.in_(players[i:i + 500])).delete(synchronize_session=False)
        session.bulk_insert_mappings(NBAOdds, fetched.odds.to_dict("records"))
        session.commit()
        info.odds = len(fetched.odds)
    if fetched.game_lines is not None and len(fetched.game_lines):
        info.game_lines = store_game_lines(session, fetched.game_lines, fetched_at=fetched.fetched_at)
    # market rates for this pool from whatever props are stored (fresh or an earlier fetch today)
    odds_df = pd.read_sql("select player, market, sportsbook, side, point, price, last_update from odds", session.connection())
    if len(odds_df):
        pool = {r["name"] for r in fetched.players}
        odds_df = odds_df[odds_df.player.isin(pool)]
        if len(odds_df):
            means = market_means(odds_df)
            slate = LiveSlate(session, use_market=False)
            basis = basis_minutes_for_market(slate, read_overrides_full(), set())
            info.market_rates = store_market_rates(session, means, basis)
    refresh_ownership(session, log=log)                     # the public projection, the field and ownership: the pool or props changed
    # archive capture: players carry their starter flag so a replay can restore it
    starters = {}
    for r in session.query(StartingLineups).all():
        starters[r.player_name] = "confirmed" if r.confirmed else "expected"
    prow = [dict(r, starter=starters.get(r["name"], "bench")) for r in fetched.players]
    odds_rows = fetched.odds.to_dict("records") if fetched.odds is not None else None
    line_rows = None
    if fetched.game_lines is not None and len(fetched.game_lines):
        mine = fetched.game_lines[fetched.game_lines.home_abbr.isin(fetched.teams) | fetched.game_lines.away_abbr.isin(fetched.teams)]
        line_rows = mine[GAME_LINE_COLS].to_dict("records") if len(mine) else None
    try:
        cap = archive_slate(session, sport="nba", slate_key=str(fetched.draft_group_id), lock=fetched.lock or datetime.now(),
                            mode="classic", players=prow, odds=odds_rows, game_lines=line_rows, source="ingest", captured_at=fetched.fetched_at)
        info.capture_id = cap.id
    except Exception as e:   # noqa: BLE001
        log(f"WARNING: archive capture failed: {type(e).__name__}: {e}")
    log(f"stored {info.players} players, {info.starters} starters, {info.odds} prop rows, "
        f"{info.market_rates} market rates, {info.game_lines} line rows; capture {info.capture_id}")
    return info


# ------------------------------------------------------------------------------ project
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
    from ivan.sports.nba.projection.rates import DK_WEIGHTS
    out: dict[str, float] = {}
    for r in session.query(NBAMarketRate).all():
        out[r.player_name] = out.get(r.player_name, 0.0) + DK_WEIGHTS.get(r.stat, 0.0) * float(r.market_mean)
    return out


def refresh_ownership(session: Session, log: Log = print) -> dict:
    """The public projection, the field and ownership from the pool and props as stored (public_field.refresh_public).
    Runs with every pool refresh (store_slate); Publish never touches it, so our edits cannot move ownership."""
    try:
        from ivan.sports.nba.public_field import refresh_public
        slate = LiveSlate(session, use_market=False)
        return refresh_public(session, slate.hist, log=log)
    except Exception as e:   # noqa: BLE001 - never fails a pool refresh
        session.rollback(); log(f"public refresh failed: {type(e).__name__}: {e}"); return {}


def project_slate(session: Session, *, n_sims: int = 4000, write_capture: bool = True, write_csv: bool = True,
                  log: Log = print, source: str = "ours", params: Optional[ProjectionParams] = None, write_ownership: bool = False) -> ProjectInfo:
    info = ProjectInfo()
    slate = LiveSlate(session)
    if not slate.players:
        raise IngestError("no players in nba_player")
    overrides = read_overrides_full()
    params = params or ProjectionParams.from_config(n_sims=n_sims)
    rng = np.random.default_rng(0)
    now = datetime.now()
    frames = []
    for team in slate.teams():
        proj = slate.project(team, overrides, set(), params, rng=rng)
        slate.write(team, proj)
        proj = proj.copy(); proj["team"] = team
        frames.append(proj)
    session.commit()
    allp = pd.concat(frames)
    info.players = int(len(allp)); info.priced = int(allp["priced"].sum()) if "priced" in allp else 0
    by_name = {normalize_name(p.name): p for p in slate.players}
    market_fpts = market_fpts_by_player(session)
    rows = []
    for name, r in allp.iterrows():
        p = by_name[name]
        rows.append(dict(player_id=p.id, player_name=name, team=p.team, opponent=p.opponent, gametime=p.gametime,
                         salary=p.salary, status=p.status, starter=starter_label(slate, name, p.team),
                         minutes_proj=float(r["minutes"]), fpts_mean=float(r["fpts_mean"]), fpts_std=float(r["fpts_std"]),
                         priced=bool(r.get("priced", False)), pinned=bool(r.get("pinned", False)),
                         baseline_min=float(r["baseline"]), fpm=float(r["fpm"]), usage_mult=float(r["usage_mult"]),
                         market_fpts=market_fpts.get(name), minutes_model=params.minutes_model, minutes_role=r.get("role"),
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
            path = artifacts.projections_path(OUTPUT_DIR, now)
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


# ------------------------------------------------------------------------------ all
def run_ingest(session, draft_group_id: int, *, update_odds: bool, project: bool = True, starters: bool = True,
               n_sims: int = 4000, log: Log = print) -> dict:
    fetched = fetch_slate(session, draft_group_id, update_odds=update_odds, starters=starters, log=log)
    stored = store_slate(session, fetched, log=log)
    projected = project_slate(session, n_sims=n_sims, log=log) if project else None
    return {"fetched": fetched, "stored": stored, "projected": projected}
