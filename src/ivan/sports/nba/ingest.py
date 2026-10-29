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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Callable, Optional

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
        lock = None
        if s.get("start_utc"):
            try:                                              # "2026-10-25T23:00:00.0000000Z" -> local naive datetime
                from datetime import timezone
                lock = datetime.fromisoformat(str(s["start_utc"])[:19]).replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
            except ValueError:
                lock = None
        out.append({"id": int(s["id"]), "start": s.get("start"), "name": s.get("name"), "contests": s.get("contests"),
                    "games": s.get("games"), "lock": lock})
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


def retire_edits(old_draft_group: int, log: Log = print, new_draft_group: Optional[int] = None) -> None:
    """
    Edits are per slate: when the pool moves to another slate the minute / rate / starter pins and the drawn rotation grids
    are filed under data/nba/input/edits/<old draft group>/ and the live files start empty (a pin on a name would otherwise
    follow him onto a slate where he has another role). With `new_draft_group`, that slate's filed edits come back, so
    switching away and back loses nothing.
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
        log(f"filed {moved} edit file(s) of slate {old_draft_group} under {keep.relative_to(INPUT_DIR.parent)}")
    if new_draft_group is not None:
        back = INPUT_DIR / "edits" / str(new_draft_group)
        if back.is_dir():
            n = 0
            for f in back.iterdir():
                if f.name == MINUTES_OVERRIDES_CSV.name:
                    shutil.move(str(f), str(MINUTES_OVERRIDES_CSV)); n += 1
                elif f.suffix == ".json":
                    ROTATIONS_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(ROTATIONS_DIR / f.name)); n += 1
            if n:
                log(f"restored {n} edit file(s) of slate {new_draft_group}")


KEEP_SLATE_DAYS = 2   # a switch keeps the lineups and sims of slates this many days back (today's other slates side by side)


def isolate_slate(session, draft_group_id: int, log: Log = print, keep_days: int = KEEP_SLATE_DAYS) -> dict:
    """
    Loading a slate keeps each slate's own work: lineups, pool snapshots, simulation results and contest reports belong to
    their draft group, and generate / the field sampler / the sim read only the loaded slate's (field_generation.
    slate_lineups). What a switch deletes is the work of slates older than `keep_days` (by the day of their contests or
    their pool snapshot's lock) and rows of no known slate, so two of today's slates can be worked side by side and coming
    back to one finds its lineups and sims (until 2026-10-29 every other slate's work was deleted, 2026-10-26: Monday's
    desk still carried Sunday's 12k lineups and sims). The pool, props, captures, news and box scores are never touched.
    """
    from sqlalchemy import text
    from ivan.sports.nba.models import ContestSimReport, Lineup, LineupPool, LineupSimulationResult
    dg = int(draft_group_id)
    cutoff = (datetime.now() - timedelta(days=int(keep_days))).date().isoformat()
    days = {}
    for d, day in session.execute(text("select draft_group_id, min(date(date)) from historical_contest_information group by draft_group_id")).all():
        if d is not None and day:
            days[int(d)] = str(day)
    for d, day in session.execute(text("select draft_group_id, min(date(lock)) from lineup_pool group by draft_group_id")).all():
        if d is not None and day and int(d) not in days:
            days[int(d)] = str(day)
    have = {int(d) for (d,) in session.execute(text("select distinct draft_group_id from lineup where draft_group_id is not null")).all()}
    old = sorted(d for d in have | {int(x) for (x,) in session.execute(text("select distinct draft_group_id from lineup_pool")).all() if x is not None}
                 if d != dg and (d not in days or days[d] < cutoff))
    n_l = session.query(Lineup).filter(Lineup.draft_group_id.in_(old) | Lineup.draft_group_id.is_(None)).delete(synchronize_session=False)
    n_p = session.query(LineupPool).filter(LineupPool.draft_group_id.in_(old)).delete(synchronize_session=False) if old else 0
    n_s = session.execute(text("DELETE FROM lineup_simulation_results WHERE lineup_id NOT IN (SELECT id FROM lineup)")).rowcount
    n_r = 0
    if old:
        n_r = session.execute(text("DELETE FROM contest_sim_report WHERE contest_id IN (SELECT contest_id FROM historical_contest_information WHERE draft_group_id IN ("
                                   + ",".join(str(int(x)) for x in old) + "))")).rowcount
    session.commit()
    out = dict(lineups=int(n_l or 0), pool=int(n_p or 0), sims=int(n_s or 0), reports=int(n_r or 0), slates=old)
    if any(v for k, v in out.items() if k != "slates"):
        log(f"cleared older slates' work ({', '.join(map(str, old)) or 'no slate'}): {out['lineups']} lineups, {out['sims']} sim results, "
            f"{out['reports']} reports, {out['pool']} pool rows")
    return out


def locally_edited_teams(session: Session) -> set:
    """On a consumer: the teams whose players this machine pinned or whose grid it painted (its own layer, not the published one)."""
    from ivan.sports.nba.projection.rotation import ROTATIONS_DIR
    local = read_overrides_full(layered=False) if MINUTES_OVERRIDES_CSV.exists() else {}
    teams = set()
    if local:
        by_name = {normalize_name(p.name): p.team for p in session.query(NBAPlayer).all()}
        teams |= {by_name[n] for n in local if n in by_name and by_name[n]}
    if ROTATIONS_DIR.is_dir():
        teams |= {f.stem for f in ROTATIONS_DIR.glob("*.json")}
    return teams


def after_pull(session: Session, previous_draft_group: Optional[int], *, n_sims: int = 2000, log: Log = print) -> dict:
    """
    A consumer, after feed.pull replaced the pool. When the published slate moved to another draft group, this machine's own
    edits of the old slate are filed and the new slate's come back (retire_edits: a pin on a name must not follow him onto
    another slate, 2026-10-27: the server still held Sunday's pins when Tuesday's slate arrived) and the other slate's lineups
    and sims are cleared (isolate_slate), exactly as the producer does on a slate switch. Then only the teams this machine
    edited, among the slate's teams, are re-projected on top of the published projection; the rest stands as published.
    Returns {moved, teams, players}.
    """
    row = session.query(NBAPlayer.draft_group_id).first()
    current = int(row[0]) if row and row[0] else None
    moved = previous_draft_group is not None and current is not None and int(previous_draft_group) != current
    if moved:
        log(f"the published slate moved from {previous_draft_group} to {current}")
        retire_edits(int(previous_draft_group), log=log, new_draft_group=current)
        isolate_slate(session, current, log=log)
    slate_teams = {t for (t,) in session.query(NBAPlayer.team).distinct().all() if t}
    mine = locally_edited_teams(session) & slate_teams
    players = 0
    if mine:
        info = project_slate(session, n_sims=n_sims, write_capture=False, write_csv=False, log=log, teams=mine)
        players = info.players
    return dict(moved=moved, teams=mine, players=players, draft_group=current)


PULLED_LIVE_BUNDLE = PROJECT_ROOT / "data" / "feed-state" / "tmp" / "live.sql.gz"     # what feed.pull last fetched and applied


def consumer_publish(session: Session, *, n_sims: int = 2000, log: Log = print) -> dict:
    """
    A consumer's Publish. The live bundle last pulled is applied again, so the pool holds the laptop's projection exactly as
    published, then only the slate's teams this machine edited (its own pins and grids) are projected on top. Clearing an
    edit and publishing therefore gives the published numbers back, and the teams never touched are never re-simulated
    (a re-simulation differs from the published numbers by a few tenths). Returns {restored, teams, players}.
    """
    from ivan.core import feed
    restored = False
    if PULLED_LIVE_BUNDLE.exists():
        feed.apply("nba", PULLED_LIVE_BUNDLE, log=lambda m: None)
        session.expire_all()
        restored = True
    else:
        log("no pulled bundle on disk: the pool is projected as it stands")
    slate_teams = {t for (t,) in session.query(NBAPlayer.team).distinct().all() if t}
    mine = locally_edited_teams(session) & slate_teams
    players = 0
    if mine:
        info = project_slate(session, n_sims=n_sims, log=log, teams=mine)
        players = info.players
    return dict(restored=restored, teams=mine, players=players)


def refresh_pool(session: Session, draft_group_id: int, *, update_odds: bool = False, starters: bool = True, n_sims: int = 1000,
                 log: Log = print, field: bool = False, news: bool = True) -> tuple:
    """
    The desk's Refresh and slate switch: fetch DraftKings (and the props with update_odds), then replace the pool, project it
    and write its public projection in ONE transaction. Nothing downstream can see a half-written or unprojected pool
    (2026-10-23: a sim started during a rewrite found no point guard). The public field and ownership are NOT built here
    (field=True does, after the commit): the sim builds the field it needs, the desk's "Field + ownership" button /
    scripts/nba/field_ownership.py builds it on demand (2026-10-28: thin wiring). Rotowire's news is read after the commit
    (news=False skips it). Returns (StoreInfo, ProjectInfo, the news check's result or None, the changes per team).
    The command line (scripts/nba/ingest.py -> run_ingest) takes the same path.
    """
    out = run_ingest(session, int(draft_group_id), update_odds=update_odds, starters=starters, n_sims=n_sims, log=log, field=field, news=news)
    return out["stored"], out["projected"], out["news"], out["changes"]


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
    refresh_public_projection(session, log=log)             # the public projection: the pool or props changed (the field follows the commit)
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


def refresh_public_projection(session: Session, log: Log = print) -> dict:
    """The public projection alone (public_field.refresh_public_projection), a second or two; the field is not touched."""
    try:
        from ivan.sports.nba.public_field import refresh_public_projection as _refresh
        slate = LiveSlate(session, use_starters=False, use_market=False)
        return _refresh(session, slate.hist, log=log)
    except Exception as e:   # noqa: BLE001 - the public side is a by-product; the pool and the projection stand
        session.rollback(); log(f"public projection failed: {type(e).__name__}: {e}"); return {}


def refresh_field(session: Session, log: Log = print) -> dict:
    """The public field and ownership on the stored public projection (public_field.refresh_field_ownership): about a
    minute when a public number moved since the cached field, at once otherwise. Run after the pool is committed."""
    try:
        from ivan.sports.nba.public_field import refresh_field_ownership
        return refresh_field_ownership(session, log=log)
    except Exception as e:   # noqa: BLE001
        session.rollback(); log(f"field refresh failed: {type(e).__name__}: {e}"); return {}


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
                  log: Log = print, source: str = "ours", params: Optional[ProjectionParams] = None, write_ownership: bool = False,
                  teams: Optional[Iterable[str]] = None) -> ProjectInfo:
    info = ProjectInfo()
    slate = LiveSlate(session)
    if not slate.players:
        raise IngestError("no players in nba_player")
    overrides = read_overrides_full()
    params = params or ProjectionParams.from_config(n_sims=n_sims)
    rng = np.random.default_rng(0)
    now = datetime.now()
    frames = []
    only = None if teams is None else {str(t) for t in teams}
    for team in slate.teams():
        if only is not None and team not in only:
            continue                                        # a consumer re-projects only the teams it edited: the rest stay as published
        proj = slate.project(team, overrides, set(), params, rng=rng)
        slate.write(team, proj)
        proj = proj.copy(); proj["team"] = team
        frames.append(proj)
    session.commit()
    if not frames:                                          # the requested teams are not on this slate: nothing to project, nothing changed
        log(f"no team of the slate among {sorted(only) if only else 'none'}; the projection stands")
        return info
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
               n_sims: int = 4000, log: Log = print, fetched: Optional[SlateFetch] = None, field: bool = False, news: bool = True) -> dict:
    """
    Fetch, store and project one slate the way the desk does it: the fetch happens outside the database, then the pool is
    replaced, projected and its public projection written in one transaction, so a desk serving the same database from
    another process never sees a half-written pool (it notices the new stamp and shows "new data"). The public field and
    ownership are not built unless field=True (refresh_field after the commit, about 45s at 3,000 entries): the sim
    builds the field it needs and the desk's "Field + ownership" button builds it on demand. When the pool moves to another draft group the old
    slate's pins and drawn grids are filed away first (retire_edits). project=False stores only; the desk then shows an
    unprojected pool until something projects it. `fetched`: a slate already fetched by the caller (the autopilot tick
    attaches the game lines to it) is stored as given instead of fetching again. news=True (the default) also reads
    Rotowire's news page after the commit (newsfeed.check, free, about a second; a failure is logged, never raised), so
    every refresh brings the headlines with the statuses. out["changes"] is what the refresh moved per team
    (ivan.sports.nba.changes: statuses, the lineups, the pool, projections that moved, new headlines), also logged.
    """
    from ivan.sports.nba import changes as CH
    try:
        before = CH.snapshot(session)
    except Exception:   # noqa: BLE001 - the diff is a convenience
        before = None
    if fetched is None:
        fetched = fetch_slate(session, int(draft_group_id), update_odds=update_odds, starters=starters, log=log)
    current = session.query(NBAPlayer.draft_group_id).first()
    switching = bool(current and current[0] and int(current[0]) != int(draft_group_id))
    if switching:
        retire_edits(int(current[0]), log=log, new_draft_group=int(draft_group_id))   # pins and grids belong to their slate; the new slate's come back
    try:
        with one_transaction(session):
            stored = store_slate(session, fetched, log=log)
            projected = project_slate(session, n_sims=n_sims, log=log) if project else None
    except BaseException:
        if switching:
            retire_edits(int(draft_group_id), log=log, new_draft_group=int(current[0]))   # the pool did not move: the edits go back where they were
        raise
    if switching:
        isolate_slate(session, int(draft_group_id), log=log)
    ownership = refresh_field(session, log=log) if field else None
    news_r = check_news(session, log=log) if news else None
    try:
        moved = CH.diff(before, CH.snapshot(session), (news_r or {}).get("new"))
    except Exception:   # noqa: BLE001
        moved = {}
    for team, items in sorted(moved.items()):
        log(f"changed {team}: " + "; ".join(x["text"] for x in items))
    return {"fetched": fetched, "stored": stored, "projected": projected, "ownership": ownership, "news": news_r, "changes": moved}


def check_news(session, log: Log = print) -> Optional[dict]:
    """Rotowire's news page into nba_news (newsfeed.check) for a refresh: None and a log line when it fails."""
    from ivan.sports.nba import newsfeed
    try:
        return newsfeed.check(session, log=log)
    except Exception as e:   # noqa: BLE001 - the pool is the refresh; the news is a convenience on top
        log(f"news check failed: {type(e).__name__}: {e}")
        return None


def restore_from_archive(session, draft_group_id: int, *, capture_id: Optional[int] = None, n_sims: int = 1000, log: Log = print) -> dict:
    """
    Load a past slate onto the desk from its archive capture (the last one taken before lock, else the last one; or
    `capture_id`): the pool as it was (salaries, statuses, starters), its props and game lines, then the projection with
    the model as it is now. Nothing is fetched. One transaction, like refresh_pool; the slate's filed edits come back.
    No capture row is written (the scorecard reads the ones taken before lock).
    """
    from ivan.core import archive
    caps = [c for c in archive.captures(session, "nba") if str(c.slate_key) == str(draft_group_id)]
    if not caps:
        raise IngestError(f"no archive capture for draft group {draft_group_id}")
    if capture_id is not None:
        cap = next((c for c in caps if int(c.id) == int(capture_id)), None)
        if cap is None:
            raise IngestError(f"capture {capture_id} is not a capture of draft group {draft_group_id}")
    else:
        pre = [c for c in caps if c.captured_at <= c.lock]
        cap = max(pre or caps, key=lambda c: c.captured_at)
    blobs = archive.load_blobs(session, cap.id)
    players, odds_rows, line_rows = blobs.get("players", []), blobs.get("odds", []), blobs.get("game_lines", [])
    if not players:
        raise IngestError(f"capture {cap.id} holds no players")
    current = session.query(NBAPlayer.draft_group_id).first()
    switching = bool(current and current[0] and int(current[0]) != int(draft_group_id))
    if switching:
        retire_edits(int(current[0]), log=log, new_draft_group=int(draft_group_id))
    cols = set(NBAPlayer.__table__.columns.keys()) - {"fpts_mean", "fpts_std", "fpts_updated_at", "ownership", "minutes_proj", "fpts_quantiles",
                                                      "minutes_role", "p_out", "rates", "minutes_p10", "minutes_p90", "rotation_cells", "boundary_sd",
                                                      "pub_fpts", "pub_updated_at"}
    try:
        with one_transaction(session):
            session.query(NBAPlayer).delete(synchronize_session=False)
            for r in players:
                row = {k: v for k, v in r.items() if k in cols}
                if isinstance(row.get("gametime"), str):
                    row["gametime"] = datetime.fromisoformat(row["gametime"])
                if str(row.get("status")) in ("None", "none", ""):
                    row["status"] = None
                row["draft_group_id"] = int(draft_group_id)
                session.add(NBAPlayer(**row))
            session.commit()
            session.query(StartingLineups).delete(synchronize_session=False)
            n_st = 0
            for r in players:
                if str(r.get("starter") or "") in ("confirmed", "expected"):
                    session.add(StartingLineups(player_name=r["name"], team=r.get("team"), position=r.get("position"),
                                                confirmed=(r.get("starter") == "confirmed"), fetched_at=cap.captured_at)); n_st += 1
            session.commit()
            if odds_rows:
                odf = pd.DataFrame(odds_rows)
                recs = odf.to_dict("records")
                for r in recs:                                        # the blob holds ISO strings; sqlite binds datetimes, not pandas Timestamps
                    lu = pd.to_datetime(r.get("last_update"), errors="coerce", utc=True)
                    r["last_update"] = None if pd.isna(lu) else lu.tz_localize(None).to_pydatetime()
                names = [str(x) for x in odf.player.dropna().unique()]
                for i in range(0, len(names), 500):
                    session.query(NBAOdds).filter(NBAOdds.player.in_(names[i:i + 500])).delete(synchronize_session=False)
                session.bulk_insert_mappings(NBAOdds, recs)
                session.commit()
            if line_rows:
                store_game_lines(session, pd.DataFrame(line_rows), fetched_at=cap.captured_at)
            odds_df = pd.read_sql("select player, market, sportsbook, side, point, price, last_update from odds", session.connection())
            pool = {r["name"] for r in players}
            odds_df = odds_df[odds_df.player.isin(pool)] if len(odds_df) else odds_df
            if len(odds_df):
                means = market_means(odds_df)
                slate = LiveSlate(session, use_market=False)
                basis = basis_minutes_for_market(slate, read_overrides_full(), set())
                store_market_rates(session, means, basis)
            refresh_public_projection(session, log=log)        # the field and ownership: the sim or the Field + ownership button
            info = project_slate(session, n_sims=n_sims, log=log, write_capture=False)
    except BaseException:
        if switching:
            retire_edits(int(draft_group_id), log=log, new_draft_group=int(current[0]))   # the pool did not move: the edits go back
        raise
    if switching:
        isolate_slate(session, int(draft_group_id), log=log)
    log(f"restored slate {draft_group_id} from capture {cap.id} ({cap.source}, {cap.captured_at:%m-%d %H:%M}): {len(players)} players, "
        f"{n_st} starters, {len(odds_rows)} prop rows, {len(line_rows)} line rows; projected {info.players}")
    return dict(capture=cap.id, source=cap.source, captured_at=cap.captured_at, players=len(players), starters=n_st, odds=len(odds_rows), lines=len(line_rows))
