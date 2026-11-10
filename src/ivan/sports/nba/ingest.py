"""
The slate pipeline: fetch, store, project. The desk's Refresh, scripts/nba/ingest.py and the autopilot tick all run it
(run_ingest); nothing here prompts.

    run_ingest        fetch (data_fetch.slate.fetch_slate), then in ONE transaction: store_slate, void the mappings a scratch
                      made stale, project_slate; then the slate's record, the public field when asked, Rotowire's news
    store_slate       the pool replaced (nba_player), the starters, the status marks, the props and game lines
                      (data_clean.odds), this pool's market rates, the public projection, the archive capture
    project_slate     every team mapped (layer 1: projection.engine through live.LiveSlate: minutes, grid, rates, into
                      nba_player and the projection store), then every game drawn once on the mapping (layer 2:
                      projection.worlds: the World's fantasy points, minutes and stats per player, laid over the pool), then
                      the capture rows and the projections CSV (projection.record)
    project_team      THE way one team is projected (the desk's edits): its mapping, then its game drawn again, both teams'
                      pool rows and store records written
    capture_pool      the snapshot: record the projection as it stands without projecting anything
    market rates      market_rates_current / ensure_market_rates / refresh_market_rates: this pool's rates from the odds
                      table's rows for its own games, rebuilt whenever a props fetch moved the table
    after_pull        a consumer, after the feed replaced the pool: its own edited teams re-projected on top
    restore_from_archive   a past slate loaded from its archive capture and projected with the model as it is now
    ensure_schema     the history database's tables (a slate's own are slates.ensure_slate_schema)
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Iterable, Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.core.archive import archive_slate, ensure_tables as ensure_archive_tables
from ivan.sports.nba.names import normalize_name
from ivan.database.session import add_missing_columns
from ivan.sports.nba.data_fetch.boxscores import store_starting_lineups
from ivan.sports.nba.models import (NBABase, NBAPlayer, NBAOdds, StartingLineups)
from ivan.sports.nba import slates as S
from ivan.sports.nba.projection import store as PSTORE
from ivan.sports.nba.projection import worlds as WS
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, basis_minutes_for_market, read_overrides_full, store_market_rates
from ivan.sports.nba.slate import changes as CH
from ivan.sports.nba.projection.market import fresh_odds, market_means
from ivan.sports.nba.data_fetch.slate import IngestError, SlateFetch, fetch_slate
from ivan.sports.nba.data_clean.odds import GAME_LINE_COLS, odds_stamp, store_game_lines, store_props
from ivan.sports.nba.projection.record import ProjectInfo, record_projection

Log = Callable[[str], None]


def ensure_schema(engine) -> None:
    """The history database's tables: every NBA table but a slate's own (slates.SLATE_TABLES live in data/nba/slates/<dg>/slate.sqlite,
    slates.ensure_slate_schema). A slate table created here would be found by nobody; a history table created in a slate file would
    shadow the real one, hence the two explicit lists."""
    from ivan.sports.nba.slates import SLATE_TABLES
    tables = [t for t in NBABase.metadata.sorted_tables if t.name not in SLATE_TABLES]
    NBABase.metadata.create_all(engine, tables=tables)
    add_missing_columns(engine, NBABase.metadata, only={t.name for t in tables})
    ensure_archive_tables(engine)

@dataclass
class StoreInfo:
    players: int = 0
    starters: int = 0
    odds: int = 0
    market_rates: int = 0
    game_lines: int = 0
    capture_id: Optional[int] = None
    status_changes: list = field(default_factory=list)     # players whose status or starter flag moved in this refresh (their older props set aside)

@contextmanager
def one_transaction(session: Session):
    """
    Everything inside commits once, at the end: the pool is replaced, projected and its public side refreshed before any
    other connection sees a row of it (SQLite WAL readers keep the old pool until the commit). The functions inside call
    session.commit() as they always did; here that becomes a flush. On an error everything rolls back and the old pool stays.
    One slate writes at a time in this process and the transaction begins IMMEDIATE (slates.writing): two slates refreshing
    together share the history file, and a deferred transaction that had read it hit "database is locked" when it wrote.
    """
    real_commit = session.commit
    with S.writing(session):
        session.commit = session.flush            # type: ignore[method-assign]
        try:
            yield
            session.commit = real_commit          # type: ignore[method-assign]
            real_commit()
        except BaseException:
            session.commit = real_commit          # type: ignore[method-assign]
            session.rollback()
            raise

def locally_edited_teams(session: Session) -> set:
    """On a consumer: the teams whose players this machine pinned or whose grid it painted (its own layer of the current slate,
    not the published one)."""
    local = read_overrides_full(layered=False) if S.overrides_csv().exists() else {}
    teams = set()
    if local:
        by_name = {normalize_name(p.name): p.team for p in session.query(NBAPlayer).all()}
        teams |= {by_name[n] for n in local if n in by_name and by_name[n]}
    if S.rotations_dir().is_dir():
        teams |= {f.stem for f in S.rotations_dir().glob("*.json")}
    return teams

def after_pull(session: Session, *, n_worlds: Optional[int] = None, log: Log = print) -> dict:
    """
    A consumer, after the feed replaced a slate's pool (the current slate's session): only the teams this machine edited, among
    the slate's teams, are re-projected on top of the published projection; the rest stands as published. Slates no longer
    displace each other (each has its own directory), so nothing is filed or cleared. Returns {teams, players, draft_group}.
    """
    row = session.query(NBAPlayer.draft_group_id).first()
    current = int(row[0]) if row and row[0] else None
    slate_teams = {t for (t,) in session.query(NBAPlayer.team).distinct().all() if t}
    mine = locally_edited_teams(session) & slate_teams
    players = 0
    if mine:
        info = project_slate(session, n_worlds=n_worlds, write_capture=False, write_csv=False, log=log, teams=mine)
        players = info.players
    return dict(moved=False, teams=mine, players=players, draft_group=current)

def refresh_pool(session: Session, draft_group_id: int, *, update_odds: bool = False, starters: bool = True, n_worlds: Optional[int] = None,
                 log: Log = print, field: bool = False, news: bool = True, lines: Optional[bool] = None, by: str = "desk") -> tuple:
    """
    The desk's Refresh and slate switch: fetch DraftKings (and the props with update_odds), then replace the pool, project it
    and write its public projection in ONE transaction. Nothing downstream can see a half-written or unprojected pool
    (2026-10-23: a sim started during a rewrite found no point guard). The public field and ownership are NOT built here
    (field=True does, after the commit): the sim builds the field it needs, the desk's "Field + ownership" button /
    scripts/nba/field_ownership.py builds it on demand (2026-10-28: thin wiring). Rotowire's news is read after the commit
    (news=False skips it). Returns (StoreInfo, ProjectInfo, the news check's result or None, the changes per team).
    The command line (scripts/nba/ingest.py -> run_ingest) takes the same path.
    """
    out = run_ingest(session, int(draft_group_id), update_odds=update_odds, starters=starters, n_worlds=n_worlds, log=log, field=field, news=news, lines=lines, by=by)
    return out["stored"], out["projected"], out["news"], out["changes"]

def market_rates_current(session: Session) -> bool:
    """False when the odds table is not the one this slate's market rates were built from (the stamp store_market_rates kept on the
    rows against odds_stamp now). The rates are a snapshot taken when the slate stored them, and the table is shared by the day's
    slates and moves with every props fetch, for this slate or another (2026-11-07: a slate's rates said Traore 12.2 points from an
    earlier fetch while the table, replaced by a later one, held his 18.5 lines; the slate never knew). Without odds, or without a
    pool, the rates are as current as they can be; rates from before the stamp count as moved once."""
    n, max_id = odds_stamp(session)
    if not n:
        return True
    row = session.execute(text("select odds_rows, odds_max_id from nba_market_rate limit 1")).first()
    if row is None:
        return False
    return row[0] == n and row[1] == max_id

def ensure_market_rates(session: Session, history=None, log: Log = print) -> bool:
    """Rebuild this pool's market rates when the odds table moved since they were built (market_rates_current), past the status
    cutoffs (refresh_market_rates). The desk checks on each request (Desk.follow_odds, throttled) and project_slate before it
    projects, so a publish or an edit reads the books as they stand, not as they stood at the slate's refresh. Returns True when
    the rates were rebuilt (the caller re-reads them)."""
    if market_rates_current(session):
        return False
    players = session.query(NBAPlayer).all()
    if not players:
        return False
    names = {normalize_name(p.name): p for p in players}
    n = refresh_market_rates(session, {k: p.gametime for k, p in names.items()}, set(names), history=history)
    session.commit()
    log(f"the books moved since this slate's market rates were built: {n} rates rebuilt")
    return True

def refresh_market_rates(session: Session, gametimes: dict, pool: set, history=None) -> int:
    """This pool's market rates (nba_market_rate) from the props of its own games only (market.fresh_odds: the row's game is the
    player's game on this slate; a player the books did not price tonight gets no market rate rather than his last game's), the
    rows a book posted before the player's status or starter flag last moved left out (changes.status_cutoffs: they priced his old
    role; he projects from history until the books post again). Called by the store and by the desk's starter call. Returns the
    rates stored; with no usable row at all the table is left as it was. `history`: (box, HistoryState) already built (the desk's),
    to skip the box-score load."""
    stamp = odds_stamp(session)                               # the table the means come from, kept on the rates (market_rates_current)
    odds_df = fresh_odds(session, gametimes, cutoffs=CH.status_cutoffs(session))
    if not len(odds_df):
        return 0
    odds_df = odds_df[odds_df.player.isin(set(pool))]
    if not len(odds_df):
        return 0
    means = market_means(odds_df)
    slate = LiveSlate(session, use_market=False, history=history)
    basis = basis_minutes_for_market(slate, read_overrides_full(), set())
    return store_market_rates(session, means, basis, odds_stamp=stamp)

def store_slate(session: Session, fetched: SlateFetch, *, log: Log = print) -> StoreInfo:
    info = StoreInfo()
    try:
        before_status = CH.status_state(session)            # the statuses and the five as they stood: what the props on file priced
    except Exception:   # noqa: BLE001 - a pool from before the column
        before_status = {}
    # the pool: replace in full (one slate at a time in nba_player)
    session.query(NBAPlayer).delete(synchronize_session=False)
    for row in fetched.players:
        session.add(NBAPlayer(**row))
    session.commit()
    info.players = len(fetched.players)
    if fetched.starters is not None:
        info.starters = store_starting_lineups(session, fetched.starters, fetched_at=fetched.fetched_at)
    try:
        info.status_changes = CH.mark_status_changes(session, before_status)
        if info.status_changes:
            log(f"status moved for {', '.join(info.status_changes)}: their props posted before now are set aside until the books post again")
    except Exception as e:   # noqa: BLE001 - the marks are a refinement of the market rates, never the refresh
        log(f"status marks not written: {type(e).__name__}: {e}")
    if fetched.odds is not None:
        info.odds = store_props(session, fetched.odds)
    if fetched.game_lines is not None and len(fetched.game_lines):
        info.game_lines = store_game_lines(session, fetched.game_lines, fetched_at=fetched.fetched_at)
    info.market_rates = refresh_market_rates(session, {r["name"]: r.get("gametime") for r in fetched.players}, {r["name"] for r in fetched.players})
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

def refresh_public_projection(session: Session, log: Log = print) -> dict:
    """The public projection alone (public_field.refresh_public_projection), a second or two; the field is not touched."""
    try:
        from ivan.sports.nba.field.public_field import refresh_public_projection as _refresh
        slate = LiveSlate(session, use_starters=False, use_market=False)
        return _refresh(session, slate.hist, log=log)
    except Exception as e:   # noqa: BLE001 - the public side is a by-product; the pool and the projection stand
        session.rollback(); log(f"public projection failed: {type(e).__name__}: {e}"); return {}

def refresh_field(session: Session, log: Log = print) -> dict:
    """The public field and ownership on the stored public projection (public_field.refresh_field_ownership): about a
    minute when a public number moved since the cached field, at once otherwise. Run after the pool is committed."""
    try:
        from ivan.sports.nba.field.public_field import refresh_field_ownership
        return refresh_field_ownership(session, log=log)
    except Exception as e:   # noqa: BLE001
        session.rollback(); log(f"field refresh failed: {type(e).__name__}: {e}"); return {}

def refresh_ownership(session: Session, log: Log = print) -> dict:
    """The public projection, the field and ownership from the pool and props as stored (public_field.refresh_public).
    Runs with every pool refresh (store_slate); Publish never touches it, so our edits cannot move ownership."""
    try:
        from ivan.sports.nba.field.public_field import refresh_public
        slate = LiveSlate(session, use_market=False)
        return refresh_public(session, slate.hist, log=log)
    except Exception as e:   # noqa: BLE001 - never fails a pool refresh
        session.rollback(); log(f"public refresh failed: {type(e).__name__}: {e}"); return {}

def _map_team(slate: LiveSlate, team: str, overrides: dict, params: ProjectionParams) -> dict:
    """Layer 1 for one team: the mapping (projection.engine through the live slate) written to its nba_player rows, the
    history-only comparison beside it, its store record returned (sim empty; not saved)."""
    rng = np.random.default_rng(0)
    proj = slate.project(team, overrides, set(), params, rng=rng)
    rot = dict(getattr(slate, "_last_rotation", {}).get(team) or {})      # the grid, targets and zero-minute players behind this frame
    hist = slate.project(team, overrides, set(), replace(params, market_weight=0.0), rng=rng)
    slate.write_plan(team, proj)
    dg = next((p.draft_group_id for p in slate.players if p.draft_group_id), None)
    return PSTORE.record(team, proj, hist, rot, dg, 0)

def _draw_games(session: Session, slate: LiveSlate, pairs: Iterable[tuple], recs: dict, overrides: dict, *, n_worlds: Optional[int] = None,
                ctx=None, log: Log = print) -> list:
    """
    Layer 2 for the given games: each drawn once on the pool as the mapping left it (projection.worlds.simulate_game), the
    hand fpts pins applied, the World's columns written to BOTH teams' nba_player rows and set on their store records (recs:
    team -> record; a team's record not in recs is loaded from the store and added). Nothing is saved or committed here.
    Returns the teams drawn.
    """
    ctx = ctx or WS.context(session)
    n = int(n_worlds or WS.n_worlds())
    inputs = WS.game_inputs(session, ctx)
    done = []
    for team, opp in pairs:
        g = inputs.get(frozenset((str(team), str(opp))))
        if g is None:
            log(f"{team} vs {opp}: no projected players on either side, not drawn"); continue
        sim = WS.simulate_game(session, ctx, g, n)
        for t in (sim.home, sim.away):
            frame, samples = sim.frames[t], sim.samples[t]
            for name, ov in (overrides or {}).items():
                v = ov.get("fpts") if isinstance(ov, dict) else None
                if v is not None and name in frame.index:
                    WS.apply_fpts_pin(frame, samples, name, float(v))
            slate.write_sim(t, frame)
            rec = recs.get(t) or PSTORE.load(t)
            if rec is None:
                continue                                        # a team never mapped here: its pool rows carry the draw, no record to hold it
            recs[t] = PSTORE.set_sim(rec, frame, samples, sim.game, n)
            done.append(t)
        gi = sim.game
        log(f"{sim.away} @ {sim.home}: {n} worlds, attempts x {gi['factor'][sim.home]:.2f} / {gi['factor'][sim.away]:.2f}, "
            f"total {gi['mean_total']:.1f}" + (f" (line {gi['total']:g})" if gi.get("total") is not None else "") + f", margin {gi['mean_margin']:+.1f}")
    return done

def project_team(session: Session, slate: LiveSlate, team: str, overrides: dict, params: Optional[ProjectionParams] = None,
                 *, n_worlds: Optional[int] = None, ctx=None, log: Log = lambda m: None) -> dict:
    """
    THE way a team is projected (the desk's edits, scripts/nba/project.py): its mapping (layer 1) written to its nba_player
    rows and its store record, then its game drawn again on the pool (layer 2) and the World's numbers written to both
    teams' rows and records (an edit on one side moves the other side's rebounds, steals and blocks). Not committed: the
    caller's transaction. The draw is seeded by the pair, so a team's projection depends only on the two mappings: an
    edit on the desk and a publish give the same numbers. The opponent is mapped first when it has no record yet.
    Returns the team's store record (frame = layer 1, sim = layer 2; store.merged lays one over the other).
    """
    params = params or ProjectionParams.from_config()
    recs = {team: _map_team(slate, team, overrides, params)}
    opp = slate.opponent(team)
    if opp and PSTORE.load(opp) is None and any(p.team == opp for p in slate.players):
        recs[opp] = _map_team(slate, opp, overrides, params)
    if opp:
        _draw_games(session, slate, [(team, opp)], recs, overrides, n_worlds=n_worlds, ctx=ctx, log=log)
    for rec in recs.values():
        PSTORE.save(rec)
    return recs[team]

def project_slate(session: Session, *, n_worlds: Optional[int] = None, write_capture: bool = True, write_csv: bool = True,
                  log: Log = print, source: str = "ours", params: Optional[ProjectionParams] = None, write_ownership: bool = False,
                  teams: Optional[Iterable[str]] = None) -> ProjectInfo:
    """Project the slate (or `teams` of it): every team mapped, then every game one of them is in drawn once, commit, then the
    capture rows and the projections CSV. n_worlds defaults to config/nba.json projection.n_worlds: the one number every path draws at."""
    info = ProjectInfo()
    ensure_market_rates(session, log=log)                   # the books as they stand, not as they stood at the slate's refresh
    slate = LiveSlate(session)
    if not slate.players:
        raise IngestError("no players in nba_player")
    overrides = read_overrides_full()
    params = params or ProjectionParams.from_config()
    now = datetime.now()
    only = None if teams is None else {str(t) for t in teams}
    recs: dict = {}
    for team in slate.teams():
        if only is not None and team not in only:
            continue                                        # a consumer re-projects only the teams it edited: the rest stay as published
        recs[team] = _map_team(slate, team, overrides, params)
    mapped = set(recs)
    dg0 = next((p.draft_group_id for p in slate.players if p.draft_group_id), None)
    if not mapped:                                          # the requested teams are not on this slate: nothing to project, nothing changed
        log(f"no team of the slate among {sorted(only) if only else 'none'}; the projection stands")
        return info
    pairs = [(t, o) for t, o in WS.game_pairs(slate.players) if t in mapped or o in mapped]
    _draw_games(session, slate, pairs, recs, overrides, n_worlds=n_worlds, log=log)
    for rec in recs.values():
        PSTORE.save(rec)
    session.commit()
    PSTORE.prune(dg0, slate.teams())                        # records of another slate, or of a team no longer on this one, go
    frames = []
    for team in sorted(mapped):
        f = PSTORE.merged(recs[team]); f["team"] = team
        frames.append(f)
    return record_projection(session, slate, frames, now, source=source, write_capture=write_capture, write_csv=write_csv,
                             write_ownership=write_ownership, minutes_model=params.minutes_model, log=log, info=info)

def capture_pool(session: Session, *, log: Log = print, source: str = "ours", write_csv: bool = True) -> ProjectInfo:
    """
    Record the projection as it stands: capture rows (source "ours", the ones the scorecard scores) and the projections CSV,
    from the projection store's frames, without projecting anything (a team with no record is projected first through
    project_team). What the desk's Snapshot button and the pipeline's first step do: the pool is already the projection
    (every edit writes it), so there is nothing to publish, only to record.
    """
    info = ProjectInfo()
    slate = LiveSlate(session)
    if not slate.players:
        raise IngestError("no players in nba_player")
    dg = next((p.draft_group_id for p in slate.players if p.draft_group_id), None)
    recs = PSTORE.load_all(dg)
    overrides = None
    frames = []
    for team in slate.teams():
        rec = recs.get(team)
        if rec is None:
            overrides = overrides if overrides is not None else read_overrides_full()
            rec = project_team(session, slate, team, overrides, ProjectionParams.from_config()); session.commit()
            log(f"{team}: no stored projection, projected")
        proj = PSTORE.merged(rec); proj["team"] = team
        frames.append(proj)
    return record_projection(session, slate, frames, datetime.now(), source=source, write_capture=True, write_csv=write_csv,
                             minutes_model=ProjectionParams.from_config().minutes_model, log=log, info=info)

def run_ingest(session, draft_group_id: int, *, update_odds: bool, project: bool = True, starters: bool = True,
               n_worlds: Optional[int] = None, log: Log = print, fetched: Optional[SlateFetch] = None, field: bool = False, news: bool = True,
               lines: Optional[bool] = None, by: str = "desk") -> dict:
    """
    Fetch, store and project one slate the way the desk does it: the fetch happens outside the database, then the slate's pool is
    replaced, projected and its public projection written in one transaction, so a desk serving the same slate from another
    process never sees a half-written pool (it notices the new stamp and shows "new data"). `session` is the slate's own
    session (slates.session(dg): its slate.sqlite with the history attached); the slate is the current one for the call, so the
    edits, grids and store read and written are its own, and no other slate is touched. The public field and ownership are not
    built unless field=True (refresh_field after the commit, about 45s at 3,000 entries): the sim builds the field it needs and
    the desk's "Field + ownership" button builds it on demand. project=False stores only; the desk then shows an unprojected pool
    until something projects it. `fetched`: a slate already fetched by the caller (the autopilot tick attaches the game lines to
    it) is stored as given instead of fetching again. news=True (the default) also reads Rotowire's news page after the commit
    (newsfeed.check, free, about a second; a failure is logged, never raised) and links the slate's headlines to the slate
    (nba_slate_news), so every refresh brings the headlines with the statuses and the slate's pre-lock news is on record.
    out["changes"] is what the refresh moved per team (ivan.sports.nba.slate.changes: statuses, the lineups, the pool, projections that
    moved, new headlines), also logged. The slate's record (slates/<dg>/slate.json) notes the lock, the games and who refreshed
    it when (`by`: desk | autopilot | script | feed).
    """
    from ivan.sports.nba.slate import changes as CH
    dg = int(draft_group_id)
    with S.using(dg):
        held = session.query(NBAPlayer.draft_group_id).first()
        if held and held[0] and int(held[0]) != dg:
            raise IngestError(f"this session holds slate {held[0]}, not {dg}: open the slate's own session (slates.session)")
        try:
            before = CH.snapshot(session)
        except Exception:   # noqa: BLE001 - the diff is a convenience
            before = None
        if fetched is None:
            fetched = fetch_slate(session, dg, update_odds=update_odds, starters=starters, log=log, lines=lines)
        with one_transaction(session):
            stored = store_slate(session, fetched, log=log)
            voided = void_stale_mappings(session, log=log)
            projected = project_slate(session, n_worlds=n_worlds, log=log) if project else None
        S.create(dg, lock=fetched.lock, games=len(fetched.teams) // 2 if fetched.teams else None)
        S.touch(dg, "refreshed", by=by)
        ownership = refresh_field(session, log=log) if field else None
        news_r = check_news(session, log=log, draft_group_id=dg) if news else None
        try:
            moved = CH.diff(before, CH.snapshot(session), (news_r or {}).get("new"))
        except Exception:   # noqa: BLE001
            moved = {}
        for v in voided:                                  # a voided mapping is a change of the team: the board's dot and the refresh note say so
            moved.setdefault(v["team"], []).append(dict(kind="mapping", player=None, text=f"mapping void: {v['why']}; the model's proposal is on the grid, map the team again"))
        for team, items in sorted(moved.items()):
            log(f"changed {team}: " + "; ".join(x["text"] for x in items))
        return {"fetched": fetched, "stored": stored, "projected": projected, "ownership": ownership, "news": news_r, "changes": moved}

def void_stale_mappings(session: Session, log: Log = print) -> list:
    """The mappings a player now out or a changed five makes void (ivan.sports.nba.slate.mapping.void_stale), with this machine's pins file rewritten."""
    from ivan.sports.nba.slate import mapping as MAP
    from ivan.sports.nba.projection.live import write_overrides
    local = read_overrides_full(layered=False) if S.overrides_csv().exists() else {}
    voided = MAP.void_stale(session, local, log=log)
    if voided:
        write_overrides(local)
    return voided

def check_news(session, log: Log = print, draft_group_id: Optional[int] = None) -> Optional[dict]:
    """Rotowire's news page into nba_news (newsfeed.check) for a refresh: None and a log line when it fails. With a draft group,
    every headline of the slate's teams now on file is linked to the slate (nba_slate_news, once per headline and slate), so
    the slate's pre-lock news is nba_news joined through that table."""
    from ivan.sports.nba.data_fetch import newsfeed
    with S.WRITE_LOCK:                                      # the headlines table is shared: two slates refreshing together write it in turn
        try:
            out = newsfeed.check(session, log=log)
        except Exception as e:   # noqa: BLE001 - the pool is the refresh; the news is a convenience on top
            log(f"news check failed: {type(e).__name__}: {e}")
            return None
        if draft_group_id is not None:
            try:
                link_slate_news(session, int(draft_group_id))
            except Exception as e:   # noqa: BLE001
                log(f"news link skipped: {type(e).__name__}: {e}")
    return out

def link_slate_news(session, draft_group_id: int) -> int:
    """Link the headlines of the slate's teams to the slate (nba_slate_news). Returns how many links are new."""
    from ivan.sports.nba.models import NBANews, NBASlateNews
    teams = {t for (t,) in session.query(NBAPlayer.team).distinct().all() if t}
    if not teams:
        return 0
    have = {int(n) for (n,) in session.query(NBASlateNews.news_id).filter(NBASlateNews.draft_group_id == int(draft_group_id)).all()}
    now = datetime.now(); n = 0
    for (nid,) in session.query(NBANews.id).filter(NBANews.team.in_(sorted(teams))).all():
        if int(nid) in have:
            continue
        session.add(NBASlateNews(draft_group_id=int(draft_group_id), news_id=int(nid), seen_at=now)); n += 1
    if n:
        session.commit()
    return n

def restore_from_archive(session, draft_group_id: int, *, capture_id: Optional[int] = None, n_worlds: Optional[int] = None, log: Log = print) -> dict:
    """
    Load a past slate into its own directory from its archive capture (the last one taken before lock, else the last one; or
    `capture_id`): the pool as it was (salaries, statuses, starters), its props and game lines, then the projection with
    the model as it is now. Nothing is fetched. One transaction, like refresh_pool, on the slate's own session
    (slates.session(dg)); its edits are where they always were. No capture row is written (the scorecard reads the ones
    taken before lock).
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
    cols = set(NBAPlayer.__table__.columns.keys()) - {"fpts_mean", "fpts_std", "fpts_updated_at", "ownership", "minutes_proj", "fpts_quantiles",
                                                      "minutes_role", "p_out", "rates", "minutes_p10", "minutes_p90", "rotation_cells", "boundary_sd",
                                                      "pub_fpts", "pub_updated_at"}
    with S.using(int(draft_group_id)):
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
                lu = pd.to_datetime(odf["last_update"], errors="coerce", utc=True) if "last_update" in odf else pd.Series(pd.NaT, index=odf.index)
                odf["last_update"] = lu.dt.strftime("%Y-%m-%dT%H:%M:%SZ")        # the ISO UTC string the live table holds
                store_props(session, odf[[c for c in odf.columns if c in NBAOdds.__table__.columns.keys()]])
            if line_rows:
                store_game_lines(session, pd.DataFrame(line_rows), fetched_at=cap.captured_at)
            odds_df = fresh_odds(session, {r["name"]: r.get("gametime") for r in players})
            pool = {r["name"] for r in players}
            odds_df = odds_df[odds_df.player.isin(pool)] if len(odds_df) else odds_df
            if len(odds_df):
                means = market_means(odds_df)
                slate = LiveSlate(session, use_market=False)
                basis = basis_minutes_for_market(slate, read_overrides_full(), set())
                store_market_rates(session, means, basis)
            refresh_public_projection(session, log=log)        # the field and ownership: the sim or the Field + ownership button
            info = project_slate(session, n_worlds=n_worlds, log=log, write_capture=False)
        S.create(int(draft_group_id), lock=cap.lock, games=len({frozenset((r.get("team"), r.get("opponent"))) for r in players if r.get("team") and r.get("opponent")}) or None)
        S.touch(int(draft_group_id), "refreshed", by="archive")
    log(f"restored slate {draft_group_id} from capture {cap.id} ({cap.source}, {cap.captured_at:%m-%d %H:%M}): {len(players)} players, "
        f"{n_st} starters, {len(odds_rows)} prop rows, {len(line_rows)} line rows; projected {info.players}")
    return dict(capture=cap.id, source=cap.source, captured_at=cap.captured_at, players=len(players), starters=n_st, odds=len(odds_rows), lines=len(line_rows))
