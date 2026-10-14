# src/ivan/sports/nfl/ingest.py
"""
The pre-lock ingest as three steps, so the script, the web UI and the replay
run the same code:

    fetch_slate    DraftKings pool and/or the-odds-api props + game lines; nothing is
                   written until every fetch succeeded, so an expired key or an outage
                   leaves the previous slate intact
    store_slate    replace the working tables (nfl_player, odds, game_odds), archive the
                   capture and index the raw payloads, drop OUT/IR players
    project_slate  fit every player's FPTS distribution, project ownership (and snapshot
                   it for training), write the projections / environment / usage
                   artifacts the validation layer reads

scripts/nfl/ingest.py is argparse around run_ingest(); the replay adapter calls
project_slate() on a restored capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import func, select

from ivan.core.config import load_config, require_env
from ivan.sports.common.draftkings import ingest_players
from ivan.sports.common.odds import get_game_odds, get_odds, parse_game_odds, parse_odds
from ivan.sports.nfl.models import NFLGameOdds, NFLOdds, NFLPlayer

Log = Callable[[str], None]
ODDS_SPORT = "americanfootball_nfl"
GAME_LINE_KEY = ["game_id", "bookmaker", "market", "outcome_name", "point", "last_update"]


class IngestError(RuntimeError):
    """A fetch returned nothing usable; the database was not touched."""


@dataclass
class SlateFetch:
    """What one ingest fetched. players is None on an odds-only refresh; odds / game_lines None when odds were skipped."""
    players: Optional[list] = None
    odds: Optional[list] = None                 # parsed prop rows for the odds table
    game_lines: Optional[pd.DataFrame] = None   # parsed lines for game_odds

    @property
    def teams(self) -> set:
        return {p.team for p in (self.players or []) if p.team}


@dataclass
class StoreInfo:
    players: int = 0
    odds: int = 0
    games: int = 0
    line_rows: int = 0
    removed_out: int = 0
    raw_indexed: int = 0
    capture: Optional[dict] = None


@dataclass
class ProjectInfo:
    players: int = 0
    fit: dict = field(default_factory=dict)
    projections_path: Optional[str] = None
    environment_rows: int = 0
    usage_rows: int = 0
    artifact_error: Optional[str] = None


# ------------------------------------------------------------------------------ fetch
def fetch_slate(session, *, prelock: bool, update_odds: bool, draft_group_id: Optional[int] = None,
                api_key: Optional[str] = None, log: Log = print) -> SlateFetch:
    """
    prelock: re-fetch the DraftKings pool (prompting for the slate unless draft_group_id is given).
    update_odds: fetch player props for the pool's teams (the stored pool's teams on an odds-only refresh)
    and the game lines. Raises IngestError when a fetch comes back empty.
    """
    if not prelock and not update_odds:
        raise IngestError("nothing to fetch: prelock and update_odds are both off")
    out = SlateFetch()
    if prelock:
        log("fetching players")
        out.players = ingest_players("nfl", dedupe_by_name=False, draft_group_id=draft_group_id)
        if not out.players:
            raise IngestError("DraftKings returned no players for that slate; nothing changed.")
    if update_odds:
        key = api_key or require_env("ODDS_API_KEY")
        teams = out.teams if out.players is not None else set(session.scalars(select(NFLPlayer.team).distinct()))
        odds = get_odds(ODDS_SPORT, key, list(teams), raw_sport="nfl")
        df = parse_odds(odds)
        if df is None or len(df) == 0:
            raise IngestError("the-odds-api returned no odds (see the error above). "
                              "Check ODDS_API_KEY and your plan at the-odds-api.com; nothing changed.")
        out.odds = df.to_dict(orient="records")
        # Game lines (spread, total, moneyline) for every upcoming game: the DST
        # projection and the per-game diagnostics read them from game_odds.
        lines_json, _ = get_game_odds(ODDS_SPORT, key, markets="h2h,spreads,totals", raw_sport="nfl")
        lines = parse_game_odds(lines_json) if lines_json else None
        if lines is None or lines.empty:
            log("the-odds-api returned no game lines; DST projections will fall back to salary.")
            lines = None
        out.game_lines = lines
    return out


# ------------------------------------------------------------------------------ store
def store_slate(session, fetched: SlateFetch, *, prelock: bool, log: Log = print) -> StoreInfo:
    """
    Replace what the fetch covers (pool, props, the fetched games' lines) in one commit,
    archive the capture for replays, index the raw payloads, then drop OUT/IR players
    from a pre-lock pool.
    """
    info = StoreInfo()
    if fetched.players is not None:
        session.query(NFLPlayer).delete(synchronize_session=False)
        for p in fetched.players:
            session.merge(NFLPlayer(id=p.dk_id, name=p.name, team=p.team, opponent=p.opponent, position=p.position,
                                    roster_slot_id=p.roster_slot_id, gametime=p.gametime, salary=p.salary, status=p.status))
        info.players = len(fetched.players)
    if fetched.odds is not None:
        session.query(NFLOdds).delete(synchronize_session=False)
        session.bulk_insert_mappings(NFLOdds, fetched.odds)
        info.odds = len(fetched.odds)
    lines = fetched.game_lines
    if lines is not None and len(lines):
        lines = lines.drop_duplicates(subset=GAME_LINE_KEY)
        games = lines["game_id"].unique().tolist()
        session.query(NFLGameOdds).filter(NFLGameOdds.game_id.in_(games)).delete(synchronize_session=False)
        session.bulk_insert_mappings(NFLGameOdds, lines.to_dict(orient="records"))
        info.games, info.line_rows = len(games), len(lines)
        log(f"game lines: {info.games} games, {info.line_rows} rows")
    session.commit()

    # Archive the raw inputs of this capture (pool, props, lines) so the slate can
    # be replayed later with whatever projection code is current. Never modelled.
    from ivan.core import raw
    from ivan.core.archive import ensure_tables
    from ivan.sports.nfl.archive_capture import archive_ingest_capture
    ensure_tables(session.get_bind())
    info.raw_indexed = raw.sync_index(session, "nfl")
    log(f"raw artifacts indexed: {info.raw_indexed} new")
    cap = archive_ingest_capture(session, fetched.players, fetched.odds, lines)
    if cap is not None:
        info.capture = {"id": cap.id, "slate_key": cap.slate_key, "mode": cap.mode, "lock": cap.lock,
                        "n_players": cap.n_players, "n_odds": cap.n_odds, "n_lines": cap.n_lines}
        log(f"archived capture #{cap.id}: slate {cap.slate_key} ({cap.mode}, lock {cap.lock:%Y-%m-%d %H:%M}) "
            f"{cap.n_players} players, {cap.n_odds} odds rows, {cap.n_lines} line rows")

    if prelock:
        info.removed_out = session.query(NFLPlayer).filter(func.upper(NFLPlayer.status).in_(["OUT", "IR"])).delete(synchronize_session=False)
        session.commit()
    return info


# ------------------------------------------------------------------------------ project
def project_slate(session, config: Optional[dict] = None, *, ownership: bool = True, snapshot: bool = True,
                  artifacts_required: bool = False, log: Log = print) -> ProjectInfo:
    """
    Fit the slate (refactor.player.fit_fpts_distributions), project ownership from field
    sampling and snapshot it for training (ownership.project_and_snapshot), then write the
    projections, game-environment and usage artifacts next to each other under OUTPUT_DIR
    (or IVAN_RUN_DIR in a replay). Artifact failures are logged, not raised, in a live
    ingest; artifacts_required=True re-raises them (the replay's validation needs them).
    """
    from ivan.sports.nfl.refactor.player import fit_fpts_distributions
    cfg = config if config is not None else load_config("nfl")
    info = ProjectInfo()
    info.fit = fit_fpts_distributions(session, cfg) or {}
    if ownership:
        from ivan.sports.nfl.ownership import project_and_snapshot
        project_and_snapshot(session, cfg, snapshot=snapshot)
    info.players = session.query(NFLPlayer).filter(NFLPlayer.fpts_mean.isnot(None)).count()
    try:
        from ivan.core import artifacts
        from ivan.sports.nfl.paths import OUTPUT_DIR
        from ivan.sports.nfl.run_artifacts import write_environment, write_projections, write_usage
        path = artifacts.projections_path(OUTPUT_DIR)
        n = write_projections(session, path)
        info.projections_path = str(path)
        info.environment_rows = write_environment(session, path)
        info.usage_rows = write_usage(info.fit.get("usage"), path, info.fit.get("unpriced_volume", ""))
        log(f"projections artifact: {n} rows -> {path}; game environment {info.environment_rows} team rows; "
            f"usage {info.usage_rows} team-stat rows ({info.fit.get('unpriced_volume', '')})")
    except Exception as e:   # noqa: BLE001 - a diagnostic must not kill a live ingest
        if artifacts_required:
            raise
        info.artifact_error = f"{type(e).__name__}: {e}"
        log(f"WARNING: could not write the projections artifact: {info.artifact_error}")
    return info


# ------------------------------------------------------------------------------ all three
def run_ingest(session, *, prelock: bool, update_odds: bool, draft_group_id: Optional[int] = None,
               ownership: bool = True, snapshot: bool = True, config: Optional[dict] = None,
               log: Log = print) -> tuple[StoreInfo, ProjectInfo]:
    fetched = fetch_slate(session, prelock=prelock, update_odds=update_odds, draft_group_id=draft_group_id, log=log)
    stored = store_slate(session, fetched, prelock=prelock, log=log)
    projected = project_slate(session, config, ownership=ownership, snapshot=snapshot, log=log)
    return stored, projected
