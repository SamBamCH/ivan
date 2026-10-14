# src/ivan/sports/nfl/replay_adapter.py
"""
The NFL pipeline as the replay driver sees it (see ivan.core.replay): how to
put an archived slate back into the working tables and which scripts to run.
The stages are the live ones; nothing is reimplemented here.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable, Dict, List

import pandas as pd
from sqlalchemy import func

from ivan.core.config import config_path, load_config
from ivan.core.replay import run_script
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.models import NFLBase, ContestPayout, HistoricalContest, NFLGameOdds, NFLOdds, NFLPlayer

SCRIPTS = PROJECT_ROOT / "scripts" / "nfl"
SHOWDOWN_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
CLASSIC_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
CONTEST_TYPES = {"showdown": ("Showdown Captain Mode",), "classic": ("Classic",)}


def _dt(v):
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    d = datetime.fromisoformat(str(v))
    return d.astimezone().replace(tzinfo=None) if d.tzinfo else d


class NFLReplayAdapter:
    sport = "nfl"
    working_tables = ("nfl_player", "odds", "lineup", "lineup_simulation_results", "contest_sim_report")
    repeat_tables = ("lineup", "lineup_simulation_results", "contest_sim_report")   # rebuilt by every repeat
    metadata = NFLBase.metadata

    def __init__(self, config: str | Path | None = None):
        self.config_path = Path(config) if config else config_path("nfl")

    def models_info(self) -> dict:
        from ivan.sports.nfl.paths import models_info
        return models_info()

    # ---------------------------------------------------------------- restore
    def restore(self, session, blobs: Dict[str, List[dict]], log: Callable[[str], None]) -> dict:
        players = blobs.get("players", [])
        for r in players:
            session.add(NFLPlayer(id=int(r["id"]), roster_slot_id=int(r["roster_slot_id"]), name=r["name"], team=r["team"],
                                  opponent=r.get("opponent"), position=r["position"], gametime=_dt(r.get("gametime")),
                                  salary=int(r["salary"]), status=r.get("status")))
        session.commit()
        # the live ingest drops OUT/IR before fitting
        n_out = session.query(NFLPlayer).filter(func.upper(NFLPlayer.status).in_(["OUT", "IR"])).delete(synchronize_session=False)
        odds = blobs.get("odds", [])
        if odds:
            session.bulk_insert_mappings(NFLOdds, [{k: r.get(k) for k in ("player", "market", "sportsbook", "side", "point", "price", "last_update")}
                                                   for r in odds if r.get("point") is not None and r.get("price") is not None])
        n_lines = 0
        for r in blobs.get("game_lines", []):
            exists = session.query(NFLGameOdds.id).filter_by(game_id=r["game_id"], bookmaker=r["bookmaker"], market=r["market"],
                                                             outcome_name=r["outcome_name"], point=r.get("point"),
                                                             last_update=r.get("last_update")).first()
            if exists:
                continue
            session.add(NFLGameOdds(**{k: (_dt(v) if k == "commence_time" else v) for k, v in r.items() if k != "id"}))
            n_lines += 1
        session.commit()
        info = {"players": len(players), "removed_out": int(n_out), "odds": len(odds), "lines_added": n_lines}
        log(f"restored {info}")
        return info

    # ---------------------------------------------------------------- project
    def project(self, session, log: Callable[[str], None]) -> None:
        """The live ingest's projection step (ivan.sports.nfl.ingest.project_slate) on the restored capture; no snapshot."""
        from ivan.sports.nfl.ingest import project_slate
        info = project_slate(session, load_config("nfl", self.config_path), snapshot=False, artifacts_required=True, log=log)
        log(f"projected {info.players} players")

    # ---------------------------------------------------------------- field
    def generate(self, env: dict, n_lineups: int, jobs: int, log_path: Path, log) -> None:
        run_script([SCRIPTS / "generate_lineups.py", str(n_lineups), str(jobs), "-c", str(self.config_path)], env, log_path, log)

    # ---------------------------------------------------------------- contests
    def contests_for(self, session, lock_local: datetime, mode: str, slate_key: Optional[str] = None) -> List[dict]:
        """The slate's contests: its draft group's when any are known, else every contest of that mode at its lock."""
        # a naive local time -> UTC, with the DST rule of that date (not today's)
        lock_utc = lock_local.astimezone(timezone.utc).replace(tzinfo=None)
        rows = []
        if slate_key is not None and str(slate_key).isdigit():
            rows = session.query(HistoricalContest).filter(HistoricalContest.draft_group_id == int(slate_key)).all()
        if not rows:
            rows = (session.query(HistoricalContest)
                    .filter(HistoricalContest.date.between(lock_utc - timedelta(minutes=10), lock_utc + timedelta(minutes=10)),
                            HistoricalContest.contest_type.in_(CONTEST_TYPES.get(mode, ("Classic",)))).all())
        with_pay = {cid for (cid,) in session.query(ContestPayout.contest_id)
                    .filter(ContestPayout.contest_id.in_([r.contest_id for r in rows])).distinct().all()} if rows else set()
        return [{"contest_id": int(r.contest_id), "name": r.name or "", "entry_fee": float(r.entry_fee),
                 "total_entries": int(r.total_entries), "multi_entry": int(r.multi_entry or 0),
                 "has_payouts": int(r.contest_id) in with_pay, "draft_group_id": r.draft_group_id} for r in rows]

    def entries_csv(self, picks: List[dict], path: Path) -> int:
        """A DraftKings-style entries file for the picked contests; sim.py and choose.py read it as the real one."""
        rows = []
        k = 0
        for p in picks:
            for _ in range(int(p["entries"])):
                k += 1
                eid = int(f"9{p['contest_id']}{k:03d}")
                fee = p["entry_fee"]
                fee_txt = f"${fee:.2f}".rstrip("0").rstrip(".") if fee != int(fee) else f"${int(fee)}"
                rows.append([eid, p["name"], p["contest_id"], fee_txt])
        showdown = any("showdown" in (p["name"] or "").lower() for p in picks)
        cols = SHOWDOWN_COLS if showdown else CLASSIC_COLS
        df = pd.DataFrame([r + [""] * (len(cols) - 4) for r in rows], columns=cols)
        df.to_csv(path, index=False)
        return len(df)

    # ---------------------------------------------------------------- simulate / choose
    def simulate(self, env: dict, entries_csv: Path, n_sims: int, field_cap: int, log_path: Path, log) -> None:
        run_script([SCRIPTS / "sim.py", "--entries", str(entries_csv), "-n", str(n_sims), "-c", str(field_cap),
                    "--config", str(self.config_path)], env, log_path, log)

    def choose(self, env: dict, entries_csv: Path, out_dir: Path, weights: dict, log_path: Path, log) -> None:
        run_script([SCRIPTS / "choose.py", "--entries", str(entries_csv), "--no-prompt",
                    "--weights-json", json.dumps({"default": weights}), "-o", str(Path(out_dir) / "chosen_entries.csv")],
                   env, log_path, log)
