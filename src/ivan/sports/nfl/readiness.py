# src/ivan/sports/nfl/readiness.py
"""
A backtest plan: a window of slates, a contest pool (the replay menu), and
for every slate what the replay needs and what is there.

    plan = backtest_plan(session, since, until, menu)
    for s in plan["slates"]: s.status, s.missing, s.captures, s.picks ...

What a slate needs to be *replayable*: an archived capture (players, props,
game lines, taken before lock) and contests at lock that match the menu
(with payouts). What it needs to *settle*: the picked contests scraped
(field scores, player points and ownership), our own entries' finish when
we played it, and the week's box scores (nflverse) as the fallback for
player points. What makes the replay *honest*: an as-of model version whose
cutoff precedes the lock (ivan.core.model_store).

The slates come from two places: archived captures (ingest or backfill) and
scraped contests (draft groups) in the window; a slate with contests but no
capture is listed as needing a backfill. Everything here is read-only; the
actions it suggests are commands the page runs or shows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import pandas as pd
from sqlalchemy import text

from ivan.core import archive, model_store
from ivan.core import slates as slate_status
from ivan.core.replay import Menu, pick_contests
from ivan.database.session import read_sql
from ivan.sports.nfl import actuals
from ivan.sports.nfl.backtest_tax import utc_to_local
from ivan.sports.nfl.paths import MODELS_DIR

SPORT = "nfl"
MODE_OF_TYPE = {"Showdown Captain Mode": "showdown", "Classic": "classic"}
LOCK_TOLERANCE = timedelta(minutes=10)


@dataclass
class ContestReadiness:
    contest_id: int
    name: str
    entry_fee: float
    total_entries: int
    entries: int                      # what the menu enters
    has_payouts: bool
    max_entries: int = 1              # the contest's entry limit per user
    field_scores: bool = False        # contest_field_scores row (every entry's points)
    lineups: int = 0                  # showdown_lineup rows (the field's lineups with points; older scrapes)
    players: int = 0                  # showdown_player rows (points / ownership per player)
    players_with_points: int = 0
    players_with_ownership: int = 0
    our_entries: int = 0              # contest_entry_result rows (needs the scrape with --username)

    @property
    def scraped(self) -> bool:
        return (self.field_scores or self.lineups > 0) and self.players_with_points > 0


@dataclass
class SlateReadiness:
    slate_key: str
    mode: str
    lock: datetime
    captures: int = 0
    capture_source: Optional[str] = None
    capture_minutes_before_lock: Optional[float] = None
    capture_players: int = 0
    capture_odds: int = 0
    contests_at_lock: int = 0
    picks: List[ContestReadiness] = field(default_factory=list)
    contests: List[ContestReadiness] = field(default_factory=list)       # every contest of the slate (the picks are a subset)
    season_week: Optional[tuple] = None
    box_scores_teams: int = 0         # teams of the slate with nfl_stats rows for the week
    teams: int = 0
    box_scores_week_rows: int = 0
    models: Dict = field(default_factory=dict)
    replayed: List[str] = field(default_factory=list)      # replay roots that already hold this slate
    settle: Optional[dict] = None                          # the settle job's status row (ivan.core.slates), when it has run
    status: str = ""
    missing: List[str] = field(default_factory=list)

    @property
    def replayable(self) -> bool:
        return self.captures > 0 and bool(self.picks)

    @property
    def scraped_picks(self) -> int:
        return sum(1 for p in self.picks if p.scraped)

    @property
    def our_entries(self) -> int:
        return sum(p.our_entries for p in self.picks)

    @property
    def box_scores(self) -> bool:
        return self.teams > 0 and self.box_scores_teams >= self.teams if self.teams else self.box_scores_week_rows > 0

    @property
    def settled(self) -> bool:
        return bool(self.picks) and self.scraped_picks == len(self.picks)

    @property
    def models_label(self) -> str:
        m = self.models or {}
        return (f"as of {m['cutoff']}" if m.get("cutoff") else "live models") + (" (fallback)" if m.get("fallback") else "")


# --------------------------------------------------------------------------- queries
def _contests(session, since: datetime, until: datetime) -> pd.DataFrame:
    """Contests in the window (local lock times), with their scraped state."""
    q = text("""SELECT h.contest_id, h.draft_group_id, h.name, h.date, h.entry_fee, h.contest_type, h.total_entries, h.multi_entry,
                       (SELECT count(*) FROM payouts p WHERE p.contest_id = h.contest_id) AS n_pay,
                       (SELECT count(*) FROM contest_field_scores f WHERE f.contest_id = h.contest_id) AS n_field,
                       (SELECT count(*) FROM showdown_lineup l WHERE l.contest_id = h.contest_id) AS n_lineups,
                       (SELECT count(*) FROM showdown_player s WHERE s.contest_id = h.contest_id) AS n_players,
                       (SELECT count(*) FROM showdown_player s WHERE s.contest_id = h.contest_id AND s.fpts_actual IS NOT NULL) AS n_points,
                       (SELECT count(*) FROM showdown_player s WHERE s.contest_id = h.contest_id AND s.ownership_actual IS NOT NULL) AS n_own,
                       (SELECT count(*) FROM contest_entry_result e WHERE e.contest_id = h.contest_id) AS n_ours
                FROM historical_contest_information h WHERE h.date BETWEEN :a AND :b""")
    # contest dates are UTC; widen the window by a day each side and filter after converting
    df = read_sql(session, q.bindparams(a=(since - timedelta(days=1)).isoformat(sep=" "), b=(until + timedelta(days=1)).isoformat(sep=" ")))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.date)
    df["lock_local"] = [utc_to_local(d.to_pydatetime()) for d in df.date]
    df = df[(df.lock_local >= since) & (df.lock_local <= until)].copy()
    df["mode"] = df.contest_type.map(MODE_OF_TYPE).fillna("classic")
    return df


def _box_scores(session, season_week, teams: Sequence[str]) -> tuple[int, int]:
    """(teams of the slate with stats rows for that week, rows for the week)."""
    if season_week is None:
        return 0, 0
    s, w = season_week
    df = read_sql(session, text("SELECT team, count(*) AS n FROM nfl_stats WHERE season = :s AND week = :w GROUP BY team").bindparams(s=s, w=w))
    if df.empty:
        return 0, 0
    have = set(actuals.dk_team(t) for t in df.team)
    return sum(1 for t in teams if actuals.dk_team(t) in have), int(df.n.sum())


def _replayed_roots(output_dir, slate_dir_name: str) -> List[str]:
    out = []
    for root in sorted(output_dir.glob("replay*")):
        d = root / SPORT / slate_dir_name
        if (d / "replay.json").exists():
            out.append(root.name)
    return out


# --------------------------------------------------------------------------- the plan
def backtest_plan(session, since: datetime, until: datetime, menu: Menu, output_dir=None) -> dict:
    from ivan.core.replay import slate_dir_name
    from ivan.sports.nfl.paths import OUTPUT_DIR
    output_dir = output_dir or OUTPUT_DIR
    weeks = actuals.week_lookup(session)
    contests = _contests(session, since, until)
    caps_all = archive.captures(session, SPORT, since, until)
    chosen = {c.slate_key: c for c in archive.slates(session, SPORT, since, until)}

    # the slates: every archived slate key, plus every draft group with contests in the window
    slates: Dict[str, SlateReadiness] = {}
    for c in chosen.values():
        slates[c.slate_key] = SlateReadiness(c.slate_key, c.mode, c.lock)
    other_groups = 0
    if not contests.empty:
        for key, g in contests.groupby("draft_group_id"):
            key = str(key)
            if key not in slates and not ((g.n_points > 0) & ((g.n_field > 0) | (g.n_lineups > 0))).any():
                other_groups += 1                 # contests we never captured or scraped: nothing to test
                continue
            if key not in slates:
                mode = g["mode"].mode().iloc[0]
                slates[key] = SlateReadiness(key, mode, g.lock_local.min().to_pydatetime())

    try:
        status_rows = slate_status.by_key(session, SPORT, list(slates))
    except Exception:                                      # table not created yet
        status_rows = {}
    for key, s in slates.items():
        caps = [c for c in caps_all if c.slate_key == key]
        s.settle = status_rows[key].as_dict() if key in status_rows else None
        s.captures = len(caps)
        chosen_cap = chosen.get(key)
        if chosen_cap is not None:
            s.capture_source = chosen_cap.source
            s.capture_minutes_before_lock = round((chosen_cap.lock - chosen_cap.captured_at).total_seconds() / 60, 1)
            s.capture_players, s.capture_odds = chosen_cap.n_players, chosen_cap.n_odds
            players = archive.load_blobs(session, chosen_cap.id).get("players", [])
            teams = sorted({str(p.get("team")) for p in players if p.get("team")})
        else:
            teams = []
        # contests at lock (the replay's contests_for) and the menu's picks
        if contests.empty:
            at_lock = contests
        else:
            at_lock = contests[contests.draft_group_id.astype(str) == str(key)]
            if at_lock.empty:                   # contests scraped before draft groups were recorded: by lock time
                at_lock = contests[(contests.lock_local >= s.lock - LOCK_TOLERANCE) & (contests.lock_local <= s.lock + LOCK_TOLERANCE)
                                   & (contests["mode"] == s.mode)]
        s.contests_at_lock = int(len(at_lock))
        rows = [{"contest_id": int(r.contest_id), "name": r.name or "", "entry_fee": float(r.entry_fee), "total_entries": int(r.total_entries),
                 "multi_entry": int(r.multi_entry or 0), "has_payouts": int(r.n_pay) > 0, "draft_group_id": r.draft_group_id} for r in at_lock.itertuples()]
        picks = pick_contests(rows, menu, s.mode)
        by_id = {int(r.contest_id): r for r in at_lock.itertuples()}
        picked_ids = {int(p["contest_id"]) for p in picks}
        for c in sorted(rows, key=lambda c: -c["total_entries"]):
            r = by_id[int(c["contest_id"])]
            s.contests.append(ContestReadiness(int(c["contest_id"]), c["name"], float(c["entry_fee"]), int(c["total_entries"]),
                                               int(c.get("multi_entry") or 0) or 1, max_entries=int(c.get("multi_entry") or 0) or 1,
                                               has_payouts=bool(c["has_payouts"]), field_scores=int(r.n_field) > 0, lineups=int(r.n_lineups), players=int(r.n_players),
                                               players_with_points=int(r.n_points), players_with_ownership=int(r.n_own), our_entries=int(r.n_ours)))
        for p in picks:
            r = by_id[int(p["contest_id"])]
            s.picks.append(ContestReadiness(int(p["contest_id"]), p["name"], float(p["entry_fee"]), int(p["total_entries"]), int(p["entries"]),
                                            max_entries=int(p.get("multi_entry") or 0) or 1,
                                            has_payouts=bool(p["has_payouts"]), field_scores=int(r.n_field) > 0, lineups=int(r.n_lineups), players=int(r.n_players),
                                            players_with_points=int(r.n_points), players_with_ownership=int(r.n_own), our_entries=int(r.n_ours)))
        s.season_week = actuals.week_for_date(session, s.lock, weeks)
        s.teams = len(teams)
        s.box_scores_teams, s.box_scores_week_rows = _box_scores(session, s.season_week, teams)
        s.models = model_store.resolve(MODELS_DIR, s.lock).describe()
        if chosen_cap is not None:
            s.replayed = _replayed_roots(output_dir, slate_dir_name(chosen_cap))
        # status
        if s.captures == 0:
            s.missing.append("no archived capture (backfill_archive.py builds one from the projection snapshot; --odds costs credits)")
        if not s.picks:
            s.missing.append("no contest at lock matches the menu" if s.contests_at_lock else "no contests at lock in the database (get_contest_ids.py / get_payouts.py)")
        unscraped = [p for p in s.picks if not p.scraped]
        if unscraped:
            s.missing.append(f"{len(unscraped)} of {len(s.picks)} picked contests not scraped (get_contest_data.py --contest-ids ...)")
        if s.picks and s.our_entries == 0:
            s.missing.append("our entries' finishes not recorded (scrape with --username)")
        if not s.box_scores:
            s.missing.append("box scores for the week not loaded (fetch_actuals.py; nflverse publishes a day or two after the games)")
        if s.models.get("fallback"):
            s.missing.append(f"no as-of model version precedes {s.lock:%Y-%m-%d} (build_models.py --cutoff)")
        if not s.replayable:
            s.status = "not replayable"
        elif s.settled and s.box_scores:
            s.status = "ready"
        elif s.settled:
            s.status = "ready (contest points only)"
        else:
            s.status = "replayable, unsettled"
    ordered = sorted(slates.values(), key=lambda x: x.lock)
    n = len(ordered)
    summary = {"slates": n, "replayable": sum(1 for s in ordered if s.replayable), "settled": sum(1 for s in ordered if s.settled),
               "ready": sum(1 for s in ordered if s.status.startswith("ready")), "picks": sum(len(s.picks) for s in ordered),
               "picks_scraped": sum(s.scraped_picks for s in ordered), "our_entries": sum(s.our_entries for s in ordered),
               "models_fallback": sum(1 for s in ordered if s.models.get("fallback")),
               "other_draft_groups": other_groups,
               "unscraped_contest_ids": sorted({p.contest_id for s in ordered if s.captures for p in s.picks if not p.scraped}),
               "weeks_without_box_scores": sorted({f"{s.season_week[0]} wk {s.season_week[1]}" for s in ordered if s.season_week and not s.box_scores})}
    return {"since": since, "until": until, "slates": ordered, "summary": summary,
            "model_versions": [m.describe() for m in model_store.versions(MODELS_DIR)]}
