# src/ivan/sports/nfl/backtest_tax.py
"""
After-tax backtest of the lineups this pipeline chose, against what really
happened.

Every choose run writes chosen_entries.lineups.<stamp>.csv (one row per
player per entry, with the sim's cash % and ROI for the lineup) and, since
this module, chosen_entries.simroi.<stamp>.npz (the per-simulation ROI
trace of every pick). For each contest the last run before lock is "what we
would have submitted". Its entries are then settled, in order of preference:

  dk_history     a DraftKings contest-history export (--dk-history): the
                 real finish, points and winnings of entries actually played
  scraped_entry  our own entries in a scraped contest (contest_entry_result,
                 written by get_contest_data.py --username)
  reconstructed  the lineup's actual DraftKings points (scraped player
                 points, else nfl_stats / nfl_dst_game) placed in the real
                 field (contest_field_scores, else the showdown lineups the
                 scraper stored) and paid from the real payout table
  points_only    actual points known but no field to place them in
  no_actual      the week's box scores are not loaded yet

Winnings and losses per entry (spec section 2), the running tax ledger,
the scenario table, the per-format breakdown, the sim-vs-actual comparison
and the volume sweep all build on that list of EntryResult rows. The tax
rules themselves live in ivan.sports.nfl.tax.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nfl import actuals
from ivan.sports.nfl.choose import sim_roi_traces
from ivan.sports.nfl.models import (ContestEntryResult, ContestFieldScores, HistoricalContest, ShowdownLineup,
                                    ShowdownPlayer)
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.sports.nfl.placement import load_payout_tiers, payout_vector, place, places_paid, remove_our_copies
from ivan.sports.nfl.tax import TaxConfig, compute_tax, scenario_configs

CPT_SLOT, CPT_MULT = 511, 1.5
SLOT_CODE = {"CPT": 511, "FLEX": 512, "QB": 66, "RB": 67, "WR": 68, "TE": 69, "DST": 71}
CASH_WORDS = ("double up", "double-up", "doubleup", "50/50", "fifty", "head-to-head", "head to head", "h2h",
              "cash", "satellite", "qualifier", "3-man", "5-man", "beginner")
SOURCE_RANK = {"dk_history": 0, "scraped_entry": 1, "reconstructed": 2, "points_only": 3, "no_actual": 4}


def parse_money(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).replace("$", "").replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s or 0)
    except ValueError:
        return 0.0
    return -v if neg else v


def classify_format(name: str, places_paid_frac: Optional[float] = None) -> str:
    """cash (double-ups, 50/50s, H2H, satellites) or gpp, from the contest name, else the paid share of the field."""
    n = (name or "").lower()
    if any(w in n for w in CASH_WORDS):
        return "cash"
    if places_paid_frac is not None and places_paid_frac >= 0.4:
        return "cash"
    return "gpp"


def contest_kind(name: str, game_type: Optional[str] = None) -> str:
    s = f"{name or ''} {game_type or ''}".lower()
    return "showdown" if "showdown" in s or "captain" in s else "classic"


def utc_to_local(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def local_to_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- records
@dataclass
class EntryResult:
    contest_id: int
    contest_name: str
    entry_id: Optional[int]
    entry_fee: float
    lock: Optional[datetime]              # UTC, naive
    contest_type: str                     # showdown | classic
    format_class: str                     # cash | gpp
    payout: Optional[float] = None        # actual prize; None when it could not be settled
    points: Optional[float] = None
    rank: Optional[int] = None
    field_size: Optional[int] = None
    source: str = "no_actual"
    played: Optional[bool] = None         # True: we know it was entered (dk history / scraped own entry)
    lineup_id: Optional[int] = None
    run_stamp: Optional[str] = None
    sim_cash_pct: Optional[float] = None
    sim_avg_roi: Optional[float] = None
    sim_roi: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    note: str = ""

    @property
    def settled(self) -> bool:
        return self.payout is not None

    @property
    def winnings(self) -> float:
        return max(self.payout - self.entry_fee, 0.0) if self.settled else 0.0

    @property
    def losses(self) -> float:
        return max(self.entry_fee - self.payout, 0.0) if self.settled else 0.0

    @property
    def slate(self) -> str:
        return utc_to_local(self.lock).strftime("%Y-%m-%d %H:%M") if self.lock else "unknown"

    @property
    def tax_year(self) -> Optional[int]:
        return utc_to_local(self.lock).year if self.lock else None

    @property
    def sim_expected_payout(self) -> Optional[float]:
        if self.sim_roi is not None and np.isfinite(self.sim_roi).any():
            return float(self.entry_fee * (1.0 + np.nanmean(self.sim_roi)))
        if self.sim_avg_roi is not None:
            return float(self.entry_fee * (1.0 + self.sim_avg_roi))
        return None

    def row(self) -> dict:
        d = asdict(self)
        d.pop("sim_roi", None)
        d.update(slate=self.slate, tax_year=self.tax_year, winnings=self.winnings, losses=self.losses,
                 sim_expected_payout=self.sim_expected_payout, settled=self.settled)
        return d


def entries_frame(entries: Iterable[EntryResult]) -> pd.DataFrame:
    rows = [e.row() for e in entries]
    cols = ["tax_year", "slate", "lock", "contest_id", "contest_name", "contest_type", "format_class", "entry_id",
            "entry_fee", "payout", "winnings", "losses", "points", "rank", "field_size", "source", "played",
            "lineup_id", "run_stamp", "sim_cash_pct", "sim_avg_roi", "sim_expected_payout", "settled", "note"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- choose runs
@dataclass
class ChooseRun:
    stamp: str
    when_local: datetime
    lineups_csv: Path
    simroi_npz: Optional[Path] = None

    @property
    def when_utc(self) -> datetime:
        return local_to_utc(self.when_local)


def find_choose_runs(dirs: Sequence[Path] = (OUTPUT_DIR, OUTPUT_DIR / "web")) -> List[ChooseRun]:
    """Every step-5 run that left a per-lineup detail CSV, oldest first."""
    runs: Dict[str, ChooseRun] = {}
    for d in dirs:
        if not Path(d).exists():
            continue
        for p in Path(d).glob("chosen_entries.lineups.*.csv"):
            stamp = p.name.split(".")[2]
            try:
                when = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
            except ValueError:
                continue
            sim = p.with_name(f"chosen_entries.simroi.{stamp}.npz")
            manifest = p.with_name("replay.json")
            if not manifest.exists():
                manifest = p.parent.parent / "replay.json"
            if manifest.exists():                 # a replayed slate: the run stands in for a pick just before lock
                try:
                    when = datetime.fromisoformat(json.loads(manifest.read_text())["lock"]) - timedelta(minutes=1)
                except (KeyError, ValueError):
                    pass
            runs[f"{d}/{stamp}"] = ChooseRun(stamp, when, p, sim if sim.exists() else None)
    return sorted(runs.values(), key=lambda r: r.when_local)


def _contests_in_run(run: ChooseRun) -> List[int]:
    df = pd.read_csv(run.lineups_csv, usecols=["contest_id"])
    return sorted(int(c) for c in df.contest_id.dropna().unique())


def contest_info(session: Session, contest_ids: Iterable[int]) -> Dict[int, HistoricalContest]:
    ids = [int(c) for c in set(contest_ids)]
    out = {}
    for i in range(0, len(ids), 900):
        for hc in session.query(HistoricalContest).filter(HistoricalContest.contest_id.in_(ids[i:i + 900])).all():
            out[int(hc.contest_id)] = hc
    return out


def select_prelock_runs(runs: Sequence[ChooseRun], info: Dict[int, HistoricalContest]) -> Dict[int, ChooseRun]:
    """Per contest, the last choose run before its lock (the latest run at all when the lock is unknown)."""
    chosen: Dict[int, ChooseRun] = {}
    for run in runs:                                    # oldest first, so later runs overwrite
        for cid in _contests_in_run(run):
            lock = info[cid].date if cid in info else None
            if lock is None or run.when_utc <= lock:
                chosen[cid] = run
    return chosen


def our_entries(run: ChooseRun, contest_id: int) -> List[dict]:
    """The entries we chose for one contest in a run: fee, lineup id, sim stats, players with slots."""
    df = pd.read_csv(run.lineups_csv)
    df = df[df.contest_id == contest_id]
    traces: Dict[int, np.ndarray] = {}
    if run.simroi_npz is not None:
        z = np.load(run.simroi_npz)
        for eid, cid, roi in zip(z["entry_id"], z["contest_id"], z["roi"]):
            if int(cid) == contest_id:
                traces[int(eid)] = roi
    out = []
    for eid, g in df.groupby("entry_id", sort=False):
        first = g.iloc[0]
        players = [(str(r.slot), int(r.player_id), str(r.name), str(r.team), str(r.position), int(r.salary))
                   for r in g.sort_values("slot_no").itertuples()]
        out.append(dict(entry_id=int(eid), lineup_id=int(first.lineup_id), entry_fee=parse_money(first.entry_fee),
                        contest_name=str(first.contest_name), contest_type=str(first.format),
                        sim_cash_pct=_f(first.cash_pct), sim_avg_roi=_f(first.avg_roi),
                        sim_roi=traces.get(int(eid)), players=players))
    return out


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# --------------------------------------------------------------------------- actual points
def scraped_points(session: Session, contest_id: int) -> Dict[Tuple[int, str, str, int], float]:
    """DraftKings' own per-player points for a scraped showdown contest, keyed (roster slot, team, position, salary)."""
    rows = session.query(ShowdownPlayer).filter(ShowdownPlayer.contest_id == contest_id,
                                                ShowdownPlayer.fpts_actual.isnot(None)).all()
    return {(int(r.roster_slot_id), str(r.team), str(r.position).upper(), int(r.salary)): float(r.fpts_actual) for r in rows}


def score_lineup(players: Sequence[tuple], week_points: Dict, scraped: Dict) -> Tuple[Optional[float], List[str], List[str]]:
    """
    Actual DraftKings points of a lineup. Captain counts 1.5x. Scraped points
    (already the slot's base points) win over the box-score table. A player
    with no stat line while his team's box score is loaded scored 0 (inactive
    or no touches). Returns (score, names with no actual at all, names
    assumed 0); the score is None when any player is truly missing.
    """
    total, missing, zeroed = 0.0, [], []
    for slot, pid, name, team, position, salary in players:
        code = SLOT_CODE.get(slot.upper(), 512)
        key = (code, team, position.upper(), salary)
        base = scraped.get(key)
        if base is None:
            base = actuals.lookup_points(week_points, name, team, position)
        if base is None and actuals.team_played(week_points, team):
            base = 0.0
            zeroed.append(name)
        if base is None:
            missing.append(name)
            continue
        total += base * (CPT_MULT if code == CPT_SLOT else 1.0)
    return (None if missing else total), missing, zeroed


# --------------------------------------------------------------------------- the field
def field_scores(session: Session, contest_id: int) -> Optional[np.ndarray]:
    """Every entry's actual points in the real contest, descending, or None when the contest was never scraped."""
    row = session.get(ContestFieldScores, contest_id)
    if row is not None:
        return np.asarray(row.scores, dtype=np.float64)
    pts = {int(r.player_id): float(r.fpts_actual)
           for r in session.query(ShowdownPlayer).filter(ShowdownPlayer.contest_id == contest_id).all()
           if r.fpts_actual is not None}
    if not pts:
        return None
    scores = []
    for lu in session.query(ShowdownLineup).filter(ShowdownLineup.contest_id == contest_id).all():
        try:
            s = CPT_MULT * pts[int(lu.cpt_player_id)] + sum(pts[int(p)] for p in lu.flex_player_ids)
        except KeyError:
            continue
        scores.extend([s] * int(lu.dupe_count or 1))
    return np.sort(np.asarray(scores, dtype=np.float64))[::-1] if scores else None


# --------------------------------------------------------------------------- DK history export
DK_HISTORY_ALIASES = {
    "contest_id": ("contest_key", "contestkey", "contest_id", "contestid"),
    "entry_id": ("entry_key", "entrykey", "entry_id", "entryid"),
    "contest_name": ("entry", "contest", "contest_name", "contestname"),
    "game_type": ("game_type", "gametype"),
    "sport": ("sport",),
    "date": ("contest_date_est", "contest_date", "date", "contestdate"),
    "place": ("place", "rank"),
    "points": ("points", "fpts"),
    "winnings": ("winnings_non_ticket", "winnings", "winnings_cash"),
    "winnings_ticket": ("winnings_ticket",),
    "entry_fee": ("entry_fee", "entryfee", "fee"),
    "contest_entries": ("contest_entries", "entries", "total_entries"),
    "places_paid": ("places_paid", "placespaid"),
}


def load_dk_history(path: str | Path, sport: str = "NFL") -> pd.DataFrame:
    """
    A DraftKings contest-history export (My Contests -> Contest History ->
    download) as one row per entry with normalized columns. Only `sport`
    rows are kept; ticket winnings are added to cash winnings.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    norm = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in raw.columns}
    def col(key):
        for alias in DK_HISTORY_ALIASES[key]:
            if alias in norm:
                return raw[norm[alias]]
        return None
    need = {k: col(k) for k in ("contest_id", "entry_id", "entry_fee", "winnings")}
    missing = [k for k, v in need.items() if v is None]
    if missing:
        raise ValueError(f"{path}: cannot find column(s) {missing}; have {list(raw.columns)}")
    df = pd.DataFrame({
        "contest_id": pd.to_numeric(need["contest_id"], errors="coerce"),
        "entry_id": pd.to_numeric(need["entry_id"], errors="coerce"),
        "contest_name": col("contest_name") if col("contest_name") is not None else "",
        "game_type": col("game_type") if col("game_type") is not None else "",
        "sport": col("sport") if col("sport") is not None else sport,
        "date": pd.to_datetime(col("date"), errors="coerce") if col("date") is not None else pd.NaT,
        "place": pd.to_numeric(col("place"), errors="coerce") if col("place") is not None else np.nan,
        "points": pd.to_numeric(col("points"), errors="coerce") if col("points") is not None else np.nan,
        "entry_fee": need["entry_fee"].map(parse_money),
        "winnings": need["winnings"].map(parse_money)
                    + (col("winnings_ticket").map(parse_money) if col("winnings_ticket") is not None else 0.0),
        "contest_entries": pd.to_numeric(col("contest_entries"), errors="coerce") if col("contest_entries") is not None else np.nan,
        "places_paid": pd.to_numeric(col("places_paid"), errors="coerce") if col("places_paid") is not None else np.nan,
    })
    df = df[df.sport.astype(str).str.upper().str.strip().isin({sport.upper(), ""})]
    df = df.dropna(subset=["contest_id", "entry_id"]).copy()
    df["contest_id"] = df.contest_id.astype(int)
    df["entry_id"] = df.entry_id.astype(int)
    return df.reset_index(drop=True)


def _history_lock(row, info: Dict[int, HistoricalContest]) -> Optional[datetime]:
    hc = info.get(int(row.contest_id))
    if hc is not None and hc.date is not None:
        return hc.date
    if pd.notna(row.date):
        # DK stamps contest history in US/Eastern
        try:
            from zoneinfo import ZoneInfo
            return row.date.to_pydatetime().replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            return row.date.to_pydatetime()
    return None


# --------------------------------------------------------------------------- assembly
@dataclass
class Coverage:
    contests_chosen: int = 0
    contests_settled: int = 0
    entries: Dict[str, int] = field(default_factory=dict)
    unsettled_contests: List[dict] = field(default_factory=list)
    history_only_entries: int = 0
    notes: List[str] = field(default_factory=list)


def assemble(session: Session, runs: Optional[Sequence[ChooseRun]] = None, dk_history: Optional[pd.DataFrame] = None,
             in_field: str = "auto", since: Optional[datetime] = None, until: Optional[datetime] = None,
             played_only: bool = False, week_cache: Optional[Dict[Tuple[int, int], Dict]] = None) -> Tuple[List[EntryResult], Coverage]:
    """
    One EntryResult per entry we chose (settled by the best source available)
    plus every entry in the DK history export we have no choose run for.
    `in_field` says whether a scraped field already contains our entries
    (yes / no / auto: drop one field entry per matching score).
    """
    runs = list(find_choose_runs()) if runs is None else list(runs)
    cov = Coverage()
    hist_by_entry: Dict[int, pd.Series] = {}
    if dk_history is not None:
        for r in dk_history.itertuples(index=False):
            hist_by_entry[int(r.entry_id)] = r

    all_cids = {c for r in runs for c in _contests_in_run(r)}
    if dk_history is not None:
        all_cids |= set(dk_history.contest_id.astype(int))
    info = contest_info(session, all_cids)
    by_contest = select_prelock_runs(runs, info)
    weeks = actuals.week_lookup(session)
    week_cache = {} if week_cache is None else week_cache
    results: List[EntryResult] = []
    seen_entry_ids = set()

    def week_points(lock: Optional[datetime]) -> Dict:
        if lock is None:
            return {}
        sw = actuals.week_for_date(session, utc_to_local(lock), weeks)
        if sw is None:
            return {}
        if sw not in week_cache:
            week_cache[sw] = actuals.actual_points(session, *sw)
        return week_cache[sw]

    for cid, run in sorted(by_contest.items(), key=lambda kv: (info[kv[0]].date if kv[0] in info and info[kv[0]].date else datetime.max)):
        hc = info.get(cid)
        lock = hc.date if hc is not None else None
        if (since and lock and lock < since) or (until and lock and lock > until):
            continue
        ours = our_entries(run, cid)
        if not ours:
            continue
        cov.contests_chosen += 1
        name = hc.name if hc is not None and hc.name else ours[0]["contest_name"]
        tiers = load_payout_tiers(session, cid)
        field_n = int(hc.total_entries) if hc is not None else None
        fmt = classify_format(name, (places_paid(tiers) / field_n) if (tiers and field_n) else None)
        ctype = ours[0]["contest_type"] if ours[0]["contest_type"] in ("showdown", "classic") else contest_kind(name, hc.contest_type if hc else None)

        base = [EntryResult(contest_id=cid, contest_name=name, entry_id=o["entry_id"], entry_fee=o["entry_fee"], lock=lock,
                            contest_type=ctype, format_class=fmt, lineup_id=o["lineup_id"], run_stamp=run.stamp,
                            sim_cash_pct=o["sim_cash_pct"], sim_avg_roi=o["sim_avg_roi"], sim_roi=o["sim_roi"])
                for o in ours]
        # sim traces still in the database (the current slate) when the run predates the npz files
        if all(e.sim_roi is None for e in base):
            traces = sim_roi_traces(session, name, [o["lineup_id"] for o in ours])
            for e in base:
                e.sim_roi = traces.get(e.lineup_id)

        # 1) DK history: the real finish of entries actually played
        pending = []
        for e in base:
            h = hist_by_entry.get(e.entry_id)
            if h is not None:
                e.payout, e.points = float(h.winnings), _f(h.points)
                e.rank = int(h.place) if pd.notna(h.place) else None
                e.field_size = int(h.contest_entries) if pd.notna(h.contest_entries) else field_n
                e.source, e.played = "dk_history", True
                if h.entry_fee and abs(h.entry_fee - e.entry_fee) > 0.005:
                    e.note = f"fee in history {h.entry_fee} != chosen {e.entry_fee}"
                seen_entry_ids.add(e.entry_id)
            else:
                pending.append(e)
        # 2) our scraped entries
        if pending:
            scraped_own = {int(r.entry_id): r for r in session.query(ContestEntryResult)
                           .filter(ContestEntryResult.contest_id == cid).all()}
            still = []
            for e in pending:
                r = scraped_own.get(e.entry_id)
                if r is not None:
                    e.payout, e.points, e.rank = float(r.payout), _f(r.points), int(r.rank)
                    e.field_size = field_n
                    e.source, e.played = "scraped_entry", True
                else:
                    still.append(e)
            pending = still
        # 3) reconstruct: actual points placed in the real field
        if pending:
            wp = week_points(lock)
            sp = scraped_points(session, cid)
            field = field_scores(session, cid)
            scored = []
            for e, o in zip(base, ours):
                if e not in pending:
                    continue
                s, missing, zeroed = score_lineup(o["players"], wp, sp)
                e.points = s
                if s is None:
                    e.source = "no_actual"
                    e.note = ("no box scores for this week yet" if not wp and not sp else f"no actual for: {', '.join(missing)}")
                    if dk_history is not None and e.played is None:
                        e.played = False
                else:
                    if zeroed:
                        e.note = f"no stat line, scored 0: {', '.join(zeroed)}"
                    scored.append(e)
            if scored and field is not None and tiers:
                our_scores = np.array([e.points for e in scored])
                fld = field
                removed = 0
                if in_field in ("auto", "yes"):
                    fld, removed = remove_our_copies(field, our_scores)
                ranks, pays = place(our_scores, fld, payout_vector(tiers, fld.size + our_scores.size))
                for e, r, p in zip(scored, ranks, pays):
                    e.rank, e.payout, e.field_size = int(r), float(p), int(fld.size + our_scores.size)
                    e.source = "reconstructed"
                    if dk_history is not None and e.played is None:
                        e.played = False
                if in_field == "auto" and 0 < removed < len(scored):
                    cov.notes.append(f"{name}: {removed}/{len(scored)} of our scores found in the field and removed (in-field auto)")
            else:
                for e in scored:
                    e.source = "points_only"
                    why = "no field scores (scrape the contest)" if field is None else "no payout table (get_payouts.py --all)"
                    e.note = "; ".join(filter(None, [why, e.note]))
                    if dk_history is not None and e.played is None:
                        e.played = False
        if played_only:
            base = [e for e in base if e.played]
        results.extend(base)
        if any(e.settled for e in base):
            cov.contests_settled += 1
        else:
            cov.unsettled_contests.append({"contest_id": cid, "name": name, "lock": lock, "entries": len(base),
                                           "why": base[0].note or base[0].source})

    # entries in the history we never chose (played by hand, or before the detail CSVs existed)
    if dk_history is not None:
        for r in dk_history.itertuples(index=False):
            if int(r.entry_id) in seen_entry_ids:
                continue
            lock = _history_lock(r, info)
            if (since and lock and lock < since) or (until and lock and lock > until):
                continue
            hc = info.get(int(r.contest_id))
            name = hc.name if hc is not None and hc.name else str(r.contest_name)
            frac = (r.places_paid / r.contest_entries) if (pd.notna(r.places_paid) and pd.notna(r.contest_entries) and r.contest_entries) else None
            results.append(EntryResult(
                contest_id=int(r.contest_id), contest_name=name, entry_id=int(r.entry_id), entry_fee=float(r.entry_fee),
                lock=lock, contest_type=contest_kind(name, str(r.game_type)), format_class=classify_format(name, frac),
                payout=float(r.winnings), points=_f(r.points), rank=int(r.place) if pd.notna(r.place) else None,
                field_size=int(r.contest_entries) if pd.notna(r.contest_entries) else None,
                source="dk_history", played=True, note="no choose run for this entry"))
            cov.history_only_entries += 1

    for e in results:
        cov.entries[e.source] = cov.entries.get(e.source, 0) + 1
    results.sort(key=lambda e: (e.lock or datetime.max, e.contest_id, e.entry_id or 0))
    return results, cov


# --------------------------------------------------------------------------- ledger
def slate_totals(entries: Iterable[EntryResult]) -> pd.DataFrame:
    """Per slate (lock time, local): entries, fees, payout, W, L, P, plus the sim's expected payout."""
    rows = {}
    for e in entries:
        if not e.settled:
            continue
        r = rows.setdefault(e.slate, {"slate": e.slate, "lock": e.lock, "entries": 0, "fees": 0.0, "payout": 0.0,
                                      "W": 0.0, "L": 0.0, "sim_expected_payout": 0.0, "sim_known": 0, "contests": set()})
        r["entries"] += 1
        r["fees"] += e.entry_fee
        r["payout"] += e.payout
        r["W"] += e.winnings
        r["L"] += e.losses
        r["contests"].add(e.contest_id)
        sx = e.sim_expected_payout
        if sx is not None:
            r["sim_expected_payout"] += sx
            r["sim_known"] += 1
    out = []
    for r in sorted(rows.values(), key=lambda r: r["lock"] or datetime.max):
        r["contests"] = len(r["contests"])
        r["P"] = r["W"] - r["L"]
        out.append(r)
    return pd.DataFrame(out)


def ledger(entries: Iterable[EntryResult], cfg: TaxConfig, start_bankroll: float = 0.0,
           scenarios: Optional[Sequence[TaxConfig]] = None) -> pd.DataFrame:
    """
    The running tax picture, one row per slate: cumulative W, L, P, fees,
    bankroll, the base config's deductible / itemize / taxable / phantom /
    tax / after-tax, the reserve, and whether the tax owed so far exceeds
    the bankroll (start + P). Extra scenarios add tax[label] and
    after_tax[label] columns.
    """
    slates = slate_totals(entries)
    if slates.empty:
        return slates
    scenarios = list(scenarios) if scenarios is not None else []
    cum = {"W": 0.0, "L": 0.0, "fees": 0.0}
    out = []
    for r in slates.to_dict("records"):
        cum["W"] += r["W"]; cum["L"] += r["L"]; cum["fees"] += r["fees"]
        t = compute_tax(cum["W"], cum["L"], cfg, total_fees=cum["fees"])
        P = cum["W"] - cum["L"]
        bankroll = start_bankroll + P
        reserve = (cfg.tax_reserve_pct or 0.0) * cum["W"]
        row = {"slate": r["slate"], "lock": r["lock"], "contests": r["contests"], "entries": r["entries"],
               "slate_fees": r["fees"], "slate_W": r["W"], "slate_L": r["L"], "slate_P": r["P"],
               "cum_fees": cum["fees"], "W": cum["W"], "L": cum["L"], "P": P, "bankroll": bankroll,
               "deductible": t.deductible, "itemize": t.itemize, "taxable_gambling": t.taxable_gambling,
               "phantom": t.phantom, "est_tax": t.est_tax, "after_tax_profit": t.after_tax_profit,
               "after_tax_roi": t.after_tax_roi, "tax_reserve": reserve, "bankroll_net_of_reserve": bankroll - reserve,
               "reserve_covers_tax": reserve >= t.est_tax if cfg.tax_reserve_pct else None,
               "tax_exceeds_bankroll": t.est_tax > bankroll, "tax_shortfall": max(0.0, t.est_tax - bankroll)}
        for sc in scenarios:
            ts = compute_tax(cum["W"], cum["L"], sc, total_fees=cum["fees"])
            row[f"tax[{sc.label}]"] = ts.est_tax
            row[f"after_tax[{sc.label}]"] = ts.after_tax_profit
        out.append(row)
    return pd.DataFrame(out)


def scenario_table(entries: Iterable[EntryResult], base: TaxConfig, start_bankroll: float = 0.0) -> pd.DataFrame:
    es = [e for e in entries if e.settled]
    W, L, fees = sum(e.winnings for e in es), sum(e.losses for e in es), sum(e.entry_fee for e in es)
    rows = []
    for sc in scenario_configs(base):
        t = compute_tax(W, L, sc, total_fees=fees)
        rows.append({"scenario": sc.label, "rules": sc.describe(), "W": W, "L": L, "P": t.P, "deductible": t.deductible,
                     "itemize": t.itemize, "taxable_gambling": t.taxable_gambling, "phantom": t.phantom,
                     "est_tax": t.est_tax, "after_tax_profit": t.after_tax_profit, "after_tax_roi": t.after_tax_roi,
                     "effective_rate": t.effective_rate, "tax_exceeds_bankroll": t.est_tax > start_bankroll + t.P})
    return pd.DataFrame(rows)


def format_table(entries: Iterable[EntryResult], cfg: TaxConfig, by: str = "format_class") -> pd.DataFrame:
    """
    Per format (cash vs gpp, or contest_type, or both): entries, fees, W, L,
    P, losses per dollar of profit, and the tax the format adds: est_tax of
    the whole year minus est_tax of the year without that format.
    """
    es = [e for e in entries if e.settled]
    if not es:
        return pd.DataFrame()
    keys = by.split("+")
    key = lambda e: tuple(getattr(e, k) for k in keys)
    W_all, L_all, fees_all = sum(e.winnings for e in es), sum(e.losses for e in es), sum(e.entry_fee for e in es)
    tax_all = compute_tax(W_all, L_all, cfg, fees_all).est_tax
    rows = []
    for k in sorted({key(e) for e in es}):
        g = [e for e in es if key(e) == k]
        W, L, fees = sum(e.winnings for e in g), sum(e.losses for e in g), sum(e.entry_fee for e in g)
        P = W - L
        t_without = compute_tax(W_all - W, L_all - L, cfg, fees_all - fees).est_tax
        t_alone = compute_tax(W, L, cfg, fees)
        rows.append({**dict(zip(keys, k)), "entries": len(g), "contests": len({e.contest_id for e in g}), "fees": fees,
                     "payout": sum(e.payout for e in g), "W": W, "L": L, "P": P, "roi": (P / fees) if fees else None,
                     "cash_rate": float(np.mean([e.payout >= e.entry_fee for e in g])),
                     "losses_per_fee": (L / fees) if fees else None,
                     "losses_per_profit": (L / P) if P > 0 else None,
                     "phantom_alone": t_alone.phantom, "tax_alone": t_alone.est_tax,
                     "tax_added": tax_all - t_without, "after_tax_profit_alone": t_alone.after_tax_profit})
    return pd.DataFrame(rows)


def sim_vs_actual(entries: Iterable[EntryResult]) -> pd.DataFrame:
    """Per contest: what the sim expected of our picks against what they really paid."""
    rows = {}
    for e in entries:
        r = rows.setdefault(e.contest_id, {"slate": e.slate, "contest_id": e.contest_id, "contest_name": e.contest_name,
                                           "contest_type": e.contest_type, "format_class": e.format_class, "entries": 0,
                                           "fees": 0.0, "sim_expected_payout": 0.0, "sim_cash_pct": [], "sim_n": 0,
                                           "actual_payout": 0.0, "actual_cashed": 0, "settled": 0, "source": e.source,
                                           "sim_p_profit": []})
        r["entries"] += 1
        r["fees"] += e.entry_fee
        sx = e.sim_expected_payout
        if sx is not None:
            r["sim_expected_payout"] += sx
            r["sim_n"] += 1
        if e.sim_cash_pct is not None:
            r["sim_cash_pct"].append(e.sim_cash_pct)
        if e.settled:
            r["settled"] += 1
            r["actual_payout"] += e.payout
            r["actual_cashed"] += int(e.payout >= e.entry_fee)
    out = []
    for r in rows.values():
        r["sim_cash_pct"] = float(np.mean(r["sim_cash_pct"])) if r["sim_cash_pct"] else None
        r.pop("sim_p_profit")
        r["sim_expected_profit"] = (r["sim_expected_payout"] - r["fees"]) if r["sim_n"] == r["entries"] else None
        r["actual_profit"] = (r["actual_payout"] - r["fees"]) if r["settled"] == r["entries"] else None
        r["actual_cash_rate"] = (r["actual_cashed"] / r["settled"]) if r["settled"] else None
        out.append(r)
    return pd.DataFrame(out).sort_values(["slate", "contest_id"]).reset_index(drop=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- volume sweep
def _sim_slate_draws(entries: Sequence[EntryResult], n_draws: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    (W, L) of one slate under the sim, n_draws times. Within a contest all of
    our entries see the same simulation; contests are drawn independently
    (the simulator samples each contest's player outcomes separately, so
    same-slate correlation across contests is not available). Entries with
    no trace contribute their expected payout.
    """
    W = np.zeros(n_draws); L = np.zeros(n_draws)
    by_contest: Dict[int, List[EntryResult]] = {}
    for e in entries:
        by_contest.setdefault(e.contest_id, []).append(e)
    for es in by_contest.values():
        traced = [e for e in es if e.sim_roi is not None and np.isfinite(e.sim_roi).any()]
        if traced:
            n_sims = min(int(np.isfinite(e.sim_roi).sum()) for e in traced)
            j = rng.integers(0, n_sims, size=n_draws)
            for e in traced:
                roi = e.sim_roi[np.isfinite(e.sim_roi)][:n_sims]
                pay = e.entry_fee * (1.0 + roi[j])
                W += np.maximum(pay - e.entry_fee, 0.0); L += np.maximum(e.entry_fee - pay, 0.0)
        for e in es:
            if e not in traced:
                pay = e.sim_expected_payout
                if pay is None:
                    continue
                W += max(pay - e.entry_fee, 0.0); L += max(e.entry_fee - pay, 0.0)
    return W, L


def volume_sweep(entries: Sequence[EntryResult], cfg: TaxConfig, start_bankroll: float = 0.0,
                 multipliers: Sequence[float] = (0.25, 0.5, 1, 2, 4, 8), n_boot: int = 4000, seed: int = 0,
                 basis: str = "actual", sim_draws: int = 400) -> pd.DataFrame:
    """
    The same edge at more or less volume. A year is `multiplier` x the
    observed number of slates, each slate resampled with replacement:
      basis="actual"  from each slate's realized W and L (keeps the real
                      same-slate correlation between contests)
      basis="sim"     from the simulator's traces of the picks (the sim's
                      view of the edge, contests independent)
    Reports expected and p10/p90 after-tax profit, the chance of a losing
    year after tax, the expected phantom income and the chance that the tax
    owed exceeds the bankroll (start + P).
    """
    rng = np.random.default_rng(seed)
    settled = [e for e in entries if e.settled] if basis == "actual" else [e for e in entries if e.sim_expected_payout is not None]
    if not settled:
        return pd.DataFrame()
    slates: Dict[str, List[EntryResult]] = {}
    for e in settled:
        slates.setdefault(e.slate, []).append(e)
    keys = sorted(slates)
    n_slates = len(keys)
    fees_per = np.array([sum(e.entry_fee for e in slates[k]) for k in keys])
    if basis == "actual":
        Wm = np.array([[sum(e.winnings for e in slates[k])] for k in keys])
        Lm = np.array([[sum(e.losses for e in slates[k])] for k in keys])
    else:
        draws = [_sim_slate_draws(slates[k], sim_draws, rng) for k in keys]
        Wm = np.vstack([d[0] for d in draws]); Lm = np.vstack([d[1] for d in draws])
    n_inner = Wm.shape[1]
    rows = []
    for m in multipliers:
        n = max(1, int(round(m * n_slates)))
        idx = rng.integers(0, n_slates, size=(n_boot, n))
        jdx = rng.integers(0, n_inner, size=(n_boot, n))
        W = Wm[idx, jdx].sum(axis=1); L = Lm[idx, jdx].sum(axis=1); fees = fees_per[idx].sum(axis=1)
        P = W - L
        res = [compute_tax(w, l, cfg) for w, l in zip(W, L)]
        tax = np.array([r.est_tax for r in res]); after = P - tax
        phantom = np.array([r.phantom for r in res])
        rows.append({"basis": basis, "multiplier": m, "slates": n, "fees": float(fees.mean()),
                     "pre_tax_profit": float(P.mean()), "pre_tax_p10": float(np.percentile(P, 10)), "pre_tax_p90": float(np.percentile(P, 90)),
                     "W": float(W.mean()), "L": float(L.mean()),
                     "est_tax": float(tax.mean()), "phantom": float(phantom.mean()),
                     "after_tax_profit": float(after.mean()), "after_tax_p10": float(np.percentile(after, 10)),
                     "after_tax_p90": float(np.percentile(after, 90)),
                     "after_tax_roi": float(after.mean() / fees.mean()) if fees.mean() else None,
                     "p_pretax_loss": float((P < 0).mean()), "p_aftertax_loss": float((after < 0).mean()),
                     "p_tax_exceeds_bankroll": float((tax > start_bankroll + P).mean()),
                     "p_tax_exceeds_profit": float((tax > np.maximum(P, 0)).mean())})
    return pd.DataFrame(rows)
