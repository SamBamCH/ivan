"""
A slate is a directory: data/nba/slates/<draft group>/ holds everything the desk knows about one DraftKings slate on this
machine (docs/nba/notes/plan_slates_2026-11-03.md), so two slates never displace each other and a slate's pre-lock
record is the directory.

    slate.json        draft_group, lock, day, label, games, created_at, refreshed_at, refreshed_by, published_at | pulled_at
    slate.sqlite      the slate's own tables (SLATE_TABLES): the pool, starters, market rates, notes, lineups, sims, contests
    projection/       the projection store (projection.store), <TEAM>.pkl
    input/            entries.csv, contest_weights.json
    output/           lineups, chosen entries, exposures, late swap files, sim artifacts

The edits are the day's, not the slate's: a game is mapped once, and the 6:00 and 6:30 slates of one evening share the
mapping, while the next day starts clean (nothing of yesterday's grids, pins or scenarios applies to today's games).
data/nba/days/<YYYY-MM-DD>/ holds them, reached through the slate's `day` in slate.json (SlatePaths.edits):

    edits/            this machine's minutes_overrides.csv, rotations/<TEAM>.json, scenarios/<TEAM>/ (and _active.json)
    published/        a consumer's base layer: the laptop's edits as pulled

A slate whose day is not known yet (its record not written) keeps them under its own directory until it is.

The slate's engine opens slate.sqlite as main and ATTACHes the history database as `hist`: SQLite resolves an
unqualified table name in main first, then in the attached databases, so `nba_player` is the slate's and `nba_stats`
the shared history with no query changed. ensure_slate_schema creates the slate tables in the slate file only;
ingest.ensure_schema creates the rest in the history file only (a history table created in a slate file would shadow
the real one).

The *current slate* is a context variable: a script sets it once (`use(dg)`, from --slate), the web per request. Every
path below is a function of it; code that needs a slate with none set raises NoSlate, loudly.
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event

from ivan.database.session import get_session, init_db
from ivan.sports.nba.paths import NBA_DATA_DIR

SLATES_DIR = NBA_DATA_DIR / "slates"
DAYS_DIRNAME = "days"                 # beside the slates directory: data/nba/days/<day>/{edits,published}
_DAYS: dict = {}                      # (slates root, dg) -> day, read once from slate.json (a day never changes once known)

# the tables that belong to one slate (created in slate.sqlite); every other NBA table lives in the history database
SLATE_TABLES = ("nba_player", "starting_lineups", "nba_market_rate", "nba_player_note", "lineup", "lineup_pool",
                "lineup_simulation_results", "contest_sim_report", "historical_contest_information", "payouts",
                "classic_contest_metadata", "projection_snapshot")
# the slate tables a producer publishes (the consumer's lineups, sims and reports are its own)
SLATE_FEED_TABLES = ("nba_player", "starting_lineups", "nba_market_rate", "historical_contest_information", "payouts", "classic_contest_metadata")


class NoSlate(RuntimeError):
    pass


_current: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("ivan_nba_slate", default=None)


def use(dg: int) -> contextvars.Token:
    """Make `dg` the current slate for this context (a script, a request, a thread). Returns the token for `release`."""
    return _current.set(int(dg))


def release(token: contextvars.Token) -> None:
    _current.reset(token)


def current() -> int:
    dg = _current.get()
    if dg is None:
        raise NoSlate("no slate in context: scripts take --slate DG (ivan.sports.nba.slates.use), the web sets it from the address")
    return dg


def current_or_none() -> Optional[int]:
    return _current.get()


class using:
    """`with slates.using(dg): ...` — the slate for a block (and the thread that runs it)."""
    def __init__(self, dg: int):
        self.dg = int(dg); self.token = None

    def __enter__(self):
        self.token = use(self.dg); return self.dg

    def __exit__(self, *exc):
        release(self.token)


# ------------------------------------------------------------------------------ layout
@dataclass(frozen=True)
class SlatePaths:
    dg: int
    root: Path
    day: Optional[str] = None         # the slate's day (slate.json "day"): its edits and published layer are the day's
    days: Optional[Path] = None       # the days directory (data/nba/days)

    @property
    def meta(self) -> Path: return self.root / "slate.json"
    @property
    def db(self) -> Path: return self.root / "slate.sqlite"
    @property
    def day_root(self) -> Optional[Path]:
        """data/nba/days/<day>: where the day's edits live; None while the slate's day is unknown."""
        return self.days / self.day if self.day and self.days is not None else None
    @property
    def edits(self) -> Path: return (self.day_root or self.root) / "edits"
    @property
    def overrides_csv(self) -> Path: return self.edits / "minutes_overrides.csv"
    @property
    def rotations(self) -> Path: return self.edits / "rotations"
    @property
    def scenarios(self) -> Path: return self.edits / "scenarios"
    @property
    def published(self) -> Path: return (self.day_root or self.root) / "published"
    @property
    def published_overrides_csv(self) -> Path: return self.published / "minutes_overrides.csv"
    @property
    def published_rotations(self) -> Path: return self.published / "rotations"
    @property
    def store(self) -> Path: return self.root / "projection"
    @property
    def input(self) -> Path: return self.root / "input"
    @property
    def entries_csv(self) -> Path: return self.input / "entries.csv"
    @property
    def contest_weights_json(self) -> Path: return self.input / "contest_weights.json"
    @property
    def output(self) -> Path: return self.root / "output"
    @property
    def chosen_entries_csv(self) -> Path: return self.output / "chosen_entries.csv"
    @property
    def lateswap_best_csv(self) -> Path: return self.output / "lateswap_best.csv"

    def ensure(self) -> "SlatePaths":
        for d in (self.root, self.edits, self.rotations, self.scenarios, self.store, self.input, self.output):
            d.mkdir(parents=True, exist_ok=True)
        return self


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else SLATES_DIR          # read at call time: tests point SLATES_DIR at a temporary directory


def days_dir(root: Optional[Path] = None) -> Path:
    """The days directory beside the slates directory: data/nba/days (the edits of each day, shared by its slates)."""
    return _root(root).parent / DAYS_DIRNAME


def _day_of(dg: int, base: Path) -> Optional[str]:
    """The slate's day from its record, cached once known (None, re-read every time, until the record carries it)."""
    key = (str(base), int(dg))
    if key in _DAYS:
        return _DAYS[key]
    try:
        d = json.loads((base / str(dg) / "slate.json").read_text()).get("day")
    except (OSError, ValueError):
        d = None
    if d:
        _DAYS[key] = str(d)[:10]
        return _DAYS[key]
    return None


def paths(dg: Optional[int] = None, root: Optional[Path] = None) -> SlatePaths:
    """The layout of slate `dg` (the current slate when None) under `root` (data/nba/slates); its edits under the day's directory."""
    dg = int(dg) if dg is not None else current()
    base = _root(root)
    return SlatePaths(dg=dg, root=base / str(dg), day=_day_of(dg, base), days=base.parent / DAYS_DIRNAME)


# the per-slate paths the code reads as functions of the current slate (the constants they replace are in the docstring)
def overrides_csv() -> Path: return paths().overrides_csv                          # MINUTES_OVERRIDES_CSV
def published_overrides_csv() -> Path: return paths().published_overrides_csv      # PUBLISHED_OVERRIDES_CSV
def rotations_dir() -> Path: return paths().rotations                              # ROTATIONS_DIR
def published_rotations_dir() -> Path: return paths().published_rotations          # PUBLISHED_ROTATIONS_DIR
def scenarios_dir() -> Path: return paths().scenarios                              # SCENARIOS_DIR
def store_dir() -> Path: return paths().store                                      # projection.store.STORE_DIR


def is_consumer() -> bool:
    """This machine pulls a producer's feed (IVAN_FEED_SSH / IVAN_FEED_URL set): only then does a published layer mean anything."""
    from ivan.core import feed
    return feed.producer_source() is not None


def published_pins() -> Optional[Path]:
    """The day's published pins file (the laptop's, as pulled) when this machine is a consumer and it exists; None otherwise. A
    producer ignores a published layer it may find (2026-11-04: a test pulled the laptop's own edits back under it as 'published',
    and every toggle then cleared the local value and the copy showed through)."""
    p = paths().published_overrides_csv
    return p if is_consumer() and p.exists() else None


def published_grid(team: str) -> Optional[Path]:
    """The day's published grid file for the team, on a consumer, when it exists; None otherwise."""
    p = paths().published_rotations / f"{team}.json"
    return p if is_consumer() and p.exists() else None


def output_dir() -> Path: return paths().output                                    # OUTPUT_DIR
def entries_csv() -> Path: return paths().entries_csv                              # DEFAULT_ENTRIES_CSV
def contest_weights_json() -> Path: return paths().contest_weights_json
def chosen_entries_csv() -> Path: return paths().chosen_entries_csv                # CHOSEN_ENTRIES_CSV
def lateswap_best_csv() -> Path: return paths().lateswap_best_csv                  # LATESWAP_BEST_CSV


# ------------------------------------------------------------------------------ meta
def read_meta(dg: Optional[int] = None, root: Optional[Path] = None) -> dict:
    p = paths(dg, root).meta
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def write_meta(dg: Optional[int] = None, root: Optional[Path] = None, **fields) -> dict:
    """Merge `fields` into slate.json (datetimes as ISO text). Creates the slate's directory (not the day's: `create` does that once the
    day is known)."""
    sp = paths(dg, root)
    sp.root.mkdir(parents=True, exist_ok=True)
    meta = read_meta(sp.dg, root)
    for k, v in fields.items():
        meta[k] = v.isoformat(timespec="seconds") if isinstance(v, datetime) else (v.isoformat() if isinstance(v, date) else v)
    meta["draft_group"] = sp.dg
    tmp = sp.meta.with_suffix(".json.tmp"); tmp.write_text(json.dumps(meta, indent=1, sort_keys=True)); os.replace(tmp, sp.meta)
    if "day" in fields:
        _DAYS.pop((str(_root(root)), sp.dg), None)                     # re-read: the day is known now (or changed by a migration)
    return meta


def meta_time(meta: dict, key: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(meta[key])) if meta.get(key) else None
    except ValueError:
        return None


def touch(dg: int, what: str, by: str = "desk", root: Optional[Path] = None) -> dict:
    """Record `what` happened to the slate now: refreshed (by desk | autopilot | script | feed), published, pulled."""
    now = datetime.now()
    fields = {f"{what}_at": now}
    if what == "refreshed":
        fields["refreshed_by"] = by
    return write_meta(dg, root, **fields)


# ------------------------------------------------------------------------------ the index
def held(root: Optional[Path] = None) -> list[int]:
    """The slates this machine holds: every directory with a slate.sqlite, newest draft group first."""
    root = _root(root)
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if d.is_dir() and d.name.isdigit() and (d / "slate.sqlite").exists():
            out.append(int(d.name))
    return sorted(out, reverse=True)


def holds(dg: int, root: Optional[Path] = None) -> bool:
    return (_root(root) / str(int(dg)) / "slate.sqlite").exists()


def held_meta(root: Optional[Path] = None) -> dict[int, dict]:
    return {dg: read_meta(dg, root) for dg in held(root)}


def open_slates(now: Optional[datetime] = None, grace_minutes: int = 5, root: Optional[Path] = None) -> list[int]:
    """The held slates that have not locked (lock from slate.json; a slate with no lock known counts as open), soonest lock first."""
    now = now or datetime.now()
    rows = []
    for dg, m in held_meta(root).items():
        lock = meta_time(m, "lock")
        if lock is None or (lock - now).total_seconds() / 60 > -grace_minutes:
            rows.append((lock or datetime.max, dg))
    return [dg for _, dg in sorted(rows)]


def main_slate_today(now: Optional[datetime] = None, root: Optional[Path] = None) -> Optional[int]:
    """The slate a flat address or a script with no --slate means: today's open held slate with the most games (else the
    soonest), else the next day's, else None."""
    now = now or datetime.now()
    best = None
    for dg, m in held_meta(root).items():
        lock = meta_time(m, "lock")
        if lock is None or lock.date() < now.date() or (lock.date() == now.date() and (lock - now).total_seconds() < -5 * 60):
            continue
        key = (lock.date(), -(m.get("games") or 0), lock)
        if best is None or key < best[0]:
            best = (key, dg)
    return best[1] if best else None


def resolve(arg: Optional[str] = None, root: Optional[Path] = None, now: Optional[datetime] = None) -> int:
    """The slate a script works: --slate DG when given, else today's main held slate; NoSlate with the held list otherwise."""
    if arg not in (None, ""):
        return int(arg)
    dg = main_slate_today(now, root=root)
    if dg is None:
        have = ", ".join(str(d) for d in held(root)) or "none"
        raise NoSlate(f"no open slate today on this machine; give --slate DG (held: {have})")
    return dg


# ------------------------------------------------------------------------------ scripts
def add_arg(ap) -> None:
    """`--slate DG` on a script's parser: the slate it works (default: today's main held slate, resolve())."""
    ap.add_argument("--slate", type=int, default=None, help="the slate (DraftKings draft group) this works; default: today's main held slate")


def open(arg=None, *, create: bool = False):
    """A script's entry: resolve the slate, make it current, make sure the history schema exists, and return (dg, the slate's
    session). `create` allows a slate this machine does not hold yet (ingest); otherwise a slate not held is an error."""
    from ivan.sports.nba.ingest import ensure_schema
    dg = resolve(str(arg) if arg is not None else None)
    if not create and not holds(dg):
        have = ", ".join(str(d) for d in held()) or "none"
        raise NoSlate(f"slate {dg} is not held on this machine (held: {have}); scripts/nba/ingest.py --draft-group {dg} fetches it")
    use(dg)
    ensure_schema(init_db("nba"))
    return dg, session(dg)


def open_any(arg=None, root: Optional[Path] = None):
    """The entry of a script that spans slates (standings, the field backtest): --slate DG when given, else today's main held
    slate, else the newest held slate (2026-11-06: `standings.py status` refused to run the morning after, when no slate was
    open). Any held slate's session reaches the shared history, which is what these scripts read."""
    if arg in (None, "") and main_slate_today(root=root) is None:
        have = held(root)
        if not have:
            raise NoSlate("no slate held on this machine; scripts/nba/ingest.py --draft-group DG fetches one")
        arg = have[0]
    return open(arg)


# ------------------------------------------------------------------------------ engines
_engines: dict[Path, object] = {}
_engines_lock = threading.Lock()
WRITE_LOCK = threading.RLock()            # one slate write transaction at a time in this process (two slates share the history file)


class writing:
    """`with slates.writing(session): ...` around a write transaction: this process's other slate writers wait (WRITE_LOCK), any
    read transaction of the session is ended, and the transaction begins IMMEDIATE, taking the write lock on the slate file and
    the attached history up front. A deferred transaction that read the history and then wrote it met another writer's lock
    with SQLite's immediate "database is locked" (no busy wait, to avoid a deadlock); IMMEDIATE waits the busy timeout instead.
    Reads are untouched: the driver begins no transaction for a SELECT, so a page never holds a lock between statements."""
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        WRITE_LOCK.acquire()
        from sqlalchemy import text
        self.session.rollback()
        self.session.execute(text("BEGIN IMMEDIATE"))
        return self.session

    def __exit__(self, *exc):
        WRITE_LOCK.release()


def history_db_path() -> Path:
    """The history database file (init_db('nba'): IVAN_NBA_DB_URL or data/nba/nba_history.sqlite)."""
    url = init_db("nba").url
    if url.drivername.startswith("sqlite") and url.database:
        return Path(url.database).resolve()
    raise RuntimeError(f"the NBA history database must be a SQLite file to attach it to a slate ({url})")


def engine(dg: Optional[int] = None, root: Optional[Path] = None, history: Optional[Path] = None):
    """The slate's engine: slate.sqlite as main, the history database attached as `hist`. Cached per file."""
    sp = paths(dg, root).ensure()
    hist = Path(history) if history is not None else history_db_path()
    with _engines_lock:
        eng = _engines.get(sp.db)
        if eng is None:
            eng = create_engine(f"sqlite:///{sp.db}", connect_args={"timeout": 30, "check_same_thread": False}, echo=False, future=True)

            @event.listens_for(eng, "connect")
            def _attach(dbapi_con, _rec, _hist=str(hist)):
                dbapi_con.execute("PRAGMA journal_mode=WAL")           # readers never block a writer's commit (the history file is WAL too)
                dbapi_con.execute(f"ATTACH DATABASE '{_hist}' AS hist")
            _engines[sp.db] = eng
    return eng


def session(dg: Optional[int] = None, root: Optional[Path] = None, history: Optional[Path] = None):
    """A session on the slate's engine with its tables in place."""
    eng = engine(dg, root, history)
    ensure_slate_schema(eng)
    return get_session(eng)


def ensure_slate_schema(eng) -> None:
    """Create the slate tables in the slate file (only them: a history table created here would shadow the attached one)."""
    from ivan.database.session import add_missing_columns
    from ivan.sports.nba.models import NBABase
    tables = [t for t in NBABase.metadata.sorted_tables if t.name in SLATE_TABLES]
    NBABase.metadata.create_all(eng, tables=tables)
    add_missing_columns(eng, NBABase.metadata, only=set(SLATE_TABLES))


def create(dg: int, *, lock: Optional[datetime] = None, label: str = "", games: Optional[int] = None, root: Optional[Path] = None,
           history: Optional[Path] = None) -> SlatePaths:
    """Make the slate's directory and database (idempotent) and note what is known about it. `history`: the history database to
    attach (default: this checkout's; the feed names the consumer's)."""
    meta = read_meta(dg, root)
    fields = {}
    if not meta.get("created_at"):
        fields["created_at"] = datetime.now()
    if lock is not None:
        fields["lock"] = lock; fields["day"] = lock.date()
    if label:
        fields["label"] = label
    if games is not None:
        fields["games"] = int(games)
    write_meta(dg, root, **fields)                                     # the record first: the day decides where the edits live
    sp = paths(dg, root).ensure()
    ensure_slate_schema(engine(dg, root, history))
    return sp


def each(root: Optional[Path] = None) -> Iterator[tuple[int, dict]]:
    for dg in held(root):
        yield dg, read_meta(dg, root)
