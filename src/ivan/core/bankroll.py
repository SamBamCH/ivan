# src/ivan/core/bankroll.py
"""
The bankroll the slate book is chosen from (ivan.core.slate_book), read from what our entries made: the starting
figure plus the net of every sport's results ledger. One DraftKings balance plays every sport, so every sport's
results count, and the figure moves with them: a winning week raises the next slate's bankroll (bolder books), a
losing one lowers it (more cautious ones).

    bankroll = start + sum over sports of (won - fees) on the scored contests of the days from `since` on

config/bankroll.json: `start` (dollars; what the bankroll was before the first counted result), `since` (iso day,
or null for every result on file; set both after a deposit or a withdrawal: the balance that day, and that day) and
`sports` (null = nba, nfl and every other sport the imported DraftKings entry history holds). A contest that has
not been scored moves nothing: its fee is at risk, not lost. A consumer (the server pulling the laptop's data,
ivan.core.feed.producer_source) holds no results and shows no money: there the bankroll is the start alone.

    sport_ledger(sport, now)         one sport's results ledger (the ledgers scripts/pnl.py and the home page read)
    ledgers(now, sports)             ({sport: ledger}, {sport: error}) over the configured sports
    from_ledgers(ledgers, cfg)       the arithmetic: {"bankroll", "start", "since", "net", "sports": {sport: net}}
    current(now)                     the bankroll now, with what it is made of
    for_slate(values, log)           what a choose run uses: the explicit bankroll_dollars when a contest's weights
                                     give one (the largest), else current(); logs the line that says which
    describe(info)                   that line

    scripts/bankroll.py              the same figure in the terminal

A choose run that names bankroll_dollars (a flag, a contest's knob, a backtest's parameter) uses that figure; only a
blank one reads the results. Replays and backtests pin theirs (ivan.sports.nfl.replay_adapter), so a past slate's
book does not depend on results that came after it.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from ivan.core.utils import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "config" / "bankroll.json"
DEFAULT_START = 1000.0
BASE_SPORTS = ("nba", "nfl")
Log = Callable[[str], None]


def load_config(path: Optional[Path] = None) -> dict:
    """{"start", "since", "sports"} from config/bankroll.json (a missing file is the defaults)."""
    try:
        raw = json.loads((path or CONFIG).read_text())
    except OSError:
        raw = {}
    start = raw.get("start")
    since = raw.get("since")
    sports = raw.get("sports")
    return {"start": DEFAULT_START if start is None else float(start), "since": str(since)[:10] if since else None,
            "sports": [str(s).strip().lower() for s in sports] if sports else None}


def sport_ledger(sport: str, now: datetime) -> Optional[dict]:
    """One sport's results ledger (ivan.core.pnl.roll_up's shape); a sport with no results code of its own reads
    DraftKings' entry history alone (None when that has nothing of it)."""
    if sport == "nba":
        from ivan.sports.nba.backtest import results
    elif sport == "nfl":
        from ivan.sports.nfl import results
    else:
        from ivan.core import dk_history
        return dk_history.ledger(sport)
    return results.ledger_days(now)


def default_sports() -> list:
    """nba, nfl and every other sport the imported DraftKings entry history holds."""
    out = list(BASE_SPORTS)
    try:
        from ivan.core import dk_history
        out += [s for s in dk_history.sports() if s not in out]
    except Exception:   # noqa: BLE001 - no entry history on this machine
        pass
    return out


def ledgers(now: Optional[datetime] = None, sports: Optional[Iterable[str]] = None) -> Tuple[Dict[str, Optional[dict]], Dict[str, str]]:
    """({sport: ledger}, {sport: why its ledger could not be read}). One sport's trouble does not hide the others."""
    now = now or datetime.now()
    out, errors = {}, {}
    for s in (list(sports) if sports else default_sports()):
        try:
            out[s] = sport_ledger(s, now)
        except Exception as e:   # noqa: BLE001
            errors[s] = f"{type(e).__name__}: {e}"
    return out, errors


def _net(ledger: Optional[dict], since: Optional[str]) -> Tuple[float, int]:
    """(won - fees, scored contests) of a ledger's days from `since` on (every day when since is None)."""
    if not ledger:
        return 0.0, 0
    if since is None:
        o = ledger.get("overall") or {}
        return float(o.get("net") or 0.0), int(o.get("contests") or 0)
    days = [a for day, a in (ledger.get("days") or {}).items() if str(day)[:10] >= since]
    return float(sum(a.get("net") or 0.0 for a in days)), int(sum(a.get("contests") or 0 for a in days))


def from_ledgers(ledgers_by: Dict[str, Optional[dict]], cfg: Optional[dict] = None, errors: Optional[Dict[str, str]] = None) -> dict:
    """The bankroll from the ledgers: {"bankroll", "start", "since", "net", "contests", "sports": {sport: net}, "errors", "source"}."""
    cfg = cfg or load_config()
    by, contests = {}, 0
    for s, led in ledgers_by.items():
        net, n = _net(led, cfg["since"])
        if n:
            by[s] = net
            contests += n
    net = float(sum(by.values()))
    return {"bankroll": float(cfg["start"]) + net, "start": float(cfg["start"]), "since": cfg["since"], "net": net, "contests": contests,
            "sports": by, "errors": dict(errors or {}), "source": "results"}


def current(now: Optional[datetime] = None, cfg: Optional[dict] = None) -> dict:
    """The bankroll now: the start plus every configured sport's net (from_ledgers over ledgers)."""
    cfg = cfg or load_config()
    if is_consumer():
        return dict(from_ledgers({}, cfg), consumer=True)
    led, errors = ledgers(now, cfg["sports"])
    return from_ledgers(led, cfg, errors)


def is_consumer() -> bool:
    """The server pulling the laptop's data: the results are the laptop's, and no money is shown or read there."""
    from ivan.core import feed
    return feed.producer_source() is not None


def describe(info: dict) -> str:
    """One line: the figure and what it is made of."""
    if info.get("source") != "results":
        return f"bankroll ${info['bankroll']:,.2f} (given)"
    if info.get("consumer"):
        return f"bankroll ${info['bankroll']:,.2f} (the start in config/bankroll.json: this desk reads no results)"
    parts = ", ".join(f"{s.upper()} {'-' if v < 0 else '+'}${abs(v):,.2f}" for s, v in sorted(info["sports"].items()))
    line = (f"bankroll ${info['bankroll']:,.2f} = ${info['start']:,.2f} start {'-' if info['net'] < 0 else '+'} ${abs(info['net']):,.2f} from "
            f"{info['contests']} scored contest(s)" + (f" since {info['since']}" if info.get("since") else "") + (f" ({parts})" if parts else ""))
    if info.get("errors"):
        line += "; NOT COUNTED (results unreadable): " + "; ".join(f"{s}: {e}" for s, e in sorted(info["errors"].items()))
    return line


def for_slate(values: Iterable, log: Optional[Log] = None, now: Optional[datetime] = None) -> dict:
    """
    The bankroll of one choose run. `values` are the slate contests' bankroll_dollars (blank ones None): a figure
    given wins (the largest when they differ), with none given the bankroll is read from the results (current).
    Returns the info dict (its "bankroll" is the figure; "source" is "given" or "results") and logs describe().
    """
    log = log or (lambda *_: None)
    given = sorted({float(v) for v in values if v not in (None, "")})
    if given:
        if len(given) > 1:
            log(f"  NOTE: the slate contests carry different bankroll_dollars {given}; the slate uses ${given[-1]:,.0f}")
        info = {"bankroll": given[-1], "source": "given"}
    else:
        info = current(now)
        if info["bankroll"] <= 0:
            raise ValueError(f"the results have used up the bankroll ({describe(info)}): set start (and since) in config/bankroll.json "
                             "to the balance you are playing from, or pass --bankroll-dollars")
    log("  " + describe(info))
    return info
