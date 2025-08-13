# scripts/mlb/overnight.py
#!/usr/bin/env python3
import os, sys, re
import requests
from datetime import datetime, timezone, timedelta

# ——— Ensure repo root on PYTHONPATH —————————————————————
ROOT = os.path.abspath(os.path.join(__file__, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ——— Orchestrate the steps —————————————————————————————
from ivan.database.session import init_db, get_session
from ivan.database.management import (
    purge_stale_contest_tracking,
    counts_snapshot,             # optional diag
)
from ivan.sports.mlb.stat_fetcher        import fetch_and_store
from ivan.sports.mlb.historical_contests import get_contest_data
from ivan.sports.mlb.build_models        import build_models
from ivan.sports.common.draftkings       import get_upcoming_contest_ids
from ivan.sports.mlb.models              import HistoricalContest

SPORT = "mlb"
WINDOW_DAYS = 7


def _parse_sd(sd_str: str) -> datetime | None:
    """
    Parse '/Date(1747782300000)/' to aware UTC datetime.
    """
    m = re.search(r"/Date\((\d+)\)/", sd_str or "")
    if not m:
        return None
    ts = int(m.group(1)) / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def main() -> None:
    # DB/session
    engine = init_db(SPORT)
    db = get_session(engine)

    # 0) Purge stale tracking (> WINDOW_DAYS)
    purge_counts = purge_stale_contest_tracking(db, SPORT, days=WINDOW_DAYS)
    print(f"🧹 Purge: {purge_counts}")
    # 1) parse & model contest data (kept as-is; it can leverage ingest-status if present)
    get_contest_data()
    # 2) last night’s box scores (keep as-is)
    fetch_and_store(days_back=1)
    # 3) rebuild models (kept as-is)
    build_models()

    # 4) Save upcoming contests (window-guarded) for tracking/modeling
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(days=WINDOW_DAYS)

    sess = requests.Session()
    try:
        contests = get_upcoming_contest_ids(sess, sport="MLB", days_ahead=2) or []
    except Exception as e:
        contests = []
        print(f"[warn] get_upcoming_contest_ids failed: {type(e).__name__}: {e}")

    saved = 0
    considered = 0

    for c in contests:
        considered += 1
        cid = str(c.get("id"))
        if not cid:
            continue

        contest_date = _parse_sd(c.get("sd")) or now_utc
        # Only track contests within the last WINDOW_DAYS and not in the far past
        if contest_date < window_start:
            continue

        existing = db.query(HistoricalContest).filter_by(contest_id=cid).one_or_none()
        if existing:
            continue

        entry_fee      = float(c.get("a", 0) or 0)
        total_entries  = int(c.get("m", 0) or 0)

        hc = HistoricalContest(
            contest_id     = cid,
            draft_group_id = c.get("dg"),
            name           = c.get("n"),
            date           = contest_date,
            entry_fee      = entry_fee,
            contest_type   = c.get("gameType"),
            total_entries  = total_entries,
            multi_entry    = c.get("mec"),
        )
        db.add(hc)
        saved += 1

    db.commit()
    print(f"✅ Saved {saved} (of {considered}) upcoming contests to the database (window={WINDOW_DAYS}d).")

    # Optional window diagnostics (already-modeled, permanently-skipped)
    diag = counts_snapshot(db, SPORT, days=WINDOW_DAYS)
    print(f"📊 Window diag: {diag}")


if __name__ == "__main__":
    main()
