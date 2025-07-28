# scripts/mlb/overnight.py
#!/usr/bin/env python3
import os, sys, re
import requests
from datetime import datetime, timezone



# ——— 2) Ensure repo root on PYTHONPATH —————————————————————
ROOT = os.path.abspath(os.path.join(__file__, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ——— 3) Orchestrate the 3 steps —————————————————————————
from ivan.database.session import init_db, get_session
from ivan.sports.mlb.stat_fetcher        import fetch_and_store
from ivan.sports.mlb.historical_contests import get_contest_data
from ivan.sports.mlb.build_models        import build_models
from ivan.sports.common.draftkings       import get_upcoming_contest_ids
from ivan.sports.mlb.models              import HistoricalContest

def _parse_sd(sd_str: str) -> datetime:
    """
    Parse timestamps of the form '/Date(1747782300000)/' into a UTC datetime.
    """
    m = re.search(r"/Date\((\d+)\)/", sd_str or "")
    if not m:
        return None
    ts = int(m.group(1)) / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def main():
    # 1) last night’s box scores
    fetch_and_store(days_back=1)

    # 2) parse & model contest data
    get_contest_data()

    # 3) rebuild all models
    build_models()

    # 4) get_contest_ids
    engine = init_db("mlb")
    db = get_session(engine)

    sess = requests.Session()

    contests = get_upcoming_contest_ids(sess, sport="MLB", days_ahead=2)

    for c in contests:
        # use 'id' as contest_id
        cid = str(c["id"])
        existing = db.query(HistoricalContest).filter_by(contest_id=cid).one_or_none()

        contest_date = _parse_sd(c.get("sd")) or datetime.utcnow()
        entry_fee    = float(c.get("a", 0))
        total_entries= int(c.get("m", 0))

        if not existing:
            hc = HistoricalContest(
                contest_id    = cid,
                draft_group_id= c.get("dg"),
                name          = c.get("n"),
                date          = contest_date,
                entry_fee     = entry_fee,
                contest_type  = c.get("gameType"),
                total_entries = total_entries,
                multi_entry   = c.get("mec")
            )
            db.add(hc)

    db.commit()
    print(f"✅ Saved {len(contests)} upcoming contests to the database.")


if __name__ == "__main__":
    main()