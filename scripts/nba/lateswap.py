#!/usr/bin/env python3
"""
scripts/nba/lateswap.py

Late swap our entries after lock: for every contest in the entries file, download
the live standings (the field's locked players), complete the field's open slots,
re-optimise our entries' open slots in sampled worlds and keep the candidate with
the best expected payout. Prints each player's exposure (entries holding him) before
and after the swap. Writes data/nba/output/lateswap.<stamp>.csv (the entries export
with the swapped lineups, ready to upload) and lateswap.<stamp>.json (the desk's report). Logic: ivan.sports.nba.post_lock.lateswap.

    scripts/nba/lateswap.py                          # DraftKings export through the logged-in Chrome profile
    scripts/nba/lateswap.py --source raw             # the newest stored export per contest (no Chrome)
    scripts/nba/lateswap.py --contest 197015856 --candidates 20 --sims 2000
"""
import argparse
import os
from datetime import datetime
from multiprocessing import freeze_support

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.paths import CHROME_PROFILE_DIR, DEFAULT_ENTRIES_CSV
from ivan.sports.nba.post_lock import lateswap as LS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config)")
    ap.add_argument("--source", choices=("draftkings", "raw"), default="draftkings", help="where the field comes from")
    ap.add_argument("--headless", action="store_true", help="headless Chrome (DraftKings answers 403 to headless sessions; only for testing)")
    ap.add_argument("--candidates", type=int, default=10, help="re-optimised lineups per entry")
    ap.add_argument("--field-cap", type=int, default=5000, help="field entries completed and scored (sampled when the contest is bigger)")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--contest", type=int, action="append", default=None, help="only this contest id (repeatable)")
    ap.add_argument("--live", action="store_true", help="score started games from the live box scores (banked points + the rest of the game)")
    ap.add_argument("--now", default=None, help="pretend it is this time (ISO), for a dry run before lock")
    ap.add_argument("--changed-only", action="store_true", help="the exposure table lists only players whose entry count changed")
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    entries = a.entries
    if entries is None:
        try:
            entries = load_config("nba")["paths"]["entries_csv"]
        except Exception:   # noqa: BLE001
            entries = str(DEFAULT_ENTRIES_CSV)
    now = datetime.fromisoformat(a.now) if a.now else datetime.now()
    http = None
    if a.source == "draftkings":
        from ivan.sports.common.draftkings import build_requests_session_from_chrome
        headless = a.headless or not os.environ.get("DISPLAY")
        if headless and not a.headless:
            print("no DISPLAY: Chrome runs headless (a logged-out profile fails; log in from a terminal session first)", flush=True)
        try:
            http = build_requests_session_from_chrome(str(CHROME_PROFILE_DIR), headless=headless)
        except RuntimeError as e:
            print(f"{e}\nfalling back to the newest stored exports (--source raw)", flush=True)
            a.source = "raw"
    report = LS.lateswap_entries(session, entries, now=now, source=a.source, http_sess=http, n_candidates=a.candidates,
                                 field_cap=a.field_cap, num_sims=a.sims, only=a.contest, log=lambda m: print(m, flush=True), use_live=a.live)
    paths = LS.write_outputs(session, report, entries, log=lambda m: print(m, flush=True))
    for c in report["contests"]:
        n_imp = sum(1 for e in c["entries"] if e["improved"])
        gain = sum(e["best"]["exp_payout"] - e["original"]["exp_payout"] for e in c["entries"])
        print(f"{c['name']}: {n_imp}/{len(c['entries'])} entries improve, +${gain:.2f} expected in total", flush=True)
    if report["failed"]:
        print("failed: " + ", ".join(f"{f['contest_id']} ({f['error']})" for f in report["failed"]), flush=True)
    if report["contests"]:
        print("\nexposure before -> after the swap" + (" (changed players only)" if a.changed_only else ""), flush=True)
        print(LS.format_exposure(report["exposure"], report["n_entries"], changed_only=a.changed_only), flush=True)
    print(f"upload: {paths['upload']}", flush=True)
    return 0 if report["contests"] else 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
