#!/usr/bin/env python3
# scripts/mma/derive_contests.py
"""
Turn the scraped standings (get_contest_data.py) into fields the models can
read: per contest, data/mma/derived/contests/<id>/lineups.parquet (one row
per unique lineup: copies, salary left, summed win probability) and
players.csv (drafted %, actual points).

    scripts/mma/derive_contests.py                         every scraped contest whose card has a record
    scripts/mma/derive_contests.py --contest-ids 1,2,3
    scripts/mma/derive_contests.py --card 156055 --player-odds data/mma/input/player_odds.prev.csv

A field is laid on the card it was played on: the pool and the win
probabilities ingest.py stored for that draft group at its last run before
lock (ivan.sports.mma.cards). --card writes that record by hand for a card
ingested before records were kept: the pool from DraftKings, the Win column
of the given player_odds file, the latest --main-events fights as five
rounds. Contests whose card has no record are listed and skipped.
"""
import argparse
from pathlib import Path

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.mma.cards import save_card, stored_cards
from ivan.sports.mma.standings import derive_contests, win_probs_from_odds_csv


def record_card(draft_group: int, player_odds: str, main_events: int) -> None:
    from ivan.sports.mma.ingest import load_players_dk
    pool = load_players_dk(int(draft_group))
    if not pool:
        raise SystemExit(f"DraftKings returned no fighters for draft group {draft_group}")
    wp = win_probs_from_odds_csv(Path(player_odds), pool)
    missing = [r["name"] for r in pool if r["id"] not in wp]
    if len(missing) * 2 > len(pool):
        raise SystemExit(f"{player_odds} has Win odds for {len(pool) - len(missing)} of {len(pool)} fighters of draft group {draft_group}: "
                         "not this card's file; nothing stored")
    times = sorted({r["gametime"] for r in pool if r.get("gametime")}, reverse=True)[:max(0, main_events)]
    main = [r["id"] for r in pool if r.get("gametime") in times]
    for r in pool:
        r["win_prob"] = wp.get(r["id"])
    art = save_card(int(draft_group), pool, main, source=f"derive_contests --card, Win from {Path(player_odds).name}")
    print(f"card {draft_group}: {len(pool)} fighters stored ({art.path}); main event: "
          f"{', '.join(r['name'] for r in pool if r['id'] in main) or 'none'}"
          + (f"; no Win odds for {', '.join(missing)} (read as 0.5)" if missing else ""))


def main() -> int:
    p = argparse.ArgumentParser(description="Derive scraped MMA contest fields from the raw standings")
    p.add_argument("--config", default=str(config_path("mma")))
    p.add_argument("--contest-ids", default=None, help="comma-separated DK contest ids (default: every scraped contest)")
    p.add_argument("--card", type=int, default=None, help="store a card record by hand for this draft group first (needs --player-odds)")
    p.add_argument("--player-odds", default=None, help="with --card: that card's player_odds csv (its Win column)")
    p.add_argument("--main-events", type=int, default=None, help="with --card: how many of the latest fights are five rounds (default: slate.num_5_round_fights)")
    a = p.parse_args()

    cfg = load_config("mma", a.config)
    session = get_session(init_db("mma", cfg.get("database_url")))
    if a.card is not None:
        if not a.player_odds:
            p.error("--card needs --player-odds")
        n = a.main_events if a.main_events is not None else int(cfg.get("slate", {}).get("num_5_round_fights", 1))
        record_card(a.card, a.player_odds, n)
    ids = [int(x) for x in a.contest_ids.split(",") if x.strip()] if a.contest_ids else None
    r = derive_contests(session, ids)
    by_card: dict[int, list] = {}
    for s in r["derived"]:
        by_card.setdefault(s["draft_group"], []).append(s)
    for dg, ss in sorted(by_card.items()):
        print(f"card {dg}: {len(ss)} contests, {sum(s['entries_matched'] for s in ss):,} entries, biggest field "
              f"{max(s['entries'] for s in ss):,}, most copies of one lineup {max(s['max_copies'] for s in ss)}")
    for dg, cids in sorted(r["no_card"].items()):
        print(f"card {dg}: no card record, {len(cids)} scraped contest(s) skipped "
              f"(scripts/mma/derive_contests.py --card {dg} --player-odds <that card's file>)")
    if r["unknown"]:
        print(f"{len(r['unknown'])} contest id(s) with no row in the contest table: {r['unknown'][:8]}")
    print(f"derived {len(r['derived'])} contest(s); cards on record: {stored_cards()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
