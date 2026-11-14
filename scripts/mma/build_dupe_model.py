#!/usr/bin/env python3
# scripts/mma/build_dupe_model.py
"""
Fit the MMA duplication model on real fields and write
data/mma/models/dupe_model.json (ivan.sports.mma.dupe_model): within a
contest, how likely an entry is to land on each valid lineup of the card,
from salary left, summed win probability, the lineup's rank, the main event,
and how each of those moves with the contest's entries-per-user cap.

    scripts/mma/build_dupe_model.py                     every derived contest whose card has a record
    scripts/mma/build_dupe_model.py --contest-ids 196337922,196426859
    scripts/mma/build_dupe_model.py --report            print the model on disk, fit nothing

Run scripts/mma/get_contest_data.py --played and scripts/mma/derive_contests.py
first. The fit prints, and stores in the file, the test that matters for a
live slate: each card predicted by a fit on the other cards, the real
fields' shape next to a field sampled from that prediction, by entry cap.
The file it replaces is kept as dupe_model.<built date>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.mma import dupe_model as DM
from ivan.sports.mma.cards import load_card, stored_cards
from ivan.sports.mma.models import HistoricalContest
from ivan.sports.mma.paths import MMA_DATA_DIR
from ivan.sports.mma.standings import DERIVED_DIR

MODEL_PATH = MMA_DATA_DIR / "models" / "dupe_model.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the MMA duplication model on real contest fields")
    ap.add_argument("--config", default=str(config_path("mma")))
    ap.add_argument("--contest-ids", default=None, help="comma-separated DK contest ids (default: every derived contest)")
    ap.add_argument("--min-entries", type=int, default=DM.MIN_ENTRIES, help="leave out contests with fewer entries")
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--report", action="store_true", help="print the model file and stop")
    a = ap.parse_args()

    out = Path(a.out)
    if a.report:
        if not out.exists():
            print(f"no model at {out}")
            return 1
        model = json.loads(out.read_text())
        if int(model.get("version", 1)) < DM.VERSION:
            print(f"{out} is a version-{model.get('version', 1)} model (no entry-cap terms); coefficients: {model.get('coef')}")
            return 0
        print(DM.format_model(model))
        return 0

    cfg = load_config("mma", a.config)
    session = get_session(init_db("mma", cfg.get("database_url")))
    derived = sorted(int(p.name) for p in DERIVED_DIR.iterdir() if p.name.isdigit()) if DERIVED_DIR.exists() else []
    ids = [int(x) for x in a.contest_ids.split(",") if x.strip()] if a.contest_ids else derived
    rows = session.query(HistoricalContest).filter(HistoricalContest.contest_id.in_(ids)).all()
    contests = [{"contest_id": int(r.contest_id), "draft_group_id": int(r.draft_group_id), "entry_cap": r.multi_entry}
                for r in rows if r.draft_group_id is not None]
    cards = {dg: load_card(dg) for dg in stored_cards()}
    no_card = sorted({c["draft_group_id"] for c in contests} - set(cards))
    if no_card:
        print(f"cards without a record, their contests left out: {no_card} (scripts/mma/derive_contests.py --card)")
    model = DM.build(contests, cards, DERIVED_DIR, min_entries=a.min_entries)
    if model is None:
        print("no derived contests on a recorded card; run get_contest_data.py and derive_contests.py first")
        return 1
    print()
    print(DM.format_model(model))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            built = str(json.loads(out.read_text()).get("built_at", "unknown"))[:19].replace(":", "")
        except ValueError:
            built = "unreadable"
        keep = out.with_name(f"{out.stem}.{built}.json")
        if not keep.exists():
            out.replace(keep)
            print(f"\nprevious model kept as {keep.name}")
    out.write_text(json.dumps(model, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
