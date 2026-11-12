# tests/nfl/test_results.py
"""The NFL money (ivan.sports.nfl.results): our entries' recorded finishes against the entry fees, per contest, slate and day."""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import slates as slate_status
from ivan.sports.nfl import results as R
from ivan.sports.nfl.backtest_tax import utc_to_local
from ivan.sports.nfl.models import ContestEntryResult, HistoricalContest, NFLBase

LOCK = datetime(2026, 11, 9, 18, 0)          # UTC, mid-day in every US zone: the local day is 11-09
LOCK2 = datetime(2026, 11, 10, 18, 0)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'nfl.sqlite'}")
    NFLBase.metadata.create_all(engine)
    slate_status.ensure_tables(engine)
    s = sessionmaker(bind=engine)()
    s.add_all([
        HistoricalContest(contest_id=1, draft_group_id=700, name="Showdown $3 Play-Action", date=LOCK, entry_fee=3.0, contest_type="Showdown Captain Mode", total_entries=1000, multi_entry=20),
        HistoricalContest(contest_id=2, draft_group_id=700, name="Showdown $1 First Down", date=LOCK, entry_fee=1.0, contest_type="Showdown Captain Mode", total_entries=500, multi_entry=20),
        HistoricalContest(contest_id=3, draft_group_id=701, name="Classic $5 Huddle", date=LOCK2, entry_fee=5.0, contest_type="Classic", total_entries=200, multi_entry=1),
        HistoricalContest(contest_id=4, draft_group_id=702, name="Showdown never scraped", date=LOCK2, entry_fee=2.0, contest_type="Showdown Captain Mode", total_entries=300, multi_entry=20),
        HistoricalContest(contest_id=5, draft_group_id=702, name="Showdown scraped, not entered", date=LOCK2, entry_fee=2.0, contest_type="Showdown Captain Mode", total_entries=300, multi_entry=20),
        # contest 1: three entries at $3: one wins 10, one gets 2 back (a loss of 1), one gets nothing
        ContestEntryResult(entry_id=11, contest_id=1, username="us", rank=1, tie_size=1, points=120.5, payout=10.0, lineup="CPT A FLEX B"),
        ContestEntryResult(entry_id=12, contest_id=1, username="us", rank=40, tie_size=3, points=99.0, payout=2.0, lineup="CPT A FLEX B"),
        ContestEntryResult(entry_id=13, contest_id=1, username="us", rank=900, tie_size=1, points=40.0, payout=0.0, lineup="CPT C FLEX D"),
        ContestEntryResult(entry_id=21, contest_id=2, username="us", rank=300, tie_size=1, points=60.0, payout=0.0, lineup="CPT A FLEX B"),
        ContestEntryResult(entry_id=31, contest_id=3, username="us", rank=5, tie_size=1, points=180.0, payout=25.0, lineup="QB Q RB R"),
    ])
    s.commit()
    slate_status.upsert(s, "nfl", "702", mode="showdown", lock=utc_to_local(LOCK2), played=[4, 5], scraped=[5], status="waiting")
    slate_status.upsert(s, "nfl", "703", mode="showdown", lock=utc_to_local(LOCK2), played=[4], scraped=[], status="open")     # not past its lock: not ours to count
    yield s
    s.close()


@pytest.fixture
def chosen(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "chosen_entries.choose.20261110-100000.json").write_text(json.dumps({"4": {"entries": 20}, "5": {"entries": 20}}))
    (d / "chosen.choose.20261110-110000.json").write_text(json.dumps({"4": {"entries": 7}}))             # the later run stands
    return [d]


def test_contest_rows_are_the_recorded_finishes_against_the_fee(session):
    rows = {r["contest_id"]: r for r in R.contest_rows(session)}
    assert set(rows) == {1, 2, 3}
    c = rows[1]
    assert (c["found"], c["fees"], c["won"], c["net"], c["cashed"], c["best_rank"], c["best_points"]) == (3, 9.0, 12.0, 3.0, 2, 1, 120.5)
    assert c["gross_win"] == 7.0 and c["gross_loss"] == 4.0                       # 10 - 3 won; 1 + 3 lost: the tax's W and L
    assert c["gross_win"] - c["gross_loss"] == c["net"]
    assert c["final"] and c["mode"] == "showdown" and c["day"] == "2026-11-09" and c["draft_group"] == "700"
    assert rows[2]["roi"] == -1.0 and rows[2]["gross_loss"] == 1.0
    assert [r["contest_id"] for r in R.contest_rows(session, "700")] == [1, 2]    # one slate, biggest fees first


def test_unscored_is_played_past_lock_without_standings(session, chosen):
    rows = R.unscored_rows(session, dirs=chosen)
    assert [r["contest_id"] for r in rows] == [4]                                 # 5 has standings and no entry of ours: not entered
    u = rows[0]
    assert (u["final"], u["ours"], u["fees"], u["won"], u["net"], u["draft_group"]) == (False, 7, 14.0, None, None, "702")


def test_slate_results_and_contest_entries(session, chosen):
    res = R.slate_results(session, "700")
    assert res["complete"] and res["summary"]["contests"] == 2 and res["summary"]["net"] == pytest.approx(2.0)
    assert res["summary"]["entries"] == 4 and res["summary"]["cashed"] == 2
    cd = R.contest_entries(session, 1)
    assert [e["entry_id"] for e in cd["entries"]] == [11, 12, 13]                 # best first
    assert [e["net"] for e in cd["entries"]] == [7.0, -1.0, -3.0] and [e["copies"] for e in cd["entries"]] == [2, 2, 1]
    assert R.contest_entries(session, 4) is None and R.contest_entries(session, 999) is None


def test_ledger_days_rolls_up_per_day_and_overall(session, chosen):
    led = R.ledger_days(datetime(2026, 11, 12), session, dirs=chosen)
    ov = led["overall"]
    assert (ov["contests"], ov["entries"], ov["fees"], ov["won"], ov["net"]) == (3, 5, 15.0, 37.0, 22.0)
    assert (ov["gross_win"], ov["gross_loss"]) == (27.0, 5.0)
    assert (ov["days"], ov["slates"], ov["incomplete"], ov["unscored"], ov["fees_unscored"]) == (2, 3, 1, 1, 14.0)
    assert list(led["days"]) == ["2026-11-10", "2026-11-09"]                      # newest first
    d = led["days"]["2026-11-10"]
    assert d["net"] == 20.0 and sorted(d["slates"]) == ["701", "702"] and d["incomplete"] == 1
    assert led["slates"]["702"]["contests"] == 0 and not led["slates"]["702"]["complete"]
    assert "contest_entry_result" in led["source"]
