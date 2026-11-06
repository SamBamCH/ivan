"""A past day's readiness for the backtest (ivan.sports.nba.readiness): box scores on file, the money scored from final exports,
a failed scrape recorded; the day is the worst of its slates."""
import json
import sqlite3
from datetime import datetime

import pytest

from ivan.sports.nba import readiness as R
from ivan.sports.nba import slates as S


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SLATES_DIR", tmp_path / "slates")
    hist = tmp_path / "nba_history.sqlite"
    con = sqlite3.connect(hist)
    con.execute("create table nba_stats (id integer primary key, player_name text, team text, game_date text, minutes real)")
    con.execute("insert into nba_stats values (1, 'a b', 'AAA', '2026-11-04', 30.0)")
    con.execute("insert into nba_stats values (2, 'c d', 'BBB', '2026-11-04', 30.0)")
    con.commit(); con.close()
    monkeypatch.setattr(S, "history_db_path", lambda: hist)
    R.forget()
    return tmp_path


def _slate(dg, lock, teams):
    S.create(dg, lock=lock, games=len(teams) // 2)
    eng = S.engine(dg)
    with eng.begin() as con:
        for i, t in enumerate(teams):
            con.exec_driver_sql("insert into nba_player (id, name, team, position, roster_slot_id, gametime, salary, draft_group_id) values (?, ?, ?, ?, ?, ?, ?, ?)", (dg * 100 + i, f"p{i}", t, "SF", 1, lock.isoformat(), 5000, dg))


def _ledger(monkeypatch, complete, played=True, unscored=0):
    from ivan.sports.nba import results
    monkeypatch.setattr(results, "slate_ledger", lambda dg, now=None, force=False, root=None, session=None:
                        dict(complete=complete, entries_file=played, contests=[1] if played else [], summary=dict(unscored=unscored)))


def test_ready_when_box_scores_and_money_are_on_file(root, monkeypatch):
    _slate(155, datetime(2026, 11, 4, 18, 0), ["AAA", "BBB"])
    _ledger(monkeypatch, complete=True)
    st = R.slate_status(155, now=datetime(2026, 11, 6, 9, 0))
    assert st["state"] == "ready" and st["box"]["missing"] == []
    days = R.day_status(now=datetime(2026, 11, 6, 9, 0), force=True)
    assert days["2026-11-04"]["state"] == "ready" and days["2026-11-04"]["slates"] == [155]


def test_pending_then_failed_as_the_box_scores_stay_missing(root, monkeypatch):
    _slate(156, datetime(2026, 11, 5, 18, 0), ["AAA", "CCC"])          # CCC has no box score on 11-05
    _ledger(monkeypatch, complete=True, played=False)
    assert R.slate_status(156, now=datetime(2026, 11, 6, 9, 0))["state"] == "pending"          # 15 h after lock: the nightly brings them
    st = R.slate_status(156, now=datetime(2026, 11, 7, 9, 0))                                   # 39 h: it did not
    assert st["state"] == "failed" and "box scores missing" in st["why"] and "CCC" in st["why"]


def test_unscored_contests_are_pending_until_a_scrape_fails(root, monkeypatch):
    _slate(155, datetime(2026, 11, 4, 18, 0), ["AAA", "BBB"])
    _ledger(monkeypatch, complete=False, unscored=2)
    now = datetime(2026, 11, 6, 9, 0)
    assert R.slate_status(155, now=now)["state"] == "pending"
    p = R.scrape_record_path(155); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(at="2026-11-05T09:00:00", contests=[1, 2], ok=[1], failed=[2], login_needed=False)))
    st = R.slate_status(155, now=now)
    assert st["state"] == "failed" and "scrape" in st["why"]
    # the day takes the worst of its slates; a day of today or later has no status
    _slate(157, datetime(2026, 11, 4, 20, 0), ["AAA", "BBB"])
    _slate(158, datetime(2026, 11, 6, 20, 0), ["AAA", "BBB"])
    days = R.day_status(now=now, force=True)
    assert days["2026-11-04"]["state"] == "failed" and sorted(days["2026-11-04"]["slates"]) == [155, 157]
    assert "2026-11-06" not in days


def test_record_scrape_writes_the_held_slates_record(root):
    from ivan.sports.nba import standings as ST
    _slate(155, datetime(2026, 11, 4, 18, 0), ["AAA", "BBB"])
    plan = [dict(slate_key="155", to_scrape=[1, 2, 3]), dict(slate_key="999", to_scrape=[9])]        # 999 is not held: no record
    paths = ST.record_scrape(plan, dict(ok=[1], failed=[2, 3], login_needed=True), now=datetime(2026, 11, 5, 9, 0))
    assert paths == [R.scrape_record_path(155)]
    rec = R.read_scrape_record(155)
    assert rec["failed"] == [2, 3] and rec["ok"] == [1] and rec["login_needed"] is True and rec["at"].startswith("2026-11-05")
