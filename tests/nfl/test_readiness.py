# tests/nfl/test_readiness.py
"""The backtest plan: what each slate in a window has and lacks, on a small in-memory database."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ivan.core import archive
from ivan.core.replay import Menu, MenuItem
from ivan.sports.nfl import readiness
from ivan.sports.nfl.models import ContestFieldScores, ContestPayout, HistoricalContest, NFLBase, NFLStat, ShowdownPlayer


def _utc(local: datetime) -> datetime:
    return local.astimezone(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    NFLBase.metadata.create_all(engine)
    archive.ensure_tables(engine)
    with engine.begin() as con:                 # nfl_schedule has no ORM class (nflverse writes it with to_sql)
        con.execute(text("CREATE TABLE nfl_schedule (game_id TEXT, season INT, game_type TEXT, week INT, gameday TEXT, home_team TEXT, "
                         "away_team TEXT, home_score REAL, away_score REAL, spread_line REAL, total_line REAL)"))
    return sessionmaker(bind=engine)()


def _game(session, season, week, gameday, home, away):
    session.execute(text("INSERT INTO nfl_schedule (game_id, season, game_type, week, gameday, home_team, away_team) VALUES (:g, :s, 'REG', :w, :d, :h, :a)"),
                    {"g": f"{season}_{week}_{away}_{home}", "s": season, "w": week, "d": gameday, "h": home, "a": away})
    session.commit()


def _contest(session, cid, dg, name, lock_local, fee=3.0, size=1000, ctype="Showdown Captain Mode", payouts=True, scraped=False, ours=0):
    session.add(HistoricalContest(contest_id=cid, draft_group_id=dg, name=name, date=_utc(lock_local), entry_fee=fee, contest_type=ctype,
                                  total_entries=size, multi_entry=20))
    if payouts:
        session.add(ContestPayout(contest_id=cid, min_position=1, max_position=1, payout_amount=100.0))
    if scraped:
        session.add(ContestFieldScores(contest_id=cid, n_entries=size, scores=[10.0, 5.0]))
        session.add(ShowdownPlayer(contest_id=cid, player_id=1, team="AAA", position="QB", salary=10000, roster_slot_id=512,
                                   fpts_actual=20.0, ownership_actual=30.0))
    session.commit()


def test_plan_reports_captures_picks_scrapes_and_box_scores(session, tmp_path, monkeypatch):
    lock = datetime(2025, 11, 9, 19, 20)
    _game(session, 2025, 10, "2025-11-09", "BBB", "AAA")
    # slate 1: captured at ingest, two pool contests, one scraped, box scores for one of the two teams
    archive.archive_slate(session, sport="nfl", slate_key="111", lock=lock, mode="showdown", source="ingest", captured_at=lock - timedelta(minutes=50),
                          players=[{"id": 1, "team": "AAA"}, {"id": 2, "team": "BBB"}], odds=[{"player": "A"}], game_lines=[])
    _contest(session, 1, 111, "NFL Showdown $20K First Down [20 Entry Max]", lock, fee=1.0, size=5000, scraped=True)
    _contest(session, 2, 111, "NFL Showdown $5K Huddle [Single Entry]", lock, fee=5.0, size=1000, scraped=False)
    _contest(session, 3, 111, "NFL Showdown Satellite", lock, fee=20.0, size=50)        # not in the menu
    # slate 2: scraped contests but never captured
    _contest(session, 4, 222, "NFL Showdown $20K First Down [20 Entry Max]", lock - timedelta(hours=6), scraped=True)
    # noise: a draft group with nothing captured or scraped
    _contest(session, 5, 333, "NFL Showdown $20K First Down [20 Entry Max]", lock - timedelta(hours=3))
    session.add(NFLStat(player_id="p1", player_display_name="A", position="QB", season=2025, week=10, season_type="REG", team="AAA"))
    session.commit()
    monkeypatch.setattr(readiness, "MODELS_DIR", tmp_path / "models")
    menu = Menu(items=[MenuItem(match="First Down", entries=20), MenuItem(match="Huddle", entries=1)])
    plan = readiness.backtest_plan(session, lock - timedelta(days=1), lock + timedelta(days=1), menu, output_dir=tmp_path)
    by_key = {s.slate_key: s for s in plan["slates"]}
    assert set(by_key) == {"111", "222"} and plan["summary"]["other_draft_groups"] == 1
    s1 = by_key["111"]
    assert s1.captures == 1 and s1.capture_source == "ingest" and s1.capture_minutes_before_lock == 50.0
    assert s1.contests_at_lock == 3 and [p.contest_id for p in s1.picks] == [1, 2] and s1.picks[0].entries == 20
    assert s1.picks[0].scraped and not s1.picks[1].scraped and s1.scraped_picks == 1
    assert s1.season_week == (2025, 10) and s1.teams == 2 and s1.box_scores_teams == 1 and not s1.box_scores
    assert s1.replayable and not s1.settled and s1.status == "replayable, unsettled"
    assert s1.models["fallback"] and any("as-of" in m for m in s1.missing) and any("not scraped" in m for m in s1.missing)
    s2 = by_key["222"]
    assert s2.captures == 0 and not s2.replayable and s2.status == "not replayable" and s2.scraped_picks == 1
    assert plan["summary"]["unscraped_contest_ids"] == [2]            # only slates with a capture are worth scraping for
    assert plan["summary"]["replayable"] == 1 and plan["summary"]["picks"] == 3


def test_plan_is_ready_when_everything_is_there(session, tmp_path, monkeypatch):
    from datetime import date
    from ivan.core import model_store
    lock = datetime(2025, 11, 9, 19, 20)
    _game(session, 2025, 10, "2025-11-09", "BBB", "AAA")
    archive.archive_slate(session, sport="nfl", slate_key="111", lock=lock, mode="showdown", source="backfill", captured_at=lock - timedelta(minutes=30),
                          players=[{"id": 1, "team": "AAA"}, {"id": 2, "team": "BBB"}], odds=[], game_lines=[])
    _contest(session, 1, 111, "NFL Showdown $20K First Down [20 Entry Max]", lock, scraped=True)
    for t in ("AAA", "BBB"):
        session.add(NFLStat(player_id=t, player_display_name=t, position="QB", season=2025, week=10, season_type="REG", team=t))
    session.commit()
    model_store.write_manifest(model_store.asof_dir(tmp_path / "models", date(2025, 11, 1)), date(2025, 11, 1), {"dupe_model": {}})
    monkeypatch.setattr(readiness, "MODELS_DIR", tmp_path / "models")
    plan = readiness.backtest_plan(session, lock - timedelta(days=1), lock + timedelta(days=1),
                                   Menu(items=[MenuItem(match="First Down", entries=20)]), output_dir=tmp_path)
    s = plan["slates"][0]
    assert s.status == "ready" and s.settled and s.box_scores and not s.models["fallback"] and s.models_label == "as of 2025-11-01"
    assert s.missing == ["our entries' finishes not recorded (scrape with --username)"]
    assert plan["model_versions"][0]["cutoff"] == "2025-11-01"
