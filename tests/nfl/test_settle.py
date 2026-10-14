# tests/nfl/test_settle.py
"""The settle job: derived files from a standings export, lineup parsing, and the per-slate assessment."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ivan.core import archive, raw, slates as slate_status
from ivan.sports.nfl import settle as S
from ivan.sports.nfl.models import ContestFieldScores, HistoricalContest, NFLBase, NFLStat

CSV = """Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,501,sam (1/20),0,180.5,CPT Josh Allen FLEX Stefon Diggs FLEX Gabe Davis FLEX James Cook FLEX Dawson Knox FLEX Bills ,,Josh Allen,CPT,40.1%,60.2
2,502,other,0,150.25,CPT Stefon Diggs FLEX Josh Allen FLEX Gabe Davis FLEX James Cook FLEX Dawson Knox FLEX Bills ,,Stefon Diggs,FLEX,55.0%,30.1
3,503,sam (2/20),0,10,CPT Bills FLEX Josh Allen FLEX Gabe Davis FLEX James Cook FLEX Dawson Knox FLEX Tyler Bass ,,Bills,FLEX,20.0%,5
"""


def test_split_lineup_and_derive(tmp_path):
    assert S.split_lineup("CPT Josh Allen FLEX Stefon Diggs FLEX Bills ") == ["CPT Josh Allen", "FLEX Stefon Diggs", "FLEX Bills"]
    assert S.split_lineup("QB Jayden Daniels RB D'Andre Swift DST Falcons") == ["QB Jayden Daniels", "RB D'Andre Swift", "DST Falcons"]
    assert S.split_lineup("") == []
    m = S.derive_contest(777, CSV, out_dir=tmp_path)
    assert m["entries"] == 3 and m["slots"] == 6
    d = pd.read_parquet(tmp_path / "777" / "lineups.parquet")
    assert list(d.columns[:4]) == ["rank", "entry_id", "entry_name", "points"] and d.slot1.tolist()[0] == "CPT Josh Allen"
    scores = np.load(tmp_path / "777" / "scores.npy")
    assert scores.dtype == np.float32 and scores.tolist() == [180.5, 150.25, 10.0]


def test_derive_missing_reads_raw_standings(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "PROJECT_ROOT", tmp_path)
    raw.store("nfl", "dk/standings", 777, CSV, ext="csv", stamp="20261012-010000")
    out = tmp_path / "derived"
    assert S.derive_missing(out_dir=out, echo=lambda *_: None) == 1
    assert (out / "777" / "meta.json").exists()
    assert S.derive_missing(out_dir=out, echo=lambda *_: None) == 0            # idempotent


def _utc(local: datetime) -> datetime:
    return local.astimezone(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    NFLBase.metadata.create_all(engine)
    archive.ensure_tables(engine)
    slate_status.ensure_tables(engine)
    with engine.begin() as con:
        con.execute(text("CREATE TABLE nfl_schedule (game_id TEXT, season INT, game_type TEXT, week INT, gameday TEXT, home_team TEXT, "
                         "away_team TEXT, home_score REAL, away_score REAL, spread_line REAL, total_line REAL)"))
        con.execute(text("INSERT INTO nfl_schedule (game_id, season, game_type, week, gameday, home_team, away_team) VALUES "
                         "('2025_10_AAA_BBB', 2025, 'REG', 10, '2025-11-09', 'BBB', 'AAA')"))
    return sessionmaker(bind=engine)()


def test_assess_and_status(session, tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "PROJECT_ROOT", tmp_path)
    lock = datetime(2025, 11, 9, 19, 20)
    archive.archive_slate(session, sport="nfl", slate_key="111", lock=lock, mode="showdown", source="ingest", captured_at=lock - timedelta(minutes=40),
                          players=[{"id": 1, "team": "AAA"}, {"id": 2, "team": "BBB"}], odds=[], game_lines=[])
    for cid, name, size in ((1, "NFL Showdown $20K First Down [20 Entry Max]", 20000), (2, "NFL Showdown $5K Huddle [Single Entry]", 1000),
                            (3, "NFL Showdown $300K Play-Action", 90000), (4, "NFL Showdown Satellite", 30)):
        session.add(HistoricalContest(contest_id=cid, draft_group_id=111, name=name, date=_utc(lock), entry_fee=3.0, contest_type="Showdown Captain Mode",
                                      total_entries=size, multi_entry=20))
    session.add(ContestFieldScores(contest_id=3, n_entries=90000, scores=[1.0]))
    session.commit()
    # we played contests 1 and 2 (a choose json in the output dir)
    out = tmp_path / "out"; out.mkdir()
    (out / "chosen_entries.choose.20251109-180000.json").write_text('{"1": {}, "2": {}}')
    monkeypatch.setattr(S, "played_contests", lambda since=None, dirs=(): {1: "20251109-180000", 2: "20251109-180000"})
    items = S.assess(session, lock - timedelta(days=1), lock + timedelta(days=1), public_n=2, now=lock + timedelta(hours=12))
    assert len(items) == 1
    s = items[0]
    assert s.played == [1, 2] and s.public == [3, 1] and s.contests == [1, 2, 3]
    assert s.scraped == [3] and s.to_scrape == [1, 2] and not s.box_scores and s.status == "waiting"
    assert any("without standings" in m for m in s.missing) and any("box scores" in m for m in s.missing)
    # before the grace period the slate is open
    assert S.assess(session, lock - timedelta(days=1), lock + timedelta(days=1), now=lock + timedelta(hours=1))[0].status == "open"
    # with standings for everything and the week's box scores it settles
    for t in ("AAA", "BBB"):
        session.add(NFLStat(player_id=t, player_display_name=t, position="QB", season=2025, week=10, season_type="REG", team=t))
    session.commit()
    raw.store("nfl", "dk/standings", 1, CSV, ext="csv"); raw.store("nfl", "dk/standings", 2, CSV, ext="csv")
    items = S.assess(session, lock - timedelta(days=1), lock + timedelta(days=1), public_n=2, now=lock + timedelta(hours=12))
    assert items[0].status == "settled" and items[0].scraped == [1, 2, 3]
    S.write_status(session, items)
    row = slate_status.by_key(session, "nfl", ["111"])["111"].as_dict()
    assert row["status"] == "settled" and row["played"] == [1, 2] and row["settled_at"] is not None and row["box_scores"]


CLASSIC_CSV = """Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,1,a (1/20),0,200,QB Jameis Winston RB Jahmyr Gibbs DST Browns,,Jahmyr Gibbs,RB,22.48%,40.70
2,2,b,0,100,QB Jameis Winston RB Jahmyr Gibbs DST Browns,,Jameis Winston,QB,24.59%,26.28
3,3,c,0,50,QB Jameis Winston RB Jahmyr Gibbs DST Browns,,Browns,DST,9.10%,12.00
,,,,,,,Nobody Known,WR,1.00%,0.00
"""


def test_classic_player_rows_from_the_export(session, monkeypatch):
    from ivan.sports.nfl.models import ShowdownPlayer
    t = S.classic_player_table(CLASSIC_CSV)
    assert t.name.tolist() == ["Jahmyr Gibbs", "Jameis Winston", "Browns", "Nobody Known"] and t.ownership.tolist()[0] == 22.48
    session.add(HistoricalContest(contest_id=9, draft_group_id=5, name="x", date=datetime(2026, 10, 11, 17), entry_fee=1.0, contest_type="Classic",
                                  total_entries=100, multi_entry=20))
    session.commit()
    cap = [{"id": 1, "roster_slot_id": 67, "name": "jahmyr gibbs", "team": "DET", "position": "RB", "salary": 9000},
           {"id": 2, "roster_slot_id": 70, "name": "jahmyr gibbs", "team": "DET", "position": "RB", "salary": 9000},
           {"id": 3, "roster_slot_id": 66, "name": "jameis winston", "team": "NYG", "position": "QB", "salary": 5000},
           {"id": 4, "roster_slot_id": 71, "name": "browns", "team": "CLE", "position": "DST", "salary": 3000}]
    session.execute(text("CREATE TABLE IF NOT EXISTS projection_snapshot_dummy (x INT)"))
    n = S.derive_classic_players(session, 9, cap, datetime(2026, 10, 11, 12), CLASSIC_CSV, echo=lambda *_: None)
    assert n == 4
    rows = session.query(ShowdownPlayer).filter_by(contest_id=9).order_by(ShowdownPlayer.player_id).all()
    assert [(r.player_id, r.roster_slot_id, r.ownership_actual, r.fpts_actual) for r in rows] == [
        (1, 67, 22.48, 40.7), (2, 70, 22.48, 40.7), (3, 66, 24.59, 26.28), (4, 71, 9.1, 12.0)]
    assert S.derive_classic_players(session, 9, cap, datetime(2026, 10, 11, 12), CLASSIC_CSV, echo=lambda *_: None) == 4   # replaces, no duplicates


SHOWDOWN_CSV = """Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,1,a,0,60.09,CPT Tyler Huntley FLEX Derrick Henry,,Derrick Henry,FLEX,39.55%,24.00
2,2,b,0,52.04,CPT Derrick Henry FLEX Tyler Huntley,,Tyler Huntley,FLEX,36.38%,24.04
,,,,,,,Derrick Henry,CPT,12.66%,36.00
,,,,,,,Tyler Huntley,CPT,5.71%,36.06
,,,,,,,Only Captain,CPT,0.10%,15.00
"""


def test_showdown_export_keeps_base_points(session):
    from ivan.sports.nfl.models import ShowdownPlayer
    base = S.base_points_from_export(SHOWDOWN_CSV)
    assert base == {"derrick henry": 24.0, "tyler huntley": 24.04, "only captain": 10.0}
    session.add(HistoricalContest(contest_id=9, draft_group_id=5, name="x", date=datetime(2026, 10, 11, 17), entry_fee=1.0,
                                  contest_type="Showdown Captain Mode", total_entries=100, multi_entry=20))
    for pid, slot, sal, pts in ((1, 511, 16500, 36.0), (2, 512, 11000, 36.0), (3, 511, 13200, 36.06), (4, 512, 8800, 36.06)):
        session.add(ShowdownPlayer(contest_id=9, player_id=pid, roster_slot_id=slot, salary=sal, position="RB" if sal in (16500, 11000) else "QB",
                                   team="BAL", ownership_actual=10.0, fpts_actual=pts))
    session.commit()
    cap = [{"id": 1, "roster_slot_id": 511, "name": "derrick henry", "team": "BAL", "position": "RB", "salary": 16500},
           {"id": 2, "roster_slot_id": 512, "name": "derrick henry", "team": "BAL", "position": "RB", "salary": 11000},
           {"id": 3, "roster_slot_id": 511, "name": "tyler huntley", "team": "BAL", "position": "QB", "salary": 13200},
           {"id": 4, "roster_slot_id": 512, "name": "tyler huntley", "team": "BAL", "position": "QB", "salary": 8800}]
    assert S.repair_showdown_points(session, 9, cap, SHOWDOWN_CSV, echo=lambda *_: None) == 4
    pts = {r.player_id: r.fpts_actual for r in session.query(ShowdownPlayer).filter_by(contest_id=9).all()}
    assert pts == {1: 24.0, 2: 24.0, 3: 24.04, 4: 24.04}
    assert S.repair_showdown_points(session, 9, cap, SHOWDOWN_CSV, echo=lambda *_: None) == 0      # already right
