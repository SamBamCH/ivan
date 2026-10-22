"""The DraftKings standings export parser and the derived field (ivan.sports.nba.standings)."""
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import archive
from ivan.sports.nba import standings as S
from ivan.sports.nba.models import HistoricalContest, NBABase, NBAProjectionCapture

EXPORT = ("Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS\n"
          "1,11,user1 (1/20),0,352.5,PG Luka Doncic SG Amen Thompson SF Kawhi Leonard PF Jalen Johnson C Bam Adebayo G Cam Spencer F Kevin Durant UTIL Jock Landale,,Luka Doncic,PG,61.2%,70.5\n"
          "2,12,user2,0,340.0,PG Luka Doncic SG Amen Thompson SF Kawhi Leonard PF Jalen Johnson C Bam Adebayo G Cam Spencer F Kevin Durant UTIL Jock Landale,,Cam Spencer,G,55.0%,30.2\n"
          "3,13,user3,0,301.0,PG Cam Spencer SG Amen Thompson SF Kevin Durant PF Jalen Johnson C Jock Landale G Luka Doncic F Kawhi Leonard UTIL Nikola Vucevic,,Nikola Vucevic,C,12.0%,28.0\n"
          "4,14,user4,0,,PG LOCKED SG LOCKED SF LOCKED PF LOCKED C LOCKED G LOCKED F LOCKED UTIL LOCKED,,,,,\n")
NAMES = ["Luka Doncic", "Amen Thompson", "Kawhi Leonard", "Jalen Johnson", "Bam Adebayo", "Cam Spencer", "Kevin Durant", "Jock Landale", "Nikola Vucevic"]


def test_parse_export_entries_and_side_table():
    ent, side = S.parse_export(EXPORT)
    assert len(ent) == 4 and ent.names.iloc[0][:2] == ["Luka Doncic", "Amen Thompson"] and ent.names.iloc[3] == [None] * 8
    assert list(side.name) == ["Luka Doncic", "Cam Spencer", "Nikola Vucevic"] and side.drafted_pct.iloc[0] == 61.2 and side.fpts.iloc[2] == 28.0
    assert [s for s, _ in S.parse_lineup("PG Ja Morant SG Desmond Bane UTIL Jaren Jackson Jr.")] == ["PG", "SG", "UTIL"]


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine); archive.ensure_tables(engine)
    return sessionmaker(bind=engine)()


def test_derive_contest_matches_names_and_counts_copies(session, tmp_path):
    lock = datetime(2026, 10, 21, 18, 0)
    players = [dict(id=100 + i, name=n, team="T%d" % (i % 4), salary=4000 + 500 * i, position="PG") for i, n in enumerate(NAMES)]
    archive.archive_slate(session, sport="nba", slate_key="555", lock=lock, mode="classic", players=players, source="ingest", captured_at=datetime(2026, 10, 21, 17, 45))
    session.add(HistoricalContest(contest_id=9, draft_group_id=555, name="Big", date=lock, entry_fee=5.0, contest_type="Classic", total_entries=4, multi_entry=20))
    for i, n in enumerate(NAMES):
        session.add(NBAProjectionCapture(captured_at=datetime(2026, 10, 21, 17, 45), draft_group_id=555, lock=lock, source="ours", player_id=100 + i,
                                         player_name=n.lower(), minutes_proj=30.0, fpts_mean=30.0 + i, fpts_std=8.0, salary=4000 + 500 * i, starter="expected"))
    session.commit()
    s = S.derive_contest(session, 9, text=EXPORT, out_dir=tmp_path, log=lambda m: None)
    assert s["entries"] == 4 and s["entries_matched"] == 3 and s["unique"] == 2 and s["max_copies"] == 2 and s["short_entries"] == 1
    lu, side, summary = S.load_field(9, tmp_path)
    assert lu.copies.tolist() == [2, 1] and lu.salary.iloc[0] == sum(4000 + 500 * i for i in range(8)) and lu.proj_sum.iloc[0] == pytest.approx(sum(30.0 + i for i in range(8)))
    assert side.loc[side.name == "Cam Spencer", "fpts_proj"].iloc[0] == 35.0 and side.loc[side.name == "Nikola Vucevic", "team"].iloc[0] == "T0"
    assert S.derived_ids(tmp_path) == {9}
