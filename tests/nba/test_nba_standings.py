"""The DraftKings standings export parser and the derived field (ivan.sports.nba.standings)."""
from datetime import datetime

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
    s = S.derive_contest(session, 9, text=EXPORT, out_dir=tmp_path, log=lambda m: None, allow_partial=True)   # one of four entries is LOCKED
    assert s["entries"] == 4 and s["entries_matched"] == 3 and s["unique"] == 2 and s["max_copies"] == 2 and s["short_entries"] == 1
    assert s["locked_entries"] == 1 and s["final"] is False
    lu, side, summary = S.load_field(9, tmp_path)
    assert lu.copies.tolist() == [2, 1] and lu.salary.iloc[0] == sum(4000 + 500 * i for i in range(8)) and lu.proj_sum.iloc[0] == pytest.approx(sum(30.0 + i for i in range(8)))
    assert side.loc[side.name == "Cam Spencer", "fpts_proj"].iloc[0] == 35.0 and side.loc[side.name == "Nikola Vucevic", "team"].iloc[0] == "T0"
    assert S.derived_ids(tmp_path) == {9}


FINAL = "\n".join(EXPORT.splitlines()[:4]) + "\n"          # the same export without the LOCKED entry


def test_a_mid_slate_export_is_not_a_field(session, tmp_path):
    ent, _ = S.parse_export(EXPORT)
    assert S.locked_entries(ent) == 1 and not S.export_is_final(ent)            # 1 of 4 entries locked: 25% > PARTIAL_MAX_SHARE
    assert S.export_is_final(S.parse_export(FINAL)[0])
    lock = datetime(2026, 10, 21, 18, 0)
    players = [dict(id=100 + i, name=n, team="T%d" % (i % 4), salary=4000 + 500 * i, position="PG") for i, n in enumerate(NAMES)]
    archive.archive_slate(session, sport="nba", slate_key="555", lock=lock, mode="classic", players=players, source="ingest", captured_at=datetime(2026, 10, 21, 17, 45))
    session.add(HistoricalContest(contest_id=9, draft_group_id=555, name="Big", date=lock, entry_fee=5.0, contest_type="Classic", total_entries=4, multi_entry=20))
    session.commit()
    msgs = []
    # the default refuses the mid-slate export and writes nothing
    assert S.derive_contest(session, 9, text=EXPORT, out_dir=tmp_path, log=msgs.append) is None
    assert not (tmp_path / "9").exists() and any("mid-slate" in m for m in msgs)
    # a folder the nightly built from a partial export before the flag existed is recognised by its short-entry share and removed
    S.derive_contest(session, 9, text=EXPORT, out_dir=tmp_path, log=lambda m: None, allow_partial=True)
    assert S.derived_ids(tmp_path) == {9} and S.partial_derived_ids(tmp_path) == {9}
    assert S.derive_contest(session, 9, text=EXPORT, out_dir=tmp_path, log=lambda m: None) is None
    assert S.derived_ids(tmp_path) == set()
    # the final export derives, is flagged final, and a final folder is never removed
    s = S.derive_contest(session, 9, text=FINAL, out_dir=tmp_path, log=lambda m: None)
    assert s["final"] is True and s["locked_entries"] == 0 and s["entries_matched"] == 3
    assert S.derived_is_final(9, tmp_path) and not S.remove_partial(9, tmp_path) and S.derived_ids(tmp_path) == {9}


def test_our_contest_ids_read_the_slates_own_entries_file(tmp_path):
    """The contests we played come from data/nba/slates/<dg>/input/entries.csv (the shared pre-split path is gone)."""
    from ivan.sports.nba import slates as SL
    root = tmp_path / "slates"
    p = SL.paths(555, root)
    p.input.mkdir(parents=True); (p.root / "slate.sqlite").write_bytes(b"")
    p.entries_csv.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL\n"
                             "1,Big,9,$5,,,,,,,,\n2,Big,9,$5,,,,,,,,\n3,Small,10,$1,,,,,,,,\n")
    assert S.our_contest_ids(555, root=root) == {9, 10}
    other = tmp_path / "e.csv"; other.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL\n1,X,77,$1,,,,,,,,\n")
    assert S.our_contest_ids(556, entries_csv=other, root=root) == {77}       # a slate not held here: the file given
    assert S.our_contest_ids(556, root=root) == set()
