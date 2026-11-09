"""results: our entries by DraftKings user name, the ledger cached per slate and summed per day."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba.backtest import results as R
from ivan.sports.nba import slates as SL
from ivan.sports.nba.models import ContestPayout, HistoricalContest, NBABase
from ivan.sports.nba.data_fetch.standings import parse_export

HEAD = "Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS\n"
LINEUP = "PG Ann One SG Bob Two SF Cal Three PF Dan Four C Ed Five G Fay Six F Gus Seven UTIL Hal Eight"
FINAL = (HEAD + f"1,11,them (1/20),0,352.5,{LINEUP},,Ann One,PG,61.2%,70.5\n"
                f"2,12,SamBamCH (1/2),0,340.0,{LINEUP},,Bob Two,SG,55.0%,30.2\n"
                f"3,13,SamBamCH (2/2),0,301.0,{LINEUP.replace('Hal Eight', 'Ian Nine')},,Ian Nine,UTIL,12.0%,28.0\n"
                f"4,14,them,0,290.0,{LINEUP.replace('Hal Eight', 'Jo Ten')},,,,,\n")
MID = FINAL.replace("UTIL Hal Eight,,Ann", "UTIL LOCKED,,Ann").replace("UTIL Hal Eight,,Bob", "UTIL LOCKED,,Bob").replace("UTIL Ian Nine,,Ian", "UTIL LOCKED,,Ian")


def test_entry_user_and_mask():
    assert R.entry_user("SamBamCH (3/10)") == "SamBamCH" and R.entry_user("SamBamCH") == "SamBamCH" and R.entry_user("some one (1/2)") == "some one"
    ent, _ = parse_export(FINAL)
    assert list(R.ours_mask(ent, user="sambamch")) == [False, True, True, False]        # case does not matter
    assert list(R.ours_mask(ent, ours={"14"})) == [False, False, False, True]
    assert list(R.ours_mask(ent, ours={"14"}, user="SamBamCH")) == [False, True, True, True]


def test_score_contest_by_user_name():
    ent, _ = parse_export(FINAL)
    r = R.score_contest(dict(contest_id=9, name="Big", entry_fee=5.0, total_entries=4), set(), ent, np.array([10.0, 6.0, 2.0, 0.0]), None, user="SamBamCH")
    assert r["found"] == 2 and r["ours"] == 2 and r["fees"] == 10.0 and r["won"] == 8.0 and r["cashed"] == 2 and r["best_rank"] == 2


@pytest.fixture
def slate(tmp_path, monkeypatch):
    """A held slate 555 on 2026-11-04 under a temporary slates root, its contests 9 (ours) and 10 (ours, no export) and 77 (public)."""
    root = tmp_path / "slates"
    p = SL.paths(555, root); p.input.mkdir(parents=True)
    (p.root / "slate.sqlite").write_bytes(b"")
    (p.root / "slate.json").write_text('{"draft_group": 555, "day": "2026-11-04", "lock": "2026-11-04T18:00:00"}')
    p.entries_csv.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL\n12,Big,9,$5,,,,,,,,\n13,Big,9,$5,,,,,,,,\n20,Small,10,$1,,,,,,,,\n")
    engine = create_engine(f"sqlite:///{tmp_path / 'hist.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for cid, name, fee in ((9, "Big", 5.0), (10, "Small", 1.0), (77, "Public", 3.0)):
        s.add(HistoricalContest(contest_id=cid, draft_group_id=555, name=name, date=datetime(2026, 11, 4, 18), entry_fee=fee, contest_type="Classic", total_entries=4, multi_entry=20))
    for cid in (9, 10, 77):
        s.add(ContestPayout(contest_id=cid, min_position=1, max_position=1, payout_amount=10.0, payout_text="$10", currency="USD"))
        s.add(ContestPayout(contest_id=cid, min_position=2, max_position=3, payout_amount=4.0, payout_text="$4", currency="USD"))
    s.commit()
    monkeypatch.setattr(SL, "holds", lambda dg, root_=None: int(dg) == 555)
    monkeypatch.setattr(SL, "paths", lambda dg=None, root_=None: SL.SlatePaths(555, root / "555", "2026-11-04", root.parent / "days"))
    monkeypatch.setattr(SL, "held_meta", lambda root_=None: {555: {"day": "2026-11-04"}})
    monkeypatch.setattr(SL, "read_meta", lambda dg=None, root_=None: {"day": "2026-11-04"})
    monkeypatch.setattr(SL, "session", lambda dg=None, root_=None, history=None: s)
    monkeypatch.setattr(SL, "using", lambda dg: __import__("contextlib").nullcontext())
    monkeypatch.setattr(R, "our_user", lambda: "SamBamCH")
    return s, root


def test_compute_scores_our_contests_only(slate):
    s, root = slate
    public = FINAL.replace("SamBamCH (1/2)", "x").replace("SamBamCH (2/2)", "y")
    rows = R.compute(555, s, exports={9: ("20261105-103100", FINAL), 77: ("20261105-103100", public)}, root=root)
    by = {r["contest_id"]: r for r in rows}
    assert set(by) == {9, 10}                                       # 77: a public contest with none of ours; 10: ours, no export yet
    assert by[9]["final"] and by[9]["found"] == 2 and by[9]["won"] == 4.0 + 4.0 and by[9]["net"] == -2.0
    assert not by[10]["final"] and by[10]["ours"] == 1 and by[10]["fees"] == 1.0      # the entries file: one entry at $1, unscored until the export
    assert by[9]["side"] is not None


def test_compute_leaves_a_mid_slate_export_unscored(slate):
    s, root = slate
    rows = R.compute(555, s, exports={9: ("20261104-180600", MID)}, root=root)
    by = {r["contest_id"]: r for r in rows}
    assert not by[9]["final"] and "mid-slate" in by[9]["scored"] and by[9]["won"] is None


def test_ledger_is_cached_until_complete(slate, monkeypatch):
    s, root = slate
    now = datetime(2026, 11, 5, 9, 0)
    calls = []
    real_compute = R.compute

    def fake_compute(dg, session, ours_ids=None, user=None, exports=None, root=None):
        calls.append(1)
        return real_compute(dg, session, ours_ids, user, exports={9: ("20261105-083000", FINAL)}, root=root)
    monkeypatch.setattr(R, "compute", fake_compute)
    monkeypatch.setattr(R, "final_exports", lambda ids: {9: type("A", (), {"stamp": "20261105-083000"})()})      # what is on disk
    led = R.slate_ledger(555, now, root=root, session=s)
    assert led["draft_group"] == 555 and led["day"] == "2026-11-04" and not led["complete"]       # contest 10 has no export
    assert led["summary"]["net"] == -2.0 and led["summary"]["unscored"] == 1
    assert Path(R.ledger_path(555, root)).exists()
    R.slate_ledger(555, now, root=root, session=s)
    assert len(calls) == 1                                           # within RECHECK: not recomputed
    R.slate_ledger(555, now + R.RECHECK, root=root, session=s)
    assert len(calls) == 1                                           # exports unchanged on disk (final_exports agrees): still not recomputed
    monkeypatch.setattr(R, "final_exports", lambda ids: {9: type("A", (), {"stamp": "20261105-103100"})()})
    R.slate_ledger(555, now + 2 * R.RECHECK, root=root, session=s)
    assert len(calls) == 2                                           # a newer export: recomputed
    days = R.ledger_days(now + 2 * R.RECHECK, root=root, today=now.date())
    assert days["overall"]["net"] == -2.0 and days["days"]["2026-11-04"]["slates"] == [555] and days["days"]["2026-11-04"]["incomplete"] == 1
