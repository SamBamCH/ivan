"""The NFL slate method on the shared core: two showdown contests on one draw chosen as one book, through choose_entries."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nfl import choose as C
from ivan.sports.nfl import lineup_export as LE
from ivan.sports.nfl.models import ContestSimReport, HistoricalContest, Lineup, LineupSimulationResult, NFLBase, NFLPlayer


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed(session, worlds=80, draw="slate"):
    kick = datetime(2026, 11, 5, 20, 15)
    for pid, slot, pos in [(1, 511, "QB"), (2, 512, "QB"), (3, 512, "WR"), (4, 512, "RB"), (5, 512, "TE"), (6, 512, "K"), (7, 512, "DST"), (8, 512, "WR")]:
        session.add(NFLPlayer(id=pid, roster_slot_id=slot, name=f"p{pid}", team="A", opponent="B", position=pos, gametime=kick,
                              salary=5000, fpts_mean=10.0, fpts_p50=9.0, ownership_proj=20.0))
    for cid, name, fee in ((100, "Showdown Flat", 5.0), (200, "Showdown Top", 5.0)):
        session.add(HistoricalContest(contest_id=cid, draft_group_id=1, name=name, date=kick, entry_fee=fee,
                                      contest_type="Showdown Captain Mode", total_entries=1000, multi_entry=20))
    rng = np.random.default_rng(0)
    flex = [2, 3, 4, 5, 6, 7, 8]
    for i in range(10):
        session.add(Lineup(id=i + 1, player_ids_list=[1] + sorted(rng.choice(flex, 5, replace=False).tolist())))
    session.flush()
    for i in range(10):
        hit = rng.random(worlds) < (0.5 if i < 5 else 0.05)
        for cid, name, small, big in ((100, "Showdown Flat", 0.8, 0.8), (200, "Showdown Top", 0.1, 20.0)):
            roi = np.where(hit, small if i < 5 else big, -1.0)
            session.add(LineupSimulationResult(lineup_id=i + 1, contest_id=cid, contest_name=name, fpts=rng.normal(100, 10, worlds).tolist(),
                                               payout=(5 * (1 + roi)).tolist(), roi=roi.tolist(), rank=list(range(worlds))))
    session.add(ContestSimReport(contest_id=100, contest_name="Showdown Flat", report={"contest": {"draw": draw, "method": "montecarlo"}}))
    session.add(ContestSimReport(contest_id=200, contest_name="Showdown Top", report={"contest": {"draw": draw, "method": "montecarlo"}}))
    session.commit()


def test_choose_slate_runs_on_the_core_and_exports(session):
    seed(session)
    contests = [{"contest_id": 100, "contest_name": "Showdown Flat", "entries": 3, "fee": 5.0},
                {"contest_id": 200, "contest_name": "Showdown Top", "entries": 3, "fee": 5.0}]
    out = C.choose_slate(session, contests, {**LE.DEFAULT_WEIGHTS, "method": "slate", "bankroll_dollars": 300.0})
    assert set(out["picks"]) == {"Showdown Flat", "Showdown Top"} and all(len(v) == 3 for v in out["picks"].values())
    assert set(out["picks"]["Showdown Flat"]) <= {1, 2, 3, 4, 5}                    # steady cashers in the flat contest
    assert set(out["picks"]["Showdown Top"]) & {6, 7, 8, 9, 10}                     # a tail lineup in the top-heavy one
    book = out["book"]
    assert book["entries"] == 6 and book["fees"] == 30.0 and book["shared"] == {"method": "montecarlo", "draw": "slate"}
    assert "improve" in book and "placements" in book and len(book["contests"]) == 2
    st = out["stats"]["Showdown Flat"]
    assert st["viz"]["slate"] is book and "rooting" in st["viz"] and st["book"]["method"] == "slate"

    entries = pd.DataFrame({"Entry ID": ["11", "12", "13", "14"], "Contest Name": ["Showdown Flat"] * 2 + ["Showdown Top"] * 2,
                            "Contest ID": [100, 100, 200, 200], "Entry Fee": ["$5"] * 4})
    lines = []
    run = LE.choose_entries(session, entries, LE.weights_resolver(flags={"method": "slate", "bankroll_dollars": 300.0}), log=lines.append)
    assert run.total_sd == 4 and run.slate_book["entries"] == 4
    assert any("slate book: 4 entries" in m for m in lines) and any("chosen as one book (bankroll $300)" in m for m in lines)


def test_choose_slate_refuses_separate_draws(session):
    seed(session, draw="contest")
    with pytest.raises(ValueError, match="shared draw"):
        C.choose_slate(session, [{"contest_id": 100, "contest_name": "Showdown Flat", "entries": 1, "fee": 5.0}], {"bankroll_dollars": 100.0})
