"""The choose step (ivan.sports.nfl.lineup_export): entries parsing, weights, slot order, the files it writes."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nfl import lineup_export as LE
from ivan.sports.nfl.models import HistoricalContest, Lineup, LineupSimulationResult, NFLBase, NFLPlayer


def P(pid, slot, pos, salary, name=None, team="A"):
    return SimpleNamespace(id=pid, roster_slot_id=slot, position=pos, salary=salary, name=name or f"p{pid}", team=team)


def test_showdown_and_classic_slot_order():
    sd = [P(1, 512, "WR", 5000), P(2, 511, "QB", 9000), P(3, 512, "RB", 4000), P(4, 512, "TE", 3000), P(5, 512, "K", 3500), P(6, 512, "DST", 3000)]
    assert [p.id for p in LE.assign_player_order_showdown(sd)] == [2, 1, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        LE.assign_player_order_showdown(sd[:5])
    cl = [P(1, 67, "RB", 7000), P(2, 68, "WR", 8000), P(3, 66, "QB", 6000), P(4, 71, "DST", 3000), P(5, 68, "WR", 5000),
          P(6, 67, "RB", 6500), P(7, 69, "TE", 4000), P(8, 68, "WR", 4500), P(9, 68, "WR", 9000)]
    order = LE.assign_player_order_classic(cl)
    assert [p.position for p in order] == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"]
    assert order[7].id == 8                       # FLEX = highest-salary RB/WR/TE not already used (the 4th WR)
    with pytest.raises(ValueError):
        LE.assign_player_order_classic(cl[1:])    # no RB pair


def test_entries_and_weights(tmp_path):
    f = tmp_path / "entries.csv"
    f.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,CPT,FLEX,FLEX,FLEX,FLEX,FLEX\n"
                 "1,Showdown A,100,$5,,,,,,\n2,Showdown A,100,$5,,,,,,\n3,Big,200,$1,,,,,,\n,,,,,,,,,\n"
                 ",Instructions row DraftKings appends,,,,,,,,\n")
    e = LE.read_entries(f)
    assert len(e) == 3 and e["Contest ID"].tolist() == [100, 100, 200] and LE.parse_money("$0.50") == 0.5

    asked = []
    prompter = lambda cid, name, last: asked.append(cid) or {**last, "cash_pct": 0.9}   # noqa: E731
    w = LE.weights_resolver(prompter=prompter)
    assert w(100, "Showdown A", LE.DEFAULT_WEIGHTS)["cash_pct"] == 0.9 and asked == [100]
    w = LE.weights_resolver(flags={"cash_pct": None, "diversity_lambda": 0.3}, prompter=prompter)
    assert w(100, "Showdown A", LE.DEFAULT_WEIGHTS) == {**LE.DEFAULT_WEIGHTS, "diversity_lambda": 0.3} and asked == [100]
    w = LE.weights_resolver(overrides={"default": {"cash_pct": 0.2}, "Big": {"diversity_lambda": 0.7}, "100": {"avg_roi_win": 1.0}}, prompter=prompter)
    assert w(200, "Big", {})["diversity_lambda"] == 0.7 and w(200, "Big", {})["cash_pct"] == 0.2 and w(100, "x", {})["avg_roi_win"] == 1.0


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_choose_entries_writes_every_file(session, tmp_path):
    kick = datetime(2026, 10, 15, 19, 15)
    for pid, slot, pos, sal in [(1, 511, "QB", 15000), (2, 512, "QB", 10000), (3, 512, "WR", 8000), (4, 512, "RB", 7000),
                                (5, 512, "TE", 5000), (6, 512, "K", 4000), (7, 512, "DST", 3000)]:
        session.add(NFLPlayer(id=pid, roster_slot_id=slot, name=f"p{pid}", team="A", opponent="B", position=pos, gametime=kick,
                              salary=sal, fpts_mean=10.0, fpts_p50=9.0, ownership_proj=20.0))
    session.add(HistoricalContest(contest_id=100, draft_group_id=1, name="Showdown A", date=kick, entry_fee=5.0, contest_type="Showdown Captain Mode",
                                  total_entries=1000, multi_entry=20))
    rng = np.random.default_rng(0)
    for lid, pids in [(1, [1, 2, 3, 4, 5, 6]), (2, [1, 2, 3, 4, 5, 7]), (3, [1, 3, 4, 5, 6, 7])]:
        session.add(Lineup(id=lid, player_ids_list=pids))
        session.flush()
        roi = rng.normal(0.1, 1, 50)
        session.add(LineupSimulationResult(lineup_id=lid, contest_id=100, contest_name="Showdown A", fpts=list(rng.normal(100, 10, 50)),
                                           payout=list(np.where(roi > 0, 10.0, 0.0)), roi=list(roi), rank=list(range(50))))
    session.commit()

    entries = pd.DataFrame({"Entry ID": ["11", "12"], "Contest Name": ["Showdown A"] * 2, "Contest ID": [100, 100], "Entry Fee": ["$5"] * 2})
    run = LE.choose_entries(session, entries, LE.weights_resolver(), log=lambda m: None)
    assert run.total_sd == 2 and len(run.upload_sd) == 2 and list(run.upload_sd.columns) == LE.BASE_COLS + LE.POSITION_ORDER_SD
    assert run.upload_sd.iloc[0]["CPT"] == "p1 (1)" and len(run.lineup_rows) == 12 and run.contests[0]["chosen"] == 2
    sd, cl = run.exposures()
    assert cl.empty and sd[sd.roster_slot == "CPT"].exposure.tolist() == [1.0] and abs(sd.exposure.sum() - 6.0) < 1e-9

    paths = LE.write_outputs(run, tmp_path / "out" / "chosen_entries.csv", stamp="S", log=lambda m: None)
    assert set(paths) == {"showdown", "exposures", "lineups", "choose", "simroi"}
    assert (tmp_path / "out" / "chosen_entries.showdown.S.csv").exists()
    z = np.load(paths["simroi"])
    assert z["roi"].shape == (2, 50) and z["entry_id"].tolist() == [11, 12] and z["entry_fee"].tolist() == [5.0, 5.0]
    lineups = pd.read_csv(paths["lineups"])
    assert lineups.format.unique().tolist() == ["showdown"] and set(lineups.slot) == {"CPT", "FLEX"} and lineups.cash_pct.notna().all()
