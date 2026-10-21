"""The choose step (ivan.sports.nba.lineup_export): entries parsing, weights, slot order, the files it writes; and the chooser (choose)."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import choose as CH
from ivan.sports.nba import lineup_export as LE
from ivan.sports.nba.models import HistoricalContest, Lineup, LineupSimulationResult, NBABase, NBAPlayer


def P(pid, pos, salary=5000, gametime=None, name=None):
    return SimpleNamespace(id=pid, position=pos, salary=salary, name=name or f"p{pid}", team="A", gametime=gametime)


def test_classic_slot_order_and_late_flex():
    early, late = datetime(2026, 10, 21, 18, 0), datetime(2026, 10, 21, 21, 30)
    players = [P(1, "PG", gametime=early), P(2, "SG", gametime=early), P(3, "SF", gametime=early), P(4, "PF", gametime=early),
               P(5, "C", gametime=early), P(6, "PG/SG", gametime=late), P(7, "SF/PF", gametime=late), P(8, "C/PF", gametime=late)]
    order = LE.assign_player_order_classic(players)
    assert [p.id for p in order] == [1, 2, 3, 4, 5, 6, 7, 8]        # late games land in G, F, UTIL
    with pytest.raises(ValueError):
        LE.assign_player_order_classic(players[:7])
    with pytest.raises(ValueError):
        LE.assign_player_order_classic([P(i, "C") for i in range(8)])   # eight centers cannot fill the guards


def test_entries_and_weights(tmp_path):
    f = tmp_path / "entries.csv"
    f.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL,,Instructions\n"
                 "1,Big,100,$5,,,,,,,,,,1. Column A lists\n2,Big,100,$5,,,,,,,,,,\n3,Small,200,$1,,,,,,,,,,\n,,,,,,,,,,,,,\n"
                 ",Instructions row DraftKings appends,,,,,,,,,,,,\n")
    e = LE.read_entries(f)
    assert len(e) == 3 and e["Contest ID"].tolist() == [100, 100, 200] and LE.parse_money("$0.50") == 0.5
    assert "Instructions" not in e.columns and "UTIL" in e.columns

    asked = []
    prompter = lambda cid, name, last: asked.append(cid) or {**last, "cash_pct": 0.9}   # noqa: E731
    w = LE.weights_resolver(prompter=prompter)
    assert w(100, "Big", CH.DEFAULT_WEIGHTS)["cash_pct"] == 0.9 and asked == [100]
    w = LE.weights_resolver(flags={"cash_pct": None, "diversity_lambda": 0.3}, prompter=prompter)
    assert w(100, "Big", CH.DEFAULT_WEIGHTS) == {**CH.DEFAULT_WEIGHTS, "diversity_lambda": 0.3} and asked == [100]
    w = LE.weights_resolver(overrides={"default": {"cash_pct": 0.2}, "Small": {"method": "portfolio"}, "100": {"avg_roi_win": 1.0}}, prompter=prompter)
    assert w(200, "Small", {})["method"] == "portfolio" and w(200, "Small", {})["cash_pct"] == 0.2 and w(100, "x", {})["avg_roi_win"] == 1.0


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed_slate(session, n_lineups=7, worlds=50, seed=0):
    """Eight-man pool x n lineups (each a different 8 of 10 players) with simulated results for contest 100."""
    gt = datetime(2026, 10, 21, 19, 0)
    pos = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "C", "PG", "SF"]
    for pid in range(1, 11):
        session.add(NBAPlayer(id=pid, name=f"player {pid}", team="BOS" if pid % 2 else "NYK", opponent="NYK" if pid % 2 else "BOS", position=pos[pid - 1],
                              roster_slot_id=1, gametime=gt, salary=4000 + 500 * pid, fpts_mean=20.0 + pid, fpts_std=8.0, minutes_proj=30.0))
    session.add(HistoricalContest(contest_id=100, draft_group_id=1, name="Big", date=gt, entry_fee=5.0, contest_type="Classic", total_entries=1000, multi_entry=20))
    rng = np.random.default_rng(seed)
    combos = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 10], [9, 2, 3, 4, 5, 6, 7, 8], [1, 2, 10, 4, 5, 6, 7, 8], [9, 2, 3, 4, 8, 6, 7, 10], [1, 2, 3, 4, 5, 9, 7, 8]]
    for i in range(n_lineups):
        session.add(Lineup(id=10 + i, player_ids_list=combos[i % len(combos)]))
        roi = rng.normal(0.3 * i, 0.2, worlds)
        pay = np.where(roi > 0, 5 * (1 + roi), 0.0)
        session.add(LineupSimulationResult(lineup_id=10 + i, contest_id=100, contest_name="Big", fpts=rng.normal(250, 20, worlds).tolist(),
                                           payout=pay.tolist(), roi=roi.tolist(), rank=rng.integers(1, 1000, worlds).tolist()))
    session.commit()


def test_choose_lineup_ids_greedy_and_portfolio(session):
    seed_slate(session)
    stats = {}
    ids = CH.choose_lineup_ids(session, {"method": "greedy", "diversity_lambda": 0.0, "cash_pct": 0.0, "avg_roi_win": 1.0}, 2, 100, stats=stats)
    assert len(ids) == 2 and stats["results"] == 7 and stats["unique"] == 6           # the 7th lineup repeats the first's composition
    assert ids[0] == 15                                                                # the highest mean ROI first (16 repeats 10's composition and is dropped)
    ids_p = CH.choose_lineup_ids(session, {"method": "portfolio"}, 3, 100, stats=stats)
    assert len(ids_p) == 3 and stats["book"]["method"] == "portfolio" and stats["book"]["chosen"]["worlds"] == 50
    with pytest.raises(ValueError):
        CH.choose_lineup_ids(session, {"method": "nope"}, 1, 100)
    assert CH.choose_lineup_ids(session, {}, 1, 999) == []


def test_choose_entries_writes_every_file(session, tmp_path):
    seed_slate(session)
    f = tmp_path / "entries.csv"
    f.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL,,Instructions\n"
                 "501,Big,100,$5,,,,,,,,,,1.\n502,Big,100,$5,,,,,,,,,,\n503,Big,100,$5,,,,,,,,,,\n,,,,,,,,,,,,,\n")
    logs = []
    run, paths = LE.run_choose(session, f, tmp_path / "out" / "chosen_entries.csv", LE.weights_resolver(no_prompt=True), log=logs.append, stamp="S")
    assert run.total == 3 and set(paths) == {"classic", "exposures", "lineups", "choose", "simroi"}
    up = LE.pd.read_csv(paths["classic"])
    assert list(up.columns) == LE.BASE_COLS + LE.POSITION_ORDER and len(up) == 3 and up["Entry ID"].tolist() == [501, 502, 503]
    assert all(up.iloc[0][s].endswith(")") for s in LE.POSITION_ORDER)
    z = np.load(paths["simroi"])
    assert z["roi"].shape == (3, 50) and z["entry_id"].tolist() == [501, 502, 503]
    exp = LE.pd.read_csv(paths["exposures"])
    assert abs(exp["exposure"].sum() - 8.0) < 1e-9                                      # eight slots per entry
    assert any("chosen 3" in m for m in logs)
