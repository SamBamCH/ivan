"""The generate step's helpers (ivan.sports.nba.field.generate)."""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba.field import generate as FG
from ivan.sports.nba.models import Lineup, LineupSimulationResult, NBABase, NBAPlayer


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_quotas_split_the_remainder_first():
    assert FG.quotas(10, 4) == [3, 3, 2, 2] and FG.quotas(5, 1) == [5] and FG.quotas(2, 0) == [2]


def test_generate_settings_read_config_and_flag():
    s = FG.generate_settings({"generate": {"n_samples": 10, "jitter": 0.1}, "player_min_fpts": 12.0}, min_fpts=250)
    assert (s["n_samples"], s["jitter"], s["randomness"], s["min_fpts"], s["player_min_fpts"], s["min_fpts_frac"]) == (10, 0.1, 0.25, 250.0, 12.0, 0.9)
    assert FG.generate_settings({}, randomness=0.5)["randomness"] == 0.5


def test_deduplicate_keeps_the_first_of_each_composition(session):
    for ids in ([1, 2, 3], [3, 2, 1], [1, 2, 4], ["1", "2", "3"]):
        session.add(Lineup(player_ids_list=ids))
    session.commit()
    removed, kept = FG.deduplicate_lineups(session)
    assert (removed, kept) == (2, 2)
    assert sorted(tuple(sorted(int(i) for i in L.player_ids_list)) for L in session.query(Lineup).all()) == [(1, 2, 3), (1, 2, 4)]


def test_clear_lineups_empties_both_tables(session):
    session.add(Lineup(id=1, player_ids_list=[1, 2]))
    session.add(LineupSimulationResult(lineup_id=1, contest_id=5, contest_name="c", fpts=[1.0], payout=[0.0], roi=[-1.0], rank=[2]))
    session.commit()
    assert FG.clear_lineups(session) == (1, 1)
    assert session.query(Lineup).count() == 0 and session.query(LineupSimulationResult).count() == 0


def test_player_pool_filters_and_coerces_positions(session):
    gt = datetime(2026, 10, 21, 19, 0)
    session.add(NBAPlayer(id=1, name="a", team="BOS", opponent="NYK", position="PG/SG", roster_slot_id=1, gametime=gt, salary=8000, fpts_mean=40.0))
    session.add(NBAPlayer(id=2, name="b", team="BOS", opponent="NYK", position="C", roster_slot_id=1, gametime=gt, salary=3000, fpts_mean=5.0))
    session.commit()
    pool = FG.player_pool(session, player_min_fpts=15.0)
    assert [p.id for p in pool] == [1] and pool[0].position == ("PG", "SG") and pool[0].game_id == "BOS-NYK"


