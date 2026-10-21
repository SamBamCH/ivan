"""The generate step's helpers (ivan.sports.nba.field_generation) and the field sampler (sample_field)."""
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import field_generation as FG
from ivan.sports.nba import sample_field as SF
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
    assert (s["n_samples"], s["jitter"], s["randomness"], s["min_fpts"], s["player_min_fpts"]) == (10, 0.1, 1.0, 250.0, 12.0)


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


def test_projection_weighted_sampler_and_expand(session):
    gt = datetime(2026, 10, 21, 19, 0)
    for pid, m in ((1, 50.0), (2, 40.0), (3, 30.0), (4, 20.0)):
        session.add(NBAPlayer(id=pid, name=f"p{pid}", team="A", opponent="B", position="PG", roster_slot_id=1, gametime=gt, salary=5000, fpts_mean=m))
    session.add(Lineup(id=10, player_ids_list=[1, 2]))      # 90
    session.add(Lineup(id=11, player_ids_list=[3, 4]))      # 50
    session.commit()
    counts, meta = SF.sample_field_projection_weighted(session, 1000, exponent=1.0, rng=np.random.default_rng(0))
    assert sum(counts.values()) == 1000 and meta["candidates"] == 2 and counts[10] > counts[11]
    u_lids, pids_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends = SF.expand_sampled_field(session, {10: 2, 11: 1})
    assert u_lids == [10, 11] and unique_pids == [1, 2, 3, 4] and len(lineup_ix) == 3
    assert lineup_ix[0] == lineup_ix[1] == [0, 1] and lineup_ix[2] == [2, 3] and list(starts) == [0, 2] and list(ends) == [2, 3]
    with pytest.raises(ValueError):
        SF.sample_field(session, 10, {"field": {"sampler": "nope"}}, {})
