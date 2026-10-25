"""The generate step's helpers (ivan.sports.nfl.field_generation) and the sim's entries loop (simulate_contest)."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nfl import field_generation as FG
from ivan.sports.nfl.models import Lineup, NFLBase
from ivan.sports.nfl.simulate_contest import contests_in_entries


def test_quotas_split_the_remainder_first():
    assert FG.quotas(10, 4) == [3, 3, 2, 2] and FG.quotas(5, 1) == [5] and FG.quotas(2, 0) == [2]


def test_deduplicate_keeps_the_first_of_each_composition(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for ids in ([1, 2, 3], [3, 2, 1], [1, 2, 4], ["1", "2", "3"]):
        s.add(Lineup(player_ids_list=ids))
    s.commit()
    removed, kept = FG.deduplicate_lineups(s)
    assert (removed, kept) == (2, 2)
    assert sorted(tuple(sorted(int(i) for i in L.player_ids_list)) for L in s.query(Lineup).all()) == [(1, 2, 3), (1, 2, 4)]


def test_contests_in_entries_one_row_per_name(tmp_path):
    f = tmp_path / "entries.csv"
    f.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,CPT,FLEX,FLEX,FLEX,FLEX,FLEX\n"
                 "1, Showdown A ,100,$5,,,,,,\n2,Showdown A,100,$5,,,,,,\n3,Big,200,$1,,,,,,\n4,Big,201,$1,,,,,,\n,,,,,,,,,\n")
    c = contests_in_entries(f)
    assert list(zip(c["Contest ID"], c["Contest Name"])) == [(100, "Showdown A"), (201, "Big")]


def test_slate_field_builds_the_pool_once(tmp_path, monkeypatch):
    from datetime import datetime
    from ivan.sports.nfl import sample_field as SF
    from ivan.sports.nfl.models import NFLPlayer
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for pid, slot, sal, own in ((1, SF.CPT_SLOT, 15000, 40.0), (2, SF.FLEX_SLOT, 8000, 20.0), (3, SF.FLEX_SLOT, 6000, 10.0), (4, SF.FLEX_SLOT, 4000, 5.0)):
        s.add(NFLPlayer(id=pid, roster_slot_id=slot, name=f"p{pid}", team="A" if pid % 2 else "B", position="WR",
                        gametime=datetime(2026, 1, 1), salary=sal, fpts_mean=10.0, ownership_proj=own))
    for ids in ([1, 2, 3], [1, 2, 4], ["1", "3", "4"]):
        s.add(Lineup(player_ids_list=ids))
    s.commit()
    slate = SF.SlateField.build(s)
    assert slate.n_lineups == 3 and slate.L.shape == (3, 4) and slate.has_ownership
    assert sorted(slate.pid_lists.values()) == [[1, 2, 3], [1, 2, 4], [1, 3, 4]]
    assert slate.spends.tolist() == [29000, 27000, 25000] and (slate.cpt_col_by_row == 0).all()
    assert slate.ownership.tolist() == [40.0, 20.0, 10.0, 5.0]
    calls = []
    monkeypatch.setattr(SF, "_dupe_features", lambda *a, **k: calls.append(1) or {"qb_mates": None})
    assert slate.dupe_features(s, slate.ownership, captain=True) is slate.dupe_features(s, slate.ownership, captain=True)
    slate.dupe_features(s, slate.ownership, captain=False)
    assert len(calls) == 2                                              # one per captain flag, then memoised
