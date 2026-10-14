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
