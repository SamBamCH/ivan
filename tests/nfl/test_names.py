"""A player keeps his odds and game logs when the sources disagree on his generational suffix (actuals.resolve_name)."""
from ivan.sports.nfl.actuals import bare_name, resolve_name


def test_suffix_is_dropped_or_added_to_find_the_name():
    books = {"deebo samuel", "kyle pitts", "marvin harrison jr", "brian thomas jr", "brian thomas"}
    assert bare_name("Deebo Samuel Sr.") == "deebo samuel"
    assert resolve_name("deebo samuel sr", books) == "deebo samuel"          # DraftKings has the suffix, the books do not
    assert resolve_name("marvin harrison", books) == "marvin harrison jr"    # and the other way round
    assert resolve_name("marvin harrison jr", books) == "marvin harrison jr"
    assert resolve_name("brian thomas", books) == "brian thomas"             # the exact spelling wins
    assert resolve_name("kyle trask", books) is None


def test_another_suffix_is_another_man():
    assert resolve_name("ken griffey sr", {"ken griffey jr"}) is None
    assert resolve_name("ken griffey", {"ken griffey jr", "ken griffey sr"}) is None    # two candidates: no guess
