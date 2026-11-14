"""A player's hand calls live under one name, his NBA key, however the name was spelled when they were made."""
from ivan.sports.nba.projection import live as L


def test_write_overrides_keeps_one_row_per_player(tmp_path):
    p = tmp_path / "minutes_overrides.csv"
    L.write_overrides({"jaime jaquez": {"starter": 1.0}, "jaime jaquez jr": {"starter": 1.0, "minutes": 30.0}, "kyle kuzma": 31.0}, p)
    got = L.read_overrides_full(p, layered=False)
    assert set(got) == {"jaime jaquez", "kyle kuzma"} and got["jaime jaquez"]["starter"] == 1.0 and got["jaime jaquez"]["minutes"] == 30.0
    assert p.read_text().count("jaquez") == 1
    L.write_overrides({"kyle kuzma": 31.0}, p)                                   # cleared: he is gone, under every spelling
    assert set(L.read_overrides_full(p, layered=False)) == {"kyle kuzma"}


def test_set_override_folds_every_spelling_and_a_clear_clears_the_player(tmp_path, monkeypatch):
    from ivan.sports.nba.desk import Desk
    p = tmp_path / "minutes_overrides.csv"
    written = {}

    class D(Desk):
        def __init__(self):                                                      # the pins only: no slate behind it
            self.local_overrides = {"jaime jaquez": dict.fromkeys(L.OVERRIDE_KEYS) | {"starter": 1.0},
                                    "jaime jaquez jr": dict.fromkeys(L.OVERRIDE_KEYS) | {"starter": 1.0}}

        def write_local(self):
            L.write_overrides(self.local_overrides, p); written.update(L.read_overrides_full(p, layered=False))
    d = D()
    d.set_override("Jaime Jaquez Jr.", starter=None)
    assert d.local_overrides == {} and "jaime jaquez" not in L.read_overrides_full(p, layered=False)
    d.set_override("jaime jaquez jr", minutes=28.0)
    assert list(d.local_overrides) == ["jaime jaquez"] and L.read_overrides_full(p, layered=False)["jaime jaquez"]["minutes"] == 28.0
