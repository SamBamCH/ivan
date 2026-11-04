"""A team's mapping is void once a player it counted on is out (ivan.sports.nba.mapping)."""
import json

import pandas as pd
import pytest

from ivan.sports.nba import mapping as MAP
from ivan.sports.nba import scenarios as SC
from ivan.sports.nba import slates as S
from ivan.sports.nba.projection import rotation as ROT


@pytest.fixture
def slate(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SLATES_DIR", tmp_path / "slates")
    tok = S.use(7)
    S.paths(7).ensure()
    yield 7
    S.release(tok)


def _mapped(team, names, available):
    rows = [[True] * 36 + [False] * 12, [False] * 12 + [True] * 36, [True] * 10 + [False] * 38][:len(names)]
    grid = pd.DataFrame(rows, index=names, columns=list(range(48)))
    ROT.save_drawn(team, grid, locked=names, fingerprint=dict(available=sorted(available), out=[], status={}, starters=[]))


def test_a_new_out_voids_the_mapping_and_keeps_it_as_a_scenario(slate):
    names = ["a one", "b two", "c three"]
    _mapped("ATL", names, names)
    local = {"a one": {"minutes": 36.0}, "b two": {"minutes": 36.0, "fppm": 1.1}, "c three": {"minutes": 10.0}}
    assert MAP.stale_outs("ATL", {n: "" for n in names}, local) == []                                   # nothing moved: the mapping stands
    statuses = {"a one": "", "b two": "OUT", "c three": ""}
    assert MAP.stale_outs("ATL", statuses, local) == ["b two"]
    rep = MAP.void_mapping("ATL", ["b two"], names, local, log=lambda m: None)
    assert rep["out"] == ["b two"] and rep["pins"] == 2 and rep["label"] == "Two in"
    assert ROT.load_drawn("ATL") is None                                                                  # the grid is gone
    assert local == {"b two": {"minutes": 36.0, "fppm": 1.1}}                                             # the pins lifted; the out player's own row untouched
    sc = SC.load_scenario("ATL", rep["scenario"])
    assert sc["assumes"] == {"b two": "in"} and set(sc["overrides"]) == {"a one", "c three"} and sc["grid"]["players"] == names
    assert SC.active_slug("ATL") is None                                                                  # kept, not live
    assert MAP.stale_outs("ATL", statuses, local) == []                                                   # void once: nothing to void again


def test_a_scratch_call_is_a_new_out_too_and_published_grids_are_left_alone(slate, monkeypatch):
    names = ["a one", "b two"]
    _mapped("BOS", names, names)
    local = {"a one": {"minutes": 0.0}}                                                                  # the desk's own scratch call
    assert MAP.stale_outs("BOS", {n: "" for n in names}, local) == ["a one"]
    rep = MAP.void_mapping("BOS", ["a one"], names, local, log=lambda m: None)
    assert local == {"a one": {"minutes": 0.0}} and rep["pins"] == 0                                      # the scratch stays: it is the call
    # a consumer: the laptop's published grid is not this machine's to void
    monkeypatch.setenv("IVAN_FEED_SSH", "me@laptop")
    p = S.published_rotations_dir() / "CHI.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"team": "CHI", "players": names, "grid": [[1] * 48, [0] * 48], "locked": names,
                             "fingerprint": dict(available=names, out=[], status={}, starters=[])}))
    assert ROT.load_drawn("CHI").attrs["source"] == "published" and MAP.stale_outs("CHI", {"a one": "OUT", "b two": ""}, {}) == []
