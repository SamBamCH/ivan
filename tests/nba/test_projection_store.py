"""The projection store (ivan.sports.nba.projection.store): one record per team, written by whatever projects, read by the desk."""
import pandas as pd
import pytest

from ivan.sports.nba.projection import store as S


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    from ivan.sports.nba import slates as SL
    monkeypatch.setattr(SL, "SLATES_DIR", tmp_path)
    tok = SL.use(155)
    yield tmp_path / "155" / "projection"
    SL.release(tok)


def _frame(names, fpts):
    return pd.DataFrame({"minutes": [30.0] * len(names), "fpts_mean": fpts, "fpts_std": [8.0] * len(names)}, index=pd.Index(names, name="player_name"))


def test_save_load_roundtrip_keeps_frames_grid_and_slate(store_dir):
    grid = pd.DataFrame([[1] * 48], index=["a"], columns=range(48)); grid.attrs["locked"] = ["a"]
    rec = S.record("BOS", _frame(["a", "b"], [30.0, 20.0]), _frame(["a", "b"], [28.0, 21.0]), {"grid": grid, "zero_minutes": ["c"], "curve": None}, 155, 4000)
    p = S.save(rec)
    assert p == store_dir / "BOS.pkl" and not (store_dir / "BOS.pkl.tmp").exists()      # written atomically
    back = S.load("BOS")
    assert back["draft_group"] == 155 and back["n_worlds"] == 4000 and back["sim"] is None
    pd.testing.assert_frame_equal(back["frame"], rec["frame"]); pd.testing.assert_frame_equal(back["hist"], rec["hist"])
    assert back["rot"]["grid"].attrs["locked"] == ["a"] and back["rot"]["zero_minutes"] == ["c"]
    assert S.load("NYK") is None
    assert S.stamp()[1] == 1


def test_load_all_filters_on_the_slate_and_prune_removes_the_rest(store_dir):
    S.save(S.record("BOS", _frame(["a"], [30.0]), _frame(["a"], [30.0]), {}, 155, 4000))
    S.save(S.record("NYK", _frame(["b"], [30.0]), _frame(["b"], [30.0]), {}, 155, 4000))
    S.save(S.record("LAL", _frame(["c"], [30.0]), _frame(["c"], [30.0]), {}, 154, 4000))      # yesterday's slate
    assert set(S.load_all()) == {"BOS", "NYK", "LAL"}
    assert set(S.load_all(155)) == {"BOS", "NYK"}                                               # another slate's record is ignored, not deleted
    assert S.prune(155, ["BOS"]) == 2                                                           # LAL (other slate) and NYK (off the slate) go
    assert set(S.load_all()) == {"BOS"}
    assert S.clear() == 1 and S.load_all() == {} and S.stamp() == (None, 0)


def test_a_record_another_version_cannot_read_is_treated_as_missing(store_dir):
    store_dir.mkdir(parents=True)
    (store_dir / "BOS.pkl").write_bytes(b"not a pickle")
    assert S.load("BOS") is None and S.load_all() == {}
    assert S.prune(155, ["BOS"]) == 1                                                           # and it is cleared out
