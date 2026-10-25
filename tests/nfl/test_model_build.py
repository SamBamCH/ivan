"""The model build (ivan.sports.nfl.model_build) and the projection scoring it uses."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ivan.core import model_store
from ivan.sports.nfl import backtest_projections as BP
from ivan.sports.nfl import model_build


def test_assign_week_and_sd_factors():
    weeks = pd.DataFrame({"season": [2025, 2025], "week": [10, 11], "first": ["2025-11-06", "2025-11-13"], "last": ["2025-11-10", "2025-11-17"]})
    w = BP.assign_week(pd.Series(["2025-11-09", "2025-11-16", "2025-12-01"]), weeks)
    assert w.week.tolist()[:2] == [10, 11] and pd.isna(w.week.iloc[2])

    rng = np.random.default_rng(0)
    n = 300
    scored = pd.DataFrame({"position": ["WR"] * n + ["K"] * 10, "fpts_mean": 10.0, "fpts_std": 5.0, "fpts_std_model": 4.0,
                           "actual": np.r_[10 + rng.normal(0, 6.0, n), 8 + rng.normal(0, 1, 10)]})
    factors, counts, skipped = BP.sd_calibration_factors(scored, ["WR", "K"])
    assert counts == {"WR": n} and skipped == {"K": 10}
    assert factors["WR"] == pytest.approx(6.0 / 4.0, rel=0.15)       # z on the model sd, not the calibrated one


def test_build_runs_every_step_and_records_failures(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model_build, "build_dst_params", lambda s, out, before, log: calls.append(("dst", before)) or {"n": 1})
    monkeypatch.setattr(model_build, "build_classic_field_models", lambda s, out, cut, log: calls.append(("classic", cut)) or {})
    monkeypatch.setattr(model_build, "build_dupe_model", lambda s, out, cutoff, log: (_ for _ in ()).throw(RuntimeError("no fields")))
    monkeypatch.setattr(model_build, "build_sd_calibration", lambda s, out, since, until, log: calls.append(("sd", since, until)) or {})
    monkeypatch.setattr(model_build, "build_ownership_calibration", lambda s, out, cutoff, log: calls.append(("oc", cutoff)) or {})
    monkeypatch.setattr(model_build, "build_ownership_model", lambda s, out, cutoff, log: calls.append(("om", cutoff)) or {"built": True})
    import ivan.sports.nfl.actuals as A
    monkeypatch.setattr(A, "week_for_date", lambda s, d: (2025, 9))

    base = tmp_path / "models"
    m = model_build.build(session=None, cutoff=date(2025, 11, 1), out_dir=base / "asof" / "2025-11-01", log=lambda s: None)
    assert m["failed"] == ["dupe"] and m["built"]["dupe"]["ok"] is False and "no fields" in m["built"]["dupe"]["error"]
    assert ("dst", (2025, 9)) in calls and ("sd", model_build.SD_SINCE, "2025-10-31") in calls and ("om", "2025-11-01") in calls
    assert not any(c[0] == "dst_table" for c in calls)                      # the table is live-only
    manifest = json.loads((base / "asof" / "2025-11-01" / "manifest.json").read_text())
    assert manifest["cutoff"] == "2025-11-01" and set(manifest["built"]) == {"dst", "classic", "dupe", "sd", "ownership_calibration", "ownership_model"}
    assert model_store.resolve(base, date(2025, 11, 20)).cutoff == date(2025, 11, 1)

    with pytest.raises(ValueError):
        model_build.build(session=None, out_dir=base, only=["nope"])
    calls.clear()
    m = model_build.build(session=None, out_dir=base, only=["sd", "ownership_model"], log=lambda s: None)
    assert [c[0] for c in calls] == ["sd", "om"] and m["failed"] == [] and calls[0][2] is None


def test_ownership_training_rows_are_showdown_only(tmp_path):
    """The learned ownership model and its calibration run on showdown slates only, so classic rows (per-slot shares,
    slot_flex=0 like a captain) stay out of their training frames."""
    from datetime import datetime
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from ivan.sports.nfl import ownership_model as OM
    from ivan.sports.nfl.models import HistoricalContest, NFLBase, ShowdownPlayer
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(HistoricalContest(contest_id=1, draft_group_id=10, name="sd", date=datetime(2026, 10, 11), entry_fee=3.0, contest_type="Showdown Captain Mode", total_entries=1000, multi_entry=1))
    s.add(HistoricalContest(contest_id=2, draft_group_id=11, name="cl", date=datetime(2026, 10, 11), entry_fee=3.0, contest_type="Classic", total_entries=1000, multi_entry=1))
    pid = 0
    for cid, slots in ((1, (511, 512)), (2, (66, 70))):
        for slot in slots:
            for k in range(3):
                pid += 1
                s.add(ShowdownPlayer(contest_id=cid, player_id=pid, roster_slot_id=slot, salary=5000 + 500 * k, position="WR", team="A" if k else "B",
                                     ownership_actual=10.0 + k, ownership_proj=9.0 + k, fpts_proj=12.0 + k))
    s.commit()
    df = OM.training_frame(s)
    assert sorted(df.contest_id.unique()) == [1] and sorted(df.slot.unique()) == [511, 512] and len(df) == 6
    cal = OM.fit_calibration(s, min_n=1)
    assert set(cal["maps"]) == {"511:WR", "512:WR"} and cal["contests"] == 1 and cal["rows"] == 6
