"""Predicted vs actual duplication over a scraped field (ivan.sports.nfl.dupe_analysis)."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import archive
from ivan.sports.nfl import dupe_analysis as DA
from ivan.sports.nfl import dupe_model as DM
from ivan.sports.nfl.models import HistoricalContest, NFLBase, ShowdownPlayer

POOL = [  # (slot, name, team, opp, pos, salary, own_proj, own_actual)
    (511, "josh allen", "BUF", "LAR", "QB", 12000, 20.0, 30.0), (512, "josh allen", "BUF", "LAR", "QB", 8000, 58.0, 54.0),
    (511, "puka nacua", "LAR", "BUF", "WR", 12000, 15.0, 12.0), (512, "puka nacua", "LAR", "BUF", "WR", 8000, 47.0, 42.0),
    (512, "matthew stafford", "LAR", "BUF", "QB", 7000, 52.0, 57.0), (512, "kyren williams", "LAR", "BUF", "RB", 7000, 37.0, 38.0),
    (512, "dalton kincaid", "BUF", "LAR", "TE", 5000, 37.0, 30.0), (512, "keon coleman", "BUF", "LAR", "WR", 4000, 22.0, 25.0),
    (512, "tyler bass", "BUF", "LAR", "K", 4000, 8.0, 15.0), (512, "rams", "LAR", "BUF", "DST", 3200, 5.0, 13.0),
]
LINEUPS = {  # slot tokens -> copies in the field
    ("CPT josh allen", "FLEX puka nacua", "FLEX matthew stafford", "FLEX kyren williams", "FLEX dalton kincaid", "FLEX keon coleman"): 40,
    ("CPT puka nacua", "FLEX josh allen", "FLEX matthew stafford", "FLEX kyren williams", "FLEX dalton kincaid", "FLEX keon coleman"): 25,
    ("CPT josh allen", "FLEX puka nacua", "FLEX matthew stafford", "FLEX kyren williams", "FLEX tyler bass", "FLEX rams"): 3,
    ("CPT puka nacua", "FLEX josh allen", "FLEX kyren williams", "FLEX dalton kincaid", "FLEX tyler bass", "FLEX rams"): 1,
}


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    archive.ensure_tables(engine)
    s = sessionmaker(bind=engine)()
    lock = datetime(2026, 10, 12, 19, 15)
    s.add(HistoricalContest(contest_id=100, draft_group_id=777, name="NFL Showdown $5K test (BUF @ LAR)", date=lock, entry_fee=1.0,
                            contest_type="Showdown Captain Mode", total_entries=69, multi_entry=20))
    pid = 0
    for slot, name, team, opp, pos, sal, op, oa in POOL:
        pid += 1
        s.add(ShowdownPlayer(contest_id=100, player_id=pid, roster_slot_id=slot, salary=sal, position=pos, team=team, ownership_proj=op, ownership_actual=oa, fpts_proj=10.0))
    s.commit()
    archive.archive_slate(s, sport="nfl", slate_key="777", lock=lock, mode="showdown",
                          players=[{"id": i + 1, "roster_slot_id": p[0], "name": p[1], "team": p[2], "opponent": p[3], "position": p[4], "gametime": lock,
                                    "salary": p[5], "status": None} for i, p in enumerate(POOL)])
    derived = tmp_path / "derived"
    (derived / "100").mkdir(parents=True)
    rows = []
    for toks, n in LINEUPS.items():
        for k in range(n):
            rows.append({"rank": len(rows) + 1, "entry_id": len(rows) + 1, "entry_name": "x", "points": 100.0, **{f"slot{i + 1}": t for i, t in enumerate(toks)}})
    pd.DataFrame(rows).to_parquet(derived / "100" / "lineups.parquet")
    (derived / "100" / "meta.json").write_text(json.dumps({"contest_id": 100, "entries": len(rows)}))
    monkeypatch.setattr(DA, "DERIVED_DIR", derived)
    return s


def test_field_lineups_features_and_predictions(session):
    model = DM.ProductDupeModel(params={"showdown": {"k": 0.05, "beta": 0.6}, "classic": {"k": 1e-4, "beta": 0.1}})
    df = DA.field_lineups(session, 100, model)
    assert len(df) == 4 and df["count"].tolist() == [40, 25, 3, 1] and df["rank"].tolist() == [1, 2, 3, 4]
    top = df.iloc[0]
    assert top.cpt_name == "josh allen" and top.cpt_pos == "QB" and top.split == "3-3" and top.cpt_side == "even split"
    assert top.n_qb == 2 and top.qb_mates == 2 and top.salary_used == 12000 + 8000 + 7000 + 7000 + 5000 + 4000 and top.salary_left == 50000 - top.salary_used
    slo = sum(np.log(o / 100) for o in (20.0, 47.0, 52.0, 37.0, 37.0, 22.0))
    assert top.sum_log_own_proj == pytest.approx(slo) and top.pred_proj == pytest.approx(69 * 0.05 * np.exp(0.6 * slo))
    assert top.resid_proj == pytest.approx(np.log(40 / top.pred_proj)) and not np.isnan(top.pred_actual)
    assert df.iloc[3].qb_mates == 1 and df.iloc[2].split == "4-2" and df.iloc[2].cpt_side == "minority"   # allen + kincaid; CPT allen on the BUF 2 side

    sc = DA.scraped_contests(session)
    assert sc.contest_id.tolist() == [100] and bool(sc.has_actual_own.iloc[0]) and bool(sc.is_showdown.iloc[0])


def test_analyze_buckets_calibration_and_report(session):
    model = DM.ProductDupeModel(params={"showdown": {"k": 0.05, "beta": 0.6}, "classic": {"k": 1e-4, "beta": 0.1}})
    rep = DA.analyze(session, [100, 999], model)
    assert rep["totals"]["contests"] == 1 and rep["totals"]["entries"] == 69 and rep["totals"]["unique"] == 4
    c = rep["contests"][0]
    assert c["top1_actual"] == 40 and c["dupe_share"] == pytest.approx(68 / 69) and c["has_actual"]
    keys = [f["key"] for f in rep["features"]]
    assert "rank_bucket" in keys and "salary_left_bucket" in keys and "qb_stack" in keys
    rank = {r["bucket"]: r for r in next(f for f in rep["features"] if f["key"] == "rank_bucket")["rows"]}
    assert rank["top lineup"]["copies"] == 40 and rank["top lineup"]["share_of_entries"] == pytest.approx(40 / 69)
    assert sum(r["copies"] for r in rank.values()) == 69
    cal = rep["calibration"]["proj"]
    assert sum(r["lineups"] for r in cal) == 4 and all(r["pred_bin"][0] != "-" for r in cal)
    assert rep["top"][0]["count"] == 40 and "josh allen" in rep["top"][0]["lineup"]
    text = DA.format_report(rep)
    assert "1 contests, 69 entries, 4 unique lineups" in text and "most duplicated lineups" in text
    assert DA.format_report(DA.analyze(session, [999], model)) == "no scraped fields for those contests"
