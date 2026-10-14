# tests/core/test_raw.py
"""Raw artifacts: stored as received, indexed, never rewritten, found again."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import raw


def test_store_load_find_and_index(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "PROJECT_ROOT", tmp_path)
    a = raw.store("nfl", "dk/draftables", 154469, {"draftables": [{"id": 1}]}, stamp="20261011-105000", meta={"n": 1})
    assert a.path == "dk/draftables/154469/20261011-105000.json.gz" and a.abspath.exists() and a.bytes > 0
    assert raw.load(a.abspath) == {"draftables": [{"id": 1}]}
    assert raw.load(a.path, "nfl") == {"draftables": [{"id": 1}]}
    b = raw.store("nfl", "dk/standings", 184253394, "Rank,EntryId,Points\n1,5,100\n", ext="csv", stamp="20261012-010000")
    assert raw.load(b.abspath).startswith("Rank,EntryId")
    c = raw.store("nfl", "dk/draftables", 154469, {"draftables": []}, stamp="20261011-113000")
    # never rewritten: the same kind/key/stamp keeps the first payload
    d = raw.store("nfl", "dk/draftables", 154469, {"draftables": [{"id": 999}]}, stamp="20261011-105000")
    assert raw.load(d.abspath) == {"draftables": [{"id": 1}]}
    arts = raw.find("nfl", "dk/draftables", 154469)
    assert [x.stamp for x in arts] == ["20261011-105000", "20261011-105000", "20261011-113000"]
    assert raw.latest("nfl", "dk/draftables", 154469).stamp == "20261011-113000"
    assert raw.find("nfl", "dk/draftables", 154469, since="20261011-110000")[0].stamp == "20261011-113000"
    assert raw.latest("nfl", "odds/props") is None
    s = raw.summary("nfl")
    assert s["dk/draftables"]["n"] == 3 and s["dk/draftables"]["keys"] == 1 and s["dk/standings"]["n"] == 1

    engine = create_engine("sqlite://")
    session = sessionmaker(bind=engine)()
    assert raw.sync_index(session, "nfl") == 3          # three distinct paths (the repeat stamp shares one)
    assert raw.sync_index(session, "nfl") == 0
    rows = session.query(raw.RawArtifactRow).order_by(raw.RawArtifactRow.id).all()
    assert [r.kind for r in rows] == ["dk/draftables", "dk/standings", "dk/draftables"] and rows[1].key == "184253394"
