# tests/nfl/test_model_store.py
"""As-of model versions resolve by the replay clock; live runs keep the live directory."""
from __future__ import annotations

import os
from datetime import date, datetime

from ivan.core import model_store


def test_resolve_picks_the_newest_version_before_the_date(tmp_path):
    base = tmp_path / "models"
    base.mkdir()
    for c in ("2025-09-01", "2025-11-01", "2026-01-15"):
        model_store.write_manifest(model_store.asof_dir(base, date.fromisoformat(c)), date.fromisoformat(c), {"dupe_model": {"ok": True}})
    (base / model_store.ASOF / "junk").mkdir()                       # not a cutoff: ignored
    assert [m.cutoff.isoformat() for m in model_store.versions(base)] == ["2025-09-01", "2025-11-01", "2026-01-15"]
    live = model_store.resolve(base, None)
    assert live.path == base and live.cutoff is None and not live.fallback and live.label == "live models"
    m = model_store.resolve(base, datetime(2025, 11, 5, 12, 0))
    assert m.cutoff == date(2025, 11, 1) and m.path == base / "asof" / "2025-11-01" and m.manifest["built"]["dupe_model"]["ok"]
    assert model_store.resolve(base, date(2025, 11, 1)).cutoff == date(2025, 11, 1)        # on the cutoff day itself
    assert model_store.resolve(base, date(2026, 6, 1)).cutoff == date(2026, 1, 15)
    early = model_store.resolve(base, date(2025, 8, 1))
    assert early.path == base and early.fallback and "no as-of" in early.label
    d = m.describe()
    assert d["cutoff"] == "2025-11-01" and d["built"] == ["dupe_model"] and not d["fallback"]


def test_write_manifest_merges_builders(tmp_path):
    d = tmp_path / "v"
    model_store.write_manifest(d, date(2025, 11, 1), {"a": {"n": 1}})
    m = model_store.write_manifest(d, date(2025, 11, 1), {"b": {"n": 2}}, notes=["second pass"])
    assert set(m["built"]) == {"a", "b"} and m["notes"] == ["second pass"] and m["cutoff"] == "2025-11-01"
    assert model_store.read_manifest(d)["built"]["a"]["n"] == 1


def test_nfl_models_dir_follows_the_clock(tmp_path, monkeypatch):
    from ivan.sports.nfl import paths
    base = tmp_path / "models"
    model_store.write_manifest(model_store.asof_dir(base, date(2025, 11, 1)), date(2025, 11, 1), {})
    monkeypatch.setattr(paths, "MODELS_DIR", base)
    monkeypatch.delenv("IVAN_AS_OF", raising=False)
    assert paths.models_dir() == base
    monkeypatch.setenv("IVAN_AS_OF", "2025-11-20T12:00:00")
    assert paths.models_dir() == base / "asof" / "2025-11-01"
    assert paths.models_info()["cutoff"] == "2025-11-01"
