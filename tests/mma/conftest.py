"""Shared fixtures for the MMA tests: the sample card under tests/mma/data and a fresh SQLite per test."""
import json
from pathlib import Path

import pytest

from ivan.database.session import get_session, init_db
from ivan.sports.mma.ingest import ingest_slate
from ivan.sports.mma.models import MMABase

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    c = json.loads((ROOT / "config" / "mma.json").read_text())
    c["paths"] = {"players_csv": str(DATA / "player_ids.csv"), "fight_odds_csv": str(DATA / "fight_odds.csv"),
                  "player_odds_csv": str(DATA / "player_odds.csv"), "payouts_csv": str(DATA / "payouts.csv")}
    return c


@pytest.fixture
def session(tmp_path):
    engine = init_db("mma", f"sqlite:///{tmp_path / 'mma.sqlite'}")
    MMABase.metadata.create_all(engine)
    return get_session(engine)


@pytest.fixture
def data():
    return DATA


@pytest.fixture
def ingest():
    """ingest_slate on the sample card from CSV, no network."""
    def run(session, cfg, **kw):
        return ingest_slate(session, cfg, players_csv=cfg["paths"]["players_csv"], fetch_contests=False, **kw)
    return run
