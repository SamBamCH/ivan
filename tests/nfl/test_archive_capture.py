"""The ingest's archive step: pool, props and lines of a capture, keyed by the slate."""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import archive
from ivan.sports.nfl.archive_capture import archive_ingest_capture
from ivan.sports.nfl.models import NFLBase, NFLPlayer


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    archive.ensure_tables(engine)
    return sessionmaker(bind=engine)()


def player(dk_id, slot, name, team, gametime, dg=777):
    p = SimpleNamespace(dk_id=dk_id, name=name, team=team, opponent="B", position="QB", roster_slot_id=slot,
                        gametime=gametime, salary=10000, status=None)
    p.draft_group_id = dg
    return p


def test_capture_from_fetched_pool_and_odds_refresh(session):
    kick = datetime(2026, 10, 15, 19, 15)
    players = [player(1, 511, "qb one", "A", kick), player(2, 512, "qb one", "A", kick)]
    odds = [{"player": "qb one", "market": "player_pass_yds", "sportsbook": "dk", "side": "Over", "point": 250.5, "price": -110, "last_update": "x"}]
    lines = pd.DataFrame([{"game_id": "g1", "sport_key": "nfl", "commence_time": kick, "home_team": "A", "away_team": "B",
                           "home_abbr": "A", "away_abbr": "B", "bookmaker": "dk", "market": "totals", "last_update": "x",
                           "outcome_name": "Over", "side": "over", "point": 45.5, "price": -110}])
    cap = archive_ingest_capture(session, players, odds, lines)
    assert cap.slate_key == "777" and cap.mode == "showdown" and cap.lock == kick
    assert (cap.n_players, cap.n_odds, cap.n_lines) == (2, 1, 1)
    blobs = archive.load_blobs(session, cap.id)
    assert blobs["players"][0]["gametime"] == kick.isoformat() and blobs["game_lines"][0]["point"] == 45.5

    # an odds-only refresh reads the pool back from nfl_player and keeps the slate key
    for p in players:
        session.add(NFLPlayer(id=p.dk_id, roster_slot_id=p.roster_slot_id, name=p.name, team=p.team, opponent=p.opponent,
                              position=p.position, gametime=p.gametime, salary=p.salary))
    session.commit()
    cap2 = archive_ingest_capture(session, None, odds * 2, None)
    assert cap2.slate_key == "777" and cap2.n_odds == 2 and cap2.n_lines == 0 and cap2.note == "odds refresh"
    assert [c.id for c in archive.slates(session, "nfl")] == [cap2.id]        # latest capture before lock wins


def test_capture_without_pool_is_skipped(session):
    assert archive_ingest_capture(session, None, [], None) is None
