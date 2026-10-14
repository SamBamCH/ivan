"""The ingest's store step (ivan.sports.nfl.ingest): replace the working tables, archive, drop OUT/IR."""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import archive
from ivan.sports.nfl import ingest
from ivan.sports.nfl.models import NFLBase, NFLGameOdds, NFLOdds, NFLPlayer


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    archive.ensure_tables(engine)
    return sessionmaker(bind=engine)()


def player(dk_id, slot, name, team, gametime, status=None, dg=777):
    p = SimpleNamespace(dk_id=dk_id, name=name, team=team, opponent="B", position="QB", roster_slot_id=slot,
                        gametime=gametime, salary=10000, status=status)
    p.draft_group_id = dg
    return p


def lines(kick, point=45.5):
    return pd.DataFrame([{"game_id": "g1", "sport_key": "nfl", "commence_time": kick, "home_team": "A", "away_team": "B",
                          "home_abbr": "A", "away_abbr": "B", "bookmaker": "dk", "market": "totals", "last_update": "x",
                          "outcome_name": "Over", "side": "over", "point": point, "price": -110}])


def test_store_replaces_pool_odds_and_lines_and_drops_out(session, monkeypatch):
    monkeypatch.setattr("ivan.core.raw.sync_index", lambda s, sport: 0)   # the raw index lives under data/; not under test here
    kick = datetime(2026, 10, 15, 19, 15)
    old = SimpleNamespace(players=[player(9, 512, "stale", "Z", kick)], odds=[], game_lines=None)
    ingest.store_slate(session, ingest.SlateFetch(**vars(old)), prelock=True, log=lambda m: None)
    assert [p.name for p in session.query(NFLPlayer).all()] == ["stale"]

    fetched = ingest.SlateFetch(
        players=[player(1, 511, "qb one", "A", kick), player(2, 512, "qb one", "A", kick), player(3, 512, "hurt", "B", kick, status="OUT")],
        odds=[{"player": "qb one", "market": "player_pass_yds", "sportsbook": "dk", "side": "Over", "point": 250.5, "price": -110, "last_update": "x"}],
        game_lines=pd.concat([lines(kick), lines(kick)]),   # a duplicated snapshot row must not violate the unique constraint
    )
    logs = []
    info = ingest.store_slate(session, fetched, prelock=True, log=logs.append)
    assert info.players == 3 and info.odds == 1 and info.games == 1 and info.line_rows == 1 and info.removed_out == 1
    assert sorted(p.name for p in session.query(NFLPlayer).all()) == ["qb one", "qb one"]     # stale pool gone, OUT dropped
    assert session.query(NFLOdds).count() == 1 and session.query(NFLGameOdds).count() == 1
    assert info.capture and info.capture["slate_key"] == "777" and info.capture["mode"] == "showdown" and info.capture["lock"] == kick
    assert any("archived capture" in m for m in logs)

    # an odds-only refresh keeps the pool and replaces only the fetched games' lines
    info2 = ingest.store_slate(session, ingest.SlateFetch(odds=fetched.odds * 2, game_lines=lines(kick, 46.5)), prelock=False, log=lambda m: None)
    assert info2.players == 0 and info2.odds == 2 and session.query(NFLPlayer).count() == 2
    assert [r.point for r in session.query(NFLGameOdds).all()] == [46.5]


def test_fetch_refuses_to_do_nothing(session):
    with pytest.raises(ingest.IngestError):
        ingest.fetch_slate(session, prelock=False, update_odds=False)
