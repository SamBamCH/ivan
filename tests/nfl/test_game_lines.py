"""Game lines for a matchup: only snapshots taken by the clock count, so a replay never reads an in-game spread."""
from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nfl.models import NFLBase, NFLGameOdds
from ivan.sports.nfl.refactor.dst import matchup_lines


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'g.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _rows(last_update: str, spread: float, total: float):
    base = dict(game_id="g1", sport_key="americanfootball_nfl", commence_time=datetime(2026, 10, 11, 17, 3), home_team="Green Bay Packers",
                away_team="Chicago Bears", home_abbr="GB", away_abbr="CHI", bookmaker="draftkings", last_update=last_update, price=-110)
    return [NFLGameOdds(**base, market="spreads", outcome_name="Green Bay Packers", side="home", point=spread),
            NFLGameOdds(**base, market="spreads", outcome_name="Chicago Bears", side="away", point=-spread),
            NFLGameOdds(**base, market="totals", outcome_name="Over", side="over", point=total),
            NFLGameOdds(**base, market="totals", outcome_name="Under", side="under", point=total)]


def test_matchup_lines_ignore_snapshots_taken_after_the_clock(session):
    session.add_all(_rows("2026-10-11T15:00:00Z", -3.5, 44.5))       # the pre-game line, 10:00 local
    session.add_all(_rows("2026-10-11T19:33:00Z", -23.5, 48.5))      # fourth quarter, fetched by a later slate's ingest
    session.commit()
    # a replay pinned a minute before the noon lock sees the pre-game line
    total, home_margin, is_home = matchup_lines(session, "GB", "CHI", now=datetime(2026, 10, 11, 11, 59))
    assert (total, home_margin, is_home) == (44.5, 3.5, True)
    # the live pipeline later in the day sees the latest snapshot, as before
    total, home_margin, _ = matchup_lines(session, "CHI", "GB", now=datetime(2026, 10, 11, 15, 0))
    assert (total, home_margin) == (48.5, 23.5)
    # nothing taken by the clock: no line rather than a future one
    assert matchup_lines(session, "GB", "CHI", now=datetime(2026, 10, 11, 9, 0)) is None
