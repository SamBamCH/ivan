"""data_clean.odds.store_props replaces the odds table by game: a fetch of one slate's games leaves another slate's rows, a back-to-back's
two games sit side by side, rows without a game are replaced by player, and old games are dropped."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ivan.sports.nba.models import NBAOdds
from ivan.sports.nba.data_clean import odds as O


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _rows(game, commence, players, book="dk", home="IND", away="BOS", last_update=None):
    return [dict(player=p, market="player_points", sportsbook=book, side="Over", point=20.5, price=-110,
                 last_update=last_update or _utc(commence - timedelta(hours=2)), game_id=game, commence_time=_utc(commence) if commence else None,
                 home_abbr=home if game else None, away_abbr=away if game else None) for p in players]


def test_store_props_replaces_by_game_and_drops_old_games():
    eng = create_engine("sqlite://")
    NBAOdds.__table__.create(eng)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    thu, fri = now - timedelta(hours=20), now + timedelta(hours=4)
    with Session(eng) as s:
        legacy = _rows(None, None, ["old legacy"], last_update=_utc(now - timedelta(hours=30)))
        legacy += _rows(None, None, ["stale legacy"], last_update=_utc(now - timedelta(days=5)))
        assert O.store_props(s, pd.DataFrame(_rows("g_thu", thu, ["a", "b"]) + _rows("g_old", now - timedelta(days=4), ["z"]) + legacy)) == 5
        have = pd.read_sql("select * from odds", eng)
        assert set(have.game_id.dropna()) == {"g_thu"} and set(have.player) == {"a", "b", "old legacy"}      # g_old and the 5-day-old legacy row are gone
        # Friday's fetch: a's Friday game joins Thursday's (a back-to-back), c is new; Thursday's rows stay for Thursday's slate
        O.store_props(s, pd.DataFrame(_rows("g_fri", fri, ["a", "c"], home="IND", away="MIA")))
        have = pd.read_sql("select * from odds", eng)
        assert sorted(have[have.player == "a"].game_id) == ["g_fri", "g_thu"] and set(have.player) == {"a", "b", "c", "old legacy"}
        # a re-fetch of Friday's game replaces only its rows (b's Thursday row and a's Thursday row are untouched)
        O.store_props(s, pd.DataFrame(_rows("g_fri", fri, ["a"], book="fd")))
        have = pd.read_sql("select * from odds", eng)
        fri_rows = have[have.game_id == "g_fri"]
        assert list(fri_rows.player) == ["a"] and list(fri_rows.sportsbook) == ["fd"] and len(have[have.game_id == "g_thu"]) == 2
        # rows without a game (an old capture restored) replace that player's rows without a game only
        O.store_props(s, pd.DataFrame(_rows(None, None, ["old legacy"], book="mg", last_update=_utc(now - timedelta(hours=1)))))
        have = pd.read_sql("select * from odds", eng)
        assert list(have[have.player == "old legacy"].sportsbook) == ["mg"] and len(have) == 4
        assert O.store_props(s, None) == 0 and O.store_props(s, pd.DataFrame()) == 0
