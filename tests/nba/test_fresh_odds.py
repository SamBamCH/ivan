"""Only a prop posted for the player's game on this slate counts (projection.market.fresh_rows): by the row's game when it has one,
by a short last_update window when it does not."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from ivan.sports.common.odds import parse_odds
from ivan.sports.nba.projection import market as M


def _utc(local: datetime) -> str:
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(player, book, last_update, game=None, **kw):
    r = dict(player=player, market="player_points", sportsbook=book, side="Over", point=22.5, price=-110, last_update=last_update,
             game_id=None, commence_time=None, home_abbr=None, away_abbr=None)
    if game is not None:
        r.update(game_id=f"g{game:%d%H}", commence_time=_utc(game), home_abbr="IND", away_abbr="BOS")
    r.update(kw)
    return r


def test_rows_with_a_game_are_matched_by_the_game_not_by_when_they_were_posted():
    fri = datetime(2026, 11, 6, 19, 0); thu = datetime(2026, 11, 5, 19, 0)                           # a back-to-back
    rows = [_row("pascal siakam", "dk", _utc(thu - timedelta(hours=3)), game=thu),                   # Thursday's game, posted 27 h before Friday's tip
            _row("pascal siakam", "fd", _utc(fri - timedelta(hours=10)), game=thu),                  # Thursday's game, re-posted Friday morning: still Thursday's
            _row("pascal siakam", "mg", _utc(fri - timedelta(hours=40)), game=fri),                  # Friday's game, posted two days early: Friday's
            _row("pascal siakam", "cz", _utc(fri - timedelta(hours=1)), game=fri + timedelta(minutes=10)),   # DK's and the book's tips 10 minutes apart
            _row("obi toppin", "dk", _utc(fri - timedelta(hours=1)), game=fri)]                      # not in the pool
    df = pd.DataFrame(rows)
    kept = M.fresh_rows(df, {"pascal siakam": fri})
    assert list(kept.sportsbook) == ["mg", "cz"]
    assert list(M.fresh_rows(df, {"pascal siakam": thu}).sportsbook) == ["dk", "fd"]


def test_rows_without_a_game_count_only_in_the_fallback_window():
    tip = datetime(2026, 11, 4, 19, 0)
    rows = [_row("obi toppin", "dk", _utc(datetime(2026, 11, 2, 17, 39))),                           # Nov 2's game
            _row("pascal siakam", "dk", _utc(datetime(2026, 11, 4, 17, 28))),                        # tonight
            _row("pascal siakam", "fd", _utc(tip - timedelta(hours=M.FALLBACK_HOURS + 1))),          # too early to tell: out
            _row("pascal siakam", "mg", None),                                                       # no time: unknown game
            _row("pascal siakam", "cz", _utc(tip + timedelta(hours=3))),                             # posted after the tip: in-game lines
            _row("pascal siakam", "bv", _utc(datetime(2026, 11, 4, 17, 28)).replace("T", " ").rstrip("Z"))]   # a restored capture's format, mixed in: still parsed
    df = pd.DataFrame(rows)
    assert list(M.fresh_rows(df, {"obi toppin": tip, "pascal siakam": tip}).sportsbook) == ["dk", "bv"]
    assert M.fresh_rows(df, {}).empty and M.fresh_rows(df.iloc[0:0], {"obi toppin": tip}).empty
    assert M.FALLBACK_HOURS == 12.0
    legacy = df.drop(columns=["game_id", "commence_time", "home_abbr", "away_abbr"])                 # an archive capture from before the columns
    assert list(M.fresh_rows(legacy, {"obi toppin": tip, "pascal siakam": tip}).sportsbook) == ["dk", "bv"]
    assert len(M.fresh_rows(legacy, {"obi toppin": tip, "pascal siakam": tip}, hours=72)) == 4     # a wider window takes Nov 2's row back


def _event(eid, home, away, commence, player="Pascal Siakam"):
    return {"id": eid, "sport_key": "basketball_nba", "commence_time": commence, "home_team": home, "away_team": away,
            "bookmakers": [{"key": "dk", "markets": [{"key": "player_points", "last_update": "2026-11-06T20:00:00Z",
                                                      "outcomes": [{"name": "Over", "description": player, "point": 22.5, "price": -110},
                                                                   {"name": "Under", "description": player, "point": 22.5, "price": -110}]}]}]}


def test_parse_odds_keeps_each_rows_game_once_per_event():
    ev = _event("e1", "Indiana Pacers", "Boston Celtics", "2026-11-07T00:00:00Z")
    by_team = {"IND": ev, "BOS": ev, "NOP": _event("e2", "New Orleans Pelicans", "Minnesota Timberwolves", "2026-11-07T01:10:00Z", "Zion Williamson")}
    df = parse_odds(by_team, games=True)
    assert len(df) == 4 and set(df.game_id) == {"e1", "e2"}                                           # e1 listed under both its teams, parsed once
    assert df[df.game_id == "e1"].iloc[0][["commence_time", "home_abbr", "away_abbr"]].tolist() == ["2026-11-07T00:00:00Z", "IND", "BOS"]
    plain = parse_odds(by_team)
    assert len(plain) == 6 and "game_id" not in plain.columns                                         # the NFL's and MMA's parse is unchanged


def test_a_cutoff_drops_the_lines_posted_before_the_status_moved():
    """A player's cutoff (when his status or starter flag last moved) sets aside the rows a book posted before it; mark=True keeps
    them flagged for the desk's hover; a player without a cutoff, and a row without a posting time, are untouched."""
    tip = datetime(2026, 11, 7, 19, 0); moved = tip - timedelta(hours=2)
    rows = [_row("a one", "dk", _utc(tip - timedelta(hours=3)), game=tip),            # posted before he moved to the bench
            _row("a one", "fd", _utc(tip - timedelta(hours=1)), game=tip),            # re-priced after
            _row("a one", "mg", None, game=tip),                                       # no posting time: kept
            _row("b two", "dk", _utc(tip - timedelta(hours=3)), game=tip)]            # nothing moved for him
    df = pd.DataFrame(rows)
    kept = M.fresh_rows(df, {"a one": tip, "b two": tip}, cutoffs={"a one": moved})
    assert [(r.player, r.sportsbook) for r in kept.itertuples()] == [("a one", "fd"), ("a one", "mg"), ("b two", "dk")]
    marked = M.fresh_rows(df, {"a one": tip, "b two": tip}, cutoffs={"a one": moved}, mark=True)
    assert list(marked.stale) == [True, False, False, False] and len(marked) == 4
    assert "stale" in M.fresh_rows(df, {"a one": tip, "b two": tip}, mark=True).columns and len(M.fresh_rows(df, {"a one": tip, "b two": tip})) == 4

