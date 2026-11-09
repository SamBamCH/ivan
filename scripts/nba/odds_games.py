#!/usr/bin/env python
"""
A slate's props by game: which of its games the odds table prices, how many players and books each game has, how fresh the
lines are, and the rows of the slate's players the projection ignores (another game's, or a row without a game outside the
fallback window). The terminal view of what projection.market.fresh_rows keeps.

    venv/bin/python scripts/nba/odds_games.py [--slate DG]              # the table
    venv/bin/python scripts/nba/odds_games.py --player "pascal siakam"  # one player's rows: his game, kept or ignored, and why
    venv/bin/python scripts/nba/odds_games.py --backfill [--apply]      # rows stored without a game get it from the raw payloads

Why this exists: a props fetch replaces only the games it priced, so the odds table holds other games and other days
(a back-to-back's two games side by side; a player the books did not price tonight keeps his last priced game). A row
knows its game (game_id, commence_time, the two teams, from the odds API event) since 2026-11-07 and a slate reads only
its own games' rows; a row stored before then has no game and counts only when the book posted it within
market.FALLBACK_HOURS before the player's tip. --backfill finds those rows in raw/odds/props (every event payload as
received) and writes their game, so the live table carries no fallback rows at all. The archive captures keep the rows as
they were stored.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from ivan.core import raw
from ivan.sports.common.odds import parse_odds
from ivan.sports.nba import slates as S
from ivan.sports.nba.models import NBAOdds, NBAPlayer
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.projection.market import FALLBACK_HOURS, GAME_TOLERANCE_HOURS, fresh_rows, load_odds
from ivan.sports.nba.utils import parse_utc


def _utc1(ts) -> pd.Timestamp:
    return parse_utc(pd.Series([ts])).iloc[0]


def _local(ts) -> str:
    t = _utc1(ts)
    return "—" if pd.isna(t) else t.tz_convert(datetime.now().astimezone().tzinfo).strftime("%a %m-%d %H:%M")


def pool_frame(session) -> pd.DataFrame:
    return pd.DataFrame([dict(name=p.name, team=p.team, opponent=p.opponent, gametime=p.gametime) for p in session.query(NBAPlayer).all()],
                        columns=["name", "team", "opponent", "gametime"])


def _why(row, tip) -> str:
    """Why fresh_rows keeps or drops one row of a player whose tip is `tip` (naive local)."""
    game = _utc1(row.commence_time)
    posted = _utc1(row.last_update)
    tz = datetime.now().astimezone().tzinfo
    if pd.notna(game):
        delta = abs((game.tz_convert(tz).tz_localize(None) - tip).total_seconds()) / 3600
        return "kept: his game" if delta <= GAME_TOLERANCE_HOURS else f"other game ({_local(row.commence_time)})"
    if pd.isna(posted):
        return "no game, no update time"
    hours = (tip - posted.tz_convert(tz).tz_localize(None)).total_seconds() / 3600
    if hours < -1:
        return f"no game, posted {-hours:.0f} h after the tip"
    return f"kept: no game, posted {hours:.0f} h before" if hours <= FALLBACK_HOURS else f"no game, posted {hours:.0f} h before (window {FALLBACK_HOURS:g})"


def table(session) -> int:
    pool = pool_frame(session)
    if pool.empty:
        print("the slate has no pool"); return 1
    odds = load_odds(session)
    tips = dict(zip(pool.name, pool.gametime))
    kept = fresh_rows(odds, tips)
    team_of = dict(zip(pool.name, pool.team))
    games = pool.dropna(subset=["gametime"]).assign(key=lambda d: [frozenset((a, b)) for a, b in zip(d.team, d.opponent)]) \
                .drop_duplicates("key").sort_values("gametime")
    print(f"{'game':<12} {'tip':<16} {'rows':>7} {'players':>8} {'books':>6}  {'linked':>6}  freshest")
    for g in games.itertuples():
        mine = kept[kept.player.map(team_of).isin(g.key)]
        linked = f"{int(mine.commence_time.notna().mean() * 100)}%" if len(mine) else "—"
        print(f"{g.opponent + '@' + g.team if g.team else '?':<12} {pd.Timestamp(g.gametime):%a %m-%d %H:%M}  {len(mine):>7} {mine.player.nunique():>8} "
              f"{mine.sportsbook.nunique():>6}  {linked:>6}  {_local(mine.last_update.max()) if len(mine) else '—'}")
    others = odds[odds.player.isin(set(pool.name)) & ~odds.index.isin(kept.index)]
    if len(others):
        print(f"\nignored rows of this slate's players: {len(others)} ({others.player.nunique()} players)")
        why = others.apply(lambda r: _why(r, pd.Timestamp(tips[r.player])), axis=1)
        for reason, n in why.value_counts().items():
            print(f"  {n:>7}  {reason}")
    rest = odds[~odds.player.isin(set(pool.name))]
    print(f"\nrows of players not on this slate: {len(rest)} ({rest.player.nunique()} players, {rest.game_id.nunique()} games)")
    return 0


def player(session, name: str) -> int:
    key = normalize_name(name)
    pool = pool_frame(session)
    me = pool[pool.name == key]
    if me.empty:
        print(f"{key}: not on this slate"); return 1
    tip = pd.Timestamp(me.iloc[0].gametime)
    odds = load_odds(session)
    mine = odds[odds.player == key]
    print(f"{key} ({me.iloc[0].team} v {me.iloc[0].opponent}, tip {tip:%a %m-%d %H:%M}): {len(mine)} rows")
    if mine.empty:
        return 0
    mine = mine.assign(why=mine.apply(lambda r: _why(r, tip), axis=1))
    g = mine.groupby(["game_id", "commence_time", "why"], dropna=False).agg(rows=("player", "size"), books=("sportsbook", "nunique"),
                                                                           first=("last_update", "min"), last=("last_update", "max")).reset_index()
    for r in g.sort_values("last").itertuples():
        gid = "no game" if pd.isna(r.game_id) else str(r.game_id)[:8]
        print(f"  {gid:<9} tip {_local(r.commence_time):<16} {r.rows:>5} rows {r.books:>2} books  posted {_local(r.first)} .. {_local(r.last)}   {r.why}")
    return 0


def backfill(session, apply: bool) -> int:
    """Give rows stored without a game their game from the raw event payloads (odds/props, one per event per fetch), matched on
    (player, market, book, last_update): a book updates a market once per second at most, and a player is in one game."""
    null = pd.read_sql("select id, player, market, sportsbook, last_update from odds where game_id is null", session.get_bind())
    print(f"{len(null)} rows without a game" + ("" if len(null) else ": nothing to do"))
    if null.empty:
        return 0
    key_time = lambda s: parse_utc(s).dt.strftime("%Y-%m-%dT%H:%M:%SZ")      # a restored capture stored "2026-10-25 22:54:18", the fetch the ISO string
    null["last_update"] = key_time(null.last_update)
    oldest = parse_utc(null.last_update).min()
    since = None if pd.isna(oldest) else (oldest - pd.Timedelta(days=2)).strftime("%Y%m%d-%H%M%S")
    arts = raw.find("nba", "odds/props", since=since)
    print(f"reading {len(arts)} raw event payloads since {since or 'the beginning'}")
    found: dict[tuple, tuple] = {}
    for a in arts:
        try:
            ev = raw.load(a.path, "nba")
        except Exception as e:   # noqa: BLE001
            print(f"  skip {a.path}: {type(e).__name__}: {e}"); continue
        if not isinstance(ev, dict) or not ev.get("bookmakers"):
            continue
        df = parse_odds({a.key: ev}, games=True)
        if df.empty:
            continue
        df["last_update"] = key_time(df.last_update)
        df = pd.concat([df, df.assign(player=df.player.map(normalize_name))])      # rows stored before names.py kept the suffix ("michael porter jr")
        for r in df.drop_duplicates(["player", "market", "sportsbook", "last_update"]).itertuples():
            found[(r.player, r.market, r.sportsbook, r.last_update)] = (r.game_id, r.commence_time, r.home_abbr, r.away_abbr)
    hits = [(r.id, found.get((r.player, r.market, r.sportsbook, r.last_update))) for r in null.itertuples()]
    updates = [dict(id=int(i), game_id=g[0], commence_time=g[1], home_abbr=g[2], away_abbr=g[3]) for i, g in hits if g is not None]
    print(f"{len(updates)} of {len(null)} rows found in the payloads" + ("" if apply else " (dry run; --apply writes)"))
    if apply and updates:
        for i in range(0, len(updates), 5000):
            session.bulk_update_mappings(NBAOdds, updates[i:i + 5000])
        session.commit()
        left = session.execute(text("select count(*) from odds where game_id is null")).scalar()
        print(f"written; {left} rows still without a game")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--player", help="one player's rows, kept or ignored, and why")
    ap.add_argument("--backfill", action="store_true", help="rows stored without a game get it from the raw payloads")
    ap.add_argument("--apply", action="store_true", help="with --backfill: write (dry run otherwise)")
    a = ap.parse_args()
    try:
        _, session = S.open_any(a.slate)
    except S.NoSlate as e:
        print(e); return 1
    if a.backfill:
        return backfill(session, a.apply)
    if a.player:
        return player(session, a.player)
    return table(session)


if __name__ == "__main__":
    sys.exit(main())
