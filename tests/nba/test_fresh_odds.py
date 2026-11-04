"""Only a prop posted for the player's game on this slate counts (projection.market.fresh_rows)."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from ivan.sports.nba.projection import market as M


def _utc(local: datetime) -> str:
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_fresh_rows_keeps_tonights_props_and_drops_a_past_games():
    tip = datetime(2026, 11, 4, 19, 0)
    rows = [dict(player="obi toppin", market="player_points", sportsbook="dk", side="Over", point=14.5, price=-110, last_update=_utc(datetime(2026, 11, 2, 17, 39))),   # Nov 2's game
            dict(player="pascal siakam", market="player_points", sportsbook="dk", side="Over", point=22.5, price=-115, last_update=_utc(datetime(2026, 11, 4, 17, 28))),   # tonight
            dict(player="pascal siakam", market="player_points", sportsbook="fd", side="Over", point=22.5, price=-112, last_update=_utc(tip - timedelta(hours=35))),       # the night before: posted for tonight
            dict(player="someone gone", market="player_points", sportsbook="dk", side="Over", point=9.5, price=-110, last_update=_utc(datetime(2026, 11, 4, 17, 28))),      # not in the pool
            dict(player="pascal siakam", market="player_points", sportsbook="mg", side="Over", point=23.5, price=-110, last_update=None)]                                    # no time: unknown game
    df = pd.DataFrame(rows)
    kept = M.fresh_rows(df, {"obi toppin": tip, "pascal siakam": tip})
    assert list(kept.player) == ["pascal siakam", "pascal siakam"] and set(kept.sportsbook) == {"dk", "fd"}
    assert M.fresh_rows(df, {}).empty and M.fresh_rows(df.iloc[0:0], {"obi toppin": tip}).empty
    assert len(M.fresh_rows(df, {"obi toppin": tip, "pascal siakam": tip}, hours=72)) == 3                        # a wider window takes Nov 2's row back
