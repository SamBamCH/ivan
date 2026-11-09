"""tips: the record at a game's tip written once to the slate's directory and read back."""
from datetime import datetime

import pandas as pd

from ivan.sports.nba import slates as SL
from ivan.sports.nba.backtest import tips as T


def test_write_and_read_a_tip(tmp_path, monkeypatch):
    root = tmp_path / "slates"
    p = SL.paths(555, root); p.ensure()
    (p.root / "slate.sqlite").write_bytes(b"")
    (p.root / "slate.json").write_text('{"draft_group": 555, "day": "2026-11-04"}')
    monkeypatch.setattr(SL, "holds", lambda dg, root_=None: True)
    monkeypatch.setattr(SL, "paths", lambda dg=None, root_=None: p)
    p.overrides_csv.write_text("player,minutes\nann one,30\n")
    (p.rotations / "BOS.json").write_text('{"players": [], "grid": []}')
    tip = datetime(2026, 11, 4, 18, 0)
    players = [dict(id=1, name="ann one", team="BOS", gametime="2026-11-04T18:00:00", salary=9000, fpts_mean=40.0, rotation_cells=[1] * 48),
               dict(id=2, name="bob two", team="DEN", gametime="2026-11-04T20:00:00", salary=5000, fpts_mean=20.0)]
    odds = pd.DataFrame([dict(player="ann one", market="player_points", sportsbook="dk", side="Over", point=24.5, price=-110, last_update="2026-11-04T17:00:00")])
    lines = pd.DataFrame([dict(game_id="g1", home_abbr="BOS", away_abbr="NYK", market="spreads", point=-6.5)])
    d = T.write_tip(555, tip, datetime(2026, 11, 4, 18, 1), players, odds, lines, session=None, root=root)
    assert d is not None and T.tips_of(555, root) == [tip]
    assert T.write_tip(555, tip, datetime(2026, 11, 4, 18, 5), players, odds, lines, session=None, root=root) is None    # written once
    got = T.read_tip(555, tip, root)
    assert got["meta"]["teams_tipping"] == ["BOS"] and got["meta"]["players"] == 2 and got["meta"]["odds_rows"] == 1 and got["meta"]["edits"]["overrides"]
    assert got["players"][0]["rotation_cells"] == [1] * 48 and len(got["odds"]) == 1 and got["game_lines"][0]["home_abbr"] == "BOS"
    assert (got["edits"] / "minutes_overrides.csv").read_text().startswith("player") and (got["edits"] / "rotations" / "BOS.json").exists()
    assert T.tip_for(555, datetime(2026, 11, 4, 19, 0), root) == tip and T.tip_for(555, datetime(2026, 11, 4, 17, 0), root) is None
