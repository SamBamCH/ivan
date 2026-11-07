"""Mid-game exits and the heir rules (ivan.sports.nba.gamesim.fouls)."""
import pandas as pd

from ivan.sports.nba.gamesim import fouls as F


def _team_stints():
    """Twelve games of one team; in the last one the starting big 'b1' leaves for good at minute 30 of a close game, the backup big 'b2' takes his minutes."""
    rows = []
    for g in range(12):
        gid = f"g{g:02d}"; date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=g)
        last = g == 11
        # starters play 0-18 and 24-42 (36 min); the bench 18-24 and 42-48 in regulation; the game runs to 48
        for nm in ("b1", "g1", "g2", "w1", "w2"):
            if last and nm == "b1":
                rows += [(gid, date, nm, 0.0, 1080.0, 0), (gid, date, nm, 1440.0, 1800.0, 0)]          # out for good at minute 30
            else:
                rows += [(gid, date, nm, 0.0, 1080.0, 0), (gid, date, nm, 1440.0, 2520.0, 0)]
        for nm in ("b2", "g3", "w3", "w4", "w5"):
            if last and nm == "b2":
                rows += [(gid, date, nm, 1080.0, 1440.0, 0), (gid, date, nm, 1800.0, 2880.0, 1)]       # the backup big plays the rest
            else:
                rows += [(gid, date, nm, 1080.0, 1440.0, 0), (gid, date, nm, 2520.0, 2880.0, 1)]
        if last:
            rows += [("g11", date, "b3", 2520.0, 2880.0, 1)]                                           # the third big only closes
    df = pd.DataFrame(rows, columns=["game_id", "game_date", "player_name", "start_sec", "end_sec", "to_buzzer"])
    df["team"] = "AAA"; df["margin_out"] = 3; df["minutes"] = (df.end_sec - df.start_sec) / 60.0; df["starter"] = 0
    df["game_date"] = df.game_date.dt.date
    return df


def test_exit_found_and_heirs_measured():
    st = _team_stints()
    groups = {"b1": "big", "b2": "big", "b3": "big", "g1": "guard", "g2": "guard", "g3": "guard", "w1": "wing", "w2": "wing", "w3": "wing", "w4": "wing", "w5": "wing"}
    fouls = pd.DataFrame([dict(game_id="g11", team="AAA", player_name="b1", elapsed=1800.0, foul_no=6)])
    ex, heirs = F.exits(st, fouls, pd.DataFrame(), groups)
    assert len(ex) == 1
    e = ex.iloc[0]
    assert e.player == "b1" and e.minute == 30 and e.kind == "foul_out" and abs(e.owed - 12.0) < 1e-6      # 30-42 he usually plays
    h = heirs.set_index("heir")
    assert abs(h.loc["b2", "gain"] - 12.0) < 1e-6 and abs(h.loc["g1", "gain"]) < 1e-6                     # b2 took the twelve minutes
    assert not h.loc["b2", "on_floor"] and h.loc["g1", "on_floor"]                                         # b2 came in at the exit second: he was off the floor
    # the flow rule sends the big's share of the owed minutes to the bigs with room; b2 has 12 minutes of room, b3 none (not in the curve: not a mate)
    assert h.loc["b2", "pred_flow"] > h.loc["g1", "pred_flow"] and h.loc["b2", "pred_group"] > 0 and h.loc["w1", "pred_group"] == 0
    assert h.pred_none.eq(0).all()
    sc = F.score(ex, heirs).set_index(["cut", "rule"])
    assert sc.loc[("all", "flow"), "top_hit"] == 1.0 and sc.loc[("kind=foul_out", "group"), "exits"] == 1
    fl = F.group_flow(ex, heirs)
    row = fl[(fl.out_group == "big") & (fl.kind == "all") & (fl.heir_group == "big")].iloc[0]
    assert abs(row.share_of_owed - 1.0) < 1e-6 and abs(row.backup_share - 1.0) < 1e-6
    shares = F.midgame_shares(ex, heirs)
    assert shares.iloc[0].heir == "b2" and abs(shares.iloc[0].share - 1.0) < 1e-6


def test_blowout_and_garbage_exits_are_not_exits():
    st = _team_stints()
    st.loc[(st.game_id == "g11") & (st.player_name == "b1"), "margin_out"] = 25                         # pulled in a blowout at minute 30
    ex, _ = F.exits(st, pd.DataFrame(), pd.DataFrame(), {})
    assert ex.empty
    assert F.blowout_margin(10) == 30 and F.blowout_margin(30) == 18 and F.blowout_margin(47) == 12


def test_minute_vector_and_curve():
    s = pd.DataFrame([dict(start_sec=30.0, end_sec=150.0)])
    v = F.minute_vector(s)
    assert abs(v[0] - 0.5) < 1e-9 and v[1] == 1.0 and abs(v[2] - 0.5) < 1e-9 and v[3] == 0
    hist = F.TeamHistory(_team_stints())
    c = hist.curve("g11")
    assert set(c) >= {"b1", "b2"} and abs(c["b1"][:18].sum() - 18.0) < 1e-6 and abs(c["b2"][18:24].sum() - 6.0) < 1e-6 and hist.curve("g00") == {}
