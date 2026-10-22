"""Stint reconstruction from the play-by-play (ivan.sports.nba.pbp) and the segment grid."""
import numpy as np
import pandas as pd

from ivan.sports.nba import pbp as P


def _pbp(rows):
    cols = ["actionNumber", "period", "clock", "teamTricode", "playerName", "actionType", "subType", "description", "scoreHome", "scoreAway"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_clock_and_segments():
    assert P.elapsed_at(1, "PT12M00.00S") == 0 and P.elapsed_at(2, "PT06M00.00S") == 1080 and P.elapsed_at(5, "PT05M00.00S") == 2880
    assert P.game_length(4) == 2880 and P.game_length(5) == 3180
    seg = P.segment_minutes(0, 720)
    assert seg[0] == 6 and seg[1] == 6 and seg[2] == 0 and P.segment_minutes(2700, 3000).sum() == 5
    assert P.garbage_threshold(60) == 12 and P.garbage_threshold(180) == 15 and P.garbage_threshold(360) == 20 and P.garbage_threshold(1000) == 99


def test_stints_follow_substitutions_and_scale_to_the_box():
    # home AAA: starters a1..a5, bench b1; away BBB: starters c1..c5, bench d1
    rows = [
        (1, 1, "PT12M00.00S", "", "", "period", "start", "", "0", "0"),
        (2, 1, "PT11M00.00S", "AAA", "A One", "Made Shot", "Jump Shot", "", "2", "0"),
        (3, 1, "PT06M00.00S", "AAA", "A One", "Substitution", "", "SUB: B One FOR A One", "", ""),
        (4, 1, "PT05M00.00S", "BBB", "C One", "Made Shot", "Jump Shot", "", "2", "3"),
        (5, 1, "PT00M00.00S", "", "", "period", "end", "", "", ""),
        (6, 2, "PT12M00.00S", "", "", "period", "start", "", "", ""),
        (7, 2, "PT11M00.00S", "AAA", "A One", "Rebound", "", "", "", ""),            # back at the period break, no SUB event
        (8, 2, "PT00M00.00S", "", "", "period", "end", "", "", ""),
        (9, 3, "PT12M00.00S", "", "", "period", "start", "", "", ""),
        (10, 4, "PT00M00.00S", "", "", "game", "end", "", "30", "20"),
    ]
    box = pd.DataFrame([dict(player_name=f"a {i}", team_abbreviation="AAA", minutes=m, position="G" if k < 5 else "BENCH", game_date="2026-01-01")
                        for k, (i, m) in enumerate(zip(("one", "two", "three", "four", "five", "six"), (42, 48, 48, 48, 48, 6)))]
                       + [dict(player_name=f"c {i}", team_abbreviation="BBB", minutes=48, position="G", game_date="2026-01-01") for i in ("one", "two", "three", "four", "five")])
    box.loc[box.player_name == "a six", "player_name"] = "b one"
    df = P.stints_from_pbp(_pbp(rows), box, "g1")
    a1 = df[df.player_name == "a one"].sort_values("stint_no")
    assert len(a1) == 2 and a1.start_sec.tolist() == [0, 720] and a1.end_sec.tolist() == [360, 2880]
    assert a1.minutes.sum() == 42 and a1.to_buzzer.tolist() == [0, 1]
    b1 = df[df.player_name == "b one"]
    assert len(b1) == 1 and b1.start_sec.iloc[0] == 360 and b1.end_sec.iloc[0] == 720 and b1.minutes.iloc[0] == 6
    assert df[df.player_name == "a one"].margin_out.tolist() == [2, 10]        # home margin +2 at his exit, +10 at the buzzer
    assert df[df.player_name == "c one"].margin_out.tolist() == [-10]           # away perspective
    assert df.groupby("player_name").minutes.sum().round(6).eq(box.set_index("player_name").minutes).all()
