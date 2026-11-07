"""Events, possession ends and five-man states from the play-by-play (ivan.sports.nba.gamesim.possessions)."""
import pandas as pd

from ivan.sports.nba.gamesim import possessions as X

COLS = ["actionNumber", "period", "clock", "teamId", "teamTricode", "personId", "playerName", "playerNameI", "actionType", "subType",
        "description", "scoreHome", "scoreAway", "shotValue", "isFieldGoal"]


def _pbp(rows):
    return pd.DataFrame([dict(zip(COLS, r)) for r in rows])


def _box():
    rows = [dict(game_id="g1", player_name=f"a {i}", team_abbreviation="AAA", position="G", minutes=48, points=0, rebounds=0, assists=0, steals=0, blocks=0,
                 turnovers=0, fgm=0, fga=0, fg3m=0, fg3a=0, ftm=0, fta=0) for i in ("one", "two", "three", "four", "five")]
    rows += [dict(r, player_name=f"c {i}", team_abbreviation="BBB") for r, i in zip(rows, ("one", "two", "three", "four", "five"))]
    return pd.DataFrame(rows)


def _stints():
    rows = [dict(game_id="g1", team="AAA", player_name=f"a {i}", start_sec=0.0, end_sec=2880.0, minutes=48.0) for i in ("one", "two", "three", "four", "five")]
    rows += [dict(game_id="g1", team="BBB", player_name=f"c {i}", start_sec=0.0, end_sec=2880.0, minutes=48.0) for i in ("one", "two", "three", "four", "five")]
    return pd.DataFrame(rows)


def test_possession_ends_and_counts():
    A, B = 1, 2
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", "Start of 1st Period", "0", "0", 0, 0),
        # AAA: made 3 with an assist -> possession ends
        (2, 1, "PT11M40.00S", A, "AAA", 11, "One", "A. One", "Made Shot", "Jump Shot", "One 25' 3PT Jump Shot (3 PTS) (Two 1 AST)", "3", "0", 3, 1),
        # BBB: miss, AAA defensive rebound -> BBB's possession ends
        (3, 1, "PT11M20.00S", B, "BBB", 21, "One", "C. One", "Missed Shot", "Jump Shot", "MISS One 15' Jump Shot", "", "", 2, 1),
        (4, 1, "PT11M18.00S", A, "AAA", 13, "Three", "A. Three", "Rebound", "Unknown", "Three REBOUND (Off:0 Def:1)", "", "", 0, 0),
        # AAA: made 2 and an and-one free throw at the same clock: the shot ends nothing, the made 1 of 1 ends the possession
        (5, 1, "PT11M00.00S", A, "AAA", 12, "Two", "A. Two", "Made Shot", "Layup", "Two Layup (2 PTS)", "5", "0", 2, 1),
        (6, 1, "PT11M00.00S", B, "BBB", 22, "Two", "C. Two", "Foul", "Shooting", "Two S.FOUL (P1.T1)", "", "", 0, 0),
        (7, 1, "PT11M00.00S", A, "AAA", 12, "Two", "A. Two", "Free Throw", "Free Throw 1 of 1", "Two Free Throw 1 of 1 (3 PTS)", "6", "0", 0, 0),
        # BBB: turnover with a steal -> ends
        (8, 1, "PT10M40.00S", B, "BBB", 23, "Three", "C. Three", "Turnover", "Bad Pass", "Three Bad Pass Turnover (P1.T1)", "", "", 0, 0),
        (9, 1, "PT10M40.00S", A, "AAA", 14, "Four", "A. Four", "", "", "Four STEAL (1 STL)", "", "", 0, 0),
        # AAA: missed 1 of 2 then a dead-ball team rebound (ignored), made 2 of 2 -> ends
        (10, 1, "PT10M20.00S", B, "BBB", 24, "Four", "C. Four", "Foul", "Personal", "Four P.FOUL (P1.T2)", "", "", 0, 0),
        (11, 1, "PT10M20.00S", A, "AAA", 15, "Five", "A. Five", "Free Throw", "Free Throw 1 of 2", "MISS Five Free Throw 1 of 2", "", "", 0, 0),
        (12, 1, "PT10M20.00S", 0, "", A, "", "", "Rebound", "Normal Rebound", "AAA Rebound", "", "", 0, 0),
        (13, 1, "PT10M20.00S", A, "AAA", 15, "Five", "A. Five", "Free Throw", "Free Throw 2 of 2", "Five Free Throw 2 of 2 (1 PTS)", "7", "0", 0, 0),
        # BBB: missed shot, blocked, BBB offensive rebound (no end), then a made 2 -> ends
        (14, 1, "PT10M00.00S", B, "BBB", 25, "Five", "C. Five", "Missed Shot", "Layup", "MISS Five Layup", "", "", 2, 1),
        (15, 1, "PT10M00.00S", A, "AAA", 11, "One", "A. One", "", "", "One BLOCK (1 BLK)", "", "", 0, 0),
        (16, 1, "PT09M58.00S", B, "BBB", 21, "One", "C. One", "Rebound", "Unknown", "One REBOUND (Off:1 Def:0)", "", "", 0, 0),
        (17, 1, "PT09M55.00S", B, "BBB", 21, "One", "C. One", "Made Shot", "Layup", "One Layup (2 PTS)", "7", "2", 2, 1),
        # a heave at the buzzer and the team rebound after it: neither is an event
        (18, 1, "PT00M00.50S", 0, "", A, "", "", "Heave", "Team Field Goal Attempt", "AAA Heave", "", "", 0, 0),
        (19, 1, "PT00M00.10S", 0, "", B, "", "", "Rebound", "Unknown", "BBB Rebound", "", "", 0, 0),
        (20, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", "End of 1st Period", "", "", 0, 0),
    ]
    ev = X.game_events(_pbp(rows), _box(), _stints(), "g1", game_date=pd.Timestamp("2026-01-01").date())
    ends = ev[ev.poss_end]
    assert ends.poss_team.tolist() == ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"]
    assert ev[(ev.kind == "fg") & (ev.player == "a two")].poss_end.tolist() == [False]          # the and-one shot ends nothing
    assert (ev.kind == "reb").sum() == 2 and ev[ev.kind == "reb"].reb_kind.tolist() == ["def", "off"]
    assert ev[ev.kind == "fg"].iloc[0].assist == "a two" and ev[ev.kind == "stl"].player.tolist() == ["a four"] and ev[ev.kind == "blk"].player.tolist() == ["a one"]
    assert ev.on_ok.all() and ev.iloc[0].on_off == "a five|a four|a one|a three|a two"
    assert ev[ev.kind == "fg"].margin.tolist() == [0, -3, 3, -7, -7]                             # the event team's lead before the event

    pg = X.player_games(ev).set_index("player")
    assert pg.loc["a one", ["pts", "fg3m", "blk"]].tolist() == [3, 1, 1] and pg.loc["a two", ["pts", "ast", "ftm"]].tolist() == [3, 1, 1]
    assert pg.loc["a five", ["fta", "ftm", "pts"]].tolist() == [2, 1, 1] and pg.loc["c one", ["oreb", "fga", "pts"]].tolist() == [1, 2, 2]
    assert pg.loc["a one", "off_poss"] == 3 and pg.loc["a one", "def_poss"] == 3 and pg.loc["c one", "off_poss"] == 3

    seg = X.team_segments(ev)
    s0 = seg[(seg.team == "AAA") & (seg.segment == 0)].iloc[0]
    assert (s0.poss, s0.pts, s0.fga, s0.fg3a, s0.fta, s0.tov, s0.dreb) == (3, 7, 2, 1, 3, 0, 1)
    assert set(seg.segment) == {0} and len(seg) == 2
    assert X.segment_of(2879) == 7 and X.segment_of(2880) == 8 and X.segment_of(3200) == 9 and X.segment_seconds(8) == 300


def test_reconcile_against_the_box():
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", "Start", "0", "0", 0, 0),
        (2, 1, "PT11M40.00S", 1, "AAA", 11, "One", "A. One", "Made Shot", "Jump Shot", "One 25' 3PT Jump Shot (3 PTS)", "3", "0", 3, 1),
        (3, 1, "PT11M20.00S", 2, "BBB", 21, "One", "C. One", "Turnover", "Bad Pass", "One Bad Pass Turnover (P1.T1)", "", "", 0, 0),
    ]
    box = _box()
    box.loc[box.player_name == "a one", ["points", "fgm", "fga", "fg3m"]] = [3, 1, 1, 1]
    box.loc[box.player_name == "c one", "turnovers"] = 1
    ev = X.game_events(_pbp(rows), box, _stints(), "g1")
    r = X.reconcile(ev, box)
    p = r["players"].set_index("stat")
    assert p.loc["pts", "exact"] == 1.0 and p.loc["tov", "total_gap"] == 0 and p.loc["fga", "mae"] == 0
    assert r["summary"]["games"] == 1 and r["teams"].poss.tolist() == [1, 1]
