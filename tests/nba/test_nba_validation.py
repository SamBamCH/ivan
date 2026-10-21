"""The projection scorecard (ivan.sports.nba.validation): the join, the error decomposition, the classes and the findings."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import validation as V
from ivan.sports.nba.models import NBABase, NBAProjectionCapture, NBAStat


def cap(name, team, minutes, fpts, fpm, starter="bench", hist_team=None, hist_games=10, pinned=False, priced=False, market=None, salary=5000):
    return dict(player_name=name, team=team, minutes_proj=minutes, fpts_mean=fpts, fpm=fpm, starter=starter, hist_team=hist_team or team,
                hist_games=hist_games, pinned=pinned, priced=priced, market_fpts=market, salary=salary, baseline_min=minutes, hist_min=minutes,
                gametime=datetime(2026, 10, 20, 19, 0))


def test_join_decomposes_every_miss():
    ours = pd.DataFrame([cap("a", "BOS", 30.0, 30.0, 1.0, "expected", priced=True, market=31.0),
                         cap("b", "BOS", 20.0, 16.0, 0.8),                     # did not play
                         cap("c", "BOS", 4.0, 3.0, 0.75, hist_team="LAL"),      # surprise 20 minutes on a new team
                         cap("d", "BOS", 12.0, 9.0, 0.75, hist_games=0)])
    act = pd.DataFrame([dict(player_name="a", team_act="BOS", min_act=36.0, fpts_act=54.0, pos_act="G"),
                        dict(player_name="b", team_act="BOS", min_act=0.0, fpts_act=0.0, pos_act="BENCH"),
                        dict(player_name="c", team_act="BOS", min_act=20.0, fpts_act=20.0, pos_act="BENCH")])
    df = V.join_actuals(ours, act, pd.DataFrame([dict(player_name="a", dk_avg_fpts=40.0)])).set_index("player_name")
    a = df.loc["a"]
    assert a.fpts_err == pytest.approx(-24.0) and a.err_minutes == pytest.approx(-6.0) and a.err_rate == pytest.approx(36 * (1.0 - 1.5))
    assert a.err_minutes + a.err_rate + a.err_other == pytest.approx(a.fpts_err)
    assert df.loc["b"].miss == "dnp" and df.loc["b"].err_rate == 0 and df.loc["b"].err_minutes == pytest.approx(16.0)
    assert df.loc["c"].miss == "surprise" and df.loc["c"].history_class == "new team"
    assert df.loc["d"].history_class == "no history" and not df.loc["d"].box_row and df.loc["d"].min_act == 0
    assert df.salary_tier.unique().tolist() == ["4-6k"]
    f = {x["text"].split(":")[0] for x in V.findings(df.reset_index())}
    assert any(t.startswith("projected to play but did not") for t in f) and any(t.startswith("played a rotation role") for t in f)
    y = V.yardsticks(df.reset_index())
    assert y["vs_dk_avg"]["n"] == 1 and y["vs_market"]["market_mae"] == pytest.approx(23.0)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_scorecard_end_to_end_and_pending(session, tmp_path):
    lock = datetime(2026, 10, 20, 19, 0)
    for t, n in ((datetime(2026, 10, 20, 12, 0), "early"), (datetime(2026, 10, 20, 18, 45), "last"), (datetime(2026, 10, 20, 19, 5), "late")):
        session.add(NBAProjectionCapture(captured_at=t, draft_group_id=1, lock=lock, source="ours", player_name="a", team="BOS", opponent="NYK",
                                         gametime=lock, salary=7000, starter="expected", minutes_proj=30.0 if n != "last" else 32.0, fpts_mean=30.0,
                                         fpm=1.0, hist_team="BOS", hist_games=20, priced=False, pinned=False))
        session.add(NBAProjectionCapture(captured_at=t, draft_group_id=1, lock=lock, source="dk_avg", player_name="a", team="BOS", fpts_mean=33.0))
    session.commit()
    assert V.pending_draft_groups(session, tmp_path) == []                      # no box scores yet
    card = V.build_scorecard(session, 1)
    assert card is not None and card.rows.empty and "no box scores" in V.format_scorecard(card)
    session.add(NBAStat(game_id="0022600001", game_date=date(2026, 10, 20), player_name="a", team_abbreviation="BOS", position="G", minutes=34.0, dk_fpts=40.0))
    session.commit()
    assert V.pending_draft_groups(session, tmp_path) == [1]
    cards = V.run_pending_scorecards(session, tmp_path, log=lambda m: None)
    assert len(cards) == 1 and cards[0].summary["captured_at"].startswith("2026-10-20T18:45")      # the last capture before lock, not the one after
    assert cards[0].rows.iloc[0].minutes_proj == 32.0 and cards[0].summary["minutes_before_lock"] == 15.0
    assert V.pending_draft_groups(session, tmp_path) == [] and (tmp_path / "2026-10-20_1.csv").exists()
    h = V.scorecard_history(tmp_path)
    assert len(h) == 1 and h.iloc[0].dk_mae == pytest.approx(7.0) and h.iloc[0].fpts_mae == pytest.approx(10.0)
