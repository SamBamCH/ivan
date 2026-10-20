"""End to end on the sample slate under tests/mma/data: ingest -> generate -> sim -> choose -> export."""

import json

import numpy as np
import pandas as pd
import pytest

from ivan.sports.mma.choose import choose_lineups, export, lineup_rows
from ivan.sports.mma.ingest import NameMatcher, load_fights
from ivan.sports.mma.market import match_events_to_fights
from ivan.sports.mma.lineup_generation import generate_candidates
from ivan.sports.mma.models import Lineup, LineupSimulationResult, MMAFight, MMAPlayer
from ivan.sports.mma.payouts import parse_payout_csv
from ivan.sports.mma.simulate_contest import rank_and_pay, resolve_contest, simulate_contest

from pathlib import Path

DATA = Path(__file__).parent / "data"


def test_ingest_builds_fights_and_projections(session, cfg, ingest):
    s = ingest(session, cfg)
    assert not s["templates_written"]
    assert s["players"] == 24 and s["fights"] == 12
    assert s["fights_with_odds"] == 12 and s["players_with_odds"] == 24
    assert s["unmatched"] == []
    assert len(s["five_round"]) == 1
    p = session.query(MMAPlayer).filter(MMAPlayer.name == "Yazmin Jauregui").one()
    assert p.win_odds == -1050 and p.win_prob == pytest.approx(1050 / 1150)
    assert p.projection > 0 and p.fight_id and p.opponent == "Demopoulos"
    f = session.get(MMAFight, p.fight_id)
    assert f.rounds in (3, 5) and f.u60_odds == 1800 and {f.fighter1_id, f.fighter2_id} >= {p.id}
    # the alias in config resolves the one odds name that differs from the DK name
    tina = session.query(MMAPlayer).filter(MMAPlayer.name == "Tina Black").one()
    assert tina.win_odds is not None
    assert session.query(MMAFight).filter(MMAFight.rounds == 5).count() == 1
    assert len(load_fights(session)) == 12


def test_name_matcher_orders():
    rows = [{"id": 1, "name": "Alateng Heili"}, {"id": 2, "name": "Raul Rosas Jr."}, {"id": 3, "name": "Ailin Perez"}]
    m = NameMatcher(rows, {"Someone Else": "Ailin Perez"})
    assert m.find("Heili Alatengheili")["id"] == 1        # shared token
    assert m.find("Raul Rosas")["id"] == 2                 # exact after dropping punctuation
    assert m.find("Pérez")["id"] == 3                      # accent-insensitive last name
    assert m.find("Someone Else")["id"] == 3               # alias
    assert m.find("Nobody Here") is None
    m = NameMatcher([{"id": 1, "name": "Melissa Croden"}, {"id": 2, "name": "Kai Kamaka III"}, {"id": 3, "name": "Allen Frye Jr"}])
    assert m.find("Melissa Gatto") is None                 # a shared first name is not a match (an earlier card's fighter)
    assert m.find("Kai Kamaka")["id"] == 2 and m.find("Allen Frye")["id"] == 3 and m.find("Frye Jr.")["id"] == 3


def test_one_lookalike_does_not_make_an_old_file_describe_this_card(session, cfg, ingest, tmp_path):
    fo, po = tmp_path / "fight_odds.csv", tmp_path / "player_odds.csv"
    s0 = ingest(session, cfg, fight_odds_csv=fo, player_odds_csv=po)
    real = pd.read_csv(po)["player"].iloc[0]
    rows = [f"{real},100,200,-150,50,1"] + [f"Fighter Nobody{i},,,-200,," for i in range(6)]       # one real name among six strangers
    po.write_text("player,1st round win,ITD,Win,sig strikes,takedowns\n" + "\n".join(rows) + "\n")
    fo.write_text("fight,u60 sec,1st round,itd\n" + "\n".join(f"Fighter Nobody{i} vs Fighter Nobody{i + 1},800,140,-400" for i in range(0, 6, 2)) + "\n")
    s = ingest(session, cfg, fight_odds_csv=fo, player_odds_csv=po)
    assert s["templates_written"] and s["players_in_csv"] == 0
    assert pd.read_csv(po)["ITD"].isna().all()               # the stale line did not leak into the fresh template


def test_rank_and_pay_pools_ties_and_conserves_the_pool():
    pay = np.array([10.0, 5.0, 3.0, 0.0, 0.0])
    scores = np.array([[50.0, 50.0, 40.0, 10.0, 10.0]])
    payouts, rois, ranks = rank_and_pay(scores, pay, entry_fee=1.0)
    assert payouts[0].tolist() == pytest.approx([7.5, 7.5, 3.0, 0.0, 0.0])
    assert ranks[0].tolist() == [1, 1, 3, 4, 4]
    assert payouts[0].sum() == pytest.approx(pay.sum())
    assert rois[0, 2] == pytest.approx(2.0)


def test_payout_csv_parses_ranges():
    df = parse_payout_csv(DATA / "payouts.csv")
    assert list(df.columns) == ["Start Rank", "End Rank", "Payout"]
    assert df.iloc[0].tolist() == [1, 1, 10000.0]
    assert (df["End Rank"] >= df["Start Rank"]).all()


def test_pipeline_end_to_end(session, cfg, ingest, tmp_path, monkeypatch):
    monkeypatch.setenv("IVAN_SEED", "7")
    ingest(session, cfg)
    r = generate_candidates(session, cfg, n_candidates=40, n_jobs=1)
    assert r["solved"] == 40 and r["new_unique"] >= 10
    pool = session.query(Lineup).all()
    fights = {f.id: (f.fighter1_id, f.fighter2_id) for f in session.query(MMAFight).all()}
    sal = {p.id: p.salary for p in session.query(MMAPlayer).all()}
    for lu in pool:
        ids = lu.player_ids_list
        assert len(ids) == 6 and 48500 <= sum(sal[i] for i in ids) <= 50000
        for a, b in fights.values():
            assert not (a in ids and b in ids)

    contest = resolve_contest(session, cfg, total_entries=47500, entry_fee=0.5, contest_name="test")
    rep = simulate_contest(session, cfg, contest, field_size=300, n_sims=60)
    assert rep["field"]["field_size"] == 300 and rep["pool_max_abs_error"] < 1e-6
    res = session.query(LineupSimulationResult).filter_by(contest_name="test").all()
    assert len(res) == rep["field"]["unique"]
    assert sum(x.dupes for x in res) == 300
    assert all(len(x.fpts) == 60 and len(x.rank) == 60 for x in res)
    assert all(0 <= x.cash_pct <= 1 for x in res)

    from ivan.sports.mma.choose import LAST_BOOK, choose_lineups, format_book
    greedy = choose_lineups(session, "test", cfg, num_lineups=5)
    port = choose_lineups(session, "test", cfg, num_lineups=5, method="portfolio", bankroll=1.5)
    assert len(greedy) == 5 and len(port) == 5 and len({r.lineup_id for r in port}) == 5
    assert LAST_BOOK["test"]["method"] == "portfolio" and LAST_BOOK["test"]["chosen"]["entries"] == 5 and "bankroll 1.5x" in format_book("test")

    picks = choose_lineups(session, "test", cfg, num_lineups=5)
    assert len(picks) == 5 and len({p.lineup_id for p in picks}) == 5
    paths = export(session, picks, out_dir=tmp_path, stamp="t")
    detail = pd.read_csv(paths["lineups"])
    assert list(detail.columns) == ["F1", "F2", "F3", "F4", "F5", "F6", "Avg Payout", "Avg ROI", "Cash %", "Salary Remaining", "Dupe Count", "Pred Dupes"]
    upload = pd.read_csv(paths["upload"])
    assert upload.shape == (5, 6)
    assert lineup_rows(session, picks).shape[0] == 5

    # a second sim of the same contest replaces the first
    simulate_contest(session, cfg, contest, field_size=100, n_sims=10)
    assert sum(x.dupes for x in session.query(LineupSimulationResult).filter_by(contest_name="test").all()) == 100


def test_new_card_writes_templates_with_api_win_odds(session, cfg, ingest, tmp_path):
    fo, po = tmp_path / "fight_odds.csv", tmp_path / "player_odds.csv"
    po.write_text("player,1st round win,ITD,Win,sig strikes,takedowns\nSomebody Else,,,-200,,\n")   # another card
    s = ingest(session, cfg, fight_odds_csv=fo, player_odds_csv=po)
    assert s["templates_written"]
    assert (tmp_path / "player_odds.prev.csv").exists()
    fights = pd.read_csv(fo)
    players = pd.read_csv(po)
    assert list(fights.columns) == ["fight", "u60 sec", "1st round", "itd"] and len(fights) == 12
    assert list(players.columns) == ["player", "1st round win", "ITD", "Win", "sig strikes", "takedowns"] and len(players) == 24
    assert players["Win"].isna().all()                       # no API key in the test
    assert s["players_with_odds"] == 0 and len(s["missing"]["player"]) == 24
    # a second run now reads the templates as (empty) overrides and leaves them alone
    s2 = ingest(session, cfg, fight_odds_csv=fo, player_odds_csv=po)
    assert not s2["templates_written"]


def test_moneyline_events_match_fights_across_transliterations():
    fights = [{"id": "pereira-zhelezniakova", "fighters": [{"id": 1, "name": "alice pereira"}, {"id": 2, "name": "daria zhelezniakova"}]},
              {"id": "godinez-souza", "fighters": [{"id": 3, "name": "loopy godinez"}, {"id": 4, "name": "ketlen souza"}]},
              {"id": "a-b", "fighters": [{"id": 5, "name": "nobody one"}, {"id": 6, "name": "nobody two"}]}]
    events = [{"home": "Darya Zheleznyakova", "away": "Alice Pereira", "prices": {"Alice Pereira": -135, "Darya Zheleznyakova": 114}},
              {"home": "Lupita Godinez", "away": "Ketlen Souza", "prices": {"Ketlen Souza": 195, "Lupita Godinez": -238}},
              {"home": "Erin Blanchfield", "away": "Jasmine Jasudavicius", "prices": {}}]
    got = match_events_to_fights(events, fights)
    assert got == {1: -135, 2: 114, 3: -238, 4: 195}


def test_fitted_dupe_weights_follow_the_model(tmp_path):
    from ivan.sports.mma.sample_field import dupe_weights
    model = {"coef": {"intercept": -19.0, "b_left_per_100": -0.0854, "b_zero_left": 0.56, "b_win": 3.07, "b_same_fight": -1.7}, "left_cap": 2000.0}
    path = tmp_path / "dupe_model.json"
    path.write_text(json.dumps(model))
    salary = {1: 9000, 2: 9000, 3: 8000, 4: 8000, 5: 8000, 6: 8000, 7: 7500}
    win = {i: 0.6 for i in salary}
    full = [1, 2, 3, 4, 5, 6]                 # 50000 exactly
    light = [1, 2, 3, 4, 5, 7]                # 49500 left 500
    cfg = {"model": "fitted", "dupe_model_path": str(path)}
    w = dupe_weights([full, light], salary, win, cfg)
    assert w[0] / w[1] == pytest.approx(np.exp(0.56 + 0.0854 * 5), rel=1e-6)
    win2 = {**win, 7: 0.9}                    # a stronger favourite in the light lineup
    w2 = dupe_weights([full, light], salary, win2, cfg)
    assert w2[1] / w[1] == pytest.approx(np.exp(3.07 * 0.3), rel=1e-6)
    fights = {1: "a", 6: "a", 2: "b", 3: "c", 4: "d", 5: "e", 7: "f"}    # 1 and 6 share a fight: only the full lineup doubles up
    w3 = dupe_weights([full, light], salary, win, cfg, fights=fights)
    assert (w3[0] / w3[1]) / (w[0] / w[1]) == pytest.approx(np.exp(-1.7), rel=1e-6)
    # a version-2 model adds the rank among the candidates, the main-event count and Gamma lumpiness (seeded)
    v2 = {"coef": {**model["coef"], "b_log_rank": -0.2, "b_main_event": 1.4}, "left_cap": 2000.0, "alpha": 2.0}
    path.write_text(json.dumps(v2))
    w4 = dupe_weights([full, light], salary, win, cfg, main_event={7})
    assert (w4[1] / w4[0]) / (w[1] / w[0]) == pytest.approx(np.exp(1.4 - 0.2 * np.log(2)), rel=1e-6)   # light holds the main-event fighter and ranks 2nd
    r1 = dupe_weights([full, light], salary, win, cfg, rng=np.random.default_rng(5)); r2 = dupe_weights([full, light], salary, win, cfg, rng=np.random.default_rng(5))
    assert np.allclose(r1, r2) and not np.allclose(r1 / r1.max(), w4 / w4.max())
    with pytest.raises(FileNotFoundError):
        dupe_weights([full], salary, win, {"model": "fitted", "dupe_model_path": str(tmp_path / "nope.json")})
    assert dupe_weights([full, light], salary, win, {"model": "original"})[0] > dupe_weights([full, light], salary, win, {"model": "original"})[1]
