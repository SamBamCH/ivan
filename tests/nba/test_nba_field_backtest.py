"""The field backtest (ivan.sports.nba.backtest.field): real fields against the model's, the metrics and the report."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ivan.sports.nba.backtest import field as FB

IDS = list(range(101, 113))                      # twelve players, eight per lineup


def _view(pub=None):
    players = pd.DataFrame(dict(id=IDS, name=[f"P{i}" for i in IDS], team=["A", "B"] * 6, opponent=["B", "A"] * 6, position=["PG/SG"] * 12,
                                salary=[3500 + 600 * k for k in range(12)], status=[""] * 11 + ["OUT"], starter=["expected"] * 5 + ["bench"] * 7,
                                gametime=[datetime(2026, 10, 25, 18, 0)] * 12))
    players["key"] = players.name.str.lower(); players["out"] = players.status.eq("OUT")
    players["pub"] = pub if pub is not None else [40.0 - 2 * k for k in range(11)] + [0.0]
    players["priced"] = [True] * 6 + [False] * 6
    return FB.SlateView(capture_id=7, slate=555, lock=datetime(2026, 10, 25, 18, 0), players=players, blobs={"players": [], "odds": [], "game_lines": []}, n_games=1,
                        captured_at=datetime(2026, 10, 25, 17, 48), odds_rows=120)


def _derived(tmp_path, lineups, copies, final=True, contest_id=9):
    d = tmp_path / str(contest_id); d.mkdir()
    L = pd.DataFrame([dict(copies=c, points=100.0, best_rank=1, salary=40000, salary_left=10000, **{f"p{j + 1}": lu[j] for j in range(8)}) for lu, c in zip(lineups, copies)])
    L.to_parquet(d / "lineups.parquet", index=False)
    (d / "players.csv").write_text("name\n")
    n = int(sum(copies))
    (d / "field.json").write_text(json.dumps(dict(contest_id=contest_id, name="Test", draft_group_id=555, entry_fee=1.0, entries=n, entries_matched=n, unique=len(lineups),
                                                  short_entries=0 if final else n, locked_entries=0 if final else n, final=final, max_copies=int(max(copies)), capture_id=7,
                                                  capture_at="2026-10-25T17:40")))
    return d


def test_ownership_and_shape_count_copies():
    field = [tuple(IDS[:8])] * 3 + [tuple(IDS[1:9])]
    own = FB.ownership(field)
    assert own[101] == 75.0 and own[109] == 25.0 and own[102] == 100.0
    Q = {i: 10.0 for i in IDS}; sal = {i: 6000 for i in IDS}
    sh = FB.shape(field, Q, 80.0, sal)
    assert sh["n"] == 4 and sh["dup_share"] == 0.75 and sh["max_copies"] == 3 and sh["unique_share"] == 0.5
    assert sh["p50"] == 1.0 and sh["salary_left_mean"] == 2000.0 and sh["salary_full"] == 0.0
    sub = FB.shape(field * 10, Q, 80.0, sal, subsample=8, seed=1)
    assert sub["n"] == 8


def test_ownership_metrics_top8_and_mae():
    real = {i: 10.0 for i in IDS[:8]}; pred = {i: 20.0 for i in IDS[:8]}
    m = FB.ownership_metrics(real, pred, IDS)
    assert m["mae"] == pytest.approx(80 / 12) and m["top8_real"] == 80.0 and m["top8_pred"] == 160.0 and m["sum_pred"] == 160.0
    assert m["top10_overlap"] >= 8


def test_final_contests_skips_partial_folders(tmp_path):
    _derived(tmp_path, [tuple(IDS[:8])], [2], final=True, contest_id=9)
    _derived(tmp_path, [tuple(IDS[:8])], [2], final=False, contest_id=10)
    (tmp_path / "notes.txt").write_text("x")
    metas = FB.final_contests(tmp_path)
    assert [m["contest_id"] for m in metas] == [9] and metas[0]["dir"] == tmp_path / "9"
    assert FB.final_contests(tmp_path, contest_ids=[10]) == []


def test_compare_contest_rows_and_report(tmp_path):
    view = _view()
    d = _derived(tmp_path, [tuple(IDS[:8]), tuple(IDS[2:10])], [3, 1])
    meta = json.loads((d / "field.json").read_text()); meta["dir"] = d
    model_field = [tuple(IDS[:8])] * 2 + [tuple(IDS[1:9])] * 2
    preds = {"model": FB.ownership(model_field), "lock": None, "softmax": {i: 8.0 for i in IDS}}
    q_opt = sum(view.pub[i] for i in IDS[:8])
    row, players = FB.compare_contest(meta, view, preds, q_opt, field_size=4, model_shape=FB.shape(model_field, view.pub, q_opt, view.salaries))
    assert row["entries"] == 4 and row["n_players"] == 11                      # the OUT player is not in the public pool and nobody held him
    assert row["capture_age_h"] == pytest.approx(0.2) and row["odds_rows"] == 120 and row["priced_share"] == pytest.approx(6 / 11)
    by = {p["id"]: p for p in players}
    assert by[101]["own_real"] == 75.0 and by[110]["own_real"] == 25.0 and by[101]["own_model"] == 50.0 and by[101]["own_lock"] is None
    assert "model_mae" in row and "softmax_mae" in row and "lock_mae" not in row
    assert row["real_max_copies"] == 3 and row["model_max_copies"] == 2 and row["real_p50"] == pytest.approx(1.0)
    contests = pd.DataFrame([row]); pl = pd.DataFrame(players)
    text = FB.report(contests, pl, {"optimizer_share": 0.45, "fitted_at": "2026-10-24"}, 4)
    assert "Ownership per contest" in text and "Calibration of the model ownership" in text and "Largest misses" in text and "P101" in text
    assert "Inputs per slate" in text and "priced the player" in text
    odds = pd.DataFrame(dict(player=["a"] * 3 + ["b"], market=["player_points", "player_rebounds", "player_assists", "player_points"], sportsbook="x", side="Over", point=1.0, price=-110))
    assert FB.priced_players(odds) == {"a"} and FB.priced_players(odds.iloc[0:0]) == set()
    cal = FB.calibration(pl, "own_model")
    assert int(cal.n.sum()) == 11 and np.isfinite(cal.bias).all()
