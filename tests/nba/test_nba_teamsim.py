"""The team page's Simulation view (ivan.sports.nba.desk.teamsim): the batched draw and its running summary."""
import numpy as np
import pandas as pd

from ivan.sports.nba.desk import teamsim as TS
from ivan.sports.nba.gamesim import profiles as PR, rotation as RO, script as SC, world as W

from test_nba_gamesim_world import _box, _players


def _ctx():
    pr = PR.ShootingProfiles.build(_box())
    fr = RO.FoulRates(pd.DataFrame(columns=["player", "game_date", "game_id", "f_num", "m_den"]))
    params = dict(SC.DEFAULTS, pace_a=0.0, pace_b=1.0, kappa=0.0, lambda_pace=0.0, q_tie=0.0, rotation="plan")
    return W.WorldContext(params=params, teams=W.LEAGUE_DEFAULTS, groups={}, profiles=pr, prof_state=pr.latest(), foul_rates=fr, foul_state=pd.DataFrame(columns=["f_num", "m_den"]))


def _game(minutes_a0=36.0):
    home = _players("AAA", 1, True); home[0].minutes = minutes_a0
    return W.GameIn(home="AAA", away="BBB", spread_home=-5.0, total=230.0, pace_home=100.0, pace_away=100.0, players={"AAA": home, "BBB": _players("BBB", 100, False)})


def test_run_fills_the_summary_batch_by_batch(monkeypatch):
    game = _game()
    monkeypatch.setattr(TS, "game_of", lambda session, ctx, team: game)
    monkeypatch.setattr(TS, "_salaries", lambda session: {p.pid: 8000.0 for t in game.players.values() for p in t})
    seen = []
    r = TS.run(None, "AAA", n=500, batch=200, ctx=_ctx(), on_batch=lambda r: seen.append((r.n, len(r.summary["AAA"]["rows"]))))
    assert [n for n, _ in seen] == [200, 400, 500] and all(k == 10 for _, k in seen)       # three batches, every player from the first
    assert r.state == "done" and r.n == 500
    s = r.summary["AAA"]; rows = {x["name"]: x for x in s["rows"]}
    assert s["rows"][0]["target"] == 36.0 and abs(rows["aaa0"]["min_med"] - 36.0) < 3.0 and abs(rows["aaa9"]["min_med"] - 12.0) < 3.0
    assert len(rows["aaa0"]["cover"]) == 48 and abs(sum(sum(x["cover"]) for x in s["rows"]) - 240.0) < 1.0   # five on the floor every minute
    assert 0.0 <= rows["aaa0"]["value"] <= 1.0 and rows["aaa0"]["p10"] < rows["aaa0"]["p50"] < rows["aaa0"]["p90"]
    assert rows["aaa0"]["d_fpts"] is None                                                   # no run before this one
    g, o = s["game"], r.summary["BBB"]["game"]
    assert abs(g["margin"] + o["margin"]) < 1e-9 and g["margin_line"] == 5.0 and o["margin_line"] == -5.0 and abs(g["total"] - 230.0) < 8.0


def test_a_second_run_measures_from_the_first_and_the_fingerprint_follows_the_inputs(monkeypatch):
    game = _game()
    monkeypatch.setattr(TS, "game_of", lambda session, ctx, team: game)
    monkeypatch.setattr(TS, "_salaries", lambda session: {})
    first = TS.run(None, "AAA", n=250, ctx=_ctx())
    changed = _game()
    for p in changed.players["AAA"]:
        if p.name == "aaa0":
            p.rates = dict(p.rates, points=p.rates["points"] * 1.5)                         # his points rate up by half
    assert TS.fingerprint(changed) != TS.fingerprint(game) and TS.fingerprint(_game()) == TS.fingerprint(game)
    monkeypatch.setattr(TS, "game_of", lambda session, ctx, team: changed)
    again = TS.run(None, "AAA", n=250, ctx=_ctx(), state=TS.Run(home="AAA", away="BBB", target=250, fingerprint=TS.fingerprint(changed), baseline=first.summary))
    rows = {x["name"]: x for x in again.summary["AAA"]["rows"]}
    assert rows["aaa0"]["d_fpts"] is not None and rows["aaa0"]["value"] is None             # measured from the run before; no salary, no value share
    assert rows["aaa0"]["fpts"] > {x["name"]: x for x in first.summary["AAA"]["rows"]}["aaa0"]["fpts"]
