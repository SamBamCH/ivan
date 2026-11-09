"""The rotation grid (ivan.sports.nba.projection.rotation): proposal invariants, the stint sampler's conservation and garbage rule, drawn grids."""
import numpy as np
import pandas as pd

from ivan.sports.nba.projection import rotation as R


def fake_curve(names, minutes):
    """A curve frame where each player is on for the first `minutes` of the game (crude but valid)."""
    rows = {}
    for n, m in zip(names, minutes):
        row = {k: (1.0 if k < m else 0.0) for k in range(R.MINUTES)}
        row.update(games=10, minutes=float(m), stints=2.0, played_share=1.0, last_game="2026-01-01")
        rows[n] = row
    return pd.DataFrame(rows).T


def test_scale_targets_takes_surplus_from_the_bottom_and_caps():
    t = R.scale_targets(np.array([38, 36, 34, 30, 28, 24, 20, 14, 10, 8, 6, 4.0]))
    assert abs(t.sum() - 240) < 0.1 and t[0] >= 37 and t[-1] < 4.0 and t.max() <= R.MAX_MINUTES
    t = R.scale_targets(np.array([46, 40, 30, 30, 30, 30, 30.0]))
    assert t.max() <= R.MAX_MINUTES and abs(t.sum() - 240) < 0.2            # two capped at 40, the rest share the remainder


def test_propose_keeps_five_on_the_floor_and_respects_starters_and_ceiling():
    names = [f"p{i}" for i in range(12)]
    curve = fake_curve(names, [40, 38, 36, 34, 32, 26, 22, 18, 14, 10, 6, 4])
    grid = R.propose(curve, names, starters=names[5:10])
    assert set(R.column_counts(grid)) == {5} and grid.shape == (12, 48)
    assert all(grid.loc[n, 0] for n in names[5:10])                           # the named starters open the game
    assert grid.sum(axis=1).max() <= R.MAX_MINUTES
    assert max(b - a for n in names for a, b in R.typical_stints(grid.loc[n].values.astype(float))) <= R.MAX_STINT + 2


def test_sample_worlds_conserves_minutes_and_pulls_closers_in_blowouts():
    names = [f"p{i}" for i in range(10)]
    curve = fake_curve(names, [36, 34, 32, 30, 28, 24, 20, 16, 12, 8])
    grid = R.propose(curve, names, starters=names[:5])
    rng = np.random.default_rng(0)
    d = R.sample_worlds(grid, {n: 1.5 for n in names}, 3000, rng, spread=0.0, curve=curve)
    M = d["minutes"]
    assert M.shape == (3000, 10) and np.all(M >= 0)
    no_ot = d["overtime"] == 0
    assert np.allclose(M[no_ot].sum(axis=1), 240.0, atol=1e-6)                 # every regulation world adds to 240
    base = grid.sum(axis=1).values
    close = (d["garbage_at"] < 0) & no_ot
    assert np.abs(M[close].mean(axis=0) - base).max() < 2.5                     # close games sit on the grid (jitter clipping drifts a little)
    blown = d["garbage_at"] >= 0
    assert blown.mean() > 0.05
    assert M[blown][:, :5].sum(axis=1).mean() < M[close][:, :5].sum(axis=1).mean() - 3   # closers lose minutes in blowouts
    far = R.sample_worlds(grid, {n: 1.5 for n in names}, 3000, rng, spread=-15.0, curve=curve)
    assert (far["garbage_at"] >= 0).mean() > blown.mean()                        # a big favourite blows out more often
    out = R.sample_worlds(grid, {n: 1.5 for n in names}, 500, rng, spread=0.0, p_out={names[0]: 1.0}, curve=curve)
    assert np.all(out["minutes"][:, 0] == 0) and np.allclose(out["minutes"][out["overtime"] == 0].sum(axis=1), 240.0, atol=1e-6)


def test_drawn_grid_round_trip(tmp_path, monkeypatch):
    from ivan.sports.nba import slates as S
    monkeypatch.setattr(S, "SLATES_DIR", tmp_path); tok = S.use(1)
    monkeypatch.setattr(S, "_current", S._current)                # released below
    try:
        _drawn_grid_round_trip()
    finally:
        S.release(tok)


def _drawn_grid_round_trip():
    names = ["a b", "c d"]
    grid = pd.DataFrame([[True] * 24 + [False] * 24, [False] * 24 + [True] * 24], index=names, columns=list(range(48)))
    R.save_drawn("XYZ", grid)
    back = R.load_drawn("XYZ")
    assert back.equals(grid) and R.drawn_minutes("XYZ") == {"a b": 24.0, "c d": 24.0} and R.stints_of(back)["c d"] == [(24, 48)]
    assert R.clear_drawn("XYZ") and R.load_drawn("XYZ") is None


def test_spots_follow_the_substitution_chain():
    names = ["pg", "sg", "sf", "pf", "c", "pg2", "c2"]
    grid = pd.DataFrame(False, index=names, columns=list(range(48)))
    for n in ["pg", "sg", "sf", "pf", "c"]:
        grid.loc[n, list(range(0, 10))] = True
    grid.loc["pg", list(range(16, 48))] = True; grid.loc["pg2", list(range(10, 16))] = True     # pg2 takes the point guard spot, pg comes back
    grid.loc["c", list(range(20, 48))] = True; grid.loc["c2", list(range(10, 20))] = True
    for n in ["sg", "sf", "pf"]:
        grid.loc[n, list(range(10, 48))] = True
    groups = {"pg": "guard", "pg2": "guard", "sg": "guard", "sf": "wing", "pf": "wing", "c": "big", "c2": "big"}
    pos = {"pg": "PG", "sg": "SG", "sf": "SF", "pf": "PF", "c": "C", "pg2": "PG", "c2": "C"}
    S, owners = R.spots_of(grid, starters=["c", "pf", "sf", "sg", "pg"], groups=groups, positions=pos)
    assert owners == ["pg", "sg", "sf", "pf", "c"]
    assert S.loc["pg", 0] == 0 and S.loc["c", 0] == 4
    assert S.loc["pg2", 12] == 0 and S.loc["c2", 12] == 4                        # two swaps in minute 10: the guard takes the guard's spot, the big the big's
    assert S.loc["pg", 20] == 0 and S.loc["c", 30] == 4 and S.loc["pg2", 20] == -1


def test_recorded_pairs_and_habits_drive_the_spot_chain():
    names = ["pg", "sg", "sf", "pf", "c", "b1", "b2"]
    grid = pd.DataFrame(False, index=names, columns=list(range(48)))
    for n in ["pg", "sg", "sf", "pf", "c"]:
        grid.loc[n, list(range(0, 10))] = True
    # at minute 10 two swaps at once: b1 and b2 come on for pg and sf; both bench players are wings by position
    grid.loc["b1", list(range(10, 48))] = True; grid.loc["b2", list(range(10, 48))] = True
    for n in ["sg", "pf", "c"]:
        grid.loc[n, list(range(10, 48))] = True
    groups = {"pg": "guard", "sg": "guard", "sf": "wing", "pf": "wing", "c": "big", "b1": "wing", "b2": "wing"}
    pos = {"pg": "PG", "sg": "SG", "sf": "SF", "pf": "PF", "c": "C", "b1": "SF", "b2": "SF"}
    # position alone cannot tell b1 from b2; the recorded pairs say b2 came in for the point guard
    pairs = [dict(player_out="pg", player_in="b2", elapsed=595.0), dict(player_out="sf", player_in="b1", elapsed=595.0)]
    S, _ = R.spots_of(grid, starters=["pg", "sg", "sf", "pf", "c"], groups=groups, positions=pos, pairs=pairs)
    assert S.loc["b2", 12] == 0 and S.loc["b1", 12] == 2
    # no pairs, but the coach's habit: b2 usually comes in for the point guard
    S2, _ = R.spots_of(grid, starters=["pg", "sg", "sf", "pf", "c"], groups=groups, positions=pos, shares={"pg": {"b2": 0.8, "b1": 0.2}, "sf": {"b1": 0.9}})
    assert S2.loc["b2", 12] == 0 and S2.loc["b1", 12] == 2


def test_dropdown_facts_final_score_and_new_starters(tmp_path):
    """game_scores reads each side's points from the box; starter_changes counts the five who had not started the game before,
    looking one game past the list for the oldest row and giving None where a five is unknown."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from ivan.sports.nba.models import NBABase, NBAStat
    from ivan.sports.nba.data_fetch.pbp import NBAStint
    from datetime import date
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    fives = {"g1": ["a", "b", "c", "d", "e"], "g2": ["a", "b", "c", "d", "f"], "g3": ["a", "b", "g", "h", "f"], "g4": ["a", "b", "g", "h", "f"]}
    days = {"g1": date(2026, 10, 22), "g2": date(2026, 10, 24), "g3": date(2026, 10, 26), "g4": date(2026, 10, 28)}
    for gid, five in fives.items():
        for n in five + ["x"]:
            s.add(NBAStint(game_id=gid, game_date=days[gid], team="BOS", player_name=n, stint_no=1, start_sec=0, end_sec=600, minutes=10, starter=int(n in five)))
        s.add(NBAStint(game_id=gid, game_date=days[gid], team="NYK", player_name="opp", stint_no=1, start_sec=0, end_sec=600, minutes=10, starter=1))
        for t, pts in (("BOS", 50), ("NYK", 40 if gid != "g3" else 60)):
            for k in range(2):
                s.add(NBAStat(game_id=gid, game_date=days[gid], player_name=f"{t}{k}", team_abbreviation=t, minutes=20, points=pts))
    s.commit()
    games = R.team_games(s, "BOS", since=date(2026, 10, 23))          # g4, g3, g2 newest first; g1 is the one before the list
    assert [g["game_id"] for g in games] == ["g4", "g3", "g2"] and games[0]["opp"] == "NYK"
    sc = R.game_scores(s, [g["game_id"] for g in games])
    assert sc["g4"] == {"BOS": 100, "NYK": 80} and sc["g3"] == {"BOS": 100, "NYK": 120}
    assert R.starter_changes(s, "BOS", games) == {"g4": 0, "g3": 2, "g2": 1}
    assert R.starter_changes(s, "BOS", R.team_games(s, "BOS", since=date(2026, 10, 1)))["g1"] is None     # nothing before the opener
    assert R.starter_changes(s, "BOS", []) == {}
