"""One spelling for every NBA source (ivan.sports.nba.names): the suffix dropped, the aliases applied, the shared rules kept."""
import sqlite3

from ivan.sports.nba import names as N


def test_suffixes_and_aliases_meet_on_one_key():
    assert N.normalize_name("Jimmy Butler III") == "jimmy butler" == N.normalize_name("Jimmy Butler")
    assert N.normalize_name("Gary Payton II") == "gary payton" and N.normalize_name("Kelly Oubre Jr.") == "kelly oubre"
    assert N.normalize_name("Robert Williams III") == "robert williams" and N.normalize_name("Tim Hardaway Jr.") == "tim hardaway"
    assert N.normalize_name("Gregory Jackson II") == "gg jackson" == N.normalize_name("GG Jackson")
    assert N.normalize_name("Yang Hansen") == "hansen yang" == N.normalize_name("Hansen Yang")
    assert N.normalize_name("Nicolas Claxton") == "nic claxton" == N.normalize_name("Nic Claxton")
    assert N.normalize_name("Luka Dončić") == "luka doncic"                       # the shared rules still apply
    assert N.normalize_name("V") == "v" and N.normalize_name("") == ""             # a suffix alone is a name, never emptied
    assert N.normalize_name(N.normalize_name("Trey Murphy III")) == N.normalize_name("Trey Murphy III")   # idempotent


def test_rekey_rewrites_rows_and_leaves_conflicts(tmp_path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("normalize_names", str(__import__("pathlib").Path(__file__).parents[2] / "scripts" / "nba" / "normalize_names.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules["normalize_names"] = spec.loader.exec_module(mod) or mod
    db = tmp_path / "h.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table nba_stats (id integer primary key, game_id text, player_name text, unique(game_id, player_name))")
    con.executemany("insert into nba_stats (game_id, player_name) values (?, ?)",
                    [("g1", "jimmy butler iii"), ("g2", "jimmy butler iii"), ("g1", "kelly oubre jr"), ("g3", "gary payton ii"), ("g3", "gary payton")])
    con.commit(); con.close()
    lines = []
    mod.run_file(db, [("nba_stats", "player_name")], apply=False, log=lines.append)
    assert any("'jimmy butler iii' -> 'jimmy butler' (2 rows)" in x for x in lines)
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from nba_stats where player_name = 'jimmy butler iii'").fetchone()[0] == 2        # a dry run writes nothing
    con.close()
    lines = []
    mod.run_file(db, [("nba_stats", "player_name")], apply=True, log=lines.append)
    con = sqlite3.connect(db)
    names = sorted(r[0] for r in con.execute("select player_name from nba_stats"))
    assert names == ["gary payton", "gary payton ii", "jimmy butler", "jimmy butler", "kelly oubre"]   # the g3 pair conflicts: left, listed
    assert any("left" in x for x in lines)
    con.close()


def test_rekey_store_and_edit_files(tmp_path):
    import importlib.util, sys, json, pickle
    import pandas as pd
    spec = importlib.util.spec_from_file_location("normalize_names", str(__import__("pathlib").Path(__file__).parents[2] / "scripts" / "nba" / "normalize_names.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # a team code, free text and a stat name are never touched; a normalized name with a suffix is
    assert mod.maybe("ORL") == "ORL" and mod.maybe("fpts_mean") == "fpts_mean" and mod.maybe("Carter out tonight") == "Carter out tonight"
    assert mod.maybe("wendell carter jr") == "wendell carter" and mod.maybe("wendell carter") == "wendell carter"
    # the projection store record: the frames' index and the rotation inputs
    store = tmp_path / "projection"; store.mkdir()
    frame = pd.DataFrame({"fpts_mean": [30.0, 10.0]}, index=["wendell carter jr", "jevon carter"])
    grid = pd.DataFrame([[True] * 48, [False] * 48], index=["wendell carter jr", "jevon carter"], columns=list(range(48)))
    rec = dict(team="ORL", frame=frame, hist=frame.copy(), rot=dict(grid=grid, targets={"wendell carter jr": 31.0}, zero_minutes=["gary payton ii"], note="ORL"))
    with open(store / "ORL.pkl", "wb") as f:
        pickle.dump(rec, f)
    assert mod.run_store(store, apply=False, log=lambda m: None) == 1
    with open(store / "ORL.pkl", "rb") as f:
        assert "wendell carter jr" in pickle.load(f)["frame"].index                                 # a dry run writes nothing
    assert mod.run_store(store, apply=True, log=lambda m: None) == 1
    with open(store / "ORL.pkl", "rb") as f:
        back = pickle.load(f)
    assert list(back["frame"].index) == ["wendell carter", "jevon carter"] and list(back["rot"]["grid"].index)[0] == "wendell carter"
    assert back["rot"]["targets"] == {"wendell carter": 31.0} and back["rot"]["zero_minutes"] == ["gary payton"] and back["rot"]["note"] == "ORL"
    assert mod.run_store(store, apply=True, log=lambda m: None) == 0                                # idempotent
    # the day's edits: the pins CSV and a drawn grid
    day = tmp_path / "days" / "2026-11-05"; (day / "edits" / "rotations").mkdir(parents=True)
    pd.DataFrame({"player": ["wendell carter jr", "anthony black"], "minutes": [31.0, 30.0]}).to_csv(day / "edits" / "minutes_overrides.csv", index=False)
    (day / "edits" / "rotations" / "ORL.json").write_text(json.dumps(dict(team="ORL", players=["wendell carter jr", "anthony black"], grid=[[1] * 48, [0] * 48],
                                                                             locked=["wendell carter jr"], fingerprint=dict(statuses={"wendell carter jr": "None"}))))
    assert mod.run_edits(day, apply=True, log=lambda m: None) == 2
    assert pd.read_csv(day / "edits" / "minutes_overrides.csv")["player"].tolist() == ["wendell carter", "anthony black"]
    g = json.loads((day / "edits" / "rotations" / "ORL.json").read_text())
    assert g["players"] == ["wendell carter", "anthony black"] and g["locked"] == ["wendell carter"] and g["fingerprint"]["statuses"] == {"wendell carter": "None"} and g["team"] == "ORL"
    assert mod.run_edits(day, apply=True, log=lambda m: None) == 0
