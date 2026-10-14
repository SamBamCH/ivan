"""The replay clock, the slate archive and the contest menu."""
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import archive, clock
from ivan.core.replay import Menu, MenuItem, pick_contests


def test_clock_env_roundtrip(monkeypatch):
    monkeypatch.delenv(clock.ENV, raising=False)
    assert clock.as_of() is None and not clock.is_replay()
    with clock.frozen(datetime(2025, 11, 23, 19, 19)):
        assert clock.now() == datetime(2025, 11, 23, 19, 19) and clock.is_replay()
        with clock.frozen(datetime(2025, 11, 2, 12, 0)):
            assert clock.now().month == 11 and clock.now().day == 2
        assert clock.now().day == 23
    assert clock.as_of() is None


def test_history_cutoff_follows_clock(monkeypatch):
    from ivan.sports.nfl.actuals import before_cutoff
    df = pd.DataFrame({"season": [2024, 2025, 2025, 2025], "week": [18, 11, 12, 13], "v": [1, 2, 3, 4]})
    assert before_cutoff(df, None).v.tolist() == [1, 2, 3, 4]
    assert before_cutoff(df, (2025, 12)).v.tolist() == [1, 2]


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'a.sqlite'}", future=True)
    archive.ensure_tables(engine)
    return sessionmaker(bind=engine)()


def test_archive_roundtrip_and_capture_selection(session):
    lock = datetime(2025, 11, 23, 19, 20)
    players = [{"id": 1, "roster_slot_id": 511, "name": "a b", "team": "TB", "opponent": "LAR", "position": "QB",
                "gametime": lock, "salary": 15000, "status": None}]
    early = archive.archive_slate(session, sport="nfl", slate_key="123", lock=lock, mode="showdown", players=players,
                                  odds=[{"player": "a b", "market": "player_pass_yds", "sportsbook": "dk", "side": "Over", "point": 250.5, "price": -110}],
                                  captured_at=lock - timedelta(hours=5))
    late = archive.archive_slate(session, sport="nfl", slate_key="123", lock=lock, mode="showdown", players=players,
                                 captured_at=lock - timedelta(minutes=20))
    after = archive.archive_slate(session, sport="nfl", slate_key="123", lock=lock, mode="showdown", players=players,
                                  captured_at=lock + timedelta(hours=1))
    other = archive.archive_slate(session, sport="nba", slate_key="9", lock=lock, mode="classic", players=players)
    picked = archive.slates(session, "nfl")
    assert [c.id for c in picked] == [late.id]                      # last capture before lock, nfl only
    blobs = archive.load_blobs(session, early.id)
    assert blobs["players"][0]["gametime"] == lock.isoformat() and blobs["odds"][0]["point"] == 250.5
    assert archive.load_blobs(session, late.id)["odds"] == []
    # a backfill is used only when no ingest capture exists
    bf = archive.archive_slate(session, sport="nfl", slate_key="123", lock=lock, mode="showdown", players=players, source="backfill",
                               captured_at=datetime(2026, 10, 9))
    assert [c.id for c in archive.slates(session, "nfl")] == [late.id]
    assert [c.id for c in archive.slates(session, "nfl", prefer="backfill")] == [bf.id]
    assert archive.delete_captures(session, "nfl", "123", source="backfill") == 1
    assert len(archive.captures(session, "nfl")) == 3
    assert archive.slates(session, "nfl", since=lock + timedelta(days=1)) == []


def test_menu_picks_largest_match_and_caps_entries(tmp_path):
    contests = [
        {"contest_id": 1, "name": "NFL Showdown $100K mini-MAX [150 Entry Max] (A @ B)", "entry_fee": 0.5, "total_entries": 200000, "multi_entry": 150, "has_payouts": True},
        {"contest_id": 2, "name": "NFL Showdown $5K mini-MAX [150 Entry Max] (A @ B)", "entry_fee": 0.5, "total_entries": 10000, "multi_entry": 150, "has_payouts": True},
        {"contest_id": 3, "name": "NFL Showdown $70K First Down [20 Entry Max] (A @ B)", "entry_fee": 1.0, "total_entries": 80000, "multi_entry": 20, "has_payouts": True},
        {"contest_id": 4, "name": "NFL Showdown $1.5M Thursday Night Showdown [$500K to 1st] (A @ B)", "entry_fee": 20.0, "total_entries": 90000, "multi_entry": 150, "has_payouts": True},
        {"contest_id": 5, "name": "NFL Showdown $80K Huddle [Single Entry] (A @ B)", "entry_fee": 5.0, "total_entries": 15000, "multi_entry": 1, "has_payouts": False},
    ]
    menu = Menu(items=[MenuItem("mini-MAX", 50), MenuItem("First Down", 40), MenuItem("Night Showdown", 2), MenuItem("Huddle", 1),
                       MenuItem("Play-Action", 10)], max_fee=5)
    picks = pick_contests(contests, menu, "showdown")
    assert [(p["contest_id"], p["entries"]) for p in picks] == [(1, 50), (3, 20)]   # fee cap drops 4, no payouts drops 5
    p = tmp_path / "m.json"
    p.write_text('{"contests": [{"match": "huddle", "entries": 3, "mode": "classic"}], "weights": {"cash_pct": 1}}')
    m = Menu.load(p)
    assert m.weights == {"cash_pct": 1} and pick_contests(contests, m, "showdown") == []
    contests[4]["has_payouts"] = True
    assert [(x["contest_id"], x["entries"]) for x in pick_contests(contests, m, "classic")] == [(5, 1)]


# --------------------------------------------------------------------------- seeds and repeats
def test_seed_for_defaults_when_unset_and_derives_when_set(monkeypatch):
    from ivan.core import seed
    monkeypatch.delenv(seed.ENV, raising=False)
    assert seed.seed_for("field/sample", 42) == 42 and seed.seed_for("sim/1") is None
    monkeypatch.setenv(seed.ENV, "7")
    a, b = seed.seed_for("field/sample", 42), seed.seed_for("sim/1")
    assert a != 42 and b is not None and a != b and seed.seed_for("field/sample", 42) == a
    monkeypatch.setenv(seed.ENV, "8")
    assert seed.seed_for("field/sample", 42) != a


class FakeAdapter:
    """A pipeline that writes what the real stages would leave behind and records the seed each stage ran with."""
    sport = "fake"
    working_tables = ()
    repeat_tables = ()

    def __init__(self):
        self.calls = []

    def restore(self, session, blobs, log):
        return {"players": len(blobs.get("players", []))}

    def project(self, session, log):
        import os
        self.calls.append(("project", os.environ.get("IVAN_SEED")))

    def generate(self, env, n_lineups, jobs, log_path, log):
        self.calls.append(("generate", env.get("IVAN_SEED")))

    def contests_for(self, session, lock_local, mode, slate_key=None):
        return [{"contest_id": 1, "name": "Fake Showdown mini-MAX", "entry_fee": 1.0, "total_entries": 100, "multi_entry": 20, "has_payouts": True}]

    def entries_csv(self, picks, path):
        path.write_text("Entry ID,Contest Name,Contest ID,Entry Fee\n")
        return sum(p["entries"] for p in picks)

    def simulate(self, env, entries_csv, n_sims, field_cap, log_path, log):
        self.calls.append(("simulate", env.get("IVAN_SEED")))

    def choose(self, env, entries_csv, out_dir, weights, log_path, log):
        self.calls.append(("choose", env.get("IVAN_SEED")))
        stamp = f"20251123-1919{len(self.calls):02d}"
        (out_dir / f"chosen_entries.lineups.{stamp}.csv").write_text("entry_id,contest_id\n")


def test_run_replay_repeats_and_resume(tmp_path):
    from ivan.core import replay as rp
    live = tmp_path / "live.sqlite"
    engine = create_engine(f"sqlite:///{live}", future=True)
    archive.ensure_tables(engine)
    s = sessionmaker(bind=engine)()
    lock = datetime(2025, 11, 23, 19, 20)
    archive.archive_slate(s, sport="fake", slate_key="5", lock=lock, mode="showdown",
                          players=[{"id": 1, "roster_slot_id": 511, "name": "a", "team": "T", "opponent": "O", "position": "QB",
                                    "gametime": lock, "salary": 1, "status": None}])
    s.close()
    menu = Menu(items=[MenuItem("mini-MAX", 5)])
    out = tmp_path / "out"
    ad = FakeAdapter()
    opts = rp.ReplayOptions(repeats=2, seed=10, skip_done=True)
    ms = rp.run_replay(ad, live, tmp_path / "replay.sqlite", out, menu, opts=opts, log=lambda *_: None)
    assert [m["status"] for m in ms] == ["ok"]
    slate_dir = out / "fake" / "20251123-1920_showdown_5"
    assert sorted(p.name for p in slate_dir.iterdir() if p.is_dir()) == ["rep_000", "rep_001"]
    assert [r["seed"] for r in ms[0]["repeats"]] == [10, 11]
    assert ("project", "10") in ad.calls and ("generate", "10") in ad.calls and ("choose", "11") in ad.calls
    assert rp.repeat_dirs([slate_dir]) == {0: [slate_dir / "rep_000"], 1: [slate_dir / "rep_001"]}

    # asking for more repeats only adds the missing ones; asking for the same count skips the slate
    ad.calls.clear()
    ms = rp.run_replay(ad, live, tmp_path / "replay.sqlite", out, menu, opts=rp.ReplayOptions(repeats=3, seed=10), log=lambda *_: None)
    assert [c for c in ad.calls if c[0] == "choose"] == [("choose", "12")]
    assert [r["k"] for r in ms[0]["repeats"]] == [0, 1, 2] and (slate_dir / "rep_002").is_dir()
    ad.calls.clear()
    rp.run_replay(ad, live, tmp_path / "replay.sqlite", out, menu, opts=rp.ReplayOptions(repeats=3, seed=10), log=lambda *_: None)
    assert ad.calls == []

    # a single, unseeded replay keeps the flat layout and reads back as repeat 0
    out2 = tmp_path / "out2"
    ms = rp.run_replay(ad, live, tmp_path / "replay.sqlite", out2, menu, opts=rp.ReplayOptions(), log=lambda *_: None)
    d2 = out2 / "fake" / "20251123-1920_showdown_5"
    assert ms[0]["status"] == "ok" and "choose" in ms[0]["stages"] and list(d2.glob("chosen_entries.lineups.*.csv"))
    assert ("generate", None) in ad.calls and rp.repeat_dirs([d2]) == {0: [d2]}
