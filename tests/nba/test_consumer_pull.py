"""A consumer after a feed pull (ingest.after_pull): its own edits are filed when the published slate moves, another slate's
lineups are cleared, only the slate's edited teams are re-projected, and edits on teams not on the slate do not blow up
(2026-10-27: the server's Refresh said 'feed pull failed: No objects to concatenate' with Sunday's pins on Tuesday's slate)."""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import ingest
from ivan.sports.nba.models import Lineup, NBABase, NBAPlayer
from ivan.sports.nba.projection import live, rotation


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    inp = tmp_path / "input"; inp.mkdir()
    ov = inp / "minutes_overrides.csv"; rot = inp / "rotations"
    monkeypatch.setattr(ingest, "INPUT_DIR", inp); monkeypatch.setattr(ingest, "MINUTES_OVERRIDES_CSV", ov)
    monkeypatch.setattr(live, "MINUTES_OVERRIDES_CSV", ov); monkeypatch.setattr(live, "PUBLISHED_OVERRIDES_CSV", inp / "published" / "minutes_overrides.csv")
    monkeypatch.setattr(rotation, "ROTATIONS_DIR", rot); monkeypatch.setattr(rotation, "PUBLISHED_ROTATIONS_DIR", inp / "published" / "rotations")
    return dict(input=inp, overrides=ov, rotations=rot)


def seed(session, draft_group: int, teams=("BOS", "NYK")):
    gt = datetime(2026, 10, 27, 18, 30)
    for i, name in enumerate(["jayson tatum", "jalen brunson", "derrick white", "karl anthony towns"], start=1):
        session.add(NBAPlayer(id=i, name=name, team=teams[i % 2], opponent=teams[(i + 1) % 2], position="PG", roster_slot_id=1, gametime=gt,
                              salary=5000, fpts_mean=30.0, fpts_std=8.0, minutes_proj=30.0, draft_group_id=draft_group))
    session.commit()


def test_slate_move_files_the_old_edits_and_clears_the_other_slates_lineups(session, paths, monkeypatch):
    seed(session, 2)                                                     # the pull brought slate 2; this machine had slate 1
    paths["overrides"].write_text("player,minutes\njayson tatum,36\n")   # Sunday's pin on a name that plays again today
    paths["rotations"].mkdir(); (paths["rotations"] / "BOS.json").write_text("{}")
    session.add(Lineup(id=1, draft_group_id=1, player_ids_list=[1, 2, 3, 4])); session.add(Lineup(id=2, draft_group_id=2, player_ids_list=[1, 2, 3, 4])); session.commit()
    calls = []
    monkeypatch.setattr(ingest, "project_slate", lambda s, **kw: calls.append(kw) or ingest.ProjectInfo())
    r = ingest.after_pull(session, 1, log=lambda m: None)
    assert r["moved"] is True and r["draft_group"] == 2
    assert not paths["overrides"].exists() and not list(paths["rotations"].glob("*.json"))          # filed away, not applied to slate 2
    assert (paths["input"] / "edits" / "1" / "minutes_overrides.csv").exists() and (paths["input"] / "edits" / "1" / "BOS.json").exists()
    assert [lu.id for lu in session.query(Lineup).all()] == [2]                                      # the other slate's lineups are gone
    assert r["teams"] == set() and calls == []                                                       # nothing of its own left: as published


def test_edits_on_teams_not_on_the_slate_are_not_an_error(session, paths, monkeypatch):
    seed(session, 2)
    paths["rotations"].mkdir(); (paths["rotations"] / "LAL.json").write_text("{}")                   # a grid for a team that is not playing
    calls = []
    monkeypatch.setattr(ingest, "project_slate", lambda s, **kw: calls.append(kw) or ingest.ProjectInfo())
    r = ingest.after_pull(session, 2, log=lambda m: None)                                             # same slate: nothing filed
    assert r["moved"] is False and r["teams"] == set() and calls == []
    assert (paths["rotations"] / "LAL.json").exists()


def test_only_the_slates_edited_teams_are_reprojected(session, paths, monkeypatch):
    seed(session, 2)
    paths["overrides"].write_text("player,minutes\njalen brunson,38\n")                              # seeded on BOS, on the slate
    paths["rotations"].mkdir(); (paths["rotations"] / "LAL.json").write_text("{}")
    calls = []
    monkeypatch.setattr(ingest, "project_slate", lambda s, **kw: calls.append(kw) or ingest.ProjectInfo(players=2))
    r = ingest.after_pull(session, 2, log=lambda m: None)
    assert r["teams"] == {"BOS"} and r["players"] == 2 and calls[0]["teams"] == {"BOS"} and calls[0]["write_capture"] is False


def test_project_slate_with_no_matching_team_returns_quietly(session, paths, monkeypatch):
    seed(session, 2)

    class Slate:                                                          # the pool without the history a real LiveSlate loads
        def __init__(self, s, **kw):
            self.players = s.query(NBAPlayer).all()

        def teams(self):
            return ["BOS", "NYK"]

        def project(self, *a, **kw):
            raise AssertionError("no team should be projected")

    monkeypatch.setattr(ingest, "LiveSlate", Slate)
    info = ingest.project_slate(session, n_sims=10, teams={"LAL"}, log=lambda m: None)
    assert info.players == 0


def test_consumer_publish_restores_the_bundle_then_projects_only_its_edited_teams(session, paths, monkeypatch, tmp_path):
    seed(session, 2)
    paths["overrides"].write_text("player,minutes\njalen brunson,38\n")                              # BOS, on the slate
    gz = tmp_path / "live.sql.gz"; gz.write_bytes(b"x")
    monkeypatch.setattr(ingest, "PULLED_LIVE_BUNDLE", gz)
    applied, calls = [], []
    from ivan.core import feed
    monkeypatch.setattr(feed, "apply", lambda sport, path, **kw: applied.append((sport, path)) or [])
    monkeypatch.setattr(ingest, "project_slate", lambda s, **kw: calls.append(kw) or ingest.ProjectInfo(players=2))
    r = ingest.consumer_publish(session, log=lambda m: None)
    assert applied == [("nba", gz)] and r["restored"] is True and r["teams"] == {"BOS"} and calls[0]["teams"] == {"BOS"} and r["players"] == 2
    paths["overrides"].unlink()                                                                       # the edit cleared: nothing projected, the bundle stands
    r = ingest.consumer_publish(session, log=lambda m: None)
    assert len(applied) == 2 and r["teams"] == set() and len(calls) == 1
