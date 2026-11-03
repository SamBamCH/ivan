"""A consumer after a feed pull (ingest.after_pull) on one slate's own directory: only the slate's edited teams are re-projected,
edits on teams not on the slate do not blow up (2026-10-27: the server's Refresh said 'feed pull failed: No objects to concatenate'
with Sunday's pins on Tuesday's slate), and nothing of another slate is touched (every slate has its own directory now)."""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import ingest
from ivan.sports.nba import slates as S
from ivan.sports.nba.models import NBABase, NBAPlayer


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SLATES_DIR", tmp_path / "slates")
    tok = S.use(2)
    sp = S.paths(2).ensure()
    yield dict(overrides=sp.overrides_csv, rotations=sp.rotations, published=sp.published)
    S.release(tok)


def seed(session, draft_group: int, teams=("BOS", "NYK")):
    gt = datetime(2026, 10, 27, 18, 30)
    for i, name in enumerate(["jayson tatum", "jalen brunson", "derrick white", "karl anthony towns"], start=1):
        session.add(NBAPlayer(id=i, name=name, team=teams[i % 2], opponent=teams[(i + 1) % 2], position="PG", roster_slot_id=1, gametime=gt,
                              salary=5000, fpts_mean=30.0, fpts_std=8.0, minutes_proj=30.0, draft_group_id=draft_group))
    session.commit()


def test_only_the_edited_teams_are_reprojected(session, paths, monkeypatch):
    seed(session, 2)
    paths["overrides"].write_text("player,minutes\njayson tatum,36\n")        # this machine's pin on one team's player (seeded as NYK)
    calls = []
    monkeypatch.setattr(ingest, "project_slate", lambda s, **kw: calls.append(kw) or ingest.ProjectInfo())
    r = ingest.after_pull(session, log=lambda m: None)
    assert r["moved"] is False and r["draft_group"] == 2
    assert r["teams"] == {"NYK"} and calls and calls[0]["teams"] == {"NYK"}        # the other team stands as published
    assert paths["overrides"].exists()                                               # nothing filed: the slate keeps its edits


def test_edits_on_teams_not_on_the_slate_are_not_an_error(session, paths, monkeypatch):
    seed(session, 2)
    paths["rotations"].mkdir(parents=True, exist_ok=True); (paths["rotations"] / "LAL.json").write_text("{}")   # a grid for a team that is not playing
    calls = []
    monkeypatch.setattr(ingest, "project_slate", lambda s, **kw: calls.append(kw) or ingest.ProjectInfo())
    r = ingest.after_pull(session, log=lambda m: None)
    assert r["teams"] == set() and calls == []                                       # nothing to re-project: as published
