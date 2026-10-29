"""notes: one slate's notes, cleared by a blank, and the ones that stood at lock."""
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import notes as N
from ivan.sports.nba.models import NBAPlayerNote


def _session():
    eng = create_engine("sqlite://")
    NBAPlayerNote.__table__.create(eng)
    return sessionmaker(bind=eng)()


def test_notes_belong_to_their_slate_until_cleared():
    s = _session(); t0 = datetime(2026, 10, 29, 12, 0)
    N.set_note(s, 1, "2026-10-29", "Ausar Thompson", "ankle, limited?", team="DET", raw="Ausar Thompson", now=t0)
    assert set(N.active(s, 1)) == {"ausar thompson"} and N.active(s, 2) == {}          # another slate: nothing
    N.set_note(s, 1, "2026-10-29", "ausar thompson", "starting per beat", now=t0 + timedelta(minutes=5))
    assert N.active(s, 1)["ausar thompson"].text == "starting per beat" and len(N.at(s, 1, None)) == 1   # replaced, not added
    lock = t0 + timedelta(hours=7)
    N.set_note(s, 1, "2026-10-29", "isaiah joe", "", now=t0)                             # a blank on no note: nothing
    N.set_note(s, 1, "2026-10-29", "isaiah joe", "out of the rotation", now=t0)
    N.set_note(s, 1, "2026-10-29", "isaiah joe", "", now=t0 + timedelta(hours=1))      # cleared before lock
    N.set_note(s, 1, "2026-10-29", "jalen duren", "late note", now=lock + timedelta(minutes=1))
    assert set(N.active(s, 1)) == {"ausar thompson", "jalen duren"}
    assert [r.player_name for r in N.at(s, 1, lock)] == ["ausar thompson"]               # what stood at lock
    assert len(N.at(s, 1, None)) == 3                                                    # cleared rows kept
