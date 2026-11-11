"""A refresh replaces the pool; the ownership column (the last field build's exposure) carries over by player id (ivan.sports.nba.ingest)."""
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba import ingest as I
from ivan.sports.nba.models import NBABase, NBAPlayer


def _p(**kw):
    """A pool row with the required columns filled."""
    base = dict(opponent="X", roster_slot_id=1, gametime=datetime(2026, 11, 10, 19, 0), salary=3000)
    return NBAPlayer(**{**base, **kw})


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_ownership_carries_over_the_pool_replace(tmp_path):
    s = _session(tmp_path)
    s.add_all([_p(id=1, name="a one", team="UTA", position="PG", salary=5000, ownership=12.5), _p(id=2, name="b two", team="UTA", position="SG", salary=4000, ownership=0.0),
               _p(id=3, name="c three", team="BOS", position="C", salary=3000, ownership=40.0)])
    s.commit()
    before = I.ownership_by_id(s)
    assert before == {1: 12.5, 2: 0.0, 3: 40.0}
    s.query(NBAPlayer).delete(synchronize_session=False)                                  # what store_slate does
    s.add_all([_p(id=1, name="a one", team="UTA", position="PG", salary=5100), _p(id=3, name="c three", team="BOS", position="C", salary=3000, status="OUT"),
               _p(id=4, name="d four", team="BOS", position="PF", salary=3500)])                # 2 left the pool, 4 is new
    s.commit()
    assert I.carry_ownership(s, before) == 2
    own = {p.id: p.ownership for p in s.query(NBAPlayer).all()}
    assert own[1] == 12.5 and own[3] == 40.0 and own[4] in (None, 0.0)                     # a new player: the field never held him
    assert I.carry_ownership(s, {}) == 0                                                   # no pool before: nothing to carry
