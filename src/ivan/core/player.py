# src/ivan/core/player.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_mixin
from datetime import datetime

@declarative_mixin
class BasePlayer:
    """Shared player fields + behavior for ALL sports."""
    id          = Column(Integer,   primary_key=True)
    name        = Column(String,    nullable=True)
    team        = Column(String,    nullable=False)
    salary      = Column(Integer,   nullable=False)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}"
            f" id={self.id!r}"
            f" name={self.name!r}"
            f" team={self.team!r}>"
        )

    # if you still want abstract hooks, you can keep them—
    # but *do not* subclass ABC here.  Instead, just document:
    def enrich(self) -> None:
        """
        (Optional) post‐load calculation, e.g. computing
        fantasy points, runs, etc.
        """
        pass
