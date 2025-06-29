# src/ivan/core/player.py
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

class BasePlayer(ABC):
    id: int
    name: str
    team: str
    gametime: datetime
    salary: int
    
    #TODO: add actual abstractmethods below. 
    # @abstractmethod
    # def load_from_db(self, session) -> None:
    #     """Populate this object’s fields from whatever ORM row(s) exist."""
    #     ...

    # @abstractmethod
    # def save_to_db(self, session) -> None:
    #     """Upsert this object back into the `player` table."""
    #     ...

    # @abstractmethod
    # def enrich(self) -> None:
    #     """Any additional in‑memory calculation (e.g. run_total, ops)."""
    #     ...