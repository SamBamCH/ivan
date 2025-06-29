from abc import ABC, abstractmethod
from typing import List, Any

class BaseLineup(ABC):
    """
    Abstract base class for any sport-specific lineup.
    """

    @abstractmethod
    def player_ids(self) -> List[int]:
        """Return a list of all player IDs in this lineup."""
        ...

    @abstractmethod
    def to_dict(self, session: Any) -> dict:
        """Return a dict mapping each roster slot name to its player ID."""
        ...

    @abstractmethod
    def save(self, session: Any) -> None:
        """Persist this lineup instance to the database."""
        ...