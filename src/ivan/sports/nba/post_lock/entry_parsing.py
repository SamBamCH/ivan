
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from ivan.sports.nba.pre_lock.lineup_generation import SLOT_NAMES  # ['PG','SG','SF','PF','C','G','F','UTIL']

SLOT_COLS = [s.lower() for s in SLOT_NAMES]

@dataclass
class OriginalEntry:
    entry_id: str
    contest_id: int
    lineup_id: int
    slots: Dict[str, int]                 # e.g. {'PG': 123, 'SG': 456, ...}
    slot_names: List[str] = field(default_factory=lambda: SLOT_NAMES.copy())
    player_names_by_slot: Dict[str, str] = field(default_factory=dict)

    # Derived during processing
    locked_player_ids: Set[int] = field(default_factory=set)
    swappable_player_ids: Set[int] = field(default_factory=set)
    best_candidate: Optional['SwapCandidate'] = None

    def player_ids(self) -> List[int]:
        return [self.slots[s] for s in self.slot_names]

    def player_names(self) -> List[str]:
        return [self.player_names_by_slot.get(s, "") for s in self.slot_names]

    def locked_slot_indices(self) -> Set[int]:
        return {i for i, s in enumerate(self.slot_names) if self.slots[s] in self.locked_player_ids}

class SwapCandidate:
    def __init__(self, original_entry_id: str, original_lineup_id: int,
                 locked_player_ids: List[int], swapped_player_ids: List[int], candidate_lineup_id: int):
        self.original_entry_id = original_entry_id
        self.original_lineup_id = original_lineup_id
        self.locked_player_ids = locked_player_ids
        self.swapped_player_ids = swapped_player_ids
        self.candidate_lineup_id = candidate_lineup_id
        self.mean_fpts: Optional[float] = None
        self.mean_payout: Optional[float] = None
        self.mean_roi: Optional[float] = None
        self.mean_rank: Optional[float] = None

    def full_lineup(self) -> List[int]:
        return self.locked_player_ids + self.swapped_player_ids

