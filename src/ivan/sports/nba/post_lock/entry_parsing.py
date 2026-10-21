
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import logging
import re
import pandas as pd
from sqlalchemy.orm import Session
from ivan.sports.nba.models import NBAPlayer
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

def parse_entries_csv(csv_path: str, session: Session) -> Dict[int, List[OriginalEntry]]:
    df = pd.read_csv(csv_path, usecols=range(12))
    df.columns = [c.strip().lower() for c in df.columns]

    entries_by_contest: Dict[int, List[OriginalEntry]] = {}
    for _, row in df.iterrows():
        raw_entry_id = row.get('entry id', '')
        if pd.isna(raw_entry_id):
            logging.error(f"unparsable entry id column {raw_entry_id}")
            continue
        try:
            entry_id = str(int(float(raw_entry_id)))
        except (ValueError, TypeError):
            entry_id = str(raw_entry_id).strip()
        if not entry_id:
            logging.error("empty entry id")
            continue

        try:
            contest_id = int(row.get('contest id', '0'))
        except (ValueError, TypeError):
            continue
        if pd.isna(contest_id):
            logging.error(f"unparsable contest id for entry {entry_id}")
            continue

        # ensure there is at least one non-empty slot
        if not any((not pd.isna(row.get(col, '')) and str(row.get(col, '')).strip()) for col in SLOT_COLS):
            logging.error(f"blank lineup for entry {entry_id}")
            continue

        slots: Dict[str, int] = {}
        names: Dict[str, str] = {}
        skip_entry = False

        for slot_name, col in zip(SLOT_NAMES, SLOT_COLS):
            player_val = row.get(col, '')
            player_str = '' if pd.isna(player_val) else str(player_val).strip()
            if not player_str:
                skip_entry = True
                break

            id_match = re.search(r'\((\d+)\)', player_str)
            name_match = re.match(r'^\s*([^\(]+?)\s*\(', player_str)

            player_id: Optional[int] = None
            if id_match:
                try:
                    player_id = int(id_match.group(1))
                except ValueError:
                    player_id = None

            if player_id is None:
                # Fallback: fuzzy name lookup on full cell text
                name_text = player_str
                p = session.query(NBAPlayer).filter(NBAPlayer.name.ilike(f"%{name_text}%")).first()
                if not p:
                    logging.error(f'NBAPlayer not found for "{name_text}"')
                    skip_entry = True
                    break
                player_id = p.id
                names[slot_name] = p.name
            else:
                p = session.query(NBAPlayer).filter_by(id=player_id).first()
                if p:
                    names[slot_name] = p.name
                elif name_match:
                    names[slot_name] = name_match.group(1).strip()
                else:
                    names[slot_name] = ""

            slots[slot_name] = player_id

        if skip_entry or len(slots) != 8:
            continue

        entry = OriginalEntry(
            entry_id=entry_id,
            contest_id=contest_id,
            lineup_id=-1,
            slots=slots,
            player_names_by_slot=names,
        )
        entries_by_contest.setdefault(contest_id, []).append(entry)

    total_entries = sum(len(v) for v in entries_by_contest.values())
    if total_entries == 0:
        raise RuntimeError("No valid entries parsed from entries CSV.")
    return entries_by_contest
