from sqlalchemy.orm import Session
import re
import pandas as pd
from typing import Dict, List, Set, Optional
from ivan.sports.nba.models import NBAPlayer
import logging

class OriginalEntry:
    """Represents a single original entry from the CSV."""
    def __init__(
        self,
        entry_id: str,
        contest_id: int,
        lineup_id: int,
        player_ids: List[int],
        player_names: Optional[List[str]] = None,
    ):
        self.entry_id = entry_id
        self.contest_id = contest_id
        self.lineup_id = lineup_id
        self.player_ids = player_ids  # All 8 players in original lineup (roster order)
        self.player_names = player_names or [""] * len(player_ids)
        # Derived during processing
        self.locked_player_ids: Set[int] = set()  # Subset that are locked
        self.swappable_player_ids: Set[int] = set()  # Can be swapped out
        self.best_candidate: Optional['SwapCandidate'] = None

class SwapCandidate:
    """Represents a potential swap lineup for an entry."""
    def __init__(
        self,
        original_entry_id: str,
        original_lineup_id: int,
        locked_player_ids: List[int],
        swapped_player_ids: List[int],
        candidate_lineup_id: int,
    ):
        self.original_entry_id = original_entry_id
        self.original_lineup_id = original_lineup_id
        self.locked_player_ids = locked_player_ids
        self.swapped_player_ids = swapped_player_ids
        self.candidate_lineup_id = candidate_lineup_id
        
        # Filled during simulation
        self.mean_fpts: Optional[float] = None
        self.mean_payout: Optional[float] = None
        self.mean_roi: Optional[float] = None
        self.mean_rank: Optional[float] = None

    def full_lineup(self) -> List[int]:
        """Return complete lineup: locked + swapped players."""
        return self.locked_player_ids + self.swapped_player_ids

def parse_entries_csv(
    csv_path: str,
    session: Session,
) -> Dict[int, List[OriginalEntry]]:
    """
    Parse DraftKings entries CSV in the native export format.

    Expects columns: Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL

    Entries with blank lineups (all position columns empty) are skipped.
    """
    df = pd.read_csv(csv_path, usecols=range(12))

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    entries_by_contest: Dict[int, List[OriginalEntry]] = {}

    # Expected position columns
    position_cols = ['pg', 'sg', 'sf', 'pf', 'c', 'g', 'f', 'util']

    for _, row in df.iterrows():
        #RUN CHECKS ON COLUMNS TO MAKE SURE DATA IS CORRECT
        entry_id = str(row.get('entry id', '')).strip()
        if pd.isna(entry_id):
            logging.error(f"unparsable entry id column {row.get('entry id', '')}")
            continue

        try:
            contest_id = int(row.get('contest id', '0'))
        except (ValueError, TypeError):
            continue
        
        if pd.isna(contest_id):
            logging.error(f"unparsable entry id column {row.get('entry id', '')}")
            continue

        # Check if entry has any players (all NaN means it's blank)
        has_any_player = any(
            (not pd.isna(row.get(col, '')) and str(row.get(col, '')).strip())
            for col in position_cols
        )
        if not has_any_player:
            logging.error(f"not able to parse position columns correctly.")
            continue

        #----------------------------------------------------------------------------------

        player_ids: List[int] = []
        player_names: List[str] = []
        skip_entry = False

        for pos_idx, pos_col in enumerate(position_cols):
            player_val = row.get(pos_col, '')
            player_str = '' if pd.isna(player_val) else str(player_val).strip()

            if not player_str:
                skip_entry = True
                break

            # Extract ID and name if available
            id_match = re.search(r'\((\d+)\)', player_str)
            name_match = re.match(r'^\s*([^\(]+?)\s*\(', player_str)

            player_id: Optional[int] = None
            if id_match:
                try:
                    player_id = int(id_match.group(1))
                except ValueError:
                    player_id = None

            if player_id is None:
                # Fallback: fuzzy name lookup
                name_text = player_str
                p = session.query(NBAPlayer).filter(
                    NBAPlayer.name.ilike(f"%{name_text}%")
                ).first()
                if not p:
                    logging.error(f'nbaplayer not found for {name_text} name_fight')
                    skip_entry = True
                    break
                player_id = p.id
                player_names.append(p.name)
            else:
                # We have an ID; attempt to get name from NBAPlayer; if missing, keep CSV name
                p = session.query(NBAPlayer).filter_by(id=player_id).first()
                if p:
                    player_names.append(p.name)
                elif name_match:
                    player_names.append(name_match.group(1).strip())
                else:
                    player_names.append("")

            if player_id is not None:
                player_ids.append(player_id)
            else:
                skip_entry = True
                break

        if skip_entry or len(player_ids) != 8:
            continue

        entry = OriginalEntry(
            entry_id=entry_id,
            contest_id=contest_id,
            lineup_id=-1,
            player_ids=player_ids,
            player_names=player_names,
        )

        entries_by_contest.setdefault(contest_id, []).append(entry)

    total_entries = sum(len(v) for v in entries_by_contest.values())
    if total_entries == 0:
        raise RuntimeError("No valid entries parsed from entries CSV.")
    return entries_by_contest
