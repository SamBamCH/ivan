"""
NBA Pre-Lock Refactored Module

This module contains pre-lock (before games start) functionality:
- lineup_generation: Core ILP-based lineup generation
- fpts_samples: Monte Carlo FPTS sampling
- lineup: Batch lineup generation and deduplication
- contest: Contest simulation logic
- slate: Slate-level orchestration
- player: Player utilities (re-exports from fpts_samples)
"""

from ivan.sports.nba.refactor.pre_lock.lineup_generation import (
    FieldPlayer,
    ROSTER_STRUCTURE,
    SLOT_NAMES,
    SLOT_ELIGIBILITY,
    generate_single_classic_lineup,
)

from ivan.sports.nba.refactor.pre_lock.player import generate_player_samples

from ivan.sports.nba.refactor.pre_lock.lineup import (
    generate_lineups,
    deduplicate_lineups,
)

from ivan.sports.nba.refactor.pre_lock.slate import (
    simulate_one_slate,
    choose_lineups_for_slate,
)

from ivan.sports.nba.refactor.pre_lock.contest import (
    sample_field,
    sample_field_with_projection_model,
    simulate_one_contest,
    choose_lineups_for_contest,
)

__all__ = [
    # lineup_generation
    'FieldPlayer',
    'ROSTER_STRUCTURE',
    'SLOT_NAMES',
    'SLOT_ELIGIBILITY',
    'generate_single_classic_lineup',
    # fpts_samples
    'generate_player_samples',
    # lineup
    'generate_lineups',
    'deduplicate_lineups',
    # slate
    'simulate_one_slate',
    'choose_lineups_for_slate',
    # contest
    'sample_field',
    'sample_field_with_projection_model',
    'simulate_one_contest',
    'choose_lineups_for_contest',
]

