"""
The NBA desk's model: one Desk object holds the loaded slate (ivan.sports.nba.projection.live.LiveSlate), the projection
store's frames per team, the override layers, the news and the rotation views the pages render. No web imports: the same
object serves the web desk (ivan.web.nba) and the scripts. Built as mixins, one file per concern:

    state.py      the session, the slate, the frames, rebuild / reproject, the header
    rotation.py   tonight's grid and a played game's, the mapping fingerprint
    rows.py       a team's rows and card
    news.py       the news table, the team states
    scenarios.py  a team's saved scenarios
    lineups.py    the lineup pool's slates (the export)
"""
from __future__ import annotations

from .state import (ADD_BACK_MINUTES, CONSUMER, DeskBase, engine, FOUL_SHORT_PF, FOUL_SHORT_SHARE, MARKET_GAP, MINUTES_DEV, NEAR_TIP_MIN, QUESTION_STATUSES, ROTATION_MIN,
                    STATE_FILE, TEAM_BUDGET, short_name, utc_to_local)
from .rotation import RotationMixin
from .rows import RowsMixin
from .news import NewsMixin
from .scenarios import ScenarioMixin
from .lineups import backfill_lineup_slates, lineup_names, lineup_slates


class Desk(RotationMixin, RowsMixin, NewsMixin, ScenarioMixin, DeskBase):
    """Session + LiveSlate + the projection store's frames per team, rebuilt when the pool changes on disk."""


__all__ = ["Desk", "CONSUMER", "engine", "ADD_BACK_MINUTES", "FOUL_SHORT_PF", "FOUL_SHORT_SHARE", "MARKET_GAP", "MINUTES_DEV", "NEAR_TIP_MIN", "QUESTION_STATUSES", "ROTATION_MIN",
           "STATE_FILE", "TEAM_BUDGET", "short_name", "utc_to_local", "backfill_lineup_slates", "lineup_names", "lineup_slates"]
