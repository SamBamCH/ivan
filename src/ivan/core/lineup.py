from sqlalchemy.orm import declarative_mixin
from sqlalchemy import Column, Integer, JSON, DateTime, func

@declarative_mixin
class BaseLineup:
    """
    Shared lineup behavior for ALL sports
    """
    id               = Column(Integer, primary_key=True, autoincrement=True)
    player_ids_list  = Column(JSON,    nullable=False)
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)
