
# src/ivan/database/session.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from abc import ABCMeta

class BaseModelMeta(DeclarativeMeta, ABCMeta):
    """Metaclass that merges SQLAlchemy’s DeclarativeMeta and ABCMeta."""
    pass

Base = declarative_base(metaclass=BaseModelMeta)

# define where your SQLite files live (project root/data)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
DB_DIR = os.path.join(PROJECT_ROOT, 'data')


def init_db(sport: str, db_url: str = None):
    """
    Initialize and return an SQLAlchemy Engine.
    If db_url is provided, use it directly. Otherwise, default to sqlite:///{DB_DIR}/{sport}/{sport}_history.sqlite
    """
    if db_url:
        engine = create_engine(db_url, echo=False, future=True)
    else:
        # ensure the directory exists
        sqlite_folder = os.path.join(DB_DIR, sport)
        os.makedirs(sqlite_folder, exist_ok=True)
        sqlite_path = os.path.join(sqlite_folder, f"{sport}_history.sqlite")
        engine = create_engine(f"sqlite:///{sqlite_path}", echo=False, future=True)
    return engine


def get_session(engine):
    """
    Create and return a SQLAlchemy Session bound to the given Engine.
    """
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()
