
# src/ivan/database/session.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from abc import ABCMeta
import time
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

class BaseModelMeta(DeclarativeMeta, ABCMeta):
    """Metaclass that merges SQLAlchemy’s DeclarativeMeta and ABCMeta."""
    pass

Base = declarative_base(metaclass=BaseModelMeta)

# define where your SQLite files live (project root/data)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
DB_DIR = os.path.join(PROJECT_ROOT, 'data')

# .env at the project root may carry per-machine overrides (IVAN_<SPORT>_DB_URL)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def init_db(sport: str, db_url: str = None):
    """
    Initialize and return an SQLAlchemy Engine.
    Resolution order: explicit db_url -> $IVAN_<SPORT>_DB_URL -> sqlite:///{DB_DIR}/{sport}/{sport}_history.sqlite
    """
    db_url = db_url or os.environ.get(f"IVAN_{sport.upper()}_DB_URL")
    if db_url:
        engine = create_engine(db_url,                                 
                               connect_args={
                                    # wait up to 30s for locks to clear
                                    "timeout": 30,
                                    # allow sharing connections across threads (not strictly needed for separate procs)
                                    "check_same_thread": False,
                                },
                                echo=False, 
                                future=True)
    else:
        # ensure the directory exists
        sqlite_folder = os.path.join(DB_DIR, sport)
        os.makedirs(sqlite_folder, exist_ok=True)
        sqlite_path = os.path.join(sqlite_folder, f"{sport}_history.sqlite")
        engine = create_engine(f"sqlite:///{sqlite_path}",
                                connect_args={
                                    # wait up to 30s for locks to clear
                                    "timeout": 30,
                                    # allow sharing connections across threads (not strictly needed for separate procs)
                                    "check_same_thread": False,
                                },
                               echo=False, 
                               future=True)
    return engine


def get_session(engine):
    """
    Create and return a SQLAlchemy Session bound to the given Engine.
    """
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()

def read_sql(session, stmt):
    """
    pandas.read_sql against the session's engine using a real Connection
    (SQLAlchemy 2.x safe). Accepts a raw SQL string or a Core statement.
    """
    import pandas as pd
    if isinstance(stmt, str):
        stmt = text(stmt)
    with session.bind.connect() as conn:
        return pd.read_sql(stmt, conn)

def safe_commit(session, max_tries=5):
    for i in range(max_tries):
        try:
            session.commit()
            return
        except OperationalError as e:
            if "database is locked" in str(e).lower():
                backoff = 0.1 * (2 ** i)
                time.sleep(backoff)
            else:
                raise
    # if we reach here, we never succeeded
    raise RuntimeError("Could not commit after retries—DB still locked")


def add_missing_columns(engine, metadata) -> list:
    """
    Add columns the models declare that an existing table lacks (there is no
    migration tool here; SQLite's ALTER TABLE ADD COLUMN is a metadata-only
    change). Tables that do not exist are left to create_all. Returns the
    "table.column" names added.
    """
    from sqlalchemy import inspect
    insp = inspect(engine)
    added = []
    for table in metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col.type.compile(engine.dialect)}'
            with engine.begin() as conn:
                conn.execute(text(ddl))
            added.append(f"{table.name}.{col.name}")
    return added
