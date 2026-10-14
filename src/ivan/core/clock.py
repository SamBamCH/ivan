# src/ivan/core/clock.py
"""
The pipeline's idea of "now", so a replay can pin a past slate to its lock.

Live runs: now() is datetime.now(). Replays set IVAN_AS_OF (an ISO local
datetime, naive) in the environment, or call set_as_of(); every stage that
asks what time it is, or which games are history, goes through here.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

ENV = "IVAN_AS_OF"


def as_of() -> Optional[datetime]:
    """The frozen local time of a replay, or None when running live."""
    raw = os.environ.get(ENV)
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def now() -> datetime:
    return as_of() or datetime.now()


def is_replay() -> bool:
    return as_of() is not None


def set_as_of(when: Optional[datetime]) -> None:
    """Freeze (or, with None, unfreeze) the clock for this process and its children."""
    if when is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = when.replace(microsecond=0).isoformat()


@contextmanager
def frozen(when: Optional[datetime]):
    prev = os.environ.get(ENV)
    set_as_of(when)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = prev
