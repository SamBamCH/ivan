"""Small helpers the NBA modules share: DraftKings positions and the odds API's clock."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Tuple

import pandas as pd


def _coerce_positions(pos: Iterable[str] | str) -> Tuple[str, ...]:
    """DraftKings positions as a sorted tuple: "PF/C" -> ("C", "PF"), ["PG", "SG"] -> ("PG", "SG"), " C , " -> ("C",)."""
    if isinstance(pos, str):
        tokens = [p.strip().upper() for p in re.split(r"[\\/ ,]", pos) if p.strip()]
    else:
        tokens = [str(p).strip().upper() for p in pos if str(p).strip()]
    return tuple(sorted(set(tokens)))


def parse_utc(strings: pd.Series) -> pd.Series:
    """UTC time strings -> tz-aware UTC. format="ISO8601" because the odds table mixes "2026-10-25T22:54:18Z" (the fetch) with
    "2026-10-25 22:54:18" (captures restored before 2026-11-07); pandas infers one format from the first row and coerces the
    other to NaT, which silently dropped every restored row."""
    return pd.to_datetime(strings, errors="coerce", utc=True, format="ISO8601")


def utc_to_local(stamp):
    """A UTC time (an ISO string, a Timestamp, or a Series of them) as naive local time, like every other desk time
    (the pool's gametime is naive local)."""
    local = datetime.now().astimezone().tzinfo
    if isinstance(stamp, pd.Series):
        return parse_utc(stamp).dt.tz_convert(local).dt.tz_localize(None)
    t = pd.Timestamp(stamp)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(local).tz_localize(None).to_pydatetime()
