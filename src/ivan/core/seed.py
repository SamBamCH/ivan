# src/ivan/core/seed.py
"""
One knob for every random draw in the pipeline.

Unset (the live default) every stage keeps the behaviour it has: a fixed
seed where it had one, fresh entropy where it had none. With IVAN_SEED=<int>
each named draw gets a seed derived from it, so a run is reproducible and two
runs with different values are two independent runs. The replay driver sets
it per repeat (ivan.core.replay).
"""
from __future__ import annotations

import os
import zlib
from typing import Optional

ENV = "IVAN_SEED"


def base() -> Optional[int]:
    v = os.environ.get(ENV)
    if v is None or v == "":
        return None
    return int(v)


def seed_for(name: str, default: Optional[int] = None) -> Optional[int]:
    """The seed for the draw called `name`: `default` when IVAN_SEED is unset, else derived from it."""
    b = base()
    if b is None:
        return default
    return (b * 1_000_003 + zlib.crc32(name.encode("utf-8"))) % (2 ** 32)
