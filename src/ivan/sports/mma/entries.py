# src/ivan/sports/mma/entries.py
"""
The DraftKings entries export (data/mma/input/entries.csv): one row per entry
we hold, `Entry ID, Contest Name, Contest ID, Entry Fee, F, F, F, F, F, F`,
followed by the player pool in later columns of rows with no Entry ID.
sim.py simulates every contest in it; choose.py fills every row.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
SLOT_COLS = ["F"] * 6


def parse_money(x) -> float:
    """'$0.50' -> 0.5"""
    try:
        return float(str(x).replace("$", "").replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def load_entries(path: str | Path) -> pd.DataFrame:
    """The entry rows, cut off at the first blank Entry ID; Contest ID as int, fee as float."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"entries file not found: {path}")
    df = pd.read_csv(path, engine="python", header=0, dtype=str, keep_default_na=False, usecols=range(10))
    for col in BASE_COLS:
        if col not in df.columns:
            raise SystemExit(f"expected column {col!r} in {path}")
    blank = df["Entry ID"].str.strip() == ""
    if blank.any():
        df = df.loc[: int(blank.idxmax()) - 1].copy()
    if df.empty:
        raise SystemExit(f"no entries in {path}")
    for col in BASE_COLS:
        df[col] = df[col].str.strip()
    df["Contest ID"] = df["Contest ID"].astype(int)
    df["fee"] = df["Entry Fee"].map(parse_money)
    return df.reset_index(drop=True)


def contests_in(entries: pd.DataFrame) -> list[dict]:
    """One record per contest, in file order: id, name, how many entries we hold, fee."""
    out = []
    for cid, g in entries.groupby("Contest ID", sort=False):
        out.append({"contest_id": int(cid), "name": g["Contest Name"].iloc[0], "entries": int(len(g)),
                    "fee": float(g["fee"].iloc[0])})
    return out
