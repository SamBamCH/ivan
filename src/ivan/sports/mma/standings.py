# src/ivan/sports/mma/standings.py
"""
Real contest fields from DraftKings standings exports (raw dk/standings):
every entry's six fighters, points and rank, the fighter side table
(drafted % and actual fantasy points), and per unique lineup the copies,
salary left, summed win probability and whether it doubles up a fight.
These are what the dupe and field models are fitted on and what the
validation scores the sampled field against.

Derived files: data/mma/derived/contests/<contest id>/lineups.parquet
(one row per unique lineup) and players.csv (the side table joined to the
pool). Rebuildable from raw at any time.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ivan.core import raw
from ivan.sports.mma.ingest import NameMatcher, name_tokens
from ivan.sports.mma.odds import american_to_prob
from ivan.sports.mma.paths import MMA_DATA_DIR as DATA_DIR

DERIVED_DIR = DATA_DIR / "derived" / "contests"
SALARY_CAP = 50000


def parse_export(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, players) from an export: entries have rank, entry_id, entry_name, points, fighters (list of names)."""
    df = pd.read_csv(io.StringIO(text.lstrip("﻿")), dtype=str, keep_default_na=False)
    cols = {c.strip(): c for c in df.columns}
    ent = pd.DataFrame({"rank": pd.to_numeric(df[cols["Rank"]], errors="coerce"), "entry_id": df[cols["EntryId"]],
                        "entry_name": df[cols["EntryName"]], "points": pd.to_numeric(df[cols["Points"]], errors="coerce"),
                        "lineup": df[cols["Lineup"]]})
    ent = ent[ent.lineup.str.strip() != ""].copy()
    ent["fighters"] = ent.lineup.map(lambda s: [x.strip() for x in re.split(r"\bF\s", " " + s.strip()) if x.strip()])
    side = df[[cols["Player"], cols["%Drafted"], cols["FPTS"]]].copy()
    side.columns = ["name", "drafted_pct", "fpts"]
    side = side[side.name.str.strip() != ""]
    side["drafted_pct"] = pd.to_numeric(side.drafted_pct.str.rstrip("%"), errors="coerce")
    side["fpts"] = pd.to_numeric(side.fpts, errors="coerce")
    return ent.reset_index(drop=True), side.reset_index(drop=True)


def derive_contest(contest_id: int, pool: Iterable[dict], win_prob: dict[int, float], out_dir: Path = DERIVED_DIR,
                   text: Optional[str] = None) -> Optional[dict]:
    """
    Parse the newest export of `contest_id` and write lineups.parquet +
    players.csv. pool rows need id, name, salary, fight_id (the slate's
    draftables); win_prob maps fighter id -> probability. Returns a summary
    or None when there is no export.
    """
    if text is None:
        art = raw.latest("mma", "dk/standings", contest_id)
        if art is None:
            return None
        text = raw.load(art.path, "mma")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
    ent, side = parse_export(text)
    rows = list(pool)
    matcher = NameMatcher(rows)
    cache: dict[str, Optional[dict]] = {}

    def find(name: str) -> Optional[dict]:
        if name not in cache:
            cache[name] = matcher.find(name)
        return cache[name]

    by_id = {r["id"]: r for r in rows}
    recs = []
    unmatched: set[str] = set()
    for e in ent.itertuples():
        ids = []
        for n in e.fighters:
            r = find(n)
            if r is None:
                unmatched.add(n)
            else:
                ids.append(int(r["id"]))
        if len(ids) != 6:
            continue
        key = tuple(sorted(ids))
        recs.append((key, float(e.points) if pd.notna(e.points) else np.nan, int(e.rank) if pd.notna(e.rank) else 0))
    lu = pd.DataFrame(recs, columns=["key", "points", "rank"])
    g = lu.groupby("key").agg(copies=("key", "size"), points=("points", "first"), best_rank=("rank", "min")).reset_index()
    g["salary"] = g.key.map(lambda k: sum(by_id[i]["salary"] for i in k))
    g["salary_left"] = SALARY_CAP - g.salary
    g["sum_win"] = g.key.map(lambda k: sum(win_prob.get(i, 0.5) for i in k))
    g["same_fight"] = g.key.map(lambda k: len({by_id[i]["fight_id"] for i in k}) < len(k))
    for j in range(6):
        g[f"f{j + 1}"] = g.key.map(lambda k, j=j: k[j])
    g = g.drop(columns=["key"]).sort_values(["copies", "points"], ascending=[False, False]).reset_index(drop=True)

    side["id"] = side.name.map(lambda n: (find(n) or {}).get("id"))
    side["salary"] = side.id.map(lambda i: by_id.get(i, {}).get("salary"))
    side["win_prob"] = side.id.map(lambda i: win_prob.get(i))

    d = Path(out_dir) / str(contest_id)
    d.mkdir(parents=True, exist_ok=True)
    g.to_parquet(d / "lineups.parquet", index=False)
    side.to_csv(d / "players.csv", index=False)
    n_entries = int(g.copies.sum())
    return {"contest_id": contest_id, "entries": len(ent), "entries_matched": n_entries, "unique": len(g),
            "unmatched_names": sorted(unmatched), "same_fight_entries": int(g.loc[g.same_fight, "copies"].sum()),
            "max_copies": int(g.copies.max()) if len(g) else 0, "dir": str(d)}


def load_derived(contest_id: int, out_dir: Path = DERIVED_DIR) -> Optional[pd.DataFrame]:
    p = Path(out_dir) / str(contest_id) / "lineups.parquet"
    return pd.read_parquet(p) if p.exists() else None


def win_probs_from_odds_csv(path: Path, pool: Iterable[dict]) -> dict[int, float]:
    """Fighter id -> win probability from a player_odds.csv (its Win column) for the pool it describes."""
    rows = list(pool)
    matcher = NameMatcher(rows)
    out = {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for _, r in df.iterrows():
        m = matcher.find(r["player"])
        w = str(r.get("Win", "")).strip()
        if m and w:
            try:
                out[int(m["id"])] = american_to_prob(float(w.replace("+", "")))
            except ValueError:
                pass
    return out
