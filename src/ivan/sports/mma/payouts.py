# src/ivan/sports/mma/payouts.py
"""Payout tables for an MMA contest: from a pasted CSV, or from the DK contest table."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.mma.models import ContestPayout

_ORD = re.compile(r"(st|nd|rd|th)\b")


def parse_payout_csv(path: str | Path) -> pd.DataFrame:
    """
    'rank,prize' rows as copied from a DK contest page: `1st,"$10,000.00"`,
    `11th - 15th,$300.00`. Returns Start Rank, End Rank, Payout.
    """
    df = pd.read_csv(path, header=None, names=["Rank", "Payout"], dtype=str)
    rows = []
    for _, r in df.iterrows():
        rank_s, pay_s = str(r["Rank"]).strip(), str(r["Payout"]).strip()
        if not rank_s or rank_s.lower() == "nan" or pay_s.lower() == "nan":
            continue
        payout = float(pay_s.replace("$", "").replace(",", ""))
        rank_s = _ORD.sub("", rank_s)
        if "-" in rank_s:
            a, b = [int(x.strip()) for x in rank_s.split("-", 1)]
        else:
            a = b = int(rank_s.strip())
        rows.append((a, b, payout))
    return pd.DataFrame(rows, columns=["Start Rank", "End Rank", "Payout"])


def payouts_from_db(session: Session, contest_id: int) -> pd.DataFrame:
    rows = (session.query(ContestPayout).filter(ContestPayout.contest_id == contest_id)
            .order_by(ContestPayout.min_position).all())
    return pd.DataFrame([(r.min_position, r.max_position, r.payout_amount) for r in rows],
                        columns=["Start Rank", "End Rank", "Payout"])
