# src/ivan/sports/mlb/payouts.py

import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from ivan.sports.mlb.models import ContestPayout
from ivan.sports.common.payouts import scale_payouts_correctly  # re-exported for existing importers

def parse_payout_csv(payouts_csv: str) -> pd.DataFrame:
    """Parses the payout CSV and extracts rank ranges & payouts."""
    df = pd.read_csv(payouts_csv, header=None, names=['Rank','Payout'], dtype=str)
    payouts = []

    for _, row in df.iterrows():
        rank_str   = row['Rank'].strip()
        payout_str = row['Payout']
        if pd.isna(payout_str) or str(payout_str).lower()=='nan':
            continue

        # clean payout to float
        payout = float(str(payout_str).replace('$','').replace(',',''))

        # parse rank or rank range
        cleaned = rank_str.replace('th','') \
                          .replace('st','') \
                          .replace('nd','') \
                          .replace('rd','')
        if '-' in cleaned:
            start, end = map(int, cleaned.split('-'))
        else:
            start = end = int(cleaned)

        payouts.append((start, end, payout))

    return pd.DataFrame(payouts, columns=['Start Rank','End Rank','Payout'])


def load_and_scale_payouts(
    payouts_csv: str,
    original_field_size: int,
    num_field_lineups: int,
    entry_fee: float
) -> dict[int, float]:
    """
    Load the payouts CSV, parse it, then scale.
    """
    payouts_df = parse_payout_csv(payouts_csv)
    return scale_payouts_correctly(
        num_field_lineups, payouts_df, original_field_size, entry_fee
    )

def load_and_scale_payouts_from_db(
    session: Session,
    contest_id: int,
    num_field_lineups: int,
    original_field_size: int,
    entry_fee: float
) -> dict[int, float]:
    """
    Fetch the raw payouts for `contest_id` from the DB, then scale to 
    num_field_lineups using the same logic as `scale_payouts_correctly`.
    """
    
    # 1) pull all payout‐rows for this contest_id
    raw_rows = (
        session
        .query(ContestPayout)
        .filter(ContestPayout.contest_id == contest_id)
        .order_by(ContestPayout.min_position.asc())
        .all()
    )

    # 2) build a small DataFrame with columns ["Start Rank","End Rank","Payout"]
    payouts_list = []
    for row in raw_rows:
        # Each row has min_position, max_position, payout_amount (float)
        payouts_list.append(
            (row.min_position, row.max_position, row.payout_amount)
        )
    if not payouts_list:
        raise ValueError(f"No payouts found in DB for contest_id={contest_id}")

    payouts_df = pd.DataFrame(
        payouts_list, columns=["Start Rank","End Rank","Payout"]
    )

    # 3) now re‐use your existing scale_payouts_correctly
    from ivan.sports.mlb.payouts import scale_payouts_correctly
    return scale_payouts_correctly(
        num_field_lineups = num_field_lineups,
        payouts_df        = payouts_df,
        original_field_size = original_field_size,
        entry_fee         = entry_fee
    )
