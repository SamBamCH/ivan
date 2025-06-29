# src/ivan/sports/mlb/payouts.py

import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from ivan.sports.mlb.models import ContestPayout

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


def scale_payouts_correctly(
    num_field_lineups: int,
    payouts_df: pd.DataFrame,
    original_field_size: int,
    entry_fee: float
) -> dict[int, float]:
    """
    Scale the contest payouts while keeping rank distribution and prize pool proportional.
    Returns a dict: rank -> scaled payout.
    """
    # total prize pool in the original
    orig_prize = sum(
        (row['End Rank'] - row['Start Rank'] + 1) * row['Payout']
        for _, row in payouts_df.iterrows()
    )
    scaled_prize = orig_prize * (num_field_lineups / original_field_size)

    # first‐place keeps its share
    first_pct = payouts_df.iloc[0]['Payout'] / orig_prize
    first_payout = scaled_prize * first_pct

    scaled = {1: round(first_payout, 2)}
    total_paid = first_payout

    # now scale the rest
    for _, row in payouts_df.iloc[1:].iterrows():
        s, e, amt = row['Start Rank'], row['End Rank'], row['Payout']
        # map old ranks → new ranks
        new_s = max(2, round((s/original_field_size)*num_field_lineups))
        new_e = max(new_s, round((e/original_field_size)*num_field_lineups))

        # ensure no overlap
        if new_s in scaled:
            new_s = max(scaled.keys()) + 1
            new_e = max(new_s, new_e)

        # scale the payout amount proportionally
        new_amt = amt * (scaled_prize/orig_prize)
        # enforce a minimum cash (e.g. 1.2× entry_fee)
        new_amt = max(new_amt, round(1.2 * entry_fee, 2))

        count = new_e - new_s + 1
        total_paid += new_amt * count
        for r in range(new_s, new_e+1):
            scaled[r] = round(new_amt, 2)

    # adjust tiny rounding drift
    factor = scaled_prize / total_paid if total_paid else 1
    print('payouts scaled and loaded')
    return {r: round(a * factor, 2) for r, a in scaled.items()}


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
