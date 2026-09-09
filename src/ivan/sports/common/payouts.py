# src/ivan/sports/common/payouts.py
"""Sport-agnostic payout helpers."""
import pandas as pd


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


