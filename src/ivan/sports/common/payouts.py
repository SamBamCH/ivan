# src/ivan/sports/common/payouts.py
"""Sport-agnostic payout helpers."""
import numpy as np
import pandas as pd


def scale_payouts_by_rank_average(
    payouts_df: pd.DataFrame,
    original_field_size: int,
    num_field_lineups: int,
    decimals: int = 4,
) -> dict[int, float]:
    """
    Payout table for a simulated field of M = num_field_lineups entries that
    stands in for a real field of N = original_field_size.

    Scaled rank m represents the real ranks in ((m-1)·N/M, m·N/M] and pays
    their average prize, so the prize pool scales by exactly M/N, the paid
    share of the field and the shape of the prize curve are preserved, and a
    lineup's mean simulated payout is its expected prize at the corresponding
    real finish. (First place in a 10,000-entry stand-in for a 240,000-entry
    contest is the average of the real top 24 prizes, not the real first prize:
    the simulation cannot resolve finishes finer than N/M real places.)

    payouts_df columns: Start Rank, End Rank, Payout. Returns {rank: amount}
    for every paying scaled rank, amounts to `decimals` places.
    """
    N, M = int(original_field_size), int(num_field_lineups)
    if N <= 0 or M <= 0:
        return {}
    pay = np.zeros(N + 2, dtype=float)                      # pay[r] = prize at real rank r
    for _, row in payouts_df.iterrows():
        s, e = int(row["Start Rank"]), min(int(row["End Rank"]), N)
        if s <= e:
            pay[s:e + 1] = float(row["Payout"])
    cum = np.concatenate([[0.0], np.cumsum(pay[1:N + 1])])  # cum[k] = prizes through real rank k

    step = N / M
    b = np.arange(M + 1) * step                              # real-rank boundaries of each scaled rank
    k = np.minimum(np.floor(b).astype(int), N)
    F = cum[k] + np.where(k < N, (b - k) * pay[k + 1], 0.0)  # cumulative prize up to a fractional rank
    amounts = np.diff(F) / step
    return {m + 1: round(float(a), decimals) for m, a in enumerate(amounts) if a > 0}


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
    return {r: round(a * factor, 2) for r, a in scaled.items()}


