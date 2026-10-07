"""
Per-minute rates from priced player props.

A book's line and its over/under prices give, after de-vigging, P(X > line).
Solving a negative binomial with the stat's calibrated dispersion for the
mean that matches that probability turns every (book, line) pair into a mean
estimate; the consensus is the median across pairs. Dividing by the minutes
the player projected to at fetch time gives a forward-looking per-minute
rate: the books price the role as it stands now, and when his minutes move
after news the rate travels with him.

Markets: points, rebounds, assists, threes, steals, blocks (and their
alternate lines). Turnovers are never priced and stay on history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nbinom

from .fpts import load_calibration

MARKET_TO_STAT = {
    "player_points": "points", "player_points_alternate": "points",
    "player_rebounds": "rebounds", "player_rebounds_alternate": "rebounds",
    "player_assists": "assists", "player_assists_alternate": "assists",
    "player_threes": "fg3m", "player_threes_alternate": "fg3m",
    "player_steals": "steals", "player_steals_alternate": "steals",
    "player_blocks": "blocks", "player_blocks_alternate": "blocks",
}
NBA_MARKETS = ",".join(MARKET_TO_STAT)
MIN_PAIRS = 2          # (book, line) pairs before a market mean is trusted
ONE_SIDED_HAIRCUT = 0.955


def american_to_prob(price: float) -> float:
    price = float(price)
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def _mean_from_tail(point: float, p_over: float, k: float) -> float | None:
    """NB mean mu with P(X > point) = p_over, dispersion k."""
    p_over = float(np.clip(p_over, 0.02, 0.98))
    thresh = int(np.floor(point))           # P(X > 7.5) = P(X >= 8) = sf(7)
    f = lambda mu: nbinom.sf(thresh, k, k / (k + mu)) - p_over
    try:
        return float(brentq(f, 1e-3, 200.0))
    except ValueError:
        return None


def market_means(odds: pd.DataFrame, dispersion: dict[str, float] | None = None) -> pd.DataFrame:
    """
    odds: columns player, market, sportsbook, side, point, price (normalized
    names). Returns one row per (player, stat): mean, pairs, books, line.
    """
    k_by_stat = (dispersion or load_calibration()["dispersion"])
    df = odds[odds["market"].isin(MARKET_TO_STAT)].drop_duplicates(
        ["player", "market", "sportsbook", "side", "point", "price"]).copy()
    df["stat"] = df["market"].map(MARKET_TO_STAT)
    rows = []
    for (player, stat), g in df.groupby(["player", "stat"]):
        ests, books = [], set()
        for (book, point), pair in g.groupby(["sportsbook", "point"]):
            over = pair[pair["side"].str.lower() == "over"]["price"]
            under = pair[pair["side"].str.lower() == "under"]["price"]
            if len(over) == 0:
                continue
            po = american_to_prob(over.iloc[0])
            if len(under):
                pu = american_to_prob(under.iloc[0]); po = po / (po + pu)
            else:
                po = po * ONE_SIDED_HAIRCUT
            mu = _mean_from_tail(float(point), po, k_by_stat.get(stat, 10.0))
            if mu is not None:
                ests.append(mu); books.add(book)
        if len(ests) >= MIN_PAIRS:
            main = g[~g["market"].str.endswith("alternate")]["point"]
            rows.append(dict(player=player, stat=stat, mean=float(np.median(ests)),
                             pairs=len(ests), books=len(books),
                             line=float(main.median()) if len(main) else float("nan")))
    return pd.DataFrame(rows)


@dataclass
class MarketRates:
    """Per player: stat -> per-minute rate, with the minutes the rate was priced at."""
    rates: dict[str, dict[str, float]]
    basis_minutes: dict[str, float]
    books: dict[str, int]

    def for_player(self, name: str) -> dict[str, float] | None:
        return self.rates.get(name)


def market_rates(means: pd.DataFrame, basis_minutes: dict[str, float], min_basis: float = 8.0) -> MarketRates:
    """Rates = market mean / minutes the player projected to when the odds were fetched."""
    rates, basis, books = {}, {}, {}
    for player, g in means.groupby("player"):
        m = basis_minutes.get(player)
        if m is None or m < min_basis:
            continue
        rates[player] = {r.stat: float(r["mean"]) / m for _, r in g.iterrows()}
        basis[player] = float(m); books[player] = int(g["books"].max())
    return MarketRates(rates, basis, books)


def blend_rates(hist: dict[str, float], market: dict[str, float] | None, weight: float = 1.0) -> dict[str, float]:
    """Market rate where priced (weight 1 = market only), history elsewhere."""
    if not market:
        return dict(hist)
    out = dict(hist)
    for s, r in market.items():
        out[s] = weight * r + (1 - weight) * hist[s]
    return out
