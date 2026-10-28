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
MU_LO, MU_HI = 1e-3, 200.0    # the bracket the NB mean is searched in


def american_to_prob(price: float) -> float:
    price = float(price)
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def american_to_prob_array(price: np.ndarray) -> np.ndarray:
    """american_to_prob over an array (NaN stays NaN)."""
    price = np.asarray(price, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(price > 0, 100.0 / (price + 100.0), -price / (-price + 100.0))


def _mean_from_tail(point: float, p_over: float, k: float) -> float | None:
    """NB mean mu with P(X > point) = p_over, dispersion k (one pair; the table path is means_from_tails)."""
    out = means_from_tails(np.array([point]), np.array([p_over]), np.array([k]))
    return None if np.isnan(out[0]) else float(out[0])


def means_from_tails(points: np.ndarray, p_overs: np.ndarray, ks: np.ndarray, iterations: int = 60) -> np.ndarray:
    """
    The NB mean for every (point, p_over, k) at once: mu with P(X > point) = p_over, found by bisection on
    [MU_LO, MU_HI] (the tail is monotone in mu). NaN where the root is not bracketed (the pair is dropped, as
    brentq's ValueError was). 60 halvings of the bracket put the answer within 2e-16, brentq's own tolerance;
    the pairs are reduced to their distinct (threshold, probability, dispersion) triples first, so a slate's
    hundred thousand prop rows cost a few vectorised tail evaluations instead of one root-find each.
    """
    points = np.asarray(points, dtype=np.float64); ks = np.asarray(ks, dtype=np.float64)
    p = np.clip(np.asarray(p_overs, dtype=np.float64), 0.02, 0.98)
    thresh = np.floor(points)                                  # P(X > 7.5) = P(X >= 8) = sf(7)
    out = np.full(len(points), np.nan)
    if not len(points):
        return out
    keys = np.stack([thresh, p, ks], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    t, pp, kk = uniq[:, 0], uniq[:, 1], uniq[:, 2]
    f = lambda mu: nbinom.sf(t, kk, kk / (kk + mu)) - pp
    lo = np.full(len(t), MU_LO); hi = np.full(len(t), MU_HI)
    flo, fhi = f(lo), f(hi)
    ok = np.isfinite(flo) & np.isfinite(fhi) & (flo * fhi <= 0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        up = fm > 0                                            # f rises with mu: the root is below mid
        hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
    mu = 0.5 * (lo + hi)
    mu[~ok] = np.nan
    return mu[np.asarray(inverse).reshape(-1)]


def market_means(odds: pd.DataFrame, dispersion: dict[str, float] | None = None) -> pd.DataFrame:
    """
    odds: columns player, market, sportsbook, side, point, price (normalized
    names). Returns one row per (player, stat): mean, pairs, books, line.

    Per (player, stat, book, line): the first over price and the first under price on file; the under de-vigs the
    over, a one-sided line takes ONE_SIDED_HAIRCUT. The pair's NB mean comes from the tail (means_from_tails); a
    (player, stat) with MIN_PAIRS or more means reports their median, the books behind them and the median main
    line. Vectorised (2026-10-28): the same arithmetic as the row-by-row version, a slate in about a second.
    """
    k_by_stat = (dispersion or load_calibration()["dispersion"])
    cols = ["player", "stat", "mean", "pairs", "books", "line"]
    df = odds[odds["market"].isin(MARKET_TO_STAT)].drop_duplicates(
        ["player", "market", "sportsbook", "side", "point", "price"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["stat"] = df["market"].map(MARKET_TO_STAT)
    df["_side"] = df["side"].astype(str).str.lower()
    key = ["player", "stat", "sportsbook", "point"]
    first = df.drop_duplicates(key + ["_side"], keep="first")          # the first row per (book, line, side), in table order
    over = first[first["_side"] == "over"][key + ["price"]]
    under = first[first["_side"] == "under"][key + ["price"]].rename(columns={"price": "price_under"})
    pairs = over.merge(under, on=key, how="left")
    po = american_to_prob_array(pairs["price"].to_numpy(dtype=np.float64))
    pu = american_to_prob_array(pairs["price_under"].to_numpy(dtype=np.float64))
    two_sided = pairs["price_under"].notna().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        po = np.where(two_sided, po / (po + pu), po * ONE_SIDED_HAIRCUT)
    ks = pairs["stat"].map(lambda s: float(k_by_stat.get(s, 10.0))).to_numpy(dtype=np.float64)
    pairs["mu"] = means_from_tails(pairs["point"].to_numpy(dtype=np.float64), po, ks)
    good = pairs[pairs["mu"].notna()]
    if good.empty:
        return pd.DataFrame(columns=cols)
    agg = good.groupby(["player", "stat"]).agg(mean=("mu", "median"), pairs=("mu", "size"), books=("sportsbook", "nunique"))
    agg = agg[agg["pairs"] >= MIN_PAIRS]
    if agg.empty:
        return pd.DataFrame(columns=cols)
    main = df[~df["market"].str.endswith("alternate")].groupby(["player", "stat"])["point"].median()
    agg["line"] = main.reindex(agg.index).astype(float)
    out = agg.reset_index().sort_values(["player", "stat"]).reset_index(drop=True)
    out["mean"] = out["mean"].astype(float); out["pairs"] = out["pairs"].astype(int); out["books"] = out["books"].astype(int)
    return out[cols]


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
