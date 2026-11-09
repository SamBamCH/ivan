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

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

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


ODDS_COLS = ["player", "market", "sportsbook", "side", "point", "price", "last_update", "game_id", "commence_time", "home_abbr", "away_abbr"]
GAME_TOLERANCE_HOURS = 2.0   # a row's game is the player's game on this slate when the two tips are this close (a team never plays twice in 2 h)
FALLBACK_HOURS = 12.0        # a row without a game (stored before 2026-11-07): counts when the book posted it within this long before the tip


def parse_utc(strings: pd.Series) -> pd.Series:
    """UTC time strings -> tz-aware UTC. format="ISO8601" because the odds table mixes "2026-10-25T22:54:18Z" (the fetch) with
    "2026-10-25 22:54:18" (captures restored before 2026-11-07); pandas infers one format from the first row and coerces the
    other to NaT, which silently dropped every restored row."""
    return pd.to_datetime(strings, errors="coerce", utc=True, format="ISO8601")


def _to_local(utc_strings: pd.Series) -> pd.Series:
    """ISO UTC strings -> naive local datetimes (the pool's gametime is naive local)."""
    return parse_utc(utc_strings).dt.tz_convert(datetime.now().astimezone().tzinfo).dt.tz_localize(None)


def fresh_rows(odds: pd.DataFrame, gametimes: dict, hours: float = FALLBACK_HOURS, cutoffs: dict | None = None, mark: bool = False) -> pd.DataFrame:
    """
    Only the odds rows priced for the player's game on this slate (`gametimes`: DK name -> naive local tip).

    `cutoffs` (DK name -> naive local datetime, changes.status_cutoffs): a row the book posted (`last_update`) before the player's
    cutoff priced the role he had before his status or starter flag moved and is dropped; a row without a posting time is kept.
    `mark=True` keeps those rows and adds a boolean `stale` column instead (the desk's hover shows them set aside).

    A row that carries its game (`commence_time`, from the odds API event, since 2026-11-07) is kept exactly when that game tips
    within GAME_TOLERANCE_HOURS of the player's DK tip: the game is identified, so when it was posted does not matter. A row without
    a game (an older table row, an older archive capture) is kept when the book posted it (`last_update`) within `hours` before
    the tip and not after it; a wide window here is the lazy filter this replaced (36 h let a back-to-back's first game price the
    second). The odds table keeps rows of other games and other days, so without this a player the books did not price tonight
    kept the rates of his last priced game (2026-11-04: IND's Toppin, Walker, Huff and Nance carried Nov 2 and Oct 30 props into
    Nov 4; 2026-10-28: O'Neale on week-old props). A player with no row for his game has no market rate: the projection falls back
    to his history alone.
    """
    if odds is None or odds.empty or "last_update" not in odds.columns:
        return odds.iloc[0:0] if odds is not None else pd.DataFrame()
    tip = pd.to_datetime(odds["player"].map(lambda n: gametimes.get(n)), errors="coerce")
    posted = _to_local(odds["last_update"])
    game = _to_local(odds["commence_time"]) if "commence_time" in odds.columns else pd.Series(pd.NaT, index=odds.index)
    linked = game.notna()
    keep_linked = linked & ((game - tip).abs() <= pd.Timedelta(hours=GAME_TOLERANCE_HOURS))
    keep_legacy = ~linked & posted.notna() & (posted >= tip - pd.Timedelta(hours=hours)) & (posted <= tip + pd.Timedelta(hours=1))
    keep = tip.notna() & (keep_linked | keep_legacy)
    out = odds[keep.values]
    if cutoffs:
        cut = pd.to_datetime(out["player"].map(lambda n: cutoffs.get(n)), errors="coerce")
        was = posted[keep.values]
        stale = cut.notna().values & was.notna().values & (was.values < cut.values)
        if mark:
            out = out.copy(); out["stale"] = stale
        else:
            out = out[~stale]
    elif mark:
        out = out.copy(); out["stale"] = False
    return out


def load_odds(bind) -> pd.DataFrame:
    """Every row of the odds table with ODDS_COLS (`bind`: a session, connection or engine); the game columns NaN on a table
    that predates them."""
    if hasattr(bind, "get_bind"):                                     # a Session
        bind = bind.get_bind()
    with (bind.connect() if hasattr(bind, "connect") else nullcontext(bind)) as conn:     # an Engine, or a Connection as is
        # pragma rather than the inspector: a slate session reaches the odds table through an attached history database,
        # where SQLAlchemy's inspector (main.table_info) does not look
        have = {r[1] for r in conn.exec_driver_sql("pragma table_info(odds)").fetchall()}
        cols = [c for c in ODDS_COLS if c in have]
        odds = pd.read_sql(f"select {', '.join(cols)} from odds", conn)
    for c in ODDS_COLS:
        if c not in odds.columns:
            odds[c] = None
    return odds


def fresh_odds(session, gametimes: dict, hours: float = FALLBACK_HOURS, cutoffs: dict | None = None, mark: bool = False) -> pd.DataFrame:
    """The odds table's rows the projection may use for this pool (fresh_rows over load_odds)."""
    return fresh_rows(load_odds(session), gametimes, hours, cutoffs=cutoffs, mark=mark)


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
