"""market.py: the vectorised prop devig gives the row-by-row arithmetic's numbers."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nbinom

from ivan.sports.nba.projection import market as M

K = {"points": 12.0, "rebounds": 6.0, "assists": 5.0, "fg3m": 3.0, "steals": 2.0, "blocks": 2.0}


def _old_mean(point, p_over, k):
    p_over = float(np.clip(p_over, 0.02, 0.98)); thresh = int(np.floor(point))
    f = lambda mu: nbinom.sf(thresh, k, k / (k + mu)) - p_over
    try:
        return float(brentq(f, 1e-3, 200.0))
    except ValueError:
        return None


def _old_market_means(odds):
    """market_means as it was before 2026-10-28: a root-find per (book, line) pair."""
    df = odds[odds["market"].isin(M.MARKET_TO_STAT)].drop_duplicates(["player", "market", "sportsbook", "side", "point", "price"]).copy()
    df["stat"] = df["market"].map(M.MARKET_TO_STAT)
    rows = []
    for (player, stat), g in df.groupby(["player", "stat"]):
        ests, books = [], set()
        for (book, point), pair in g.groupby(["sportsbook", "point"]):
            over = pair[pair["side"].str.lower() == "over"]["price"]; under = pair[pair["side"].str.lower() == "under"]["price"]
            if len(over) == 0:
                continue
            po = M.american_to_prob(over.iloc[0])
            if len(under):
                pu = M.american_to_prob(under.iloc[0]); po = po / (po + pu)
            else:
                po = po * M.ONE_SIDED_HAIRCUT
            mu = _old_mean(float(point), po, K.get(stat, 10.0))
            if mu is not None:
                ests.append(mu); books.add(book)
        if len(ests) >= M.MIN_PAIRS:
            main = g[~g["market"].str.endswith("alternate")]["point"]
            rows.append(dict(player=player, stat=stat, mean=float(np.median(ests)), pairs=len(ests), books=len(books),
                             line=float(main.median()) if len(main) else float("nan")))
    return pd.DataFrame(rows)


def _odds():
    rng = np.random.default_rng(3)
    rows = []
    players = ["a smith", "b jones", "c lee", "d kim"]
    for pl in players:
        for market in ("player_points", "player_points_alternate", "player_rebounds", "player_assists", "player_threes", "player_blocks"):
            for book in ("dk", "fd", "mgm"):
                base = {"player_points": 22.5, "player_points_alternate": 27.5, "player_rebounds": 6.5, "player_assists": 4.5,
                        "player_threes": 2.5, "player_blocks": 0.5}[market]
                point = base + rng.choice([-1, 0, 1])
                over = int(rng.choice([-120, -110, -105, 100, 115, 140]))
                rows.append(dict(player=pl, market=market, sportsbook=book, side="Over", point=point, price=over))
                if rng.random() < 0.8:                                   # one-sided quotes now and then
                    rows.append(dict(player=pl, market=market, sportsbook=book, side="Under", point=point, price=-over if over < 0 else -120))
    rows.append(dict(player="e solo", market="player_points", sportsbook="dk", side="Over", point=10.5, price=-110))   # one pair: below MIN_PAIRS
    rows.append(dict(player="a smith", market="player_turnovers", sportsbook="dk", side="Over", point=2.5, price=-110))  # not a market we read
    rows.append(dict(player="a smith", market="player_points", sportsbook="dk", side="Over", point=22.5, price=-110))    # a repeat row
    return pd.DataFrame(rows)


def test_vectorised_means_match_the_root_find_pair_by_pair():
    odds = _odds()
    new = M.market_means(odds, dispersion=K)
    old = _old_market_means(odds).sort_values(["player", "stat"]).reset_index(drop=True)
    assert list(new.columns) == ["player", "stat", "mean", "pairs", "books", "line"]
    assert len(new) == len(old) > 10
    assert (new.player.values == old.player.values).all() and (new.stat.values == old.stat.values).all()
    assert (new.pairs.values == old.pairs.values).all() and (new.books.values == old.books.values).all()
    assert np.allclose(new["mean"].values, old["mean"].values, atol=1e-9, rtol=0)
    assert np.allclose(new.line.values, old.line.values, equal_nan=True)
    assert "e solo" not in set(new.player)


def test_means_from_tails_matches_brentq_and_marks_unbracketed():
    pts = np.array([22.5, 6.5, 0.5, 2.5, 150.5])
    p = np.array([0.5, 0.3, 0.9, 0.55, 0.9])
    k = np.array([12.0, 6.0, 2.0, 3.0, 12.0])
    mu = M.means_from_tails(pts, p, k)
    for i in range(4):
        assert abs(mu[i] - _old_mean(pts[i], p[i], k[i])) < 1e-9
    assert _old_mean(pts[4], p[4], k[4]) is None and np.isnan(mu[4])       # P(X > 150) = 0.9 needs a mean above the bracket


def test_empty_and_unreadable_odds_give_an_empty_table():
    assert M.market_means(pd.DataFrame(columns=["player", "market", "sportsbook", "side", "point", "price"])).empty
    odds = pd.DataFrame([dict(player="x", market="player_turnovers", sportsbook="dk", side="Over", point=2.5, price=-110)])
    out = M.market_means(odds, dispersion=K)
    assert out.empty and list(out.columns) == ["player", "stat", "mean", "pairs", "books", "line"]
