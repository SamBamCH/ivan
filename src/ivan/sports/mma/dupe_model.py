# src/ivan/sports/mma/dupe_model.py
"""
How many copies of a lineup a contest's field holds, fitted on real fields,
with the contest in it.

Each scraped contest is laid on every valid lineup of its card (the
universe, zeros included). Within a contest an entry lands on lineup L with
probability proportional to

    exp( x(L) . (b + z x g) ),      z = log(entries per user / 20)

x(L) = salary left (per $100, flat past $2,000), an exact-cap flag, the
summed win probability, the log of the lineup's rank by summed win
probability among the card's lineups, and the number of main-event
fighters. b is the lineup's pull in a 20-max contest; g is how each pull
moves with the contest's entries-per-user cap (`multi_entry` on the contest
row): a single-entry crowd and a 150-max crowd do not build alike. Contest
size and entry fee were tested next to the cap and add nothing on a card
the fit has not seen (docs/mma/notes/contest_dupes_2026-11-14.md).

The level is not a parameter: every contest's probabilities sum to one, so
the fit is a multinomial per contest (a Poisson fit with a free intercept
per contest), by damped Newton steps.

`alpha` is the lumpiness the features do not explain: copies vary around
the expectation as a Gamma(alpha, 1/alpha) multiplier per lineup (variance
mu + mu^2 / alpha, a negative binomial). It is the maximum-likelihood value
on cards the coefficients were NOT fitted on (leave one card out, fields of
4,000 entries or more), because a live slate is always a new card and the
in-card figure is too smooth. On 2026-11-14 (three cards) that is 0.67, and
a field sampled with it has the real fields' unique-lineup share, top-100
share and largest lineup on the held-out cards; a moments estimate on the
well-covered lineups (1.4) left the top a fifth too thin.

The sampler (sample_field.fitted_weights) reads the model file through the
same feature definitions, so a field sampled for a contest and the copies
predicted for our own lineups in it agree.

    scripts/mma/derive_contests.py      scraped standings -> lineups per contest
    scripts/mma/build_dupe_model.py     fit, report, write data/mma/models/dupe_model.json
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

VERSION = 3
LEFT_CAP = 2000.0          # dollars left past which the salary effect stops growing
EMAX_REF = 20.0            # the entries-per-user cap the base coefficients describe; also assumed when a contest's cap is unknown
EMAX_MAX = 150.0
MIN_ENTRIES = 100          # contests smaller than this are left out of the fit
ALPHA_MIN_FIELD = 4000     # lumpiness is measured in fields at least this big (small fields hold almost no repeats)
ALPHA_GRID = np.exp(np.linspace(np.log(0.05), np.log(50.0), 70))
COEF = ("b_left_per_100", "b_zero_left", "b_win", "b_log_rank", "b_main_event")     # the names the sampler reads, in feature order


def entry_cap_z(entry_cap) -> float:
    """log(entries per user / 20), the contest covariate; 0 (the base) when the cap is unknown."""
    try:
        cap = float(entry_cap)
    except (TypeError, ValueError):
        cap = 0.0
    if not np.isfinite(cap) or cap <= 0:
        cap = EMAX_REF
    return float(np.log(min(max(cap, 1.0), EMAX_MAX) / EMAX_REF))


def features(salary_left, sum_win, rank_win, n_main, left_cap: float = LEFT_CAP) -> np.ndarray:
    """The lineup features, one row per lineup, columns in COEF order."""
    left = np.asarray(salary_left, dtype=float)
    return np.column_stack([np.minimum(left, left_cap) / 100.0, (left <= 0).astype(float), np.asarray(sum_win, dtype=float),
                            np.log(np.asarray(rank_win, dtype=float)), np.asarray(n_main, dtype=float)])


def contest_coef(model: dict, entry_cap) -> dict:
    """The model's coefficients for one contest: base + z x the entry-cap slopes."""
    z = entry_cap_z(entry_cap)
    slopes = model.get("coef_entry_cap") or {}
    return {k: float(v) + z * float(slopes.get(k, 0.0)) for k, v in model["coef"].items()}


# ------------------------------------------------------------------ the fit
def _mu(D: np.ndarray, beta: np.ndarray, codes: np.ndarray, n_contests: int, totals: np.ndarray) -> np.ndarray:
    """Expected copies: softmax of the linear predictor within each contest, times the contest's entries."""
    eta = D @ beta
    top = np.full(n_contests, -np.inf)
    np.maximum.at(top, codes, eta)
    w = np.exp(eta - top[codes])
    return w * (totals / np.bincount(codes, weights=w, minlength=n_contests))[codes]


def fit(D: np.ndarray, y: np.ndarray, codes: np.ndarray, iters: int = 80, ridge: float = 1e-6) -> np.ndarray:
    """
    Maximum likelihood for copies ~ Multinomial(entries, softmax(D beta))
    per contest (codes = contest index per row). Newton steps on the
    standardised design, halved until the likelihood rises.
    """
    mean, sd = D.mean(axis=0), D.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (D - mean) / sd
    nC = int(codes.max()) + 1
    totals = np.bincount(codes, weights=y, minlength=nC)
    beta = np.zeros(Z.shape[1])

    def loglik(mu: np.ndarray) -> float:
        return float(np.sum(y * np.log(np.maximum(mu, 1e-300))))

    mu = _mu(Z, beta, codes, nC, totals)
    cur = loglik(mu)
    for _ in range(iters):
        grad = Z.T @ (y - mu)
        Zm = Z * mu[:, None]
        H = Z.T @ Zm
        per = np.zeros((nC, Z.shape[1]))
        np.add.at(per, codes, Zm)
        H -= (per.T / np.maximum(totals, 1e-12)) @ per
        step = np.linalg.solve(H + ridge * (np.trace(H) / len(beta)) * np.eye(len(beta)), grad)
        t, better = 1.0, False
        while t > 1e-4:
            nb = beta + t * step
            nmu = _mu(Z, nb, codes, nC, totals)
            nl = loglik(nmu)
            if nl > cur:
                better = True
                break
            t /= 2
        if not better:
            break
        gain = nl - cur
        beta, mu, cur = nb, nmu, nl
        if gain < 1e-9 * abs(cur):
            break
    return beta / sd


def deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(t - (y - mu)))


def fit_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Gamma overdispersion by maximum likelihood: copies ~ negative binomial
    with mean mu and variance mu + mu^2 / alpha, alpha searched on
    ALPHA_GRID. The top of the grid (50) is as good as Poisson.
    """
    from scipy.special import gammaln
    y, mu = np.asarray(y, dtype=float), np.maximum(np.asarray(mu, dtype=float), 1e-300)
    if not len(y):
        return float(ALPHA_GRID[-1])
    pos = y > 0
    yp, mp = y[pos], mu[pos]

    def loglik(a: float) -> float:                      # the y = 0 rows only carry a log(a / (a + mu))
        return float(np.sum(a * np.log(a / (a + mu))) + np.sum(gammaln(yp + a) - gammaln(a) + yp * np.log(mp / (a + mp))))

    return float(ALPHA_GRID[int(np.argmax([loglik(float(a)) for a in ALPHA_GRID]))])


def design(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.hstack([X, X * z[:, None]])


# ------------------------------------------------------------------ training data
def card_frame(card: dict, salary_cap: int = 50000) -> pd.DataFrame:
    """Every valid lineup of a stored card (cards.load_card): key and the feature columns."""
    from ivan.sports.mma.universe import enumerate_lineups
    U = enumerate_lineups(card["pool"], card["win_prob"], salary_cap=salary_cap, main_event=card["main_event"])
    X = features(U.salary_left, U.sum_win, U.rank_win, U.n_main)
    return pd.DataFrame({"key": U.key, **{name: X[:, j] for j, name in enumerate(COEF)}})


def training_frame(contests: list[dict], cards: dict[int, dict], derived_dir: Path, min_entries: int = MIN_ENTRIES) -> pd.DataFrame:
    """
    One row per (contest, lineup of its card's universe): the features, the
    copies the real field held (0 for most), the contest's entries on the
    universe (`n`), its draft group and entry-cap covariate. contests:
    dicts with contest_id, draft_group_id, entry_cap. Entries on lineups
    outside the universe (two fighters from one fight) are counted in
    `outside` and left out.
    """
    frames = []
    universes: dict[int, pd.DataFrame] = {}
    for c in contests:
        dg, cid = int(c["draft_group_id"]), int(c["contest_id"])
        p = Path(derived_dir) / str(cid) / "lineups.parquet"
        if dg not in cards or not p.exists():
            continue
        g = pd.read_parquet(p)
        if int(g.copies.sum()) < min_entries:
            continue
        if dg not in universes:
            universes[dg] = card_frame(cards[dg])
        g["key"] = g[[f"f{j}" for j in range(1, 7)]].apply(lambda r: tuple(sorted(int(x) for x in r)), axis=1)
        m = universes[dg].merge(g[["key", "copies"]], on="key", how="left").fillna({"copies": 0.0}).drop(columns="key")
        m["contest_id"], m["draft_group"] = cid, dg
        m["n"] = float(m.copies.sum())
        m["outside"] = float(g.copies.sum() - m.copies.sum())
        m["entry_cap"] = float(c.get("entry_cap") or 0) or EMAX_REF
        m["z"] = entry_cap_z(c.get("entry_cap"))
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fit_frame(df: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(df.contest_id)
    return fit(design(df[list(COEF)].to_numpy(dtype=float), df.z.to_numpy(dtype=float)), df.copies.to_numpy(dtype=float), codes)


def _predict_frame(df: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(df.contest_id)
    nC = int(codes.max()) + 1
    return _mu(design(df[list(COEF)].to_numpy(dtype=float), df.z.to_numpy(dtype=float)), beta, codes, nC,
               np.bincount(codes, weights=df.copies.to_numpy(dtype=float), minlength=nC))


def cap_group(entry_cap: float, n: float) -> str:
    g = "single entry" if entry_cap <= 1 else "2-14 max" if entry_cap < 20 else "20 max" if entry_cap < 150 else "150 max"
    return g + (", under 700 entries" if n < 700 else "")


def shape(copies: np.ndarray) -> dict:
    """What a field looks like: unique lineups per entry, the share of entries on the 100 most copied lineups, the most copied."""
    n = float(copies.sum())
    return {"unique_share": float((copies > 0).sum() / n), "top100_share": float(np.sort(copies)[-100:].sum() / n), "max_copies": int(copies.max())}


def held_out(df: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict]:
    """
    Leave one card out: each card's contests predicted by a fit on the other
    cards. Returns df with `mu` (the held-out expectation) and, per card,
    the deviance explained and the lumpiness measured there. With one card
    there is nothing to hold out and the in-card fit stands in.
    """
    cards = sorted(df.draft_group.unique())
    out = df.copy()
    out["mu"] = np.nan
    per_card = {}
    for dg in cards:
        test = (df.draft_group == dg).to_numpy()
        train = df[~test] if len(cards) > 1 else df
        beta = _fit_frame(train)
        mu = _predict_frame(df[test], beta)
        out.loc[test, "mu"] = mu
        y = df.copies.to_numpy(dtype=float)[test]
        n = df.n.to_numpy()[test]
        codes = pd.factorize(df.contest_id[test])[0]
        null = n / np.bincount(codes)[codes]                   # every lineup of the card equally likely
        big = n >= ALPHA_MIN_FIELD
        per_card[int(dg)] = {"contests": int(df.contest_id[test].nunique()), "entries": int(y.sum()),
                             "deviance_explained": 1 - deviance(y, mu) / deviance(y, null),
                             "alpha": fit_alpha(y[big], mu[big]) if big.any() else None}
        log(f"  card {dg} held out: {per_card[int(dg)]['contests']} contests, {int(y.sum()):,} entries, "
            f"deviance explained {per_card[int(dg)]['deviance_explained']:.3f}, alpha {per_card[int(dg)]['alpha'] or float('nan'):.2f}")
    return out, per_card


def shape_report(out: pd.DataFrame, alpha: float, rng: np.random.Generator) -> pd.DataFrame:
    """Per contest group: the real field's shape against one field sampled from the held-out expectation, with and without the lumpiness."""
    rows = []
    for cid, g in out.groupby("contest_id"):
        y, mu, n = g.copies.to_numpy(dtype=float), g.mu.to_numpy(dtype=float), int(g.n.iloc[0])
        smooth = rng.multinomial(n, mu / mu.sum())
        w = mu * rng.gamma(alpha, 1.0 / alpha, size=len(mu)) if 0 < alpha < 1e6 else mu
        lumpy = rng.multinomial(n, w / w.sum())
        rows.append({"group": cap_group(float(g.entry_cap.iloc[0]), n), "entries": n,
                     **{f"real_{k}": v for k, v in shape(y).items()}, **{f"model_{k}": v for k, v in shape(lumpy).items()},
                     **{f"smooth_{k}": v for k, v in shape(smooth).items()}})
    r = pd.DataFrame(rows)
    agg = r.groupby("group").agg(contests=("entries", "size"), entries=("entries", "sum"),
                                 **{c: (c, "mean") for c in r.columns if c not in ("group", "entries")}).reset_index()
    return agg.sort_values("entries", ascending=False).reset_index(drop=True)


def build(contests: list[dict], cards: dict[int, dict], derived_dir: Path, min_entries: int = MIN_ENTRIES, seed: int = 0, log=print) -> Optional[dict]:
    """Fit on every derived contest whose card is stored; returns the model dict (None with no data)."""
    df = training_frame(contests, cards, derived_dir, min_entries)
    if df.empty:
        return None
    n_cards = int(df.draft_group.nunique())
    log(f"{df.contest_id.nunique()} contests on {n_cards} card(s), {int(df.copies.sum()):,} entries on {len(df):,} (contest, lineup) rows")
    out, per_card = held_out(df, log)
    y, mu, n = out.copies.to_numpy(dtype=float), out.mu.to_numpy(dtype=float), out.n.to_numpy()
    big = n >= ALPHA_MIN_FIELD
    alpha = fit_alpha(y[big], mu[big]) if big.any() else fit_alpha(y, mu)
    shapes = shape_report(out, alpha, np.random.default_rng(seed))
    beta = _fit_frame(df)
    fitted = _predict_frame(df, beta)
    k = len(COEF)
    codes = pd.factorize(df.contest_id)[0]
    null = (df.n.to_numpy() / np.bincount(codes)[codes])
    per_contest = df.groupby("contest_id").agg(n=("n", "first"), outside=("outside", "first"))
    return {"version": VERSION, "features": "universe + entry cap", "built_at": datetime.now().isoformat(timespec="seconds"),
            "cards": sorted(int(c) for c in df.draft_group.unique()), "contests": sorted(int(c) for c in df.contest_id.unique()),
            "entries": int(y.sum()), "left_cap": LEFT_CAP, "entry_cap_ref": EMAX_REF,
            "coef": dict(zip(COEF, map(float, beta[:k]))), "coef_entry_cap": dict(zip(COEF, map(float, beta[k:]))),
            "alpha": float(alpha), "alpha_source": "held-out cards" if n_cards > 1 else "in-card (one card only)",
            "same_fight_share": float(per_contest.outside.sum() / max(1.0, per_contest.outside.sum() + per_contest.n.sum())),
            "fit": {"deviance_explained": 1 - deviance(df.copies.to_numpy(dtype=float), fitted) / deviance(df.copies.to_numpy(dtype=float), null),
                    "held_out_by_card": per_card, "held_out_shape": shapes.round(4).to_dict("records")}}


def format_model(model: dict) -> str:
    lines = [f"{len(model['contests'])} contests on cards {model['cards']}, {model['entries']:,} entries; "
             f"deviance explained in-sample {model['fit']['deviance_explained']:.3f}",
             f"alpha {model['alpha']:.2f} ({model['alpha_source']}); entries on same-fight lineups (outside the model) {100 * model['same_fight_share']:.1f}%",
             f"{'coefficient':<16} {'20-max':>8} {'per log cap':>12} {'single':>8} {'150-max':>8}"]
    for name in COEF:
        b, g = model["coef"][name], model["coef_entry_cap"][name]
        lines.append(f"{name:<16} {b:>8.3f} {g:>12.3f} {b + entry_cap_z(1) * g:>8.3f} {b + entry_cap_z(150) * g:>8.3f}")
    lines.append("\nheld out by card (each card predicted by a fit on the others); real field vs one sampled field:")
    lines.append(f"{'contests':<32} {'n':>3} {'unique share':>15} {'top-100 share':>15} {'max copies':>13}   (real / model; smooth = no lumpiness)")
    for r in model["fit"]["held_out_shape"]:
        lines.append(f"{r['group']:<32} {r['contests']:>3} {r['real_unique_share']:>7.3f}/{r['model_unique_share']:<7.3f} "
                     f"{r['real_top100_share']:>7.3f}/{r['model_top100_share']:<7.3f} {r['real_max_copies']:>6.0f}/{r['model_max_copies']:<6.0f}"
                     f"   smooth {r['smooth_unique_share']:.3f} / {r['smooth_top100_share']:.3f} / {r['smooth_max_copies']:.0f}")
    return "\n".join(lines)
