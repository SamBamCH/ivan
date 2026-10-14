# src/ivan/sports/nfl/backtest_repeats.py
"""
The distribution of a replayed window over repeats of the pipeline.

A replay with ReplayOptions.repeats > 1 runs generate -> simulate -> choose
several times per slate, each with its own seed, against the same archived
inputs and the same actual results. Settling every repeat gives the window's
profit as a distribution rather than one number:

  * across repeats  the pipeline's own noise - which lineups the generator
                    jittered to and which universes the sim drew - with the
                    real outcomes held fixed. Profit that only shows in some
                    repeats was luck of the draw, not edge.
  * slate bootstrap resampling the slates (with replacement) and, for each,
                    one repeat at random: the two sources of spread together.
  * leave-one-out   the window without its best slate, since a handful of
                    slates usually carries the result.

What this cannot do is sample the world: the actual scores are the ones that
happened, once. More slates in the archive are the only cure for that.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ivan.sports.nfl import backtest_tax as bt
from ivan.sports.nfl.tax import TaxConfig, compute_tax

PCTS = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------- settle every repeat
def repeat_runs(rep_dirs: Dict[int, Sequence[Path]]) -> Dict[int, List[bt.ChooseRun]]:
    """Per repeat index, the choose runs of its slate directories (see ivan.core.replay.repeat_dirs)."""
    return {int(k): bt.find_choose_runs([Path(d) for d in dirs]) for k, dirs in sorted(rep_dirs.items())}


def settle_repeats(session, runs_by_repeat: Dict[int, Sequence[bt.ChooseRun]], cfg: TaxConfig, *, in_field: str = "no",
                   since: Optional[datetime] = None, until: Optional[datetime] = None, tax_year: Optional[int] = None,
                   seeds: Optional[Dict[int, Optional[int]]] = None, echo=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Settle each repeat's picks against the actual results. Returns
      per_repeat  one row per repeat: entries, contests, slates, fees, W, L, P,
                  roi, cash_rate, sim_expected_profit, est_tax, after_tax_profit
      by_slate    one row per (repeat, slate): entries, fees, W, L, P, sim_expected_profit
      by_contest  one row per (repeat, slate, contest): entries, fees, W, L, P, cash_rate, sim_expected_profit
    """
    week_cache: dict = {}
    rows, slate_rows, contest_rows = [], [], []
    year = tax_year
    for k, runs in sorted(runs_by_repeat.items()):
        entries, cov = bt.assemble(session, list(runs), in_field=in_field, since=since, until=until, week_cache=week_cache)
        if year is None:
            years = pd.Series([e.tax_year for e in entries if e.tax_year]).value_counts()
            year = int(years.idxmax()) if len(years) else datetime.now().year
        settled = [e for e in entries if e.settled and (e.tax_year == year or e.tax_year is None)]
        fees = sum(e.entry_fee for e in settled)
        W = sum(e.winnings for e in settled)
        L = sum(e.losses for e in settled)
        t = compute_tax(W, L, cfg, fees)
        with_sim = [e for e in settled if e.sim_expected_payout is not None]
        sim_exp = sum(e.sim_expected_payout - e.entry_fee for e in with_sim)
        cash = float(np.mean([e.payout >= e.entry_fee for e in settled])) if settled else float("nan")
        rows.append({"repeat": k, "seed": (seeds or {}).get(k), "entries": len(settled),
                     "contests": len({e.contest_id for e in settled}), "slates": len({e.slate for e in settled}),
                     "unsettled_contests": len(cov.unsettled_contests),
                     "fees": fees, "W": W, "L": L, "P": W - L, "roi": (W - L) / fees if fees else float("nan"),
                     "cash_rate": cash, "sim_expected_profit": sim_exp if with_sim else float("nan"),
                     "sim_entries": len(with_sim), "deductible": t.deductible, "itemize": t.itemize,
                     "taxable_gambling": t.taxable_gambling, "phantom": t.phantom, "est_tax": t.est_tax,
                     "after_tax_profit": t.after_tax_profit, "after_tax_roi": t.after_tax_roi})
        st = bt.slate_totals(settled)
        for r in st.to_dict("records") if not st.empty else []:
            slate_rows.append({"repeat": k, "slate": r["slate"], "contests": r["contests"], "entries": r["entries"],
                               "fees": r["fees"], "W": r["W"], "L": r["L"], "P": r["P"],
                               "sim_expected_profit": r["sim_expected_payout"] - r["fees"] if r["sim_known"] else float("nan")})
        by_c: Dict[tuple, list] = {}
        for e in settled:
            by_c.setdefault((e.slate, e.contest_id), []).append(e)
        for (slate, cid), es in by_c.items():
            f = sum(e.entry_fee for e in es); w = sum(e.winnings for e in es); l = sum(e.losses for e in es)
            ws = [e for e in es if e.sim_expected_payout is not None]
            contest_rows.append({"repeat": k, "slate": slate, "contest_id": cid, "contest_name": getattr(es[0], "contest_name", ""),
                                 "entries": len(es), "fees": f, "W": w, "L": l, "P": w - l, "actual_profit": w - l,
                                 "cash_rate": float(np.mean([e.payout >= e.entry_fee for e in es])),
                                 "sim_expected_profit": sum(e.sim_expected_payout - e.entry_fee for e in ws) if ws else float("nan")})
        if echo:
            echo(f"  repeat {k:3d}: {len(settled):4d} entries  fees {fees:8.2f}  P {W - L:+9.2f}  after tax {t.after_tax_profit:+9.2f}"
                 + (f"  ({len(cov.unsettled_contests)} contests unsettled)" if cov.unsettled_contests else ""))
    cols = ["repeat", "seed", "entries", "contests", "slates", "unsettled_contests", "fees", "W", "L", "P", "roi", "cash_rate",
            "sim_expected_profit", "sim_entries", "deductible", "itemize", "taxable_gambling", "phantom", "est_tax",
            "after_tax_profit", "after_tax_roi"]
    per_repeat = pd.DataFrame(rows, columns=cols)
    by_slate = pd.DataFrame(slate_rows, columns=["repeat", "slate", "contests", "entries", "fees", "W", "L", "P", "sim_expected_profit"])
    by_contest = pd.DataFrame(contest_rows, columns=["repeat", "slate", "contest_id", "contest_name", "entries", "fees", "W", "L", "P",
                                                     "actual_profit", "cash_rate", "sim_expected_profit"])
    per_repeat.attrs["tax_year"] = year
    return per_repeat, by_slate, by_contest


# --------------------------------------------------------------------------- the distribution
def _dist(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    d = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
         "min": float(x.min()), "max": float(x.max()), "p_positive": float((x > 0).mean())}
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(x, p))
    return d


def slate_matrix(by_slate: pd.DataFrame) -> pd.DataFrame:
    """Slates x repeats of P; a slate a repeat did not settle counts 0 (it entered nothing it could settle)."""
    if by_slate.empty:
        return pd.DataFrame()
    return by_slate.pivot_table(index="slate", columns="repeat", values="P", aggfunc="sum").fillna(0.0).sort_index()


def summarize(per_repeat: pd.DataFrame, by_slate: pd.DataFrame, cfg: TaxConfig, *, n_boot: int = 4000, seed: int = 0) -> dict:
    """The window's profit as a distribution: across repeats, with a slate bootstrap on top, and without the best slate."""
    out = {"repeats": int(len(per_repeat)), "tax_year": per_repeat.attrs.get("tax_year"), "config": cfg.describe()}
    if per_repeat.empty:
        return out
    out["fees"] = _dist(per_repeat.fees.values)
    out["P"] = _dist(per_repeat.P.values)
    out["roi"] = _dist(per_repeat.roi.values)
    out["cash_rate"] = _dist(per_repeat.cash_rate.values)
    out["after_tax_profit"] = _dist(per_repeat.after_tax_profit.values)
    out["est_tax"] = _dist(per_repeat.est_tax.values)
    out["sim_expected_profit"] = _dist(per_repeat.sim_expected_profit.values)
    both = per_repeat.dropna(subset=["sim_expected_profit", "P"])
    if len(both) > 2 and both.sim_expected_profit.std() > 0 and both.P.std() > 0:
        out["corr_sim_expected_vs_actual"] = float(np.corrcoef(both.sim_expected_profit, both.P)[0, 1])
    else:
        out["corr_sim_expected_vs_actual"] = None

    mat = slate_matrix(by_slate)
    if mat.empty:
        return out
    M = mat.values                                    # slates x repeats
    n_slates, n_rep = M.shape
    per_slate = []
    for slate, row in zip(mat.index, M):
        d = _dist(row)
        fees = by_slate[by_slate.slate == slate].fees.mean()
        entries = by_slate[by_slate.slate == slate].entries.mean()
        sim = by_slate[by_slate.slate == slate].sim_expected_profit.mean()
        per_slate.append({"slate": slate, "entries": float(entries), "fees": float(fees), "mean_P": d["mean"], "sd_P": d["sd"],
                          "min_P": d["min"], "max_P": d["max"], "p_profit": d["p_positive"],
                          "sim_expected_profit": float(sim) if np.isfinite(sim) else None})
    out["per_slate"] = per_slate

    # the spread of the window across repeats vs across slates (on the per-repeat window totals)
    out["variance"] = {"pipeline_sd": out["P"]["sd"],
                       "slate_sum_sd": float(np.sqrt((M.std(axis=1, ddof=1) ** 2).sum())) if n_rep > 1 else 0.0,
                       "between_slate_sd_of_means": float(M.mean(axis=1).std(ddof=1)) if n_slates > 1 else 0.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_slates, size=(n_boot, n_slates))
    jdx = rng.integers(0, n_rep, size=(n_boot, n_slates))
    boot = M[idx, jdx].sum(axis=1)
    out["slate_bootstrap"] = {**_dist(boot), "n_boot": int(n_boot)}
    fees_per_slate = np.array([by_slate[by_slate.slate == s].fees.mean() for s in mat.index])
    boot_fees = fees_per_slate[idx].sum(axis=1)
    out["slate_bootstrap"]["roi_mean"] = float((boot / np.where(boot_fees > 0, boot_fees, np.nan)).mean()) if boot_fees.any() else None

    best = int(np.argmax(M.mean(axis=1)))
    rest = np.delete(M, best, axis=0).sum(axis=0)
    out["without_best_slate"] = {"slate": str(mat.index[best]), "best_mean_P": float(M[best].mean()), **_dist(rest)}
    loo = []
    for i in range(n_slates):
        r = np.delete(M, i, axis=0).sum(axis=0)
        loo.append({"slate": str(mat.index[i]), "mean_P": float(r.mean()), "p_profit": float((r > 0).mean())})
    out["leave_one_out"] = loo
    return out


# --------------------------------------------------------------------------- the report
def money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _line(label: str, d: dict, fmt=money) -> str:
    if not d or d.get("n", 0) == 0:
        return f"{label:18s} -"
    return (f"{label:18s} mean {fmt(d['mean']):>11s}   sd {fmt(d['sd']):>10s}   p5 {fmt(d['p5']):>11s}   median {fmt(d['p50']):>11s}"
            f"   p95 {fmt(d['p95']):>11s}   min {fmt(d['min']):>11s}   max {fmt(d['max']):>11s}   >0 in {d['p_positive']:.0%}")


def report_repeats(session, rep_dirs: Dict[int, Sequence[Path]], cfg: TaxConfig, *, out: Path, in_field: str = "no",
                   since: Optional[datetime] = None, until: Optional[datetime] = None, tax_year: Optional[int] = None,
                   n_boot: int = 4000, seed: int = 0, plot: bool = True, title: str = "", echo=print,
                   verbose: bool = False) -> dict:
    """Settle every repeat, write repeats.csv / repeats_by_slate.csv / repeats_summary.json (+ repeats.png) under out, print the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = repeat_runs(rep_dirs)
    seeds = {}
    for k, dirs in rep_dirs.items():
        for d in dirs:
            mp = Path(d) / "replay.json"
            if not mp.exists():
                mp = Path(d).parent / "replay.json"
            if mp.exists():
                for r in json.loads(mp.read_text()).get("repeats") or []:
                    if int(r.get("k", -1)) == int(k):
                        seeds[int(k)] = r.get("seed")
                break
    if title:
        echo(f"== {title}")
    echo(f"Settling {len(runs)} repeat(s) of {max((len(v) for v in rep_dirs.values()), default=0)} slate(s) ...")
    per_repeat, by_slate, by_contest = settle_repeats(session, runs, cfg, in_field=in_field, since=since, until=until, tax_year=tax_year,
                                                      seeds=seeds, echo=echo if verbose else None)
    per_repeat.to_csv(out / "repeats.csv", index=False)
    by_slate.to_csv(out / "repeats_by_slate.csv", index=False)
    by_contest.to_csv(out / "repeats_by_contest.csv", index=False)
    summ = summarize(per_repeat, by_slate, cfg, n_boot=n_boot, seed=seed)
    summ["title"] = title
    (out / "repeats_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    if per_repeat.empty:
        echo("No repeat settled anything.")
        return summ

    pct = lambda x: f"{x:+.1%}" if x is not None and np.isfinite(x) else "-"
    echo(f"\n== Distribution over {summ['repeats']} repeats of the pipeline (tax year {summ['tax_year']}; {cfg.describe()})")
    echo(f"{'entries':18s} {per_repeat.entries.min()}..{per_repeat.entries.max()} per repeat, fees {money(per_repeat.fees.mean())} per repeat")
    echo(_line("pre-tax P", summ["P"]))
    echo(_line("after-tax P", summ["after_tax_profit"]))
    echo(_line("est tax", summ["est_tax"]))
    echo(_line("ROI", summ["roi"], pct))
    echo(_line("cash rate", summ["cash_rate"], lambda x: f"{x:.1%}"))
    echo(_line("sim expected P", summ["sim_expected_profit"]))
    c = summ.get("corr_sim_expected_vs_actual")
    echo(f"{'sim vs actual':18s} corr(sim expected P, actual P) across repeats = " + (f"{c:+.2f}" if c is not None else "-")
         + "   (near 0: the sim's preference between its own picks carries no information about the realised result)")
    if "slate_bootstrap" in summ:
        b = summ["slate_bootstrap"]
        echo(f"\n== Slates resampled x repeats ({b['n_boot']} draws: which slates were played, and which repeat of each)")
        echo(_line("pre-tax P", b))
        v = summ["variance"]
        echo(f"{'spread':18s} window sd across repeats {money(v['pipeline_sd'])}   sd of per-slate means across slates {money(v['between_slate_sd_of_means'])}")
        w = summ["without_best_slate"]
        echo(f"\n== Without the best slate ({w['slate']}, mean P {money(w['best_mean_P'])})")
        echo(_line("pre-tax P", w))
        echo("\n== Per slate (P across repeats)")
        ps = pd.DataFrame(summ["per_slate"])
        show = ps.copy()
        for col in ("fees", "mean_P", "sd_P", "min_P", "max_P", "sim_expected_profit"):
            show[col] = show[col].map(money)
        show["entries"] = show.entries.map(lambda x: f"{x:.0f}")
        show["p_profit"] = show.p_profit.map(lambda x: f"{x:.0%}")
        echo(show.to_string(index=False))
        echo("\n== Leave one slate out (window mean P and share of repeats > 0 without it)")
        echo("   ".join(f"{r['slate'][:16]} {money(r['mean_P'])} ({r['p_profit']:.0%})" for r in summ["leave_one_out"]))
        if plot:
            try:
                plot_repeats(per_repeat, by_slate, summ, out / "repeats.png")
            except Exception as e:                      # a plot must never sink the report
                echo(f"(plot skipped: {e})")
    echo(f"\nWrote {out}/ (repeats.csv, repeats_by_slate.csv, repeats_summary.json" + (", repeats.png" if plot else "") + ")")
    return summ


def plot_repeats(per_repeat: pd.DataFrame, by_slate: pd.DataFrame, summ: dict, path: Path) -> None:
    """Two panels: the window's pre-tax and after-tax profit over repeats, and each slate's profit over repeats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = slate_matrix(by_slate)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    bins = min(40, max(10, len(per_repeat) // 5))
    ax1.hist(per_repeat.P, bins=bins, color="#4C72B0", alpha=0.85, label="pre-tax")
    ax1.hist(per_repeat.after_tax_profit, bins=bins, color="#DD8452", alpha=0.6, label="after tax")
    ax1.axvline(0, color="black", lw=1)
    ax1.axvline(summ["P"]["mean"], color="#4C72B0", ls="--", lw=1)
    ax1.set_title(f"window profit over {len(per_repeat)} repeats  (>0 in {summ['P']['p_positive']:.0%} pre-tax, "
                  f"{summ['after_tax_profit']['p_positive']:.0%} after tax)")
    ax1.set_xlabel("profit ($)")
    ax1.set_ylabel("repeats")
    ax1.legend(frameon=False)
    if not mat.empty:
        data = [row for row in mat.values]
        ax2.boxplot(data, vert=False, labels=[s[:16] for s in mat.index], showfliers=True, widths=0.6)
        ax2.axvline(0, color="black", lw=1)
        ax2.set_title("per-slate profit over repeats")
        ax2.set_xlabel("profit ($)")
        ax2.invert_yaxis()
    fig.suptitle(summ.get("title") or "", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
