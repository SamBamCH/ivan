# src/ivan/core/slate_book.py
"""
The slate as one book: every contest of the entries file chosen at once, in
dollars, from one bankroll. Shared by the sports; each sport's choose module
loads its own simulation rows into ContestCandidates and checks that its
contests were simulated on one shared draw (world w is the same games in
every contest), then calls choose_book.

The two inputs are the entries file (which contests, how many entries in
each, the fee) and the bankroll in dollars. Everything else follows:

  candidates   (contest, lineup) pairs. Each carries its net dollars in
               every world, fee x ROI, after the per-contest empirical-Bayes
               shrinkage (chooser.shrink_rows), so the book never ranks on
               a lineup's sampling noise.
  objective    mean over worlds of log(1 + book_w / bankroll), the Kelly
               objective for a season played one slate at a time. The log
               is what makes the picks a portfolio: a lineup that wins in
               worlds the book already wins adds little, so the picks spread
               across game scripts and across contests; the same lineup is
               worth a different vector in each contest (its payout table),
               which is how a steady casher ends up in the single-entry or
               flat-payout contests and a tail lineup in the top-heavy ones.
  selection    greedy on the objective (portfolio.pick_book: each step adds
               the pair with the largest utility gain among contests with a
               slot left; a lineup goes into a contest at most once) and
               then an exchange pass (portfolio.improve_book: one-for-one
               swaps that raise the utility, so an early pick can give way to
               a lineup it blocked and a lineup can move to the contest where
               it does more).
  honest check the slate cross-fit: select the same way on the even worlds,
               score on the odd ones, both halves on the raw rows, so the gap
               is selection alone.

choose_book returns, per contest, the picks and the stats the exports write,
and the slate book: its dollar distribution over the worlds, the exchange
pass's effect, the per-contest split with each contest's role (its picks'
cash rate, mean ROI, how many of its lineups the book also plays elsewhere),
and where every lineup went (placements).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from ivan.core.chooser import SampledDistances, fill_stats, quality_features, shrink_rows, viz_payload
from ivan.core.portfolio import book_summary, dollar_book_summary, improve_book, pick_book

METHOD = "slate"
DEFAULT_BANKROLL_DOLLARS = 1000.0
Log = Callable[[str], None]


def bankroll_dollars_of(weights: dict) -> float:
    b = float((weights or {}).get("bankroll_dollars") or DEFAULT_BANKROLL_DOLLARS)
    if b <= 0:
        raise ValueError("bankroll_dollars must be positive")
    return b


@dataclass
class ContestCandidates:
    """One contest's simulated candidates, as the sport's choose module loads them."""
    contest_id: int
    contest_name: str
    fee: float
    slots: int                                   # entries to fill, from the entries file
    lineup_ids: List[int]
    roi_raw: np.ndarray                          # N x W: each lineup's ROI in each world, unshrunk
    fpts: np.ndarray                             # N x W: simulated points (the map on the page)
    cash: np.ndarray                             # N: each lineup's cash probability
    counts: tuple = (0, 0, 0, 0, 0)              # (results, after_roi_filter, valid, unique, missing players)

    @property
    def n(self) -> int:
        return len(self.lineup_ids)


def choose_book(contests: Sequence[ContestCandidates], bankroll: float, weights: Optional[dict] = None, shared: Optional[dict] = None,
                log: Optional[Log] = None, improve: bool = True) -> dict:
    """
    Choose every contest as one book in dollars. Returns {"picks": [lineup
    ids per contest, in the contests' order], "stats": [the per-contest
    stats a choose run writes], "book": the slate book}. weights only name
    cash_pct / avg_roi_win for the quality shown beside the picks; `shared`
    is what the sport's world-axis check found (recorded in the book).
    """
    log = log or (lambda *_: None)
    weights = dict(weights or {})
    weights["method"] = METHOD
    bankroll = float(bankroll)
    if bankroll <= 0:
        raise ValueError("the bankroll must be positive")
    per = []
    for c in contests:
        live = c.n > 0 and c.roi_raw.size > 0
        roi_m, shrink = shrink_rows(c.roi_raw) if live else (np.zeros((0, 0), dtype=np.float32), {})
        raw, q, _, norm = quality_features(c.cash, roi_m, weights) if live else (np.zeros((0, 2)), np.zeros(0), None, (np.zeros(2), np.zeros(2)))
        per.append({"c": c, "live": live, "roi_m": roi_m, "shrink": shrink, "raw": raw, "q": q, "norm": norm})
    worlds = {p["roi_m"].shape[1] for p in per if p["live"]}
    if len(worlds) > 1:
        raise ValueError(f"the contests' sim results have different world counts {sorted(worlds)}; re-run the sim for the whole slate")
    W = worlds.pop() if worlds else 0

    # the slate stacked: (contest, lineup) pairs, net dollars per world
    blocks, groups, ids, qs, offsets = [], [], [], [], []
    for g, p in enumerate(per):
        offsets.append(sum(len(b) for b in blocks))
        if not p["live"]:
            continue
        blocks.append((p["roi_m"] * np.float32(p["c"].fee)).astype(np.float32))
        groups.append(np.full(p["c"].n, g, dtype=int))
        ids.extend(p["c"].lineup_ids)
        qs.append(p["q"])
    names = [c.contest_name for c in contests]
    if not blocks:
        return {"picks": [[] for _ in contests], "stats": [_empty_stats(c, weights) for c in contests], "book": None}
    net = np.vstack(blocks)
    group = np.concatenate(groups)
    q_all = np.concatenate(qs)
    slots = [p["c"].slots if p["live"] else 0 for p in per]
    fees_total = float(sum(p["c"].fee * p["c"].slots for p in per if p["live"]))
    log(f"  slate: {net.shape[0]:,} (contest, lineup) candidates x {W} shared worlds"
        + (f" ({shared.get('method')}, draw {shared.get('draw')})" if shared else "")
        + f"; {sum(slots)} entries, ${fees_total:,.2f} in fees, bankroll ${bankroll:,.0f}")
    selected, trace, diag = pick_book(net, bankroll, group, slots, q=q_all, ids=ids)
    greedy_summary = dollar_book_summary(net[selected], fees_total, bankroll)
    imp = {"passes": 0, "swaps": 0, "swapped": [], "utility_before": None, "utility_after": None}
    if improve and selected:
        selected, imp = improve_book(net, bankroll, group, selected)
        for s in imp["swapped"]:                           # the trace keeps the greedy walk; a swapped slot shows what replaced its pick
            t = trace[s["slot"]]
            t.update({"lineup_id": ids[s["in"]], "q": float(q_all[s["in"]]), "swapped_from": ids[s["out"]], "swap_gain": s["gain"]})
    if diag.get("clipped"):
        log(f"  WARNING: in {diag['clipped']} of {W} worlds the book loses more than the bankroll (the fees exceed it); utility clipped there")
    if imp["swaps"]:
        log(f"  exchange pass: {imp['swaps']} swap(s) in {imp['passes']} pass(es), utility {imp['utility_before']:.4f} -> {imp['utility_after']:.4f}")

    # quality-only comparison book: each contest's top entries by quality
    top_global = []
    for g, p in enumerate(per):
        if p["live"]:
            top_global.extend(offsets[g] + int(i) for i in np.argsort(-p["q"])[:p["c"].slots])
    chosen = dollar_book_summary(net[selected], fees_total, bankroll)
    cf = _cross_fit(per, bankroll, slots, fees_total, improve)
    book = {"method": METHOD, "bankroll_dollars": bankroll, "worlds": W, "shared": shared, "clipped_worlds": int(diag.get("clipped", 0)),
            "fees": fees_total, "entries": int(len(selected)), "chosen": chosen,
            "greedy": greedy_summary,
            "improve": {k: imp[k] for k in ("passes", "swaps", "utility_before", "utility_after")}
            | {"log_growth_before": greedy_summary["log_growth"] if greedy_summary else None, "log_growth_after": chosen["log_growth"] if chosen else None},
            "top_quality": dollar_book_summary(net[top_global], fees_total, bankroll) if top_global else None,
            "cross_fit": cf.get("slate") if cf else None, "contests": [], "placements": [], "lineups_distinct": 0, "lineups_reused": 0}

    # where every lineup went
    sel_set = set(selected)
    used: dict = {}
    for j in selected:
        used.setdefault(ids[j], []).append(names[int(group[j])])
    book["lineups_distinct"] = len(used)
    book["lineups_reused"] = sum(1 for v in used.values() if len(v) > 1)
    book["placements"] = sorted(({"lineup_id": lid, "contests": sorted(v), "n": len(v)} for lid, v in used.items()), key=lambda r: (-r["n"], r["lineup_id"]))

    picks, stats_by = [], []
    for g, p in enumerate(per):
        c = p["c"]
        stats: dict = {}
        if not p["live"]:
            picks.append([]); stats_by.append(_empty_stats(c, weights))
            continue
        N = c.n
        sel = [j - offsets[g] for j in selected if group[j] == g]
        top_q = [int(i) for i in np.argsort(-p["q"])[:c.slots]]
        trace_c = [{**t, "slate_pick": t["pick"], "pick": k + 1} for k, t in enumerate(t for t in trace if t["group"] == g)]
        dist = SampledDistances(np.asarray(c.fpts, dtype=float), keep=set(sel) | set(top_q))
        roi_m, roi_raw, raw = p["roi_m"], c.roi_raw, p["raw"]
        cf_c = (cf.get("contests") or {}).get(g) if cf else None
        book_c = {"method": METHOD, "bankroll": bankroll, "bankroll_dollars": bankroll, "shrinkage": p["shrink"],
                  "chosen": book_summary(roi_m[sel]) if sel else None, "top_quality": book_summary(roi_m[top_q]),
                  "chosen_unshrunk": book_summary(roi_raw[sel]) if sel else None, "cross_fit": cf_c}
        net_c = net[offsets[g]:offsets[g] + N]
        own_ids = [c.lineup_ids[i] for i in sel]
        book["contests"].append({"contest_id": c.contest_id, "contest_name": c.contest_name, "entries": len(sel), "fee": c.fee,
                                 "fees": c.fee * len(sel), "mean_net": float(net_c[sel].sum(axis=0).mean()) if sel else 0.0,
                                 "roi_mean": float(roi_m[sel].mean()) if sel else None,
                                 "cash_mean": float(np.asarray(c.cash, dtype=float)[sel].mean()) if sel else None,
                                 "p_loss": float((net_c[sel].sum(axis=0) < 0).mean()) if sel else None,
                                 "lineups_shared": sum(1 for lid in own_ids if len(used.get(lid, [])) > 1),
                                 "honest_roi": ((cf_c or {}).get("out_of_sample") or {}).get("mean")})
        fill_stats(stats, *c.counts[:3], N, raw[top_q], raw[sel], sel, weights, c.counts[4],
                   by_lineup={c.lineup_ids[i]: {"cash_pct": float(raw[i, 0]), "avg_roi": float(raw[i, 1])} for i in sel}, book=book_c, trace=trace_c)
        stats["viz"] = viz_payload(c.lineup_ids, raw, p["q"], dist, c.fpts, sel, top_q, trace_c, 0.0, weights, p["norm"], book_c, method=METHOD)
        stats["viz"]["slate"] = book
        picks.append(own_ids); stats_by.append(stats)
    del sel_set
    return {"picks": picks, "stats": stats_by, "book": book}


def _empty_stats(c: ContestCandidates, weights: dict) -> dict:
    stats: dict = {}
    fill_stats(stats, *c.counts[:3], 0, [], [], [], weights, c.counts[4])
    return stats


def _cross_fit(per: list, bankroll: float, slots: Sequence[int], fees_total: float, improve: bool) -> Optional[dict]:
    """
    The slate book's honest expectation: choose on the even worlds the way
    the real pass does (shrunk per contest, greedy then the exchange pass),
    score on the odd ones. Both halves are scored on the raw rows, so the gap
    is selection alone (the shrunk in-sample book is kept for reference).
    Returns the slate's in/out-of-sample dollar books and each contest's
    (keyed by its index) in/out-of-sample ROI, what validation reads.
    """
    live = [p for p in per if p["live"]]
    if not live:
        return None
    S = live[0]["c"].roi_raw.shape[1]
    if S < 20:
        return None
    fit = np.arange(S) % 2 == 0
    fit_blocks, raw_fit_blocks, score_blocks, groups, RA_by, RB_by = [], [], [], [], {}, {}
    for g, p in enumerate(per):
        if not p["live"]:
            continue
        roi_raw = p["c"].roi_raw
        RA, _ = shrink_rows(roi_raw[:, fit])
        RB = roi_raw[:, ~fit]
        RA_by[g], RB_by[g] = RA, RB
        fee = np.float32(p["c"].fee)
        fit_blocks.append((RA * fee).astype(np.float32))
        raw_fit_blocks.append((roi_raw[:, fit] * fee).astype(np.float32))
        score_blocks.append((RB * fee).astype(np.float32))
        groups.append(np.full(p["c"].n, g, dtype=int))
    NA, NB, group = np.vstack(fit_blocks), np.vstack(score_blocks), np.concatenate(groups)
    sel, _, _ = pick_book(NA, bankroll, group, slots)
    if improve and sel:
        sel, _ = improve_book(NA, bankroll, group, sel)
    sel = [int(i) for i in sel]
    ins = dollar_book_summary(np.vstack(raw_fit_blocks)[sel], fees_total, bankroll)
    ins_shrunk, oos = dollar_book_summary(NA[sel], fees_total, bankroll), dollar_book_summary(NB[sel], fees_total, bankroll)
    if ins is None or oos is None:
        return None
    out = {"slate": {"worlds_fit": int(fit.sum()), "worlds_score": int((~fit).sum()), "in_sample": ins, "in_sample_shrunk": ins_shrunk,
                     "out_of_sample": oos, "selection_bias": float(ins["mean"] - oos["mean"]),
                     "selection_bias_roi": float(ins["roi_mean"] - oos["roi_mean"])},
           "contests": {}}
    offset = 0
    for g, p in enumerate(per):
        if not p["live"]:
            continue
        N = p["c"].n
        sel_c = [j - offset for j in sel if group[j] == g]
        if sel_c:
            a, b = book_summary(RA_by[g][sel_c]), book_summary(RB_by[g][sel_c])
            out["contests"][g] = {"worlds_fit": int(fit.sum()), "worlds_score": int((~fit).sum()),
                                  "in_sample": a, "out_of_sample": b, "selection_bias": float(a["mean"] - b["mean"])}
        offset += N
    return out


def describe_book(book: Optional[dict]) -> List[str]:
    """The lines a choose run logs for the slate book (shared by the sports' exports)."""
    if not book or not book.get("chosen"):
        return []
    ch, tq, cf, imp = book["chosen"], book.get("top_quality") or {}, book.get("cross_fit"), book.get("improve") or {}
    lines = [f"  slate book: {book['entries']} entries, ${book['fees']:,.2f} in fees, {book['lineups_distinct']} distinct lineups "
             f"({book['lineups_reused']} played in more than one contest); mean net ${ch['mean']:+,.2f} ({ch['roi_mean']:+.0%}), "
             f"median ${ch['p50']:+,.2f}, p5 ${ch['p5']:+,.2f}, P(loss) {ch['p_loss']:.0%}, P(lose half) {ch['p_lose_half']:.0%}, "
             f"mean log growth {ch['log_growth']:+.4f}"
             + (f"  (quality alone: mean ${tq['mean']:+,.2f}, p5 ${tq['p5']:+,.2f}, log growth {tq['log_growth']:+.4f})" if tq else "")]
    if imp.get("swaps"):
        lines.append(f"  exchange pass: {imp['swaps']} swap(s), log growth {imp['log_growth_before']:+.4f} -> {imp['log_growth_after']:+.4f}")
    if cf:
        lines.append(f"  honest expectation (chosen on {cf['worlds_fit']} worlds, scored on the other {cf['worlds_score']}): "
                     f"mean net ${cf['out_of_sample']['mean']:+,.2f} ({cf['out_of_sample']['roi_mean']:+.0%}) vs "
                     f"${cf['in_sample']['mean']:+,.2f} in-sample; selection bias ${cf['selection_bias']:+,.2f}")
    for c in book.get("contests") or []:
        hr = c.get("honest_roi")
        lines.append(f"    {c['contest_name']}: {c['entries']} entries, mean net ${c['mean_net']:+,.2f}"
                     + (f", mean ROI {c['roi_mean']:+.0%}" if c.get("roi_mean") is not None else "")
                     + (f", cash {c['cash_mean']:.0%}" if c.get("cash_mean") is not None else "")
                     + (f", {c['lineups_shared']} of its lineups also elsewhere" if c.get("lineups_shared") else "")
                     + (f", honest {hr:+.0%}" if hr is not None else ""))
    return lines
