# src/ivan/sports/nba/contest/export.py
"""
The choose step: for every contest in a DraftKings entries file, pick lineups
from its simulation results (choose.choose_lineup_ids) and write them the way
DraftKings wants them uploaded, plus what a web UI and a backtest read:

    <base>.classic.<stamp>.csv       the upload file (entry rows, PG..UTIL filled with "Name (id)")
    <base>.exposures.<stamp>.csv     player exposure across the chosen entries
    <base>.lineups.<stamp>.csv       one row per player per chosen entry, with the lineup's sim cash % / ROI
    <base>.choose.<stamp>.json       what the chooser saw per contest (counts, book, greedy trace; the slate book under each contest it chose)
    <base>.simroi.<stamp>.npz        each pick's per-simulation ROI, for the calibration tests

read_entries() parses the entries file, weights_resolver() decides the
per-contest weights (file / flags / prompt; contest_weights() holds the ones set on
the desk's contests section), choose_entries() does the picking
and slot assignment, write_outputs() the files. scripts/nba/choose.py is
argparse around run_choose().
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ivan.core.slate_book import describe_book
from ivan.core.utils import coerce_player_ids
from ivan.sports.nba.contest.choose import DEFAULT_WEIGHTS, METHODS, choose_lineup_ids, choose_slate, sim_roi_traces
from ivan.sports.nba.models import Lineup, NBAPlayer
from ivan.sports.nba import slates as S
from ivan.sports.nba.field.solver import SLOT_ELIGIBILITY, SLOT_NAMES
from ivan.sports.nba.utils import _coerce_positions

Log = Callable[[str], None]
BASE_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
POSITION_ORDER = list(SLOT_NAMES)          # PG, SG, SF, PF, C, G, F, UTIL
WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS)


# ------------------------------------------------------------------------------ entries file
def parse_money(x) -> float:
    """'$0.50' -> 0.5"""
    try:
        return float(str(x).replace("$", "").replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def read_entries(path) -> pd.DataFrame:
    """
    The DraftKings entries export: the first twelve columns only (DK puts an
    instructions block and a ragged player table to their right), cut at the
    first blank Entry ID, Contest ID as int.
    """
    df = pd.read_csv(path, engine="python", header=0, usecols=range(len(BASE_COLS) + len(POSITION_ORDER)), dtype=str, keep_default_na=False)
    for col in BASE_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Expected column '{col}' in entries CSV but not found.")
    blank = df["Entry ID"].str.strip() == ""
    if blank.any():
        df = df.loc[: blank.idxmax() - 1].copy()
    df = df[[c for c in df.columns if c in BASE_COLS or c in POSITION_ORDER]].copy()
    df["Contest ID"] = df["Contest ID"].str.strip().astype(int)
    return df


# ------------------------------------------------------------------------------ weights
Prompter = Callable[[int, str, Dict[str, float]], Dict[str, float]]


def weights_resolver(flags: Optional[dict] = None, overrides: Optional[dict] = None, no_prompt: bool = False,
                     prompter: Optional[Prompter] = None, defaults: Optional[dict] = None):
    """
    Per-contest weight resolution, highest priority first:
      overrides[<contest id or name>]  >  overrides["default"]  >  flags  >  prompter(cid, name, last)
    Any flag, any override, no_prompt, or no prompter makes the run non-interactive (defaults fill in).
    """
    defaults = {**DEFAULT_WEIGHTS, **(defaults or {})}
    flags = {k: v for k, v in (flags or {}).items() if v is not None}
    overrides = overrides or {}
    non_interactive = no_prompt or bool(flags) or bool(overrides) or prompter is None

    def weights_for(cid: int, cname: str, last: Dict[str, float]) -> Dict[str, float]:
        per = overrides.get(str(cid)) or overrides.get(cname) or {}
        if per or non_interactive:
            return {**defaults, **flags, **overrides.get("default", {}), **per}
        return prompter(cid, cname, last)

    return weights_for


def load_weights_json(raw: Optional[str]) -> dict:
    """--weights-json: a file path or a JSON literal."""
    if not raw:
        return {}
    return json.loads(Path(raw).read_text() if Path(raw).exists() else raw)


# per-contest weights set on the desk's contests section (scripts/nba/contest_weights.py): {"<contest id>": {knob: value}};
# the desk's choose job passes the file as --weights-json, so a contest's entry wins over the config panel's knobs
def contest_weights_json() -> Path:
    """The slate's per-contest chooser knobs (slates/<dg>/input/contest_weights.json)."""
    return S.contest_weights_json()
CONTEST_KNOBS = {"method": str, "cash_pct": float, "avg_roi_win": float, "diversity_lambda": float, "bankroll": float, "bankroll_dollars": float}


def contest_weights(path: Optional[Path] = None) -> Dict[str, dict]:
    try:
        return {str(k): v for k, v in json.loads((path or contest_weights_json()).read_text()).items() if v}
    except (OSError, ValueError):
        return {}


def set_contest_weights(contest_id: int, knobs: dict, path: Optional[Path] = None) -> dict:
    """Set one contest's knobs (blank / None drops a knob back to the panel's value; no knobs left drops the contest)."""
    path = path or contest_weights_json()
    allw = contest_weights(path)
    cur = dict(allw.get(str(int(contest_id)), {}))
    for k, v in knobs.items():
        if k not in CONTEST_KNOBS:
            raise ValueError(f"unknown knob {k!r}: one of {', '.join(CONTEST_KNOBS)}")
        v = None if v is None else str(v).strip()
        if not v:
            cur.pop(k, None)
        elif k == "method":
            if v not in METHODS:
                raise ValueError(f"method: {v!r} is not one of {', '.join(METHODS)}")
            cur[k] = v
        else:
            try:
                cur[k] = float(v)
            except ValueError:
                raise ValueError(f"{k}: {v!r} is not a number")
    if cur:
        allw[str(int(contest_id))] = cur
    else:
        allw.pop(str(int(contest_id)), None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(allw, indent=1, sort_keys=True))
    return cur


# ------------------------------------------------------------------------------ slot assignment
def positions_of(p) -> Tuple[str, ...]:
    return tuple(_coerce_positions(getattr(p, "position", None) or []))


def assign_player_order_classic(player_objs: list, prefer_late_in_flex: bool = True) -> list:
    """
    The 8 players in DraftKings slot order (PG, SG, SF, PF, C, G, F, UTIL) by
    backtracking over each player's eligible slots. With prefer_late_in_flex
    the earliest games fill the fixed slots first, so the flexible slots hold
    the late games (what a swap before their tip would touch).
    """
    if len(player_objs) != len(POSITION_ORDER):
        raise ValueError(f"expected {len(POSITION_ORDER)} players, got {len(player_objs)}")
    if prefer_late_in_flex:
        def key(p):
            gt = getattr(p, "gametime", None)
            return gt.timestamp() if gt is not None and hasattr(gt, "timestamp") else 0
        player_objs = sorted(player_objs, key=key)
    pos = {id(p): positions_of(p) for p in player_objs}

    def backtrack(slot_idx: int, assignment: list, remaining: list):
        if slot_idx == len(POSITION_ORDER):
            return assignment
        allowed = SLOT_ELIGIBILITY[POSITION_ORDER[slot_idx]]
        for i, p in enumerate(remaining):
            if any(x in allowed for x in pos[id(p)]):
                r = backtrack(slot_idx + 1, assignment + [p], remaining[:i] + remaining[i + 1:])
                if r is not None:
                    return r
        return None

    out = backtrack(0, [], list(player_objs))
    if out is None:
        raise ValueError("no valid slot assignment for " + ", ".join(f"{getattr(p, 'name', p.id)} ({'/'.join(pos[id(p)])})" for p in player_objs))
    return out


# ------------------------------------------------------------------------------ the run
@dataclass
class ChooseRun:
    upload: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BASE_COLS + POSITION_ORDER))
    counts: Counter = field(default_factory=Counter)           # player_id -> entries
    info: Dict[int, dict] = field(default_factory=dict)
    total: int = 0
    lineup_rows: List[dict] = field(default_factory=list)      # one row per player per chosen entry
    choose_viz: Dict[str, dict] = field(default_factory=dict)  # per contest, what the chooser saw
    sim_rows: List[tuple] = field(default_factory=list)        # (entry_id, contest_id, lineup_id, fee, roi trace)
    contests: List[dict] = field(default_factory=list)
    slate_book: Optional[dict] = None                           # the slate method's book (dollar distribution, split, placements, cross-fit)

    def exposures(self) -> pd.DataFrame:
        """Exposure as a fraction of the chosen entries, one row per player."""
        if not self.total or not self.counts:
            return pd.DataFrame(columns=["player_id", "name", "team", "position", "salary", "exposure", "projection"])
        df = pd.DataFrame([{"player_id": pid, **self.info[pid], "exposure": n / self.total} for pid, n in self.counts.items()])
        return df[["player_id", "name", "team", "position", "salary", "exposure", "projection"]].sort_values("exposure", ascending=False)


def _name(p) -> str:
    return getattr(p, "name", "") or ""


def _claims_by_lineup(session, by_lineup: dict) -> Dict[str, dict]:
    """{lineup id: {players: DK ids, cash_pct, avg_roi}}: each chosen lineup's simulated claim with its players, so the slate's
    backtest (slate_backtest) can set every entry's result beside the claim made for it after the lineup rows are gone."""
    from ivan.sports.nba.models import Lineup
    ids = [int(i) for i in by_lineup]
    if not ids:
        return {}
    players = {int(l.id): [int(x) for x in (l.player_ids_list or [])] for l in session.query(Lineup).filter(Lineup.id.in_(ids)).all()}
    return {str(i): dict(players=sorted(players.get(int(i), [])), cash_pct=v.get("cash_pct"), avg_roi=v.get("avg_roi")) for i, v in by_lineup.items()}


def choose_entries(session, entries: pd.DataFrame, weights_for, log: Log = print) -> ChooseRun:
    """Pick and slot-assign lineups for every contest in the entries frame (read_entries)."""
    run = ChooseRun()
    last = dict(DEFAULT_WEIGHTS)
    groups = [(int(cid), group) for cid, group in entries.groupby("Contest ID", sort=False)]
    weights_by: Dict[int, Dict[str, float]] = {}
    for cid, group in groups:                              # every contest's weights first: the slate method needs them all
        last = weights_for(cid, group["Contest Name"].iloc[0].strip(), last)
        weights_by[cid] = last

    # the slate method: one pass over every contest that asks for it, as one book in dollars (ivan.core.slate_book)
    slate_contests = [{"contest_id": cid, "contest_name": group["Contest Name"].iloc[0].strip(), "entries": len(group),
                       "fee": parse_money(group["Entry Fee"].iloc[0])}
                      for cid, group in groups if str(weights_by[cid].get("method") or "greedy") == "slate"]
    slate = None
    if slate_contests:
        w0 = weights_by[slate_contests[0]["contest_id"]]
        banks = {float(weights_by[c["contest_id"]].get("bankroll_dollars") or DEFAULT_WEIGHTS["bankroll_dollars"]) for c in slate_contests}
        if len(banks) > 1:
            log(f"  NOTE: the slate contests carry different bankroll_dollars {sorted(banks)}; the slate uses ${max(banks):,.0f}")
        log(f"\n=== Slate: {len(slate_contests)} contest(s) chosen as one book (bankroll ${max(banks):,.0f}) ===")
        slate = choose_slate(session, slate_contests, {**w0, "bankroll_dollars": max(banks)}, log=log)
        run.slate_book = slate.get("book")
        for line in describe_book(run.slate_book):
            log(line)

    for cid, group in groups:
        n_entries = len(group)
        cname = group["Contest Name"].iloc[0].strip()
        log(f"\n=== Contest {cid} \"{cname}\" -> {n_entries} entries ===")
        w = weights_by[cid]
        method = str(w.get("method") or "greedy")
        log(f"  method: {method}" + (f" (bankroll {w.get('bankroll', 3.0)}x fees)" if method == "portfolio" else "")
            + (f" (slate book, bankroll ${float(w.get('bankroll_dollars') or DEFAULT_WEIGHTS['bankroll_dollars']):,.0f})" if method == "slate" else "")
            + f"  weights: cash_pct={w['cash_pct']} avg_roi_win={w['avg_roi_win']} diversity_lambda={w['diversity_lambda']}")
        stats: dict = {}
        if method == "slate" and slate is not None:
            chosen_ids = list(slate["picks"].get(cid) or [])
            stats = dict(slate["stats"].get(cid) or {})
        else:
            chosen_ids = choose_lineup_ids(session, w, n_entries, cid, stats=stats)
        if stats:
            log(f"  candidates: {stats['results']:,} simulated"
                + (f" -> {stats['after_roi_filter']:,} with ROI >= 0" if stats['after_roi_filter'] != stats['results'] else "")
                + (f" -> {stats['valid']:,} with all players still on the slate" if stats['valid'] != stats['after_roi_filter'] else "")
                + f" -> {stats['unique']:,} unique")
            if stats["missing_players"]:
                log(f"  WARNING: {stats['missing_players']} player id(s) in simulated lineups are no longer on the slate")
        run.contests.append({"contest_id": cid, "contest_name": cname, "entries": n_entries, "weights": w,
                             **{k: stats.get(k) for k in ("results", "after_roi_filter", "valid", "unique", "chosen", "chosen_roi", "chosen_cash",
                                                          "top_quality_roi", "top_quality_cash", "missing_players")}})
        if not chosen_ids:
            log(f"  WARNING: no lineups chosen for contest {cid}; skipping")
            continue
        run.choose_viz[str(cid)] = {"contest_name": cname, "entries": n_entries, "book": stats.get("book"), "trace": stats.get("trace"),
                                    **{k: stats.get(k) for k in ("results", "unique", "chosen", "chosen_roi", "chosen_cash", "top_quality_roi", "top_quality_cash")},
                                    "by_lineup": _claims_by_lineup(session, stats.get("by_lineup") or {}),
                                    **({"slate": run.slate_book} if method == "slate" and run.slate_book else {})}     # the slate book, under each of its contests (the json's keys stay contest ids)
        log(f"  chosen {stats['chosen']}: mean ROI {stats['chosen_roi']:+.1%}, cash {stats['chosen_cash']:.1%}"
            f"  (top-{stats['chosen']} by quality alone: ROI {stats['top_quality_roi']:+.1%}, cash {stats['top_quality_cash']:.1%})")
        book = stats.get("book") or {}
        if book.get("chosen"):
            b, t = book["chosen"], book.get("top_quality") or {}
            log(f"  book over {b['worlds']} sims: median ROI {b['p50']:+.0%}, p5 {b['p5']:+.0%}, P(loss) {b['p_loss']:.0%}, P(lose half) {b['p_lose_half']:.0%}"
                + (f"  (quality alone: median {t['p50']:+.0%}, P(lose half) {t['p_lose_half']:.0%})" if t else ""))
            cf = book.get("cross_fit")
            if cf:
                log(f"  honest expectation (chosen on {cf['worlds_fit']} sims, scored on the other {cf['worlds_score']}): mean ROI {cf['out_of_sample']['mean']:+.0%} "
                    f"vs {cf['in_sample']['mean']:+.0%} in-sample; selection bias {cf['selection_bias']:+.0%}")
        traces = sim_roi_traces(session, cid, chosen_ids)
        lineups = {L.id: L for L in session.query(Lineup).filter(Lineup.id.in_(chosen_ids)).all()}
        all_pids = {pid for L in lineups.values() for pid in coerce_player_ids(L.player_ids_list)}
        players = {p.id: p for p in session.query(NBAPlayer).filter(NBAPlayer.id.in_(list(all_pids))).all()}
        by_lineup = stats.get("by_lineup", {})
        rows = group.reset_index(drop=True)
        for i, lid in enumerate(chosen_ids):
            lineup = lineups.get(lid)
            if not lineup:
                continue
            pids = coerce_player_ids(lineup.player_ids_list)
            objs = [players[pid] for pid in pids if pid in players]
            if len(objs) != len(POSITION_ORDER):
                log(f"  WARNING: lineup {lid} has {len(objs)} players on the slate; skipped")
                continue
            base = rows.iloc[i % n_entries][BASE_COLS].to_list()
            run.sim_rows.append((int(base[0]), cid, int(lid), parse_money(base[3]), traces.get(lid)))
            assigned = assign_player_order_classic(objs)
            run.upload.loc[len(run.upload)] = base + [f"{_name(pl)} ({pl.id})" for pl in assigned]
            run.total += 1
            sim = by_lineup.get(lid, {})
            for k, (slot, pl) in enumerate(zip(POSITION_ORDER, assigned)):
                run.counts[pl.id] += 1
                run.info[pl.id] = {"name": _name(pl), "team": pl.team, "position": pl.position, "salary": pl.salary, "projection": pl.fpts_mean}
                run.lineup_rows.append({"entry_id": base[0], "contest_name": base[1], "contest_id": base[2], "entry_fee": base[3], "format": "classic",
                                        "lineup_id": lid, "slot_no": k, "slot": slot, "player_id": pl.id, "name": _name(pl), "team": pl.team,
                                        "opponent": pl.opponent, "position": pl.position, "salary": pl.salary, "projection": pl.fpts_mean,
                                        "minutes_proj": pl.minutes_proj, "cash_pct": sim.get("cash_pct"), "avg_roi": sim.get("avg_roi")})
    return run


# ------------------------------------------------------------------------------ outputs
def format_exposures(df: pd.DataFrame, header: str, top_n: int = 30) -> str:
    if df is None or df.empty:
        return ""
    lines = [f"\n{header}", f"  {'player':28s} {'team':4s} {'pos':6s} {'salary':>6s} {'proj':>6s} {'exp':>6s}"]
    for _, r in df.head(top_n).iterrows():
        proj = f"{r['projection']:.1f}" if r.get("projection") is not None and pd.notna(r.get("projection")) else "-"
        lines.append(f"  {str(r['name'])[:28]:28s} {str(r['team'])[:4]:4s} {str(r['position'])[:6]:6s} {int(r['salary'] or 0):6d} {proj:>6s} {100 * r['exposure']:5.1f}%")
    return "\n".join(lines)


def write_outputs(run: ChooseRun, output_csv, stamp: Optional[str] = None, log: Log = print) -> Dict[str, str]:
    """Write the upload CSV and the exposures / lineups / choose / simroi files next to it. Returns {kind: path}."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    root, ext = os.path.splitext(str(output_csv))
    Path(root).parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    if not run.upload.empty:
        paths["classic"] = f"{root}.classic.{stamp}{ext}"
        run.upload.to_csv(paths["classic"], index=False)
        log(f"wrote {len(run.upload)} classic rows -> {paths['classic']}")
    else:
        log("WARNING: no rows written; no valid chosen lineups were found")
    exp = run.exposures()
    if len(exp):
        paths["exposures"] = f"{root}.exposures.{stamp}{ext}"
        exp.assign(format="classic").to_csv(paths["exposures"], index=False)
    if run.lineup_rows:
        paths["lineups"] = f"{root}.lineups.{stamp}{ext}"
        pd.DataFrame(run.lineup_rows).to_csv(paths["lineups"], index=False)
    if run.choose_viz:
        paths["choose"] = f"{root}.choose.{stamp}.json"
        Path(paths["choose"]).write_text(json.dumps(run.choose_viz, default=_json_default))
    if run.sim_rows:
        n_sims = max((len(t) for *_, t in run.sim_rows if t is not None), default=0)
        roi = np.full((len(run.sim_rows), n_sims), np.nan, dtype=np.float32)
        for k, (*_, t) in enumerate(run.sim_rows):
            if t is not None:
                roi[k, :len(t)] = t[:n_sims]
        paths["simroi"] = f"{root}.simroi.{stamp}.npz"
        np.savez_compressed(paths["simroi"], entry_id=np.array([r[0] for r in run.sim_rows], dtype=np.int64),
                            contest_id=np.array([r[1] for r in run.sim_rows], dtype=np.int64),
                            lineup_id=np.array([r[2] for r in run.sim_rows], dtype=np.int64),
                            entry_fee=np.array([r[3] for r in run.sim_rows], dtype=np.float64), roi=roi)
        log(f"wrote per-simulation ROI ({len(run.sim_rows)} entries x {n_sims} sims) -> {paths['simroi']}")
    for kind in ("exposures", "lineups", "choose"):
        if kind in paths:
            log(f"wrote {kind} -> {paths[kind]}")
    return paths


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_choose(session, entries_path, output_csv, weights_for, log: Log = print, stamp: Optional[str] = None) -> Tuple[ChooseRun, Dict[str, str]]:
    """read_entries -> choose_entries -> write_outputs; returns the run and the files written."""
    log(f"Loading entries from {entries_path}...")
    run = choose_entries(session, read_entries(entries_path), weights_for, log)
    paths = write_outputs(run, output_csv, stamp=stamp, log=log)
    log(format_exposures(run.exposures(), f"[Exposures] (n={run.total} chosen lineups)") or "\n[Exposures] none")
    return run, paths
