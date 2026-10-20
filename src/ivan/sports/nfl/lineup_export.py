# src/ivan/sports/nfl/lineup_export.py
"""
The choose step: for every contest in a DraftKings entries file, pick lineups
from its simulation results (choose.choose_lineup_ids) and write them the way
DraftKings wants them uploaded, plus what the web UI and the tax backtest read:

    <base>.showdown.<stamp>.csv / <base>.classic.<stamp>.csv   the upload files
    <base>.exposures.<stamp>.csv     player exposure per slot (showdown) / position (classic)
    <base>.lineups.<stamp>.csv       one row per player per chosen entry, with the lineup's sim cash % / ROI
    <base>.choose.<stamp>.json       what the chooser saw per contest (candidates, picks, greedy trace)
    <base>.simroi.<stamp>.npz        each pick's per-world expected ROI (roi) and one realized ROI per
                                     world (roi_draw) for the calibration tests (the sim tables are cleared weekly)

read_entries() parses the entries file, weights_resolver() decides the per-contest
weights (file / flags / prompt), choose_entries() does the picking and slot
assignment, write_outputs() the files. scripts/nfl/choose.py is argparse around
run_choose().
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

from ivan.sports.nfl.choose import choose_lineup_ids, sim_roi_traces
from ivan.sports.nfl.models import Lineup, NFLPlayer

Log = Callable[[str], None]

# DraftKings upload order
POSITION_ORDER_SD = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
POSITION_ORDER_CLASSIC = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SLOT_CODE = {"CPT": 511, "FLEX": 512, "QB": 66, "RB": 67, "WR": 68, "TE": 69, "DST": 71}
CODE_TO_SLOT = {v: k for k, v in SLOT_CODE.items()}
BASE_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
DEFAULT_WEIGHTS = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1, "method": "greedy", "bankroll": 3.0}
# method: "greedy" (quality + diversity, lineup by lineup) or "portfolio" (log utility of the whole book over the
# simulated worlds; bankroll = its scale in multiples of the entries' fees). See ivan.sports.nfl.choose.
WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS)
EXPOSURE_ORDER = ["CPT", "FLEX", "QB", "RB", "WR", "TE", "DST"]


# ------------------------------------------------------------------------------ entries file
def parse_money(x) -> float:
    """'$0.50' -> 0.5"""
    try:
        return float(str(x).replace("$", "").replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def read_entries(path) -> pd.DataFrame:
    """
    The DraftKings entries export: the first ten columns (showdown's; classic has more
    but only BASE_COLS are used), cut at the first blank Entry ID, Contest ID as int.
    """
    df = pd.read_csv(path, engine="python", header=0, usecols=range(10), dtype=str, keep_default_na=False)
    for col in BASE_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Expected column '{col}' in entries CSV but not found.")
    blank = df["Entry ID"].str.strip() == ""
    if blank.any():
        df = df.loc[: blank.idxmax() - 1].copy()
    df["Contest ID"] = df["Contest ID"].astype(int)
    return df


# ------------------------------------------------------------------------------ weights
Prompter = Callable[[int, str, Dict[str, float]], Dict[str, float]]


def weights_resolver(flags: Optional[dict] = None, overrides: Optional[dict] = None, no_prompt: bool = False,
                     prompter: Optional[Prompter] = None, defaults: Optional[dict] = None):
    """
    Per-contest weight resolution, highest priority first:
      overrides[<contest id or name>]  >  overrides["default"]  >  flags  >  prompter(cid, name, last)
    overrides is the --weights-json mapping; flags the CLI weights given. Any flag, any
    override, no_prompt, or no prompter makes the run non-interactive (defaults fill in).
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


# ------------------------------------------------------------------------------ slot assignment
def _rs(p) -> Optional[int]:
    try:
        return int(getattr(p, "roster_slot_id"))
    except (TypeError, ValueError):
        return None


def _sal(p) -> int:
    try:
        return int(getattr(p, "salary") or 0)
    except (TypeError, ValueError):
        return 0


def _team(p) -> str:
    for attr in ("team", "team_abbrev", "team_abbreviation", "TeamAbbrev", "team_short", "team_code"):
        v = getattr(p, attr, None)
        if v:
            return str(v)
    return "UNK"


def _pos(p) -> str:
    v = getattr(p, "position", None)
    if v:
        return str(v).upper()
    return CODE_TO_SLOT.get(_rs(p), "?")


def _name(p) -> str:
    for attr in ("name", "player_name", "player_display_name", "full_name", "display_name"):
        v = getattr(p, attr, None)
        if v:
            return str(v)
    return f"#{getattr(p, 'id', '??')}"


def assign_player_order_showdown(player_objs: list, slots: list[str] = POSITION_ORDER_SD) -> list:
    """Fill showdown slots strictly by roster_slot_id (511 CPT, 512 FLEX)."""
    assigned: list = [None] * len(slots)
    used: set[int] = set()
    want_cpt, want_flex = slots.count("CPT"), slots.count("FLEX")
    have_cpt = sum(1 for p in player_objs if _rs(p) == SLOT_CODE["CPT"])
    have_flex = sum(1 for p in player_objs if _rs(p) == SLOT_CODE["FLEX"])
    if have_cpt < want_cpt:
        raise ValueError(f"Expected at least {want_cpt} CPT(511) in lineup but found {have_cpt}.")
    if have_flex < want_flex:
        raise ValueError(f"Expected at least {want_flex} FLEX(512) in lineup but found {have_flex}.")
    for idx, slot_req in enumerate(slots):
        code = SLOT_CODE[slot_req]
        for p in player_objs:
            if p.id not in used and _rs(p) == code:
                assigned[idx] = p
                used.add(p.id)
                break
        if assigned[idx] is None:
            remain = [(p.id, _rs(p)) for p in player_objs if p.id not in used]
            raise ValueError(f"Cannot fill {slot_req}; remaining {remain}")
    return assigned


def assign_player_order_classic(player_objs: list) -> list:
    """
    Classic slots in exact upload order QB,RB,RB,WR,WR,WR,TE,FLEX,DST, by roster_slot_id
    when present, else by position; FLEX is the highest-salary remaining RB/WR/TE.
    """
    qbs, rbs, wrs, tes, dsts, flex_eligible = [], [], [], [], [], []
    for p in player_objs:
        slot, pos = _rs(p), _pos(p)
        if slot == SLOT_CODE["QB"] or pos == "QB":
            qbs.append(p)
        elif slot == SLOT_CODE["RB"] or pos == "RB":
            rbs.append(p); flex_eligible.append(p)
        elif slot == SLOT_CODE["WR"] or pos == "WR":
            wrs.append(p); flex_eligible.append(p)
        elif slot == SLOT_CODE["TE"] or pos == "TE":
            tes.append(p); flex_eligible.append(p)
        elif slot == SLOT_CODE["DST"] or pos in {"DST", "D/ST", "DEF"}:
            dsts.append(p)
    key = lambda p: -_sal(p)   # noqa: E731
    for group in (qbs, rbs, wrs, tes, dsts, flex_eligible):
        group.sort(key=key)
    if not qbs: raise ValueError("Classic lineup missing QB.")                     # noqa: E701
    if len(rbs) < 2: raise ValueError("Classic lineup requires 2 RB.")            # noqa: E701
    if len(wrs) < 3: raise ValueError("Classic lineup requires 3 WR.")            # noqa: E701
    if len(tes) < 1: raise ValueError("Classic lineup requires TE.")              # noqa: E701
    if not dsts: raise ValueError("Classic lineup missing DST.")                   # noqa: E701
    order = [qbs[0]] + rbs[:2] + wrs[:3] + [tes[0]]
    used_ids = {p.id for p in order}
    flex_pick = next((p for p in flex_eligible if p.id not in used_ids), None)
    if flex_pick is None:
        raise ValueError("No remaining RB/WR/TE to fill FLEX.")
    order += [flex_pick, dsts[0]]
    if len(order) != 9:
        raise ValueError(f"Classic assignment produced {len(order)} players, expected 9.")
    return order


# ------------------------------------------------------------------------------ choosing
@dataclass
class ChooseRun:
    upload_sd: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BASE_COLS + POSITION_ORDER_SD))
    upload_classic: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BASE_COLS + POSITION_ORDER_CLASSIC))
    sd_counts: Counter = field(default_factory=Counter)        # (player_id, slot label) -> entries
    sd_info: Dict[Tuple[int, str], dict] = field(default_factory=dict)
    classic_counts: Counter = field(default_factory=Counter)   # player_id -> entries
    classic_info: Dict[int, dict] = field(default_factory=dict)
    total_sd: int = 0
    total_classic: int = 0
    lineup_rows: List[dict] = field(default_factory=list)      # one row per player per chosen entry
    choose_viz: Dict[str, dict] = field(default_factory=dict)  # per contest, what the chooser saw
    sim_rows: List[tuple] = field(default_factory=list)        # (entry_id, contest_id, lineup_id, fee, roi trace, realized draws)
    contests: List[dict] = field(default_factory=list)         # per contest summary (stats)

    def exposures(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(showdown exposures by roster slot, classic exposures by position), exposure as a fraction of entries."""
        sd = pd.DataFrame([{"roster_slot": slot, "player_id": pid, **self.sd_info[(pid, slot)], "exposure": n / self.total_sd}
                           for (pid, slot), n in self.sd_counts.items()]) if self.total_sd and self.sd_counts else pd.DataFrame()
        cl = pd.DataFrame([{"player_id": pid, **self.classic_info[pid], "exposure": n / self.total_classic}
                           for pid, n in self.classic_counts.items()]) if self.total_classic and self.classic_counts else pd.DataFrame()
        cols_sd = ["roster_slot", "player_id", "name", "team", "position", "salary", "exposure", "ownership", "projection"]
        cols_cl = ["position", "player_id", "name", "team", "salary", "exposure", "ownership", "projection"]
        return (sd[cols_sd] if len(sd) else sd), (cl[cols_cl] if len(cl) else cl)


def choose_entries(session, entries: pd.DataFrame, weights_for, log: Log = print) -> ChooseRun:
    """Pick and slot-assign lineups for every contest in the entries frame (read_entries)."""
    run = ChooseRun()
    last = dict(DEFAULT_WEIGHTS)
    for cid, group in entries.groupby("Contest ID", sort=False):
        cid = int(cid)
        n_entries = len(group)
        cname = group["Contest Name"].iloc[0].strip()
        log(f"\n=== Contest {cid} \"{cname}\" -> {n_entries} entries ===")
        w = weights_for(cid, cname, last)
        last = w
        method = str(w.get("method") or "greedy")
        log(f"  method: {method}" + (f" (bankroll {w.get('bankroll', 3.0)}x fees)" if method == "portfolio" else "")
            + f"  weights: cash_pct={w['cash_pct']} avg_roi_win={w['avg_roi_win']} diversity_lambda={w['diversity_lambda']}")

        stats: dict = {}
        chosen_ids = choose_lineup_ids(session=session, weights=w, num_lineups=n_entries, contest_name=cname, stats=stats)
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
        if stats.get("viz"):
            run.choose_viz[str(cid)] = {"contest_name": cname, "entries": n_entries, **stats["viz"]}
        log(f"  chosen {stats['chosen']}: mean ROI {stats['chosen_roi']:+.1%}, cash {stats['chosen_cash']:.1%}"
            f"  (top-{stats['chosen']} by quality alone: ROI {stats['top_quality_roi']:+.1%}, cash {stats['top_quality_cash']:.1%})")
        book = stats.get("book") or {}
        if book.get("chosen"):
            b, t = book["chosen"], book.get("top_quality") or {}
            log(f"  book over {b['worlds']} sim worlds: median ROI {b['p50']:+.0%}, p5 {b['p5']:+.0%}, P(loss) {b['p_loss']:.0%}, "
                f"P(lose half) {b['p_lose_half']:.0%}" + (f"  (quality alone: median {t['p50']:+.0%}, p5 {t['p5']:+.0%}, P(lose half) {t['p_lose_half']:.0%})" if t else ""))
            sh = book.get("shrinkage") or {}
            if sh.get("tau") is not None:
                log(f"  shrinkage: per-lineup ROI se median {sh['se_median']:.2f} vs spread of true means {sh['tau']:.2f}; "
                    f"top-20 claimed mean {sh['top_mean_before']:+.0%} -> {sh['top_mean_after']:+.0%} after shrinking (median weight on a top-decile lineup's own estimate {sh['lambda_top_decile']:.2f})")
            cf = book.get("cross_fit")
            if cf:
                log(f"  honest expectation (chosen on {cf['worlds_fit']} worlds, scored on the other {cf['worlds_score']}): mean ROI {cf['out_of_sample']['mean']:+.0%} "
                    f"vs {cf['in_sample']['mean']:+.0%} in-sample; selection bias {cf['selection_bias']:+.0%}")

        traces = sim_roi_traces(session, cname, chosen_ids)
        draws = sim_roi_traces(session, cname, chosen_ids, column="roi_draw")
        lineups = {L.id: L for L in session.query(Lineup).filter(Lineup.id.in_(chosen_ids)).all()}
        all_pids = {pid for L in lineups.values() for pid in L.player_ids_list}
        players = {p.id: p for p in session.query(NFLPlayer).filter(NFLPlayer.id.in_(all_pids)).all()}
        by_lineup = stats.get("by_lineup", {})
        rows = group.reset_index(drop=True)
        for i, lid in enumerate(chosen_ids):
            lineup = lineups.get(lid)
            if not lineup:
                continue
            pids = list(lineup.player_ids_list)
            objs = [players[pid] for pid in pids]
            base = rows.iloc[i % n_entries][BASE_COLS].to_list()
            run.sim_rows.append((int(base[0]), cid, int(lid), parse_money(base[3]), traces.get(lid), draws.get(lid)))
            if len(pids) == 6:
                fmt, slots = "showdown", POSITION_ORDER_SD
                assigned = assign_player_order_showdown(objs, slots)
                run.upload_sd.loc[len(run.upload_sd)] = base + [f"{_name(pl)} ({pl.id})" for pl in assigned]
                run.total_sd += 1
                for pl in assigned:
                    slot_label = CODE_TO_SLOT.get(_rs(pl), "UNK")
                    run.sd_counts[(pl.id, slot_label)] += 1
                    run.sd_info[(pl.id, slot_label)] = {"name": _name(pl), "team": _team(pl), "position": _pos(pl), "salary": _sal(pl),
                                                        "ownership": pl.ownership_proj, "projection": pl.fpts_p50}
            elif len(pids) == 9:
                fmt, slots = "classic", POSITION_ORDER_CLASSIC
                assigned = assign_player_order_classic(objs)
                run.upload_classic.loc[len(run.upload_classic)] = base + [f"{_name(pl)} ({pl.id})" for pl in assigned]
                run.total_classic += 1
                for pl in assigned:
                    run.classic_counts[pl.id] += 1
                    run.classic_info[pl.id] = {"name": _name(pl), "team": _team(pl), "position": _pos(pl), "salary": _sal(pl),
                                               "ownership": pl.ownership_proj, "projection": pl.fpts_p50}
            else:
                raise ValueError(f"Unsupported lineup size {len(pids)} for lineup {lid} (expected 6 or 9).")
            sim = by_lineup.get(lid, {})
            for k, (slot, pl) in enumerate(zip(slots, assigned)):
                run.lineup_rows.append({
                    "entry_id": base[0], "contest_name": base[1], "contest_id": base[2], "entry_fee": base[3], "format": fmt,
                    "lineup_id": lid, "slot_no": k, "slot": slot, "player_id": pl.id, "name": _name(pl), "team": _team(pl),
                    "opponent": getattr(pl, "opponent", None), "position": _pos(pl), "salary": _sal(pl),
                    "projection": pl.fpts_mean, "ownership": pl.ownership_proj, "cash_pct": sim.get("cash_pct"), "avg_roi": sim.get("avg_roi")})
    return run


# ------------------------------------------------------------------------------ outputs
def format_exposures(df: pd.DataFrame, header: str, by_col: str) -> str:
    """The exposure table as text, grouped by roster slot or position."""
    if df.empty:
        return ""
    groups = sorted(df[by_col].unique().tolist(), key=lambda g: EXPOSURE_ORDER.index(g) if g in EXPOSURE_ORDER else len(EXPOSURE_ORDER))
    out = [f"\n{header}"]
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_columns", None):
        for g in groups:
            sub = df[df[by_col] == g].copy().sort_values("exposure", ascending=False)
            if sub.empty:
                continue
            sub["exposure"] = (sub["exposure"] * 100).round(2).astype(str) + "%"
            sub["ownership"] = sub["ownership"].round(2).astype(str) + "%"
            out.append(f"\n[{g}]\n" + sub[["name", "team", "position", "salary", "exposure", "ownership", "projection"]].to_string(index=False))
    return "\n".join(out)


def write_outputs(run: ChooseRun, output_csv, stamp: Optional[str] = None, log: Log = print) -> Dict[str, str]:
    """Write the upload CSVs and the exposures / lineups / choose / simroi files next to them. Returns {kind: path}."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    root, ext = os.path.splitext(str(output_csv))
    Path(root).parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for kind, df in (("showdown", run.upload_sd), ("classic", run.upload_classic)):
        if not df.empty:
            paths[kind] = f"{root}.{kind}.{stamp}{ext}"
            df.to_csv(paths[kind], index=False)
            log(f"wrote {len(df)} {kind} rows -> {paths[kind]}")
    if not paths:
        log("WARNING: no rows written; no valid chosen lineups were found")
    sd, cl = run.exposures()
    frames = ([sd.assign(format="showdown")] if len(sd) else []) + ([cl.assign(format="classic")] if len(cl) else [])
    if frames:
        paths["exposures"] = f"{root}.exposures.{stamp}{ext}"
        pd.concat(frames, ignore_index=True).to_csv(paths["exposures"], index=False)
    if run.lineup_rows:
        paths["lineups"] = f"{root}.lineups.{stamp}{ext}"
        pd.DataFrame(run.lineup_rows).to_csv(paths["lineups"], index=False)
    if run.choose_viz:
        paths["choose"] = f"{root}.choose.{stamp}.json"
        Path(paths["choose"]).write_text(json.dumps(run.choose_viz))
    if run.sim_rows:
        n_sims = max((len(t) for *_, t, _d in run.sim_rows if t is not None), default=0)
        roi = np.full((len(run.sim_rows), n_sims), np.nan, dtype=np.float32)
        roi_draw = np.full((len(run.sim_rows), n_sims), np.nan, dtype=np.float32)
        for k, (*_, t, dr) in enumerate(run.sim_rows):
            if t is not None:
                roi[k, :len(t)] = t[:n_sims]
            if dr is not None:
                roi_draw[k, :len(dr)] = dr[:n_sims]
        paths["simroi"] = f"{root}.simroi.{stamp}.npz"
        np.savez_compressed(paths["simroi"], entry_id=np.array([r[0] for r in run.sim_rows], dtype=np.int64),
                            contest_id=np.array([r[1] for r in run.sim_rows], dtype=np.int64),
                            lineup_id=np.array([r[2] for r in run.sim_rows], dtype=np.int64),
                            entry_fee=np.array([r[3] for r in run.sim_rows], dtype=np.float64), roi=roi, roi_draw=roi_draw)
        log(f"wrote per-world ROI ({len(run.sim_rows)} entries x {n_sims} worlds, expected + one realized draw) -> {paths['simroi']}")
    for kind in ("exposures", "lineups", "choose"):
        if kind in paths:
            log(f"wrote {kind} -> {paths[kind]}")
    return paths


def run_choose(session, entries_path, output_csv, weights_for, log: Log = print,
               stamp: Optional[str] = None) -> Tuple[ChooseRun, Dict[str, str]]:
    """read_entries -> choose_entries -> write_outputs; returns the run and the files written (stamp: the files' timestamp, default now)."""
    log(f"Loading entries from {entries_path}...")
    run = choose_entries(session, read_entries(entries_path), weights_for, log)
    paths = write_outputs(run, output_csv, stamp=stamp, log=log)
    sd, cl = run.exposures()
    log(format_exposures(sd, f"[Showdown exposures] (n={run.total_sd} showdown lineups)", "roster_slot") or "\n[Showdown exposures] none")
    log(format_exposures(cl, f"[Classic exposures] (n={run.total_classic} classic lineups)", "position") or "\n[Classic exposures] none")
    return run, paths
