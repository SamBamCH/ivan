# src/ivan/sports/nba/post_lock/lateswap.py
"""
Late swap: after the slate locks, re-optimise the slots of our entries whose
games have not tipped yet, against the contest's real field.

lock_status(session, now)        every slate player: locked (his game has started) or open
our_entries(session, entries)    the DraftKings entries export -> OriginalEntry per contest
field_export(session, cid, ...)  the contest's standings export (DraftKings, through the
                                 logged-in Chrome profile; or the newest stored raw export)
complete_field(...)              each field entry's eight players: the LOCKED names are
                                 known, the open slots are filled by the optimiser in a
                                 sampled world (what that entrant will most likely hold)
candidates(...)                  for one of our entries: the original plus N re-optimised
                                 lineups, locked players held in their slots
evaluate(...)                    expected payout of every candidate against the field, in
                                 num_sims worlds; ties ignored (the candidate ranks above
                                 the field entries it beats)
lateswap_contest / lateswap_entries   the whole thing; write_outputs writes the upload CSV
                                 (the entries export with swapped lineups) and a json report
                                 the desk reads: data/nba/output/lateswap.<stamp>.{csv,json}

With use_live (--live), players in started games score from ingame.update: the points they
have banked (NBA's live box score, live.py) plus a draw of the rest of their game under that
game's live score; without it they score from their pre-game draws.
scripts/nba/lateswap.py is the command line; the desk's late swap tab runs it as a job.
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.core.seed import seed_for
from ivan.core.utils import normalize_name
from ivan.sports.nba import contests as C
from ivan.sports.nba.lineup_export import BASE_COLS, POSITION_ORDER, assign_player_order_classic, read_entries
from ivan.sports.nba.models import NBAPlayer
from ivan.sports.nba.paths import OUTPUT_DIR
from ivan.sports.nba.post_lock.entry_parsing import OriginalEntry
from ivan.sports.nba.pre_lock.lineup_generation import SLOT_NAMES, FieldPlayer, generate_single_classic_lineup
from ivan.sports.nba.pre_lock.player import generate_player_samples
from ivan.sports.nba.simulate_contest import payout_tiers, player_samples_matrix, scale_payouts
from ivan.sports.nba.standings import parse_export
from ivan.sports.nba.utils import _coerce_positions

Log = Callable[[str], None]
SPORT = "nba"
WORKERS = 8
FIELD_WORLDS = 500             # sampled worlds the field's open slots are completed in (one per entry, drawn at random)


# ----------------------------------------------------------------------------- the slate
def slate_players(session: Session) -> Dict[int, FieldPlayer]:
    out: Dict[int, FieldPlayer] = {}
    for rp in session.query(NBAPlayer).all():
        if not rp.position or rp.salary is None:
            continue
        team, opp = rp.team or "", rp.opponent
        out[int(rp.id)] = FieldPlayer(id=int(rp.id), name=rp.name, salary=int(rp.salary), team=team, opponent=opp,
                                      position=tuple(_coerce_positions(rp.position)), fpts_mean=float(rp.fpts_mean or 0.0),
                                      game_id="-".join(sorted([team.upper(), opp.upper()])) if team and opp else None, gametime=rp.gametime)
    return out


def is_locked(fp: FieldPlayer, now: datetime) -> bool:
    return bool(fp.gametime is not None and fp.gametime <= now)


def lock_status(session: Session, now: Optional[datetime] = None) -> dict:
    """{players: {id: locked}, games_started, games_total, next_tip}: what a swap can still touch."""
    now = now or datetime.now()
    fps = slate_players(session)
    tips = sorted({fp.gametime for fp in fps.values() if fp.gametime is not None})
    started = [t for t in tips if t <= now]
    return dict(players={pid: is_locked(fp, now) for pid, fp in fps.items()}, games_started=len(started), games_total=len(tips),
                next_tip=next((t for t in tips if t > now), None), now=now)


def name_map(session: Session) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in session.query(NBAPlayer).all():
        out[normalize_name(p.name)] = int(p.id)
    return out


def our_entries(session: Session, entries_csv) -> Dict[int, List[OriginalEntry]]:
    """The entries export (lineup_export.read_entries: DK's instruction block cut off) -> OriginalEntry per contest; ids from 'Name (id)'."""
    df = read_entries(entries_csv)
    out: Dict[int, List[OriginalEntry]] = {}
    for _, r in df.iterrows():
        slots, names = {}, {}
        for slot in POSITION_ORDER:
            m = str(r[slot]).strip()
            if "(" not in m:
                break
            slots[slot] = int(m[m.rfind("(") + 1:m.rfind(")")]); names[slot] = m[:m.rfind("(")].strip()
        if len(slots) != 8:
            continue                                               # a reservation without a lineup
        out.setdefault(int(r["Contest ID"]), []).append(OriginalEntry(entry_id=str(r["Entry ID"]).strip(), contest_id=int(r["Contest ID"]),
                                                                    lineup_id=-1, slots=slots, player_names_by_slot=names))
    return out


# ----------------------------------------------------------------------------- the field
def field_export(session: Session, contest_id: int, source: str = "draftkings", http_sess=None, log: Log = print) -> Optional[pd.DataFrame]:
    """
    The contest's standings export as a DataFrame. source: "draftkings" downloads it now
    (http_sess from ivan.sports.common.draftkings.build_requests_session_from_chrome; the
    export is also kept raw under dk/standings); "raw" takes the newest stored export.
    """
    if source == "draftkings" and http_sess is not None:
        from ivan.sports.common.draftkings import download_contest_dataframe
        df = download_contest_dataframe(http_sess, int(contest_id), raw_sport=SPORT)
        if df is not None and not df.empty:
            return df
        log(f"{contest_id}: DraftKings returned no export; trying the newest stored one")
    art = raw.latest(SPORT, "dk/standings", contest_id)
    if art is None:
        return None
    text = raw.load(art.path, SPORT)
    import io
    log(f"{contest_id}: stored export {art.stamp}")
    return pd.read_csv(io.StringIO(str(text).lstrip("﻿")), dtype=str, keep_default_na=False, low_memory=False)


def _resolve(names: Dict[str, int], label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    n = normalize_name(label)
    if n in names:
        return names[n]
    last = n.split()[-1] if n else ""
    cands = [k for k in names if k.split()[-1] == last]
    if len(cands) == 1:
        return names[cands[0]]
    first = n.split()[0][:3] if n else ""
    cands = [k for k in cands if k.split()[0][:3] == first]
    return names[cands[0]] if len(cands) == 1 else None


def _complete_one(pairs: List[Tuple[str, Optional[str]]], names: Dict[str, int], fps: Dict[int, FieldPlayer],
                  scores: Dict[int, float], now: datetime) -> Optional[List[int]]:
    """One field entry: the named (locked) players held in their slots, the open slots solved in `scores`."""
    locked: Dict[int, str] = {}
    for slot, label in pairs:
        if label is None:
            continue
        pid = _resolve(names, label)
        if pid is None:
            return None
        locked[pid] = slot
    pool = [fp for fp in fps.values() if fp.id in locked or not is_locked(fp, now)]
    if len(pool) < 8:
        return None
    sc = {fp.id: scores.get(fp.id, fp.fpts_mean or 0.0) for fp in pool}
    return generate_single_classic_lineup(pool, base_scores=sc, locked_assignments=locked or None, max_salary_per_team=50000)


def _complete_chunk(args) -> List[List[int]]:
    rows, names, fps, worlds, pids, now, seed, bonus = args
    rng = np.random.default_rng(seed)
    out = []
    for pairs in rows:
        w = worlds[rng.integers(0, worlds.shape[0])]
        ids = _complete_one(pairs, names, fps, {pid: float(w[j]) + bonus.get(pid, 0.0) for j, pid in enumerate(pids)}, now)
        if ids:
            out.append(ids)
    return out


def actual_ownership(export: pd.DataFrame, names: Dict[str, int]) -> Dict[int, float]:
    """player id -> DraftKings' %Drafted from the export's player table (empty when the export has none)."""
    _, side = parse_export(export.to_csv(index=False))
    out: Dict[int, float] = {}
    for r in side.itertuples():
        pid = _resolve(names, r.name)
        if pid is not None and pd.notna(r.drafted_pct):
            out[pid] = float(r.drafted_pct)
    return out


OWN_ROUNDS = 4                 # passes of the ownership matching (complete, compare with %Drafted, nudge, complete again)
OWN_STEP = 0.8                 # fantasy points of bonus per point of ownership gap per pass


def complete_field(session: Session, export: pd.DataFrame, fps: Dict[int, FieldPlayer], names: Dict[str, int], now: datetime,
                   cap: int = 5000, seed: Optional[int] = None, log: Log = print, workers: int = WORKERS,
                   target_own: Optional[Dict[int, float]] = None, rounds: int = OWN_ROUNDS) -> Tuple[List[List[int]], dict]:
    """
    Every field entry as eight player ids (copies kept): see the module doc. Returns (lineups, meta).

    The started players in every entry are DraftKings' own (the export names them), so their field ownership is the real one.
    With target_own (%Drafted per player, actual_ownership), the hidden slots are filled so the open players' ownership
    matches DraftKings' too: each pass adds a per-player bonus proportional to the gap and completes the field again.
    """
    from ivan.sports.nba.standings import parse_lineup
    ent, _ = parse_export(export.to_csv(index=False))
    rows = [parse_lineup(s) for s in ent.lineup.tolist()]
    rows = [r for r in rows if len(r) == 8]
    total = len(rows)
    rng = np.random.default_rng(seed)
    if total > cap:
        rows = [rows[i] for i in rng.choice(total, size=int(cap), replace=False)]
    pids = sorted(fps)
    samples = generate_player_samples(session, n_samples=FIELD_WORLDS, randomness=1.0, seed=seed)
    worlds = np.zeros((FIELD_WORLDS, len(pids)), dtype=np.float32)
    for j, pid in enumerate(pids):
        v = samples.get(pid)
        worlds[:, j] = v[:FIELD_WORLDS] if isinstance(v, np.ndarray) and len(v) >= FIELD_WORLDS else float(fps[pid].fpts_mean or 0.0)
    n = max(1, min(workers, len(rows)))
    chunks = [rows[i::n] for i in range(n)]
    open_ids = [pid for pid, fp in fps.items() if not is_locked(fp, now)]
    targets = {pid: float(target_own[pid]) for pid in open_ids if target_own and pid in target_own}
    bonus: Dict[int, float] = {}
    gap_by_round = []
    for rnd in range(max(1, rounds if targets else 1)):
        out: List[List[int]] = []
        with ProcessPoolExecutor(max_workers=n) as exe:
            futs = [exe.submit(_complete_chunk, (c, names, fps, worlds, pids, now, (seed or 0) + k, bonus)) for k, c in enumerate(chunks)]
            for f in as_completed(futs):
                out.extend(f.result())
        if not targets or not out:
            break
        held = {}
        for ln in out:
            for pid in ln:
                held[pid] = held.get(pid, 0) + 1
        own = {pid: 100.0 * held.get(pid, 0) / len(out) for pid in targets}
        gap = {pid: targets[pid] - own[pid] for pid in targets}
        gap_by_round.append(round(float(np.mean([abs(g) for g in gap.values()])), 2))
        log(f"field ownership pass {rnd + 1}: open players' mean gap to %Drafted {gap_by_round[-1]:.2f} pts")
        if rnd + 1 < rounds:
            for pid, g in gap.items():
                bonus[pid] = bonus.get(pid, 0.0) + OWN_STEP * g
    meta = dict(field_entries=total, used=len(rows), completed=len(out), dropped=len(rows) - len(out),
                own_targets=len(targets), own_gap=gap_by_round)
    log(f"field: {total:,} entries in the export, {len(rows):,} used, {len(out):,} completed ({meta['dropped']} dropped: names not on the slate)"
        + ("" if targets else "; no %Drafted for the open players, hidden slots follow the projections only"))
    return out, meta


# ----------------------------------------------------------------------------- our candidates
def candidates(entry: OriginalEntry, fps: Dict[int, FieldPlayer], worlds: np.ndarray, pids: List[int], now: datetime,
               n: int = 10, jitter: float = 0.25, rng: Optional[np.random.Generator] = None) -> List[List[int]]:
    """The original lineup first, then up to n distinct re-optimised ones (locked players held in their slots)."""
    rng = rng or np.random.default_rng()
    orig = entry.player_ids()
    locked = {pid: slot for slot, pid in entry.slots.items() if pid in fps and is_locked(fps[pid], now)}
    entry.locked_player_ids = set(locked)
    out, seen = [orig], {tuple(sorted(orig))}
    if len(locked) == 8:
        return out
    pool = [fp for fp in fps.values() if fp.id in locked or not is_locked(fp, now)]
    if len(pool) < 8:
        return out
    col = {pid: j for j, pid in enumerate(pids)}
    for ridx in rng.permutation(worlds.shape[0]):
        w = worlds[ridx]
        sc = {fp.id: float(w[col[fp.id]]) + (rng.normal(0.0, jitter) if jitter and fp.id not in locked else 0.0) for fp in pool if fp.id in col}
        ids = generate_single_classic_lineup(pool, base_scores=sc, locked_assignments=locked or None, max_salary_per_team=50000)
        if not ids:
            continue
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key); out.append(ids)
        if len(out) > n:
            break
    return out


# ----------------------------------------------------------------------------- scoring
def live_samples(session: Session, n: int, live: pd.DataFrame, fps: Dict[int, FieldPlayer], names: Dict[str, int],
                 seed: Optional[int] = None, log: Log = print) -> Tuple[Dict[int, np.ndarray], dict]:
    """
    Every slate player's draws for scoring: the pre-game sampler's, with each player of a started game replaced by
    ingame.update (banked points + the rest of his game under that game's live script). Returns (samples, summary).
    """
    from ivan.sports.nba.post_lock import ingame
    from ivan.sports.nba.projection.team_minutes import load_params
    base = generate_player_samples(session, n_samples=int(n), randomness=1.0, seed=seed)
    if live is None or live.empty:
        return base, dict(updated=0, games=0)
    beta = (load_params() or {}).get("beta") or ingame.DEFAULT_BETA
    rng = np.random.default_rng(None if seed is None else seed + 7)
    prior = {int(p.id): p for p in session.query(NBAPlayer).all()}
    out = dict(base); updated = 0; games = 0
    for gid, g in live.groupby("game_id"):
        teams = list(dict.fromkeys(g.team))
        el = float(g.elapsed.max()); left = max(0.0, 1.0 - el / 2880.0); final = bool(g.final.any())
        m0 = float(g[g.team == teams[0]].margin.iloc[0])
        draw = ingame.margin_draws(m0, left, int(n), rng)
        margin_of = {teams[0]: draw, **({teams[1]: -draw} if len(teams) > 1 else {})}
        games += 1
        for r in g.itertuples():
            pid = _resolve(names, r.player_name)
            if pid is None or pid not in base or pid not in prior:
                continue
            pp = prior[pid]
            S = np.asarray(base[pid], dtype=np.float64)[: int(n)]
            out[pid] = ingame.update(S, float(pp.fpts_mean or S.mean()), float(pp.minutes_proj or 0.0), float(r.fpts), float(r.minutes),
                                     left, margin_of.get(r.team, draw), bool(r.starter), rng, beta=beta, final=final).astype(np.float32)
            updated += 1
    log(f"in-game update: {updated} players in {games} started games")
    return out, dict(updated=updated, games=games)


def evaluate(session: Session, contest, field: List[List[int]], cands_by_entry: Dict[str, List[List[int]]], num_sims: int,
             seed: Optional[int] = None, log: Log = print, samples: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, List[dict]]:
    """
    entry_id -> one dict per candidate: ids, exp_payout, exp_fpts, p_cash, exp_rank. Payouts are the
    contest's tiers scaled to (field + 1); a candidate's rank in a world is one more than the field
    entries that outscore it.
    """
    pids = sorted({p for ln in field for p in ln} | {p for cs in cands_by_entry.values() for ln in cs for p in ln})
    col = {pid: i for i, pid in enumerate(pids)}
    if samples is None:
        P, ns, _, missing = player_samples_matrix(session, col, num_sims, seed=seed)
    else:                                                          # live-updated draws (live_samples)
        ns = int(num_sims); missing = [pid for pid in col if pid not in samples]
        P = np.zeros((ns, len(col)), dtype=np.float32)
        for pid, j in col.items():
            if pid in samples:
                P[:, j] = np.asarray(samples[pid], dtype=np.float32)[:ns]
    if ns == 0:
        raise ValueError("the player sampler returned nothing; project the slate first")
    if missing:
        log(f"WARNING: {len(missing)} players on the field have no projection; they score 0")
    Mf = len(field)
    ix = np.array([[col[p] for p in ln] for ln in field], dtype=np.int64)
    F = np.sort(P[:, ix].sum(axis=2), axis=1)                                   # worlds x field, ascending
    tiers = payout_tiers(session, int(contest.contest_id))
    if not tiers:
        raise ValueError(f"no payouts for contest {contest.contest_id}")
    scaled = scale_payouts(tiers, Mf + 1, int(contest.total_entries), float(contest.entry_fee))
    pay = np.zeros(Mf + 2)
    for r, a in scaled.items():
        if 1 <= r <= Mf + 1:
            pay[r] = float(a)
    out: Dict[str, List[dict]] = {}
    for eid, cs in cands_by_entry.items():
        rows = []
        for ln in cs:
            s = P[:, [col[p] for p in ln]].sum(axis=1)
            beaten = np.array([np.searchsorted(F[i], s[i], side="left") for i in range(ns)])     # field entries below us
            rank = Mf - beaten + 1
            pays = pay[rank]
            rows.append(dict(ids=[int(p) for p in ln], exp_payout=float(pays.mean()), exp_fpts=float(s.mean()),
                             p_cash=float((pays > 0).mean()), exp_rank=float(rank.mean())))
        out[eid] = rows
    return out


# ----------------------------------------------------------------------------- the run
def lateswap_contest(session: Session, contest_id: int, entries: List[OriginalEntry], fps: Dict[int, FieldPlayer], names: Dict[str, int],
                     now: datetime, source: str = "draftkings", http_sess=None, n_candidates: int = 10, field_cap: int = 5000,
                     num_sims: int = 1000, log: Log = print, live: Optional[pd.DataFrame] = None) -> Optional[dict]:
    contest = C.ensure_contest(session, int(contest_id), log=log)
    if contest is None:
        log(f"{contest_id}: not in the database and DraftKings has nothing for it"); return None
    if not payout_tiers(session, int(contest_id)):
        C.fetch_payouts(session, [int(contest_id)], log=log)
    export = field_export(session, int(contest_id), source=source, http_sess=http_sess, log=log)
    if export is None or export.empty:
        log(f"{contest_id}: no standings export (log in to DraftKings, or scrape it first)"); return None
    seed = seed_for(f"lateswap/{contest_id}")
    own = actual_ownership(export, names)
    field, fmeta = complete_field(session, export, fps, names, now, cap=field_cap, seed=seed, log=log, target_own=own)
    fmeta["ownership"] = {str(pid): round(v, 2) for pid, v in own.items()}
    if not field:
        log(f"{contest_id}: the field could not be completed"); return None
    pids = sorted(fps)
    worlds_n = max(4 * n_candidates, 200)
    samples = generate_player_samples(session, n_samples=worlds_n, randomness=1.0, seed=seed)
    worlds = np.zeros((worlds_n, len(pids)), dtype=np.float32)
    for j, pid in enumerate(pids):
        v = samples.get(pid)
        worlds[:, j] = v[:worlds_n] if isinstance(v, np.ndarray) and len(v) >= worlds_n else float(fps[pid].fpts_mean or 0.0)
    rng = np.random.default_rng(seed)
    cands = {e.entry_id: candidates(e, fps, worlds, pids, now, n=n_candidates, rng=rng) for e in entries}
    log(f"{contest_id} {contest.name}: {len(entries)} entries, {sum(len(v) - 1 for v in cands.values())} candidates, field {len(field):,}, {num_sims} sims")
    samples = live_samples(session, num_sims, live, fps, names, seed=seed, log=log)[0] if live is not None else None
    scored = evaluate(session, contest, field, cands, num_sims, seed=seed, log=log, samples=samples)
    rows = []
    for e in entries:
        sc = scored.get(e.entry_id) or []
        if not sc:
            continue
        orig = sc[0]
        keep = [r for r in sc if set(e.locked_player_ids) <= set(r["ids"])]     # a candidate missing a locked player is never chosen
        best = max(keep or [orig], key=lambda r: r["exp_payout"])
        swaps = [(o, b) for o, b in zip([p for p in orig["ids"] if p not in best["ids"]], [p for p in best["ids"] if p not in orig["ids"]])]
        rows.append(dict(entry_id=e.entry_id, original=orig, best=best, improved=best is not orig and best["exp_payout"] > orig["exp_payout"],
                         locked=sorted(e.locked_player_ids), swaps=swaps, candidates=len(sc) - 1))
    return dict(contest_id=int(contest_id), name=contest.name, entry_fee=float(contest.entry_fee), total_entries=int(contest.total_entries),
                field=fmeta, entries=rows)


def lateswap_entries(session: Session, entries_csv, now: Optional[datetime] = None, source: str = "draftkings", http_sess=None,
                     n_candidates: int = 10, field_cap: int = 5000, num_sims: int = 1000, only: Optional[Sequence[int]] = None,
                     log: Log = print, use_live: bool = False) -> dict:
    now = now or datetime.now()
    fps = slate_players(session); names = name_map(session)
    by_contest = our_entries(session, entries_csv)
    status = lock_status(session, now)
    log(f"{status['games_started']} of {status['games_total']} games started at {now:%H:%M}" + (f"; next tip {status['next_tip']:%H:%M}" if status["next_tip"] else ""))
    out = dict(when=now.isoformat(timespec="seconds"), games_started=status["games_started"], games_total=status["games_total"], source=source,
               settings=dict(candidates=n_candidates, field_cap=field_cap, sims=num_sims, live=use_live), contests=[], failed=[])
    live = None
    if use_live:
        from ivan.sports.nba.post_lock import live as LIVE
        try:
            live = LIVE.live_state(teams={fp.team for fp in fps.values()}, log=log)
            log(f"live box scores: {live.game_id.nunique() if len(live) else 0} started games")
        except Exception as e:   # noqa: BLE001 - without live data the swap still runs on pre-game draws
            log(f"live box scores unavailable ({type(e).__name__}: {e}); scoring on pre-game draws"); live = None
    for cid, group in by_contest.items():
        if only and int(cid) not in {int(x) for x in only}:
            continue
        try:
            r = lateswap_contest(session, int(cid), group, fps, names, now, source=source, http_sess=http_sess,
                                 n_candidates=n_candidates, field_cap=field_cap, num_sims=num_sims, log=log, live=live)
        except Exception as e:   # noqa: BLE001 - one contest failing must not lose the others
            import traceback
            session.rollback()
            log(f"FAILED contest {cid}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            out["failed"].append(dict(contest_id=int(cid), error=f"{type(e).__name__}: {e}")); continue
        if r is None:
            out["failed"].append(dict(contest_id=int(cid), error="no field")); continue
        out["contests"].append(r)
    return out


# ----------------------------------------------------------------------------- outputs
def write_outputs(session: Session, report: dict, entries_csv, out_dir: Path = OUTPUT_DIR, stamp: Optional[str] = None, log: Log = print) -> Dict[str, str]:
    """The upload CSV (the entries export with every improved entry's new lineup, slot order by eligibility) and the json report."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fps = slate_players(session)
    meta = {int(p.id): p for p in session.query(NBAPlayer).all()}
    best_by_entry = {str(e["entry_id"]): e for c in report["contests"] for e in c["entries"]}
    df = read_entries(entries_csv)
    swapped = 0
    for i, row in df.iterrows():
        e = best_by_entry.get(str(row["Entry ID"]).strip())
        if not e or not e["improved"]:
            continue
        ids = e["best"]["ids"]
        # keep the locked players in the slots DraftKings has them in; slot the rest around them
        slots = {}
        for slot in POSITION_ORDER:
            m = str(row[slot])
            pid = int(m[m.rfind("(") + 1:m.rfind(")")]) if "(" in m else None
            if pid in e["locked"]:
                slots[slot] = pid
        free = [fps[p] for p in ids if p not in slots.values() and p in fps]
        free_slots = [s for s in POSITION_ORDER if s not in slots]
        try:
            ordered = _slot(free, free_slots)
        except ValueError as ex:
            log(f"entry {row['Entry ID']}: {ex}; left as is"); continue
        for slot, fp in zip(free_slots, ordered):
            slots[slot] = fp.id
        before = {slot: str(row[slot]) for slot in POSITION_ORDER}
        locked_slots = [sl for sl in POSITION_ORDER if sl in slots and slots[sl] in e["locked"]]
        if len(locked_slots) != len(e["locked"]) or sorted(slots.values()) != sorted(ids):
            log(f"entry {row['Entry ID']}: the swap would move or drop a locked player; left as is"); continue
        for slot in POSITION_ORDER:
            p = meta[slots[slot]]
            df.at[i, slot] = f"{p.name} ({p.id})"
        if any(str(df.at[i, sl]) != before[sl] and _pid(before[sl]) != _pid(str(df.at[i, sl])) for sl in locked_slots):
            for sl in POSITION_ORDER:                              # belt and braces: never upload a changed locked slot
                df.at[i, sl] = before[sl]
            log(f"entry {row['Entry ID']}: a locked slot changed; restored"); continue
        swapped += 1
    csv_path = out_dir / f"lateswap.{stamp}.csv"
    df[BASE_COLS + POSITION_ORDER].to_csv(csv_path, index=False)
    for c in report["contests"]:                                   # names for the desk
        for e in c["entries"]:
            e["names"] = {str(pid): meta[pid].name for pid in set(e["original"]["ids"]) | set(e["best"]["ids"]) if pid in meta}
    report["swapped"] = swapped; report["upload"] = csv_path.name; report["stamp"] = stamp
    js = out_dir / f"lateswap.{stamp}.json"
    js.write_text(json.dumps(report, default=str))
    log(f"wrote {swapped} swapped entries -> {csv_path} and the report -> {js}")
    return {"upload": str(csv_path), "report": str(js)}


def _pid(cell: str) -> Optional[int]:
    return int(cell[cell.rfind("(") + 1:cell.rfind(")")]) if "(" in cell else None


def _slot(players: List[FieldPlayer], slots: List[str]) -> List[FieldPlayer]:
    """Players into the given open slots by eligibility (backtracking), latest games in the flexible slots."""
    from ivan.sports.nba.pre_lock.lineup_generation import SLOT_ELIGIBILITY
    players = sorted(players, key=lambda p: p.gametime.timestamp() if p.gametime is not None else 0)

    def bt(k, acc, rem):
        if k == len(slots):
            return acc
        for i, p in enumerate(rem):
            if any(x in SLOT_ELIGIBILITY[slots[k]] for x in p.position):
                r = bt(k + 1, acc + [p], rem[:i] + rem[i + 1:])
                if r is not None:
                    return r
        return None
    out = bt(0, [], list(players))
    if out is None:
        raise ValueError("no slot assignment for " + ", ".join(f"{p.name} ({'/'.join(p.position)})" for p in players))
    return out


def latest_report(out_dir: Path = OUTPUT_DIR) -> Optional[dict]:
    for js in sorted(out_dir.glob("lateswap.*.json"), reverse=True):
        try:
            return json.loads(js.read_text())
        except ValueError:
            continue
    return None
