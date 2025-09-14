from __future__ import annotations
import json
import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, Tuple, List
from sqlalchemy.orm import Session
from concurrent.futures import ProcessPoolExecutor, as_completed


from ivan.sports.nfl.models import HistoricalContest, ContestPayout, Lineup, LineupSimulationResult
from ivan.sports.nfl.sample_field import (
    sample_field,
    build_lineup_matrix,
    build_player_samples,
)
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_impose
# Reuse your MLB payout scaler (sport-agnostic)
from ivan.sports.mlb.payouts import scale_payouts_correctly

# --- DIAGNOSTICS: recap, CSVs, and plots for the sampled field ----------------
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

# --- force headless, non-interactive backend before importing pyplot ---
import os
os.environ["MPLBACKEND"] = "Agg"   # global hammer in case other modules import pyplot

import matplotlib
matplotlib.use("Agg", force=True)  # ensure Agg backend
import matplotlib.pyplot as plt
plt.ioff()  # extra safety: disable interactive mode
# ----------------------------------------------------------------------

def _safe_getattr(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default

def _fetch_player_meta(session, pids: list[int]) -> dict[int, dict]:
    from ivan.sports.nfl.models import NFLPlayer  # adjust if path differs
    rows = (
        session.query(NFLPlayer)
        .filter(NFLPlayer.id.in_(pids))
        .all()
    )
    meta = {}
    for r in rows:
        name = _safe_getattr(r, "player_display_name", "display_name", "name", default=str(r.id))
        team = _safe_getattr(r, "team", "team_abbrev", "team_abbr", "defteam", default=None)
        pos  = _safe_getattr(r, "position", "pos", default=None)
        slot = _safe_getattr(r, "roster_slot_id", default=None)
        meta[int(r.id)] = {
            "name": name,
            "team": team,
            "pos": pos,
            "roster_slot": slot,
            "slot_label": ("CPT" if slot == 511 else "FLEX" if slot == 512 else str(slot) if slot is not None else None),
        }
    return meta

def _lineup_player_names_for(lid: int, session, pid_to_meta: dict[int, dict]) -> list[str]:
    from ivan.sports.nfl.models import Lineup
    row = session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.id == lid).one()
    pids = _coerce_player_ids(row.player_ids_list)
    names = []
    for pid in pids:
        m = pid_to_meta.get(pid, {})
        nm = m.get("name", str(pid))
        slot = m.get("slot_label")
        if slot:
            names.append(f"{nm} ({slot})")
        else:
            names.append(nm)
    return names

def _plot_bar(x_labels: list[str], values: np.ndarray, title: str, out_path: Path):
    plt.figure()
    plt.bar(range(len(values)), values)  # no specific colors/styles per policy
    plt.xticks(range(len(values)), x_labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_hist(int_values: np.ndarray, title: str, out_path: Path):
    if int_values.size == 0:
        return
    bins = np.arange(int(int_values.min()), int(int_values.max()) + 2) - 0.5  # center on integers
    plt.figure()
    plt.hist(int_values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def recap_and_plots_for_sampled_field(
    session,
    contest_id: int,
    counts_by_lineup: dict[int, int],
    lineup_ix: list[list[int]],
    unique_pids: list[int],
    outdir: str | Path = "data/nfl/diagnostics",
    top_n_print: int = 10,
    top_n_plot: int = 30,
):
    """
    Prints a recap, saves CSVs, and writes a few PNG plots:
      - field_dupes.csv                    (lineup_id, dupes)
      - exposures_by_pid.csv               (per (pid, CPT/FLEX))
      - exposures_by_player_merged.csv     (CPT+FLEX merged by name)
      - dupes_hist.png
      - exposures_topN_overall.png
      - exposures_topN_CPT.png
      - exposures_topN_FLEX.png
    """
    outdir = Path(outdir) / f"contest_{contest_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- basic dupe recap ----------
    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    dupe_counts = np.array([int(counts_by_lineup[l]) for l in u_lids], dtype=np.int64)
    M = int(dupe_counts.sum())
    unique_lineups = int((dupe_counts > 0).sum())
    solo = int((dupe_counts == 1).sum())
    dupe_rate = 1.0 - (unique_lineups / max(1, M))

    print("\n=== Sampled Field Recap ===")
    print(f"Total entries (M):        {M:,}")
    print(f"Unique lineups:           {unique_lineups:,}  ({unique_lineups / max(1, M):.1%} of field)")
    print(f"Solo lineups (no dupes):  {solo:,}  ({solo / max(1, M):.1%} of field)")
    print(f"Dupe rate:                {dupe_rate:.1%}")
    if dupe_counts.size:
        print(f"Max dupes of a single lineup: {dupe_counts.max():,}")
        qs = np.quantile(dupe_counts, [0.50, 0.75, 0.90, 0.95, 0.99])
        print("Dupes quantiles (50/75/90/95/99%):", " / ".join(f"{int(x)}" for x in qs))

    # Save dupes table
    dupes_df = pd.DataFrame({"lineup_id": u_lids, "dupes": dupe_counts})
    dupes_df.sort_values("dupes", ascending=False).to_csv(outdir / "field_dupes.csv", index=False)

    # Print top-N duped with player names
    pid_meta = _fetch_player_meta(session, unique_pids)
    print(f"\nTop {min(top_n_print, len(u_lids))} most-duped lineups:")
    for lid, c in dupes_df.sort_values("dupes", ascending=False).head(top_n_print).values:
        names = _lineup_player_names_for(int(lid), session, pid_meta)
        print(f"  lineup {int(lid)}  dupes={int(c):>4}  ->  " + ", ".join(names))

    # Histogram of dupes
    _plot_hist(dupe_counts, f"Lineup dupes distribution (M={M:,})", outdir / "dupes_hist.png")

    # ---------- player exposures ----------
    nP = len(unique_pids)
    counts_per_pid_idx = np.zeros(nP, dtype=np.int64)
    for idxs in lineup_ix:
        if not idxs:  # None or empty
            continue
        # vectorized bump for all players in this entry
        counts_per_pid_idx[np.array(idxs, dtype=int)] += 1

    exposures_rows = []
    for j, pid in enumerate(unique_pids):
        cnt = int(counts_per_pid_idx[j])
        exp = cnt / max(1, M)
        meta = pid_meta.get(pid, {})
        exposures_rows.append({
            "pid": int(pid),
            "name": meta.get("name", str(pid)),
            "team": meta.get("team"),
            "pos": meta.get("pos"),
            "roster_slot": meta.get("roster_slot"),
            "slot_label": meta.get("slot_label"),
            "picked_count": cnt,
            "exposure_pct": exp,
        })
    exposures_df = pd.DataFrame(exposures_rows).sort_values(["exposure_pct", "picked_count"], ascending=False)
    exposures_df.to_csv(outdir / "exposures_by_pid.csv", index=False)

    # merged by player name (CPT+FLEX combined)
    merged = (
        exposures_df
        .groupby(["name", "team", "pos"], dropna=False, as_index=False)
        .agg(picked_count=("picked_count", "sum"))
    )
    merged["exposure_pct"] = merged["picked_count"] / max(1, M)
    merged.sort_values(["exposure_pct", "picked_count"], ascending=False, inplace=True)
    merged.to_csv(outdir / "exposures_by_player_merged.csv", index=False)

    # Print quick top-10
    print("\nTop 10 player exposures (merged CPT+FLEX):")
    for _, row in merged.head(10).iterrows():
        nm = row["name"]
        tm = row.get("team", None)
        pct = row["exposure_pct"]
        cnt = int(row["picked_count"])
        print(f"  {nm:<24} {tm or '—':<4}  {pct:6.2%}  ({cnt:,}/{M:,})")

    # ---------- plots for exposures ----------
    # overall (merged) top N
    top = merged.head(top_n_plot)
    _plot_bar(
        x_labels=top["name"].tolist(),
        values=top["exposure_pct"].to_numpy(),
        title=f"Top {len(top)} player exposures (merged) — M={M:,}",
        out_path=outdir / "exposures_topN_overall.png",
    )

    # CPT-only
    cpt = exposures_df[exposures_df["slot_label"] == "CPT"].sort_values("exposure_pct", ascending=False).head(top_n_plot)
    _plot_bar(
        x_labels=cpt["name"].tolist(),
        values=cpt["exposure_pct"].to_numpy(),
        title=f"Top {len(cpt)} CPT exposures — M={M:,}",
        out_path=outdir / "exposures_topN_CPT.png",
    )

    # FLEX-only
    flex = exposures_df[exposures_df["slot_label"] == "FLEX"].sort_values("exposure_pct", ascending=False).head(top_n_plot)
    _plot_bar(
        x_labels=flex["name"].tolist(),
        values=flex["exposure_pct"].to_numpy(),
        title=f"Top {len(flex)} FLEX exposures — M={M:,}",
        out_path=outdir / "exposures_topN_FLEX.png",
    )

    print(f"\nDiagnostics saved to: {outdir}")


def _coerce_player_ids(val) -> List[int]:
    import json as _json
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            data = _json.loads(val)
            if isinstance(data, list):
                return [int(x) for x in data]
        except Exception:
            return []
    return []

def _expand_sampled_field(
        session: Session,
        counts_by_lineup: Dict[int, int],
) -> Tuple[List[int], List[List[int]], List[int], Dict[int, int], np.ndarray, np.ndarray]:
    """
    Turn sampled counts (with replacement) into: 
      - entry_lineup_ids:   one per entry (length M)
      - lineup_ix:          list of per-entry player-index lists (using pid_to_idx)
      - unique_pids:        compact list of player IDs used across the sampled field
      - pid_to_idx:         map pid -> [0..nP-1]
      - starts, ends:       per-unique-lineup slices in the expanded entries (for aggregations)
    """

    if not counts_by_lineup:
        print('no counts_by_lineup returned from sampling algorithm.')
        return [], [], [], {}, np.array([], dtype=int), np.array([], dtype=int)

    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    cnts   = np.array([int(counts_by_lineup[lid]) for lid in u_lids], dtype=np.int64)
    starts = np.cumsum(np.r_[0, cnts[:-1]])
    ends   = starts + cnts
    M      = int(cnts.sum())

    # fetch lineups

    rows = (session.query(Lineup.id, Lineup.player_ids_list)
                  .filter(Lineup.id.in_(u_lids))
                  .all())
    
    pid_lists_by_lid = {int(lid): _coerce_player_ids(j) for (lid, j) in rows}

    all_pids = set()
    for lid in u_lids:
        all_pids.update(pid_lists_by_lid.get(lid, []))
    unique_pids = sorted(all_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    entry_lineup_ids: List[int]     = [0] * M
    lineup_ix: List[List[int]]      = [None] * M
    for u, lid in enumerate(u_lids): 
        pids = pid_lists_by_lid.get(lid, [])
        idxs = [pid_to_idx[p] for p in pids if p in pid_to_idx]
        a, b = int(starts[u]), int(ends[u])
        for pos in range(a, b): 
            entry_lineup_ids[pos] = lid
            lineup_ix[pos] = idxs

    return entry_lineup_ids, lineup_ix, unique_pids, pid_to_idx, starts, ends  

def _load_and_scale_payouts_from_db(
        session: Session,
        contest_id: int, 
        num_field_lineups: int, 
        original_field_size: int,
        entry_fee: float,
) -> Dict[int, float]: 
    """
    scale payouts to desired contest size
    """
    rows = (session.query(ContestPayout)
                .filter(ContestPayout.contest_id == contest_id)
                .order_by(ContestPayout.min_position.asc())
                .all())
    
    if not rows:
        raise ValueError(f"No payouts found for contest_id={contest_id}")
    
    payouts_df = pd.DataFrame(
        [(r.min_position, r.max_position, r.payout_amount) for r in rows],
        columns=["Start Rank", "End Rank", "Payout"]
    )

    print(f'original payouts_df = {payouts_df}')

    print(f'scaling parameters: num_field_lineups={num_field_lineups}, original_field_size={original_field_size}, entry_fee={entry_fee}')
    return scale_payouts_correctly(
        num_field_lineups=num_field_lineups,
        payouts_df=payouts_df,
        original_field_size=original_field_size,
        entry_fee=entry_fee
    )
def _build_player_samples_matrix(
        session: Session,
        pid_to_idx: Dict[int, int],
        model_dir: str | Path = "data/nfl/models",
        n_samples: int = 1000,
) -> np.ndarray:
    """
    NFL scoring matrix P: [num_sims x num_players] aligned to 'pid_to_idx'.
    Ensures every column has exactly 'num_sims' rows by slicing/tiling as needed.
    """
    # If the caller specified num_sims, force the sampler to produce that many

    raw = generate_correlated_samples_impose(
        session,
        model_dir=model_dir,
        n_samples=n_samples
    )
    # fall back to the first vector length in raw
    ns = 0
    for v in raw.values():
        if isinstance(v, np.ndarray):
            ns = int(v.shape[0])
            print(f'n_sums = {ns}')
            break
    if ns == 0:
        return np.zeros((0, len(pid_to_idx)), dtype=np.float32)

    nP = len(pid_to_idx)
    P  = np.zeros((ns, nP), dtype=np.float32)

    for pid, col in pid_to_idx.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray):
            continue
        vv = v.astype(np.float32, copy=False)

        # Normalize length to ns
        if vv.shape[0] < ns:
            reps = int(np.ceil(ns / vv.shape[0]))
            vv = np.tile(vv, reps)[:ns]
        elif vv.shape[0] > ns:
            vv = vv[:ns]

        P[:, col] = vv

    return P, ns


def _simulate_chunk(
    start: int, 
    end: int,
    lineup_ix: List[List[int]],
    P_chunk: np.ndarray,
    entry_fee: float,
    scaled_payouts: Dict[int, float],
) -> Dict[str, np.ndarray]:

    slice_len = end - start
    M = len(lineup_ix)

    # work in float64 for stability at scale
    scores  = np.zeros((slice_len, M), dtype=np.float64)
    payouts = np.zeros_like(scores)
    rois    = np.zeros_like(scores)
    ranks   = np.zeros((slice_len, M), dtype=np.int32)

    # zero-based payout array: index 0 -> rank 1
    pay_arr = np.zeros(M, dtype=np.float64)
    for r, a in scaled_payouts.items():
        if 1 <= r <= M:
            pay_arr[r - 1] = float(a)
    pay_csum = np.cumsum(pay_arr, dtype=np.float64)

    atol = 1e-8  # tighter since float64

    for local_i in range(slice_len):
        row = P_chunk[local_i]  # [nP]

        # 1) score each entry for THIS sim
        for m in range(M):
            idxs = lineup_ix[m]
            if idxs:  # list truthiness is fine
                scores[local_i, m] = row[idxs].sum(dtype=np.float64)

        # 2) sort by score desc
        order = np.argsort(-scores[local_i], kind="stable")
        sc_sorted = scores[local_i][order]

        # 3) tie groups via explicit boundaries
        diffs = np.abs(np.diff(sc_sorted))
        edges = np.concatenate(([0], np.nonzero(diffs > atol)[0] + 1, [M]))

        start_rank = edges[:-1] + 1                  # 1-indexed
        end_rank   = edges[1:]                       # 1-indexed
        sizes      = end_rank - (start_rank - 1)     # == edges[1:] - edges[:-1]

        # --- FIXED: vectorized pool sums with prefix sums (no array "if") ---
        j2   = end_rank - 1                          # 0-based inclusive end
        j1m1 = start_rank - 2                        # one before start (can be -1)
        pools = pay_csum[j2] - np.where(j1m1 >= 0, pay_csum[j1m1], 0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            per_entry = np.divide(
                pools, sizes,
                out=np.zeros_like(pools, dtype=np.float64),
                where=sizes > 0
            )

        # write per group
        for g in range(len(start_rank)):
            i0, i1 = edges[g], edges[g+1]
            idx_in_field = order[i0:i1]
            if idx_in_field.size == 0:
                continue
            payouts[local_i, idx_in_field] = per_entry[g]
            ranks[local_i, idx_in_field] = int(start_rank[g])

        # 5) ROI
        rois[local_i] = (payouts[local_i] - entry_fee) / entry_fee

        # ======= invariants / anomaly prints =======
        total_payout_this_sim = float(payouts[local_i].sum())
        expected_pool = float(pay_arr.sum())
        if not np.isfinite(total_payout_this_sim) or abs(total_payout_this_sim - expected_pool) > 1e-4:
            print(f"[WARN sim {start + local_i}] payout conservation: got={total_payout_this_sim:.6f}, "
                  f"expected={expected_pool:.6f}")

        for g in range(len(start_rank)):
            r1, r2, sz = int(start_rank[g]), int(end_rank[g]), int(sizes[g])
            if sz <= 0:
                print(f"[WARN sim {start + local_i}] non-positive group size at ranks {r1}..{r2}")
                continue
            sl = pay_arr[r1-1:r2]
            if sl.size == 0:
                print(f"[WARN sim {start + local_i}] empty pay slice {r1}..{r2} (size={sz})")
                continue
            pe = float(per_entry[g])
            lo, hi = float(sl.min()), float(sl.max())
            if (pe < lo - 1e-8 or pe > hi + 1e-8) and (lo != hi):
                print(f"[WARN sim {start + local_i}] per-entry {pe:.6f} outside [{lo:.6f},{hi:.6f}] "
                      f"for tie ranks {r1}..{r2} (size={sz})")

        mx = float(payouts[local_i].max())
        top = float(pay_arr[0]) if M else 0.0
        if mx > top + 1e-8:
            print(f"[WARN sim {start + local_i}] max payout {mx:.6f} > top prize {top:.6f}")

    return {
        "start": start,
        "end": end,
        "scores": scores.astype(np.float32, copy=False),
        "payouts": payouts.astype(np.float32, copy=False),
        "rois": rois.astype(np.float32, copy=False),
        "ranks": ranks,
    }



def simulate_one_contest(
    session: Session,
    contest_id: int,
    contest_name: str,
    config_path: str,
    field_cap: int,
    num_sims: int = 1000,
) -> None:
    # 1) contest info
    cfg      = json.load(open(config_path, encoding="utf-8"))
    contest  = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    mode     = str('showdown' if contest.contest_type == 'Showdown Captain Mode' else 'classic')
    field_sz = int(contest.total_entries)
    capped_field_sz = min(field_cap, field_sz)
    entry_fee = float(contest.entry_fee)

    # 1) sample the field (WITH replacement). This returns counts per lineup_id.
    counts_by_lineup, meta = sample_field(
        session,             
        field_size=capped_field_sz,
        config=cfg,
        mode=mode
    )
    total = sum(counts_by_lineup.values())
    assert total == capped_field_sz, (total, capped_field_sz)
    entry_lids, lineup_ix, unique_pids, pid_to_idx, starts, ends = _expand_sampled_field(session, counts_by_lineup)

    recap_and_plots_for_sampled_field(
        session=session,
        contest_id=contest_id,
        counts_by_lineup=counts_by_lineup,
        lineup_ix=lineup_ix,
        unique_pids=unique_pids,
        outdir="data/nfl/diagnostics",
        top_n_print=10,
        top_n_plot=30,
    )

    M = len(entry_lids)
    if M == 0:
        print(f"[{contest_id}] no sampled entries; nothing to simulate.")
    
    # 2) payouts scaled to M (simulated field size)
    scaled = _load_and_scale_payouts_from_db(
        session=session,
        contest_id=contest_id,
        num_field_lineups=M,
        original_field_size=field_sz,   # scale from original → simulated
        entry_fee=entry_fee,
    )

    # --- debug/inspection: top payouts + last paid position ---
    paid_items = [(rank, amt) for rank, amt in scaled.items() if amt > 0]
    top10 = sorted(paid_items, key=lambda x: (-x[1], x[0]))[:10]

    print("\nTop 10 payouts (rank → payout):")
    for r, a in top10:
        print(f"  {r:>6} → ${a:,.2f}")

    # last position that pays (max rank with payout > 0)
    last_paid = max((r for r, a in scaled.items() if a > 0), default=None)
    if last_paid is None:
        print("No paid positions (all payouts are $0).")
    else:
        print(f"Last paid position: {last_paid}  (pays ${scaled[last_paid]:,.2f})")

    # (optional) how many places are paid and what percent of the field that is
    paid_count = sum(1 for r in range(1, M + 1) if scaled.get(r, 0.0) > 0.0)
    print(f"Paid places: {paid_count}/{M} ({paid_count / max(1, M):.1%})\n")
    # 3) correlated NFL projections for the compact player set

    P, num_sims = _build_player_samples_matrix(
        session=session,
        pid_to_idx=pid_to_idx,
        model_dir="data/nfl/models",
        n_samples=num_sims
    )

    # 4) run simulations in parallel (same “chunk” pattern as MLB)
    chunk_size   = math.ceil(num_sims / min(8, num_sims))   # mirror MLB limiter
    full_scores  = np.zeros((num_sims, M), dtype=np.float32)
    full_payouts = np.zeros_like(full_scores)
    full_rois    = np.zeros_like(full_scores)
    full_ranks   = np.zeros((num_sims, M), dtype=int)

    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = []
        for start in range(0, num_sims, chunk_size):
            end = min(start + chunk_size, num_sims)
            futures.append(
                exe.submit(
                    _simulate_chunk,
                    start,
                    end, 
                    lineup_ix,
                    P[start:end],
                    float(entry_fee),
                    scaled
                )
            )
        for fut in as_completed(futures):
            part = fut.result()
            s, e = part["start"], part["end"]
            full_scores[s:e]  = part["scores"]
            full_payouts[s:e] = part["payouts"]
            full_rois[s:e]    = part["rois"]
            full_ranks[s:e]   = part["ranks"]

    
    # 5) persist per-UNIQUE-lineup aggregates (dupe-aware)
    #    We aggregate across an individual lineup’s duplicate entries using starts/ends.
    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    counts_U = np.array([int(counts_by_lineup[l]) for l in u_lids], dtype=np.int64)

    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        session.add(LineupSimulationResult(
            lineup_id    = int(lid),
            contest_id   = contest_id,
            contest_name = contest_name,
            fpts         = full_scores[:, a:b].mean(axis=1).tolist(),
            payout       = full_payouts[:, a:b].sum(axis=1).tolist(),   # total payout over dupes
            roi          = full_rois[:, a:b].mean(axis=1).tolist(),     # mean ROI per entry
            rank         = full_ranks[:, a:b].mean(axis=1).tolist(),    # mean rank across dupes
        ))

    session.commit()
    print(f"Done (NFL): {num_sims} sims × {M} entries | unique lineups={len(u_lids)} for contest {contest_id}")
        



