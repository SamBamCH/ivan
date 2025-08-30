import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sqlalchemy import func
from sqlalchemy.orm import aliased
from ivan.sports.mlb.models import Lineup, MLBPlayer
from ivan.core.utils import get_config_value

PITCHER_POS = {"SP", "RP", "P"}

class DummyPlayer:
    def __init__(self, pid: int):
        self.id = pid
        self.name = str(pid)      # ensure name is a string
        self.position = []        # empty list so `any(...)` won’t blow up
        self.team = ""            # empty string so formatting works
        self.lp = 0
        self.adj_r = 1
        self.adj_p = 1
        self.fpts_projection = 0.0
        self.stdv = 0.0
        self.ownership = 0.0

def fetch_latest_map(session, all_ids):
    sub = (
        session.query(
            MLBPlayer.id,
            func.max(MLBPlayer.created_at).label("max_dt")
        )
        .filter(MLBPlayer.id.in_(all_ids))
        .group_by(MLBPlayer.id)
        .subquery()
    )
    P2 = aliased(MLBPlayer)
    latest = (
        session.query(P2)
               .join(sub, (P2.id==sub.c.id)&(P2.created_at==sub.c.max_dt))
               .all()
    )
    return {p.id:p for p in latest}

def sample_lineups_fast(
    session,
    n_lineups: int,
    config: dict,
    stack_dist: dict[str, float],
    proj_bin_probs: np.ndarray,
    allowed_player_ids: set[int] | None = None,   # NEW: valid pool filter
) -> list[Lineup]:
    """
    Fast sampler that:
      • Matches a target projection-sum distribution by filling each bin's target count,
        sampling WITH replacement inside the bin (enables top-end duplication).
      • Optionally filters OUT lineups that include players not in `allowed_player_ids`.
        If none provided, no filtering is applied.

    Config (optional):
      - field.proj_exponent: float (default 1.5). Scales per-lineup weights inside bins.
      - field.strict_player_pool: bool (default True). If True, exclude invalid lineups entirely.
      - field.invalid_lineup_penalty: float (default 1e-6). If strict=False, weight penalty for invalid lineups.
    """
    # 1) fetch recent lineups
    cutoff   = datetime.utcnow() - timedelta(hours=24)
    all_lups = session.query(Lineup).filter(Lineup.created_at>=cutoff).all()
    if not all_lups:
        raise ValueError("No recent lineups available.")

    # 2) latest player data & target ownership (for reporting only)
    all_ids    = {pid for lup in all_lups for pid in lup.player_ids_list}
    latest_map = fetch_latest_map(session, all_ids)
    target_own = {
        pid: (latest_map.get(pid, DummyPlayer(pid)).ownership or 0.0)/100.0
        for pid in all_ids
    }

    # 3) build pool & compute raw projections + stacks
    pool      = [(lup, lup.player_ids_list) for lup in all_lups]
    N         = len(pool)
    proj_vals = np.zeros(N, dtype=float)
    stacks    = [None]*N

    # NEW: valid lineup mask (all players in allowed pool)
    if allowed_player_ids is not None:
        allowed_player_ids = set(allowed_player_ids)
        def _is_valid_lineup(pids: list[int]) -> bool:
            return all((pid in allowed_player_ids) for pid in pids)
        valid_mask = np.array([_is_valid_lineup(pids) for _, pids in pool], dtype=bool)
    else:
        valid_mask = np.ones(N, dtype=bool)

    for i, (_, pids) in enumerate(pool):
        players      = [latest_map.get(pid, DummyPlayer(pid)) for pid in pids]
        proj_vals[i] = sum(p.fpts_projection for p in players)
        cnt          = Counter(p.team for p in players)
        top_two      = sorted(cnt.values(), reverse=True)
        stacks[i]    = f"{top_two[0]}-{top_two[1] if len(top_two)>1 else 0}"

    # 4) normalize projections to [0,1]
    pmin, pmax = proj_vals.min(), proj_vals.max()
    if pmax > pmin:
        norm_proj = (proj_vals - pmin)/(pmax - pmin)
    else:
        norm_proj = np.zeros_like(proj_vals)

    # 5) assign bins
    alpha_proj = get_config_value(config, "field.proj_exponent") or 1.5
    strict_pool = bool(get_config_value(config, "field.strict_player_pool"))
    invalid_pen = float(get_config_value(config, "field.invalid_lineup_penalty") or 1e-6)

    B         = len(proj_bin_probs)
    bin_edges = np.linspace(0,1,B+1)
    proj_bins = np.clip(np.digitize(norm_proj, bin_edges)-1, 0, B-1)

    # 6) stack weights and pool weights (independent of projection bin)
    w_stack = np.array([stack_dist.get(s, 1e-6) for s in stacks], dtype=float)

    # If not strict, give invalid a tiny weight; if strict, exclude them later.
    pool_penalty = np.where(valid_mask, 1.0, (0.0 if strict_pool else invalid_pen))

    # 7) compute integer target counts per bin that sum to n_lineups
    raw_counts = proj_bin_probs * float(n_lineups)
    base = np.floor(raw_counts).astype(int)
    remainder = n_lineups - base.sum()
    if remainder != 0:
        # distribute remainder by largest fractional parts
        frac = raw_counts - base
        order = np.argsort(frac)[::-1] if remainder > 0 else np.argsort(frac)
        for b in order[:abs(remainder)]:
            base[b] += 1 if remainder > 0 else -1
    targets = base  # length B, sums to n_lineups

    # Pre-index lineups by bin for fast sampling
    idx_by_bin = [np.where(proj_bins == b)[0] for b in range(B)]
    valid_idx_by_bin = [idx[valid_mask[idx]] for idx in idx_by_bin]

    # 8) sample per bin WITH replacement (enables duplication when bin is thin)
    chosen_idx_list: list[int] = []
    for b in range(B):
        k = int(targets[b])
        if k <= 0:
            continue

        # Prefer valid lineups in this bin
        cand_idx = valid_idx_by_bin[b]
        if len(cand_idx) == 0:
            # Fall back: if strict, expand search to any other bin that has valid;
            # otherwise, allow invalid in this bin with penalty weight.
            if strict_pool:
                # nearest non-empty valid bin search
                found = None
                for d in range(1, B):
                    for nb in (b-d, b+d):
                        if 0 <= nb < B and len(valid_idx_by_bin[nb]) > 0:
                            found = valid_idx_by_bin[nb]
                            break
                    if found is not None:
                        break
                cand_idx = found if found is not None else np.where(valid_mask)[0]
            else:
                cand_idx = idx_by_bin[b]  # include invalid with small penalty

        # Build weights inside candidate set:
        #   - projection bin alignment (sharpen via alpha_proj)
        #   - stack weights
        #   - pool penalty (0 if strict & invalid, tiny if soft)
        bin_center_weight = (proj_bin_probs[b] ** alpha_proj)
        w = (w_stack[cand_idx].clip(min=1e-12)) * bin_center_weight * pool_penalty[cand_idx]
        if not np.isfinite(w).any() or w.sum() <= 0:
            # fallback uniform if pathological
            w = np.ones_like(cand_idx, dtype=float)
        w = w / w.sum()

        # WITH replacement sampling so duplication is possible
        picks = np.random.choice(cand_idx, size=k, replace=True, p=w)
        chosen_idx_list.extend(picks.tolist())

    # (Paranoia) if rounding issues left us short/long, adjust
    if len(chosen_idx_list) < n_lineups:
        # top off from global valid pool (or full pool)
        surplus = n_lineups - len(chosen_idx_list)
        global_cand = np.where(valid_mask)[0] if strict_pool else np.arange(N)
        gw = (w_stack[global_cand].clip(min=1e-12)) * pool_penalty[global_cand]
        gw = gw / gw.sum()
        extra = np.random.choice(global_cand, size=surplus, replace=True, p=gw)
        chosen_idx_list.extend(extra.tolist())
    elif len(chosen_idx_list) > n_lineups:
        chosen_idx_list = chosen_idx_list[:n_lineups]

    chosen_idx = np.array(chosen_idx_list, dtype=int)

    # 9) attach players and collect chosen lineups
    chosen: list[Lineup] = []
    for idx in chosen_idx:
        lup, pids = pool[idx]
        lup.players = [latest_map.get(pid, DummyPlayer(pid)) for pid in pids]
        chosen.append(lup)

    # 10) compute exposures & deviations (unchanged)
    exposures = Counter(p.id for lup in chosen for p in lup.players)
    total     = len(chosen)
    pid2player = {p.id:p for lup in chosen for p in lup.players}

    pitchers, batters = {}, {}
    for pid, cnt in exposures.items():
        expf    = cnt/total
        proj_own= target_own.get(pid,0.0)
        pl      = pid2player[pid]
        entry   = (expf, proj_own, getattr(pl,"name",f"PID {pid}"), pl.team or "NA")
        if any(pos in PITCHER_POS for pos in pl.position):
            pitchers[pid] = entry
        else:
            batters[pid]  = entry

    def print_exposures(title, dct, top_n=None):
        print(f"\n{title}")
        print(f"{'Name':<20}  {'Team':<5}  {'Exposure':>10}  {'ProjOwn':>10}")
        items = sorted(dct.values(), key=lambda x:x[0], reverse=True)
        if top_n: items = items[:top_n]
        for expf, projf, nm, tm in items:
            print(f"{nm:<20}  {tm:<5}  {expf:>9.2%}  {projf:>9.2%}")

    def print_deviators(title, dct, top_n=5):
        print(f"\n{title}")
        print(f"{'Name':<20}  {'Team':<5}  {'Exposure':>10}  {'ProjOwn':>10}  {'Dev':>8}")
        devs = [(abs(expf-projf), expf, projf, nm, tm) for expf,projf,nm,tm in dct.values()]
        for dev, expf, projf, nm, tm in sorted(devs, reverse=True)[:top_n]:
            print(f"{nm:<20}  {tm:<5}  {expf:>9.2%}  {projf:>9.2%}  {dev:>7.2%}")

    print_exposures("ALL Pitcher Exposures",   pitchers)
    print_exposures("Top 20 Batter Exposures",  batters, top_n=5)
    print_deviators("Top 5 Pitcher Deviations", pitchers)
    print_deviators("Top 5 Batter Deviations",  batters)

    # 11) plot projection histogram vs target (same as before)
    chosen_proj = np.array([sum(p.fpts_projection for p in lup.players) for lup in chosen])
    if pmax>pmin:
        chosen_norm = (chosen_proj - pmin)/(pmax - pmin)
    else:
        chosen_norm = np.zeros_like(chosen_proj)

    plt.hist(chosen_norm, bins=bin_edges, edgecolor="black", alpha=0.7, label="Sampled")
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(centers, proj_bin_probs*total, "--", linewidth=2, label="Target")
    plt.title("Lineup Normalized Projection Sum")
    plt.xlabel("Normalized Projection (0–1)")
    plt.ylabel("Number of Lineups")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return chosen

