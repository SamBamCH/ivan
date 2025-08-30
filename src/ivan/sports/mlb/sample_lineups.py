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
    proj_bin_probs: np.ndarray
) -> list[Lineup]:
    """
    Fast approximate sampler: one‐shot weighted draw (no per‐draw penalties).
    Matches:
      • normalized projection‐sum distribution (proj_bin_probs over [0,1])
      • stack‐type distribution (stack_dist)
    Then prints exposures & deviations and plots projection histogram.
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

    # 5) build one‐shot weight vector
    alpha_proj = get_config_value(config, "field.proj_exponent") or 1.5
    B          = len(proj_bin_probs)
    bin_edges  = np.linspace(0,1,B+1)
    proj_bins  = np.clip(np.digitize(norm_proj, bin_edges)-1, 0, B-1)
    w_proj     = proj_bin_probs[proj_bins] ** alpha_proj
    w_stack    = np.array([stack_dist.get(s,1e-6) for s in stacks])
    w          = w_proj * w_stack
    w          = np.clip(w, 1e-12, None)
    w         /= w.sum()

    # 6) sample all at once (with replacement)
    chosen_idx = np.random.choice(N, size=n_lineups, replace=True, p=w)

    # 7) attach players and collect chosen lineups
    chosen = []
    for idx in chosen_idx:
        lup, pids = pool[idx]
        lup.players = [latest_map.get(pid, DummyPlayer(pid)) for pid in pids]
        chosen.append(lup)

    # 8) compute exposures & deviations
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

    # 9) plot projection histogram vs target
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
