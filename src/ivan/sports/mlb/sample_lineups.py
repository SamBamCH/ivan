import numpy as np
import random
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sqlalchemy import func
from sqlalchemy.orm import aliased
from ivan.sports.mlb.models import Lineup, MLBPlayer
from ivan.core.utils import get_config_value, show_player_exposures, show_team_exposures

PITCHER_POS = {"SP", "RP", "P"}

class DummyPlayer:
    """Zero‑projection/ownership placeholder for missing players."""
    def __init__(self, id_):
        self.id              = id_
        self.name            = "Unknown"
        self.team            = "N/A"
        self.position        = ["Unknown"]
        self.salary          = 0
        self.fpts_projection = 0.0
        self.ownership       = 0.0
        self.ops             = 0.0
        self.adjusted_params = (0, 0)


def diagnostic_report(
    own_avg,
    proj_norm,
    stack_labels,
    stack_dist,
    final_inds,
    bw_own,
    bw_proj
):
    """
    Generates diagnostics comparing predicted vs actual distributions:
      1) Ownership KDE vs sampled histogram
      2) Projection KDE vs sampled histogram
      3) Stack-type predicted vs actual bar chart

    Parameters:
    - own_avg: array-like of avg ownership per lineup
    - proj_norm: array-like of normalized projection sums per lineup
    - stack_labels: list of stack-type labels (one per lineup)
    - stack_dist: dict mapping each stack-type string → predicted probability
    - final_inds: list/array of selected lineup indices
    - bw_own: float bandwidth for ownership KDE
    - bw_proj: float bandwidth for projection KDE
    """
    own_avg    = np.asarray(own_avg)
    proj_norm  = np.asarray(proj_norm)
    final_inds = np.asarray(final_inds)

    # Ownership plot
    x_own = np.linspace(own_avg.min(), own_avg.max(), 200)
    kde_own = gaussian_kde(own_avg, bw_method=bw_own)
    plt.figure(figsize=(6, 4))
    plt.plot(x_own, kde_own(x_own), label='Predicted KDE (Own)')
    plt.hist(own_avg[final_inds], bins=30, density=True, alpha=0.6, label='Actual Sampled (Own)')
    plt.title('Ownership: Predicted vs Actual')
    plt.xlabel('Avg Ownership')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Projection plot
    x_proj = np.linspace(proj_norm.min(), proj_norm.max(), 200)
    kde_proj = gaussian_kde(proj_norm, bw_method=bw_proj)
    plt.figure(figsize=(6, 4))
    plt.plot(x_proj, kde_proj(x_proj), label='Predicted KDE (Proj)')
    plt.hist(proj_norm[final_inds], bins=30, density=True, alpha=0.6, label='Actual Sampled (Proj)')
    plt.title('Projection: Predicted vs Actual')
    plt.xlabel('Normalized Projection')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Stack-type distribution
    stack_types = sorted(stack_dist.keys())
    y_pred = np.array([stack_dist.get(st, 0.0) for st in stack_types])
    counts = Counter(stack_labels[i] for i in final_inds)
    total = len(final_inds)
    y_act = np.array([counts.get(st, 0)/total for st in stack_types])

    x = np.arange(len(stack_types))
    w = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - w/2, y_pred, width=w, label='Predicted')
    plt.bar(x + w/2, y_act, width=w, label='Actual')
    plt.xticks(x, stack_types, rotation=45, ha='right')
    plt.title('Stack-type: Predicted vs Actual')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.show()


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
               .join(sub, (P2.id == sub.c.id) & (P2.created_at == sub.c.max_dt))
               .all()
    )
    return {p.id: p for p in latest}


def sample_lineups_fast(
    session,
    n_lineups,
    config,
    own_bin_probs,
    proj_bin_probs,
    stack_dist,
    bin_edges
):
    """
    Fast sampler that:
      • Matches fixed-bin ownership, projection, and stack distributions
      • Injects noise for exploration
      • Penalizes individual player exposure deviations
    """
    # 1) fetch recent lineups
    cutoff = datetime.utcnow() - timedelta(hours=24)
    all_lups = session.query(Lineup).filter(Lineup.created_at >= cutoff).all()
    if len(all_lups) < n_lineups:
        raise ValueError(f"Only {len(all_lups)} lineups available, need {n_lineups}.")

    # 2) fetch latest player info
    all_ids = {pid for lup in all_lups for pid in lup.player_ids_list}
    latest_map = fetch_latest_map(session, all_ids)
    target_own = {pid: (latest_map.get(pid, DummyPlayer(pid)).ownership or 0)/100.0 for pid in all_ids}

    # 3) compute per-lineup metrics
    proj_vals, own_vals, stacks, records = [], [], [], []
    for lup in all_lups:
        players = [latest_map.get(pid, DummyPlayer(pid)) for pid in lup.player_ids_list]
        proj_vals.append(sum(p.fpts_projection or 0.0 for p in players))
        own_vals.append(np.mean([p.ownership/100.0 for p in players]))
        cnt=Counter(p.team for p in players);
        top=sorted(cnt.values(), reverse=True)
        stacks.append(f"{top[0]}-{top[1] if len(top)>1 else 0}")
        records.append((lup, [p.id for p in players]))

    proj_vals=np.array(proj_vals); max_p=proj_vals.max() or 1.0
    proj_norm=proj_vals/max_p; own_vals=np.array(own_vals)

    # 4) bins
    own_bins=np.clip(np.digitize(own_vals, bin_edges)-1,0,len(own_bin_probs)-1)
    proj_bins=np.clip(np.digitize(proj_norm,bin_edges)-1,0,len(proj_bin_probs)-1)

    # 5) base weights
    base_w=own_bin_probs[own_bins]*proj_bin_probs[proj_bins]*np.array([stack_dist.get(s,1e-6) for s in stacks])
    w=np.clip(base_w,1e-12,None); w/=w.sum()

    # 6) noise
    sigma=get_config_value(config,'field.noise_sigma') or 0.1
    w+=sigma*np.random.rand(len(w)); w=np.clip(w,1e-12,None); w/=w.sum()

    # 7) incidence matrix
    player_ids=sorted(all_ids); pid2idx={pid:i for i,pid in enumerate(player_ids)}
    N=len(records); P=len(player_ids)
    M=np.zeros((N,P),dtype=int)
    for i,(_,pids) in enumerate(records):
        for pid in pids: M[i,pid2idx[pid]]=1

    # 8) expected exposures
    exp_j=w@M

    # 9) player penalties
    beta=get_config_value(config,'field.penalty_beta') or 10.0
    dev=exp_j-np.array([target_own[pid] for pid in player_ids])
    player_pen=np.exp(-beta*dev**2)

    #10) lineup penalties
    lineup_pen=np.prod(np.where(M,player_pen,1.0),axis=1)

    #11) combine and sample
    w*=lineup_pen; w=np.clip(w,1e-12,None); w/=w.sum()
    chosen_idx=np.random.choice(N,n_lineups,replace=False,p=w)

    #12) collect
    chosen=[]
    for idx in chosen_idx:
        lup,_=records[idx]
        lup.players=[latest_map.get(pid,DummyPlayer(pid)) for pid in lup.player_ids_list]
        chosen.append(lup)

    #13) reporting + diagnostics
    lists=[l.player_ids_list for l in chosen]
    show_player_exposures(session,lists)
    show_team_exposures(session,lists,stack_size=4)
    diagnostic_report(
        own_avg=own_vals,
        proj_norm=proj_norm,
        stack_labels=stacks,
        stack_dist=stack_dist,
        final_inds=chosen_idx,
        bw_own=None,
        bw_proj=None
    )

    return chosen