import random
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

from sqlalchemy import func
from sqlalchemy.orm import aliased
import matplotlib.pyplot as plt

from ivan.sports.mlb.models import Lineup, MLBPlayer
from ivan.core.utils import get_config_value, show_player_exposures, show_team_exposures

PITCHER_POS = {"SP", "RP", "P"}


class DummyPlayer:
    """Zero‐projection/ownership placeholder for missing players."""
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


def diagnostic_report(session, chosen_lineups):
    # pull in the player rows for all lineups
    all_pids = {pid for lup in chosen_lineups for pid in lup.player_ids_list}
    players = session.query(MLBPlayer).filter(MLBPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}

    ownership_avgs = []
    proj_sums      = []
    stack_types    = []

    for lup in chosen_lineups:
        pids = lup.player_ids_list
        proj = sum(player_map[pid].fpts_projection for pid in pids)
        proj_sums.append(proj)

        own = sum(player_map[pid].ownership / 100 for pid in pids) / len(pids)
        ownership_avgs.append(own)

        counts = Counter()
        for pid in pids:
            p = player_map[pid]
            if not any(pos in PITCHER_POS for pos in p.position):
                counts[p.team] += 1
        top_two = sorted(counts.values(), reverse=True)[:2]
        c1, c2 = top_two[0], (top_two[1] if len(top_two) > 1 else 0)
        stack_types.append(f"{c1}-{c2}")

    plt.figure()
    plt.hist(ownership_avgs, bins=20)
    plt.title("Field: Lineup Average Ownership")
    plt.xlabel("Avg Ownership")
    plt.ylabel("Count")
    plt.show()

    plt.figure()
    plt.hist(proj_sums, bins=20)
    plt.title("Field: Lineup Projection Sums")
    plt.xlabel("Total Projected FPTS")
    plt.ylabel("Count")
    plt.show()

    counts = Counter(stack_types)
    total  = len(stack_types)
    probs  = {stype: cnt / total for stype, cnt in counts.items()}

    import json
    print("\nstack_probabilities:")
    print(json.dumps(probs, indent=4))


def fetch_latest_map(session, all_ids):
    """
    Return a dict {player_id: MLBPlayer} where each MLBPlayer is
    the row with the max(created_at) for that id.
    """
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
    slate_features: dict,
    kde_regressor_path: str = None
):
    """
    slate_features must contain exactly the 9 keys:
       num_games, total_entries, max_entries, entry_fee,
       top_owned, top_5_owned, top_10_owned, first_pct, top5_pct
    kde_regressor_path: path to your pickle of (scaler,rf_bw_own,rf_bw_proj,rf_max_proj)
    """
        # --- load regressors & predict KDE params ---
    import pickle
    from scipy.stats import gaussian_kde

    models = pickle.load(open(kde_regressor_path, 'rb'))
    scaler    = models['scaler']
    rf_bw_own = models['bw_own']
    rf_bw_proj= models['bw_proj']
    rf_max    = models['max_proj']

    # assemble feature‐vector and standardize
    X_star = np.array([[ slate_features[k] for k in [
        'num_games','total_entries','max_entries','entry_fee',
        'top_owned','top_5_owned','top_10_owned','first_pct','top5_pct'
    ]]], dtype=float)
    Xs     = scaler.transform(X_star)

    # predict: these become your KDE bandwidths & normalization
    bw_own   = float(rf_bw_own .predict(Xs)[0])
    bw_proj  = float(rf_bw_proj.predict(Xs)[0])
    max_proj = float(rf_max   .predict(Xs)[0])
    # 1) Fetch recent lineups
    cutoff = datetime.utcnow() - timedelta(hours=24)
    all_lups = (
        session.query(Lineup)
               .filter(Lineup.created_at >= cutoff)
               .all()
    )
    if len(all_lups) < n_lineups:
        raise ValueError(f"Only {len(all_lups)} lineups available, need {n_lineups}.")

    # 2) Fetch latest player data
    all_ids = {pid for lup in all_lups for pid in lup.player_ids_list}
    latest_map = fetch_latest_map(session, all_ids)

    # 3) Build record list
    records = []
    for lup in all_lups:
        players_objs = [latest_map.get(pid, DummyPlayer(pid)) for pid in lup.player_ids_list]
        proj_sum = sum(p.fpts_projection or 0.0 for p in players_objs)
        pids     = [p.id for p in players_objs]
        records.append({"lineup": lup, "players": players_objs, "proj_sum": proj_sum, "pids": pids})

    # 4) Prefilter top K by projection
    pre_mul = get_config_value(config, "field.prefilter_multiplier") or 5
    K       = max(n_lineups, min(len(records), int(n_lineups * pre_mul)))
    records.sort(key=lambda r: r["proj_sum"], reverse=True)
    records = records[:K]

    # 5) Build target ownership map
    target_own = {}
    for rec in records:
        for p in rec["players"]:
            target_own[p.id] = max(0.0, getattr(p, "ownership", 0.0)) / 100.0

    # 6) Configurable weights & penalty
    proj_param = get_config_value(config, "field.projection_weight") or 1.0
    own_param  = get_config_value(config, "field.ownership_weight")  or 1.0
    beta       = get_config_value(config, "field.penalty_beta")      or 10.0
    R_iters    = get_config_value(config, "field.reweight_iters")    or 4

    # 7) Prepare arrays
    N_total           = len(records)
    lineup_objs       = [rec["lineup"] for rec in records]
    players_objs_list = [rec["players"] for rec in records]
    pids_list         = [rec["pids"] for rec in records]
    proj_vals         = np.array([rec["proj_sum"] for rec in records], dtype=float)
    max_proj          = proj_vals.max() if proj_vals.max() > 0 else 1.0

    # ownership average per lineup
    own_avg = np.array([
        np.mean([target_own.get(pid, 0.0) for pid in pids]) if pids else 0.0
        for pids in pids_list
    ], dtype=float)

    #  — continuous‐KDE weighting —
    proj_norm = proj_vals / max_proj

    # build KDEs on the field itself
    kde_own  = gaussian_kde(own_avg,    bw_method=bw_own)
    kde_proj = gaussian_kde(proj_norm,  bw_method=bw_proj)

    # evaluate each lineup's PDF value
    p_own  = kde_own.evaluate(own_avg)    # shape (N_total,)
    p_proj = kde_proj.evaluate(proj_norm) # shape (N_total,)

    # combine by your configurable exponents
    w = (p_proj ** proj_param) * (p_own ** own_param)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()

    # 9) Iterative re-weighting with squared over-exposure penalty
    for _ in range(R_iters):
        # a) sample batch
        chosen_inds = np.random.choice(N_total, size=n_lineups, replace=False, p=w)

        # b) compute e_hat
        count_j = Counter()
        for idx in chosen_inds:
            for pid in pids_list[idx]:
                count_j[pid] += 1

        # c) compute per-player overage
        over_j = {
            pid: max(0.0, count_j.get(pid, 0) / n_lineups - t_j)
            for pid, t_j in target_own.items()
        }

        # d) compute lineup-level penalty (squared to amplify)
        over_sum = np.array([
            sum(over_j.get(pid, 0.0) for pid in pids)
            for pids in pids_list
        ], dtype=float)

        penalty = np.exp(-beta * (over_sum ** 2))

        # e) update weights
        w = w * penalty
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()

        # — Log-normal jitter
    sigma = get_config_value(config, "field.noise_sigma") or 0.1
    log_w = np.log(w)
    noise = np.random.normal(0, sigma, size=log_w.shape)
    w = np.exp(log_w + noise)
    w = np.clip(w, 1e-12, None)
    w /= w.sum()

    # 10) Final draw
    final_inds = np.random.choice(N_total, size=n_lineups, replace=False, p=w)

    # 11) Build chosen list
    chosen = []
    for idx in final_inds:
        lup = lineup_objs[idx]
        lup.players = players_objs_list[idx]
        chosen.append(lup)

    # 12) Report
    sel_id_lists = [lup.player_ids_list for lup in chosen]
    print("\n--- EXPOSURES FOR SELECTED LINEUPS ---")
    show_player_exposures(session, sel_id_lists)
    show_team_exposures(session, sel_id_lists, stack_size=4)
    print("--- END SELECTED-LINEUP EXPOSURES ---\n")

    return chosen
