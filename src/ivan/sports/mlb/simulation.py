import math
import json
import pickle
from pathlib import Path
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import nbinom, norm
from sqlalchemy.orm import Session
from sqlalchemy import func


from ivan.database.session import init_db, get_session, Base
from ivan.sports.mlb.models import LineupSimulationResult, MLBPlayer, HistoricalContest, ContestPayout
from ivan.sports.mlb.sample_lineups import sample_lineups_fast
from ivan.sports.mlb.payouts import load_and_scale_payouts_from_db
from ivan.core.utils import project_root, match_contest_id


class DummyPlayer:
    def __init__(self, pid: int):
        self.id = pid
        self.position = ""
        self.team = None
        self.lp = 0
        self.adj_r = 1
        self.adj_p = 1
        self.fpts_projection = 0.0
        self.stdv = 0.0
        self.ownership = 0.0


def load_copulas():
    models_dir = Path(project_root()) / "data" / "mlb" / "models"
    with open(models_dir / "team_copulas.pkl", "rb") as f:
        team_copulas = pickle.load(f)
    return team_copulas


def _simulate_chunk(
    start: int,
    end: int,
    lineup_ix: list[list[int]],
    unique_players: list,
    team_copulas: dict[str, any],
    entry_fee: float,
    scaled_payouts: dict[int, float]
) -> dict:
    M = len(lineup_ix)
    slice_len = end - start
    scores = np.zeros((slice_len, M), dtype=float)
    payouts = np.zeros_like(scores)
    rois = np.zeros_like(scores)
    ranks = np.zeros((slice_len, M), dtype=int)

    # sample copulas for this chunk
    chunk_draws = {team: cop.sample(slice_len) for team, cop in team_copulas.items()}

    for local_i, sim in enumerate(range(start, end)):
        # simulate batters
        fpts = {}
        by_team = {}
        for p in unique_players:
            if "P" not in p.position:
                by_team.setdefault(p.team, []).append(p)
        for team, bats in by_team.items():
            draws = chunk_draws.get(team)
            if draws is not None and draws.shape[1] >= len(bats):
                z = draws[local_i]
                u = norm.cdf(z)
                for p in sorted(bats, key=lambda x: x.lp):
                    fpts[p] = float(nbinom.ppf(u[p.lp - 1], p.adj_r, p.adj_p))
            else:
                for p in bats:
                    fpts[p] = float(nbinom.rvs(p.adj_r, p.adj_p))

        # simulate pitchers
        for p in unique_players:
            if "P" in p.position:
                mu, sd = p.fpts_projection, getattr(p, 'stdv', 0.0)
                fpts[p] = max(0.0, np.random.normal(mu, sd))

        # compute scores for each lineup
        for idx, ix in enumerate(lineup_ix):
            scores[local_i, idx] = sum(fpts[unique_players[i]] for i in ix)

        # ranking and payouts (once per sim)
        order = np.argsort(-scores[local_i])
        rank_positions = np.empty(M, dtype=int)
        for rpos, lid in enumerate(order, start=1):
            rank_positions[lid] = rpos
        ranks[local_i] = rank_positions
        for ridx in range(M):
            pay = scaled_payouts.get(rank_positions[ridx], 0.0)
            payouts[local_i, ridx] = pay
            rois[local_i, ridx] = (pay - entry_fee) / entry_fee

    return {"start": start, "end": end, "scores": scores, "payouts": payouts, "rois": rois, "ranks": ranks}

def plot_scaled_field_metrics(session, chosen_lineups):
    # 1) pull in MLBPlayer rows
    all_pids = {pid for lup in chosen_lineups for pid in lup.player_ids_list}
    players = session.query(MLBPlayer).filter(MLBPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}

    # 2) compute raw metrics
    proj_sums = []
    own_sums  = []
    for lup in chosen_lineups:
        psum = sum(player_map.get(pid, DummyPlayer(pid)).fpts_projection or 0.0
                   for pid in lup.player_ids_list)
        osum = sum(player_map.get(pid, DummyPlayer(pid)).ownership/100.0
                   for pid in lup.player_ids_list)
        proj_sums.append(psum)
        own_sums .append(osum)

    # 3) print out raw values & summary stats
    print("\n--- RAW FIELD METRICS ---")
    print("First 10 projection sums:", proj_sums[:10])
    print("First 10 ownership sums :", own_sums [:10])

    for name, vals in [("Projection", proj_sums), ("Ownership", own_sums)]:
        arr = np.array(vals)
        print(f"{name:10} — min: {arr.min():.3f}, "
              f"mean: {arr.mean():.3f}, median: {np.median(arr):.3f}, "
              f"max: {arr.max():.3f}")

    # 4) now scale to [0,1] exactly as before
    def scale(v):
        lo, hi = min(v), max(v)
        rng    = hi - lo if hi>lo else 1.0
        return [(x-lo)/rng for x in v]

    scaled_proj = scale(proj_sums)
    scaled_own  = scale(own_sums)

    # 5) plot (anchored 0–1)
    plt.figure()
    plt.hist(scaled_proj, bins=20, range=(0,1))
    plt.title("Scaled Projections (0–1)")
    plt.xlabel("Scaled Projection")
    plt.ylabel("Count")
    plt.xlim(0,1)
    plt.show()

    plt.figure()
    plt.hist(scaled_own, bins=20, range=(0,1))
    plt.title("Scaled Ownership (0–1)")
    plt.xlabel("Scaled Ownership Sum")
    plt.ylabel("Count")
    plt.xlim(0,1)
    plt.show()



def simulate_one_contest(
    session: Session,
    contest_id: int,
    contest_name: str,
    num_sims: int,
    config_path: str
) -> None:
    # 1) load config and canonical contest
    cfg = json.load(open(config_path, encoding='utf-8'))
    actual_cid = match_contest_id(session, contest_id, contest_name)
    contest = session.query(HistoricalContest).filter_by(contest_id=actual_cid).one()

    # 2) derive slate-level features
    players = session.query(MLBPlayer).all()
    teams = {p.team for p in players}
    owns = [(p.ownership or 0.0)/100.0 for p in players]
    top_owned   = max(owns) if owns else 0.0
    top5_owned  = np.mean(sorted(owns, reverse=True)[:5]) if owns else 0.0
    top10_owned = np.mean(sorted(owns, reverse=True)[:10]) if owns else 0.0

    payouts = session.query(ContestPayout).filter_by(contest_id=actual_cid).all()
    total_pool = contest.entry_fee * contest.total_entries
    first_pct = next((p.payout_amount for p in payouts if p.min_position == 1), 0.0) / total_pool
    top5_pct  = sum(p.payout_amount for p in payouts if p.min_position <= 5) / total_pool

    slate_features = np.array([[  
        len(teams)//2,
        contest.total_entries,
        contest.multi_entry,
        float(contest.entry_fee),
        top_owned, top5_owned, top10_owned,
        first_pct, top5_pct
    ]], dtype=float)

    # 3) load bin regressors and predict distributions
    bin_fp = Path(project_root()) / "data" / "mlb" / "models" / "bin_regressors.pkl"
    with open(bin_fp, "rb") as f:
        bin_md = pickle.load(f)
    scaler        = bin_md['scaler']
    Xs_new        = scaler.transform(slate_features)
    proj_bin_probs = bin_md['proj_model'].predict(Xs_new)[0]
    stack_dist     = dict(zip(
        bin_md['stack_types'],
        bin_md['stack_model'].predict(Xs_new)[0]
    ))

    # 4) sample lineups
    chosen = sample_lineups_fast(
        session=session,
<<<<<<< Updated upstream
        n_lineups=min(contest.total_entries, 50000), #TODO: wrap this lower for worse systems. 35000 lineups x 1000 sims uses like 16 gb of RAM, 10000 x1000 should be able to run on any system i would think. 
=======
        n_lineups=min(contest.total_entries, 15000),
>>>>>>> Stashed changes
        config=cfg,
        stack_dist=stack_dist,
        proj_bin_probs=proj_bin_probs,
    )

    # 5) load scaled payouts
    M = len(chosen)
    scaled = load_and_scale_payouts_from_db(
        session=session,
        contest_id=actual_cid,
        num_field_lineups=M,
        original_field_size=contest.total_entries,
        entry_fee=float(contest.entry_fee)
    )

    # 6) prepare for simulation
    unique_players = []
    for lup in chosen:
        for p in lup.players:
            if p not in unique_players:
                unique_players.append(p)

    # map each player object to its index
    p2idx = {p: idx for idx, p in enumerate(unique_players)}

    # now build the index‐lists for each lineup
    lineup_ix = [
        [p2idx[p] for p in lup.players]
        for lup in chosen
    ]

    # 7) run simulations in parallel
    chunk_size = math.ceil(num_sims / min(8, num_sims)) #TODO: pull this out to config file
    full_scores  = np.zeros((num_sims, M))
    full_payouts = np.zeros_like(full_scores)
    full_rois    = np.zeros_like(full_scores)
    full_ranks   = np.zeros((num_sims, M), dtype=int)

    with ProcessPoolExecutor(max_workers=min(8, num_sims)) as exe: #TODO: pull this out to config file
        futures = []
        copulas = load_copulas()
        for start in range(0, num_sims, chunk_size):
            end = min(start + chunk_size, num_sims)
            futures.append(
                exe.submit(
                    _simulate_chunk,
                    start, end,
                    lineup_ix,
                    unique_players,
                    copulas,
                    float(contest.entry_fee),
                    scaled
                )
            )
        for fut in as_completed(futures):
            part = fut.result()
            s, e = part['start'], part['end']
            full_scores[s:e]   = part['scores']
            full_payouts[s:e]  = part['payouts']
            full_rois[s:e]     = part['rois']
            full_ranks[s:e]    = part['ranks']

    # 8) persist
    for idx, l in enumerate(chosen):
        session.add(LineupSimulationResult(
            lineup_id    = l.id,
            contest_id   = actual_cid,
            contest_name = contest_name,
            fpts         = full_scores[:, idx].tolist(),
            payout       = full_payouts[:, idx].tolist(),
            roi          = full_rois[:, idx].tolist(),
            rank         = full_ranks[:, idx].tolist()
        ))
    session.commit()
    print(f"Done: {num_sims} sims × {M} lineups for contest {actual_cid}")