##game.py

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import NFLStat
from .team import TEAM10_SLOTS, build_team_correlation


def _default_cross_correlation() -> np.ndarray:
    """
    Default cross-team Spearman correlation when insufficient league data.

    - Offense vs offense: small positive (game flow)
    - K vs offense: small positive (shootouts raise attempts/XP)
    - K vs K: small positive
    - Offense/K vs opposing DST: negative
    - DST vs DST: small positive
    """
    n = 10
    corr = np.zeros((n, n))
    slot_idx = {s: i for i, s in enumerate(TEAM10_SLOTS)}

    offense = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE1', 'TE2']
    k = 'K'
    dst = 'DST'

    # Offense vs offense (A vs B)
    for s1 in offense:
        for s2 in offense:
            corr[slot_idx[s1], slot_idx[s2]] = 0.15

    # Kicker dependencies
    for s in offense:
        corr[slot_idx[k], slot_idx[s]] = 0.10
    corr[slot_idx[k], slot_idx[k]] = 0.10

    # DST interactions
    for s in offense:
        corr[slot_idx[s], slot_idx[dst]] = -0.20
    corr[slot_idx[k], slot_idx[dst]] = -0.15

    # DST vs DST
    corr[slot_idx[dst], slot_idx[dst]] = 0.05

    return corr

def build_generic_cross_team_correlation(
    session: Session,
    min_games: int = 50,
    min_pair_obs: int = 10,
) -> Tuple[np.ndarray, bool]:
    """
    Fit a single 10x10 cross-team Spearman correlation from all games league-wide.
    Uses season-level slot labels per team to align players into TEAM10_SLOTS.
    Returns (R, is_valid). Falls back to defaults if not enough data.
    """
    stats = session.query(NFLStat).filter(NFLStat.fantasy_points_ppr.isnot(None)).all()
    if not stats:
        return _default_cross_correlation(), False

    df = pd.DataFrame([{
        'season': s.season,
        'week': s.week,
        'team': s.team,
        'opponent_team': s.opponent_team,
        'player_name': s.player_display_name,
        'position': s.position,
        'fantasy_points_ppr': s.fantasy_points_ppr,
    } for s in stats])

    from .team import _assign_season_slot_labels, _apply_slot_labels_to_game, TEAM10_SLOTS

    slot_idx = {s: i for i, s in enumerate(TEAM10_SLOTS)}

    # Build season-level slot maps per team
    season_team_slots: Dict[Tuple[int, str], Dict[str, str]] = {}
    for (season, team), team_df in df.groupby(['season', 'team']):
        season_team_slots[(season, team)] = _assign_season_slot_labels(team_df)

    # Identify unique games (season, week, a<b)
    uniq_games = set()
    for (season, week, team), g in df.groupby(['season', 'week', 'team']):
        opp = g['opponent_team'].dropna().astype(str)
        if opp.empty:
            continue
        o = opp.iloc[0]
        a, b = sorted([team, o])
        uniq_games.add((season, week, a, b))

    X_a, X_b = [], []
    for season, week, a, b in sorted(uniq_games):
        slots_a = season_team_slots.get((season, a), {})
        slots_b = season_team_slots.get((season, b), {})
        if not slots_a or not slots_b:
            continue

        game_a = df[(df['season'] == season) & (df['week'] == week) & (df['team'] == a)]
        game_b = df[(df['season'] == season) & (df['week'] == week) & (df['team'] == b)]

        game_a = _apply_slot_labels_to_game(game_a, slots_a)
        game_b = _apply_slot_labels_to_game(game_b, slots_b)

        arr_a = np.full(10, np.nan)
        arr_b = np.full(10, np.nan)

        # Fill per slot (fallback: ensure DST is captured even if not assigned by season labeling)
        for _, row in game_a.iterrows():
            slot = row.get('slot_label')
            if not slot and str(row.get('position')) == 'DST':
                slot = 'DST'
            if slot in slot_idx:
                arr_a[slot_idx[slot]] = row['fantasy_points_ppr']

        for _, row in game_b.iterrows():
            slot = row.get('slot_label')
            if not slot and str(row.get('position')) == 'DST':
                slot = 'DST'
            if slot in slot_idx:
                arr_b[slot_idx[slot]] = row['fantasy_points_ppr']

        X_a.append(arr_a)
        X_b.append(arr_b)

    if len(X_a) < min_games:
        print(f"  Only {len(X_a)} league games available for generic cross-team correlation; using defaults")
        return _default_cross_correlation(), False

    X_a = np.array(X_a)
    X_b = np.array(X_b)

    from scipy.stats import spearmanr
    n = 10
    R = _default_cross_correlation()
    for i in range(n):
        for j in range(n):
            mask = ~np.isnan(X_a[:, i]) & ~np.isnan(X_b[:, j])
            if mask.sum() < min_pair_obs:
                continue
            rho, _ = spearmanr(X_a[mask, i], X_b[mask, j])
            if not np.isnan(rho):
                R[i, j] = rho

    print(f"  Built generic cross-team correlation from {len(X_a)} games")
    return R, True


def generate_correlated_samples(
    session: Session,
    players: List,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    randomness: float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Generate correlated samples by GAME (matchup). Uses:
      - Team-specific within-team Spearman correlations (converted to Pearson)
      - Cross-team Spearman correlations for head-to-head (converted to Pearson)
      - Normal(μ, σ) marginals with σ scaled by `randomness`

    Returns:
        Dict[player_id -> np.ndarray(n_samples,)]
    """
    rng = np.random.default_rng(seed)
    randomness = float(np.clip(randomness, 0.0, 1.0))

    slot_idx = {s: i for i, s in enumerate(TEAM10_SLOTS)}

    def s2p(Rs: np.ndarray) -> np.ndarray:
        return 2.0 * np.sin(np.clip(np.pi * Rs / 6.0, -np.pi / 2, np.pi / 2))

    def near_spd(A: np.ndarray) -> np.ndarray:
        B = (A + A.T) / 2.0
        w, V = np.linalg.eigh(B)
        w[w < 1e-8] = 1e-8
        return (V * w) @ V.T

    slot_sigma_default = {
        "QB": 6.0, "RB1": 6.0, "RB2": 5.5, "WR1": 6.5, "WR2": 6.0, "WR3": 5.5,
        "TE1": 5.5, "TE2": 5.0, "K": 3.0, "DST": 6.0,
    }

    # Build team->players and team->opponent from provided players
    by_team = defaultdict(list)
    team_to_opp: Dict[str, Optional[str]] = {}
    for p in players:
        t = getattr(p, "team", None)
        if not t:
            continue
        by_team[t].append(p)
        if t not in team_to_opp and getattr(p, "opponent", None):
            team_to_opp[t] = p.opponent

    # Build unique matchups
    seen = set()
    matchups: List[Tuple[str, str]] = []
    for t, o in team_to_opp.items():
        if not o:
            continue
        a, b = sorted([t, o])
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        matchups.append(key)

    out: Dict[int, np.ndarray] = {}

    team_corr_cache: Dict[str, np.ndarray] = {}
    
    # Build generic cross-team Spearman (once), convert to Pearson
    Rx_generic_S, _ = build_generic_cross_team_correlation(session)
    Rx_generic = s2p(Rx_generic_S)

    for a, b in matchups:
        A_all = by_team.get(a, [])
        B_all = by_team.get(b, [])

        # Keep only players with modeled slots for correlation; others will fall back later
        A = [p for p in A_all if getattr(p, "slot_label", None) in slot_idx]
        B = [p for p in B_all if getattr(p, "slot_label", None) in slot_idx]

        if not A and not B:
            continue

        # Stable order: by slot then mean desc
        def sort_team(pl):
            ordered = []
            for s in TEAM10_SLOTS:
                g = [p for p in pl if p.slot_label == s]
                g.sort(key=lambda x: (float(x.fpts_mean) if x.fpts_mean is not None else -1e9), reverse=True)
                ordered.extend(g)
            return ordered

        A = sort_team(A)
        B = sort_team(B)
        players_game = A + B
        if not players_game:
            continue

        # Build μ and σ
        mu = []
        sigma = []
        for p in players_game:
            prm = tuple(p.fpts_dist_params) if isinstance(p.fpts_dist_params, list) else p.fpts_dist_params
            mu_i = float(p.fpts_mean if p.fpts_mean is not None else (prm[0] if prm else 0.0))
            if getattr(p, "slot_label", None) in slot_sigma_default:
                default_sd = slot_sigma_default[p.slot_label]
            else:
                default_sd = 6.0
            sd_i = float(prm[1]) if (prm and len(prm) >= 2 and prm[1] is not None) else float(default_sd)
            mu.append(mu_i)
            sigma.append(sd_i)

        mu = np.array(mu, dtype=float)
        sigma = np.array(sigma, dtype=float)

        # Slot indices per player
        slots_A = [slot_idx[p.slot_label] for p in A]
        slots_B = [slot_idx[p.slot_label] for p in B]

        # Pull or build within-team Spearman, then convert to Pearson
        if a not in team_corr_cache:
            Rw_a_S, _ = build_team_correlation(session, a)
            Rw_a = s2p(Rw_a_S); np.fill_diagonal(Rw_a, 1.0)
            team_corr_cache[a] = Rw_a
        if b not in team_corr_cache:
            Rw_b_S, _ = build_team_correlation(session, b)
            Rw_b = s2p(Rw_b_S); np.fill_diagonal(Rw_b, 1.0)
            team_corr_cache[b] = Rw_b

            

        # Assemble correlation blocks (allow multiple players per slot)
        Corr_A = Rw_a[np.ix_(slots_A, slots_A)] if slots_A else np.zeros((0, 0))
        Corr_B = Rw_b[np.ix_(slots_B, slots_B)] if slots_B else np.zeros((0, 0))
        Corr_X = Rx_generic[np.ix_(slots_A, slots_B)] if (slots_A and slots_B) else np.zeros((len(slots_A), len(slots_B)))

        if len(players_game) == 1:
            Corr = np.array([[1.0]])
        else:
            Corr = np.block([[Corr_A, Corr_X],
                             [Corr_X.T, Corr_B]])
            Corr = np.clip(Corr, -0.999, 0.999)
            Corr = near_spd(Corr)

        # Covariance and sampling
        D = np.diag(sigma)
        Sigma = D @ Corr @ D

        if randomness == 0.0:
            X = np.tile(mu, (n_samples, 1))
        else:
            Sigma_scaled = Sigma * (randomness ** 2)
            Z = rng.multivariate_normal(mean=np.zeros(len(players_game)), cov=Sigma_scaled, size=n_samples)
            X = Z + mu

        # Store samples
        for j, p in enumerate(players_game):
            samples = X[:, j].astype(np.float32, copy=False)
            if getattr(p, "roster_slot_id", None) == 511:
                samples = samples * 1.5
            out[p.id] = samples

    # Independent fallback for any players not covered above (no matchup or unknown slot)
    for p in players:
        if p.id in out:
            continue

        print(f'fallback to independent sampling for player {p.name}')
        prm = tuple(p.fpts_dist_params) if isinstance(p.fpts_dist_params, list) else p.fpts_dist_params
        mu_i = float(p.fpts_mean if p.fpts_mean is not None else (prm[0] if prm else 0.0))
        sd_i = float(prm[1]) if (prm and len(prm) >= 2 and prm[1] is not None) else 6.0
        if randomness == 0.0:
            samples = np.full(n_samples, mu_i, dtype=np.float32)
        else:
            samples = rng.normal(loc=mu_i, scale=(sd_i * randomness), size=n_samples).astype(np.float32, copy=False)
        if getattr(p, "roster_slot_id", None) == 511:
            samples = samples * 1.5
        out[p.id] = samples

    return out
