from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import json
import numpy as np
from sqlalchemy.orm import Session

from ivan.sports.nfl.paths import MODELS_DIR
from ivan.sports.nfl.models import NFLPlayer

TEAM10 = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "TE2", "K", "DST"]
SLOT_IDX = {s: i for i, s in enumerate(TEAM10)}
CPT_SLOT = 511
CPT_MULTIPLIER = 1.5

# Fallback sigma per slot when a player has no fitted distribution.
SLOT_SIGMA_DEFAULT = {
    "QB": 6.0, "RB1": 6.0, "RB2": 5.5, "WR1": 6.5, "WR2": 6.0, "WR3": 5.5,
    "TE1": 5.5, "TE2": 5.0, "K": 3.0, "DST": 6.0,
}


def _spearman_to_pearson(Rs: np.ndarray) -> np.ndarray:
    """Bivariate-normal relationship between Spearman and Pearson correlation."""
    return 2.0 * np.sin(np.clip(np.pi * Rs / 6.0, -np.pi / 2, np.pi / 2))


def _near_spd(A: np.ndarray) -> np.ndarray:
    """
    Nearest well-conditioned correlation matrix: symmetrise, floor the
    eigenvalues, then rescale back to a unit diagonal so the projection does
    not change any player's variance.
    """
    B = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < 1e-8] = 1e-8
    C = (eigvec * eigval) @ eigvec.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


# Two *different* players pooled into the same depth slot (a team's third and
# fourth WR both carry WR3) are not the same player: instead of the slot's
# self-correlation of 1.0 they get the team's adjacent-depth correlation.
POOLED_NEIGHBOUR = {"RB1": ("RB1", "RB2"), "RB2": ("RB1", "RB2"), "WR1": ("WR1", "WR2"), "WR2": ("WR2", "WR3"),
                    "WR3": ("WR2", "WR3"), "TE1": ("TE1", "TE2"), "TE2": ("TE1", "TE2")}


def _pooled_pair_corr(Rw: np.ndarray, slot: str) -> float:
    pair = POOLED_NEIGHBOUR.get(slot)
    return float(Rw[SLOT_IDX[pair[0]], SLOT_IDX[pair[1]]]) if pair else 0.0


def _separate_pooled_players(requested: np.ndarray, team_players: list, offset: int, Rw: np.ndarray) -> int:
    """In place: same-slot pairs of different people get the pooled correlation. Returns pairs changed."""
    changed = 0
    for i in range(len(team_players)):
        for j in range(i + 1, len(team_players)):
            a, b = team_players[i], team_players[j]
            if a.slot_label == b.slot_label and a.name != b.name:
                requested[offset + i, offset + j] = requested[offset + j, offset + i] = _pooled_pair_corr(Rw, a.slot_label)
                changed += 1
    return changed


def _mu_sigma(p: NFLPlayer, default_sigma: float) -> Tuple[float, float]:
    prm = tuple(p.fpts_dist_params) if isinstance(p.fpts_dist_params, list) else p.fpts_dist_params
    mu = float(p.fpts_mean if p.fpts_mean is not None else (prm[0] if prm else 0.0))
    sd = float(prm[1]) if (prm and len(prm) >= 2 and prm[1] is not None) else float(default_sigma)
    return mu, sd


def generate_correlated_samples_gaussian(
    session: Session,
    model_dir: str | Path = MODELS_DIR,
    n_samples: int = 1000,
    seed: int | None = None,
    use_hgv20: bool = False,
    use_team_specific_corr: bool = True,
    randomness: float = 1.0,
    return_diagnostics: bool = False,
) -> Dict[int, np.ndarray] | Tuple[Dict[int, np.ndarray], dict]:
    """
    Treat every player's fantasy points as Normal(mu, sigma) and draw one
    multivariate normal per *game* with a slot-structured covariance: a 10x10
    within-team correlation for each side plus a 10x10 cross-team block.

    Players need NFLPlayer.slot_label in TEAM10 (set by assign_slots at ingest)
    and a paired opponent on the slate; anything else is sampled independently.
    Captain rows (roster_slot_id 511) are the same draw times 1.5.

    randomness scales the spread (0 = every sample at the mean, 1 = full
    distribution). With return_diagnostics=True the second return value records
    what was actually sampled, so a run can be verified afterwards:

        {"n_samples", "randomness", "method",
         "games": [{"teams", "player_ids", "slots", "mu", "sigma",
                    "requested_corr", "corr", "sources"}],
         "independent": [{"id", "reason"}]}

    where `requested_corr` is the correlation assembled from the model and
    `corr` is what was sampled from after clipping and the SPD projection.
    """
    rng = np.random.default_rng(seed)
    model_dir = Path(model_dir)
    randomness = float(np.clip(randomness, 0.0, 1.0))

    players = session.query(NFLPlayer).all()
    if not players:
        return ({}, {"games": [], "independent": []}) if return_diagnostics else {}

    team_to_opp: Dict[str, str] = {}
    by_team: Dict[str, list] = defaultdict(list)
    for p in players:
        by_team[p.team].append(p)
        if p.team not in team_to_opp and p.opponent:
            team_to_opp[p.team] = p.opponent

    seen, matchups = set(), []
    for t, o in team_to_opp.items():
        if not o:
            continue
        a, b = sorted([t, o])
        if (a, b) in seen:
            continue
        seen.add((a, b))
        matchups.append((a, b))

    # --- correlation source -----------------------------------------------------------
    team_corr_cache: Dict[str, Tuple[np.ndarray, bool]] = {}
    cross_corr_cache: Dict[Tuple[str, str], Tuple[np.ndarray, bool]] = {}
    Rw = Rx = None
    if use_team_specific_corr:
        from ivan.sports.nfl.refactor.team_correlation import build_team_correlation, build_cross_team_correlation
        method = "team_specific"
    elif use_hgv20:
        from .correlated_sampling_validation import print_hgv20_spearman
        hgv = print_hgv20_spearman(model_dir, n=20_000)
        if isinstance(hgv, tuple) and len(hgv) >= 2:
            Rw_S, Rx_S = hgv[0], hgv[1]
        elif isinstance(hgv, dict):
            Rw_S, Rx_S = np.asarray(hgv["within"], float), np.asarray(hgv["cross"], float)
        else:
            raise RuntimeError("print_hgv20_spearman returned an unexpected type.")
        Rw, Rx = _spearman_to_pearson(Rw_S), _spearman_to_pearson(Rx_S)
        np.fill_diagonal(Rw, 1.0)
        method = "hgv20"
    else:
        d = json.loads((model_dir / "team10_spearman.json").read_text())
        Rw, Rx = _spearman_to_pearson(np.array(d["within"], float)), _spearman_to_pearson(np.array(d["cross"], float))
        np.fill_diagonal(Rw, 1.0)
        method = "team10_spearman.json"

    def within(team: str) -> Tuple[np.ndarray, str]:
        if not use_team_specific_corr:
            return Rw, method
        if team not in team_corr_cache:
            R_S, ok = build_team_correlation(session, team)
            R = _spearman_to_pearson(R_S)
            np.fill_diagonal(R, 1.0)
            team_corr_cache[team] = (R, ok)
        R, ok = team_corr_cache[team]
        return R, ("history" if ok else "identity")

    def cross(a: str, b: str) -> Tuple[np.ndarray, str]:
        if not use_team_specific_corr:
            return Rx, method
        key = (a, b) if a < b else (b, a)
        if key not in cross_corr_cache:
            R_S, ok = build_cross_team_correlation(session, a, b)
            cross_corr_cache[key] = (_spearman_to_pearson(R_S), ok)
        R, ok = cross_corr_cache[key]
        return R, ("history" if ok else "default")

    def sort_team(pl: list) -> list:
        seq = []
        for s in TEAM10:
            group = [p for p in pl if p.slot_label == s]
            group.sort(key=lambda p: (p.fpts_mean if p.fpts_mean is not None else -1e9), reverse=True)
            seq.extend(group)
        return seq

    # --- one multivariate normal per game ----------------------------------------------
    out: Dict[int, np.ndarray] = {}
    games: list[dict] = []
    for a, b in matchups:
        A = sort_team([p for p in by_team.get(a, []) if p.slot_label in SLOT_IDX])
        B = sort_team([p for p in by_team.get(b, []) if p.slot_label in SLOT_IDX])
        players_game = A + B
        if not players_game:
            continue

        mu, sigma = [], []
        for p in players_game:
            m, s = _mu_sigma(p, SLOT_SIGMA_DEFAULT[p.slot_label])
            mu.append(m)
            sigma.append(s)
        mu, sigma = np.array(mu, float), np.array(sigma, float)

        slots_A = [SLOT_IDX[p.slot_label] for p in A]
        slots_B = [SLOT_IDX[p.slot_label] for p in B]
        Rw_a, src_a = within(a)
        Rw_b, src_b = within(b)
        Rx_ab, src_x = cross(a, b)
        requested = np.block([[Rw_a[np.ix_(slots_A, slots_A)], Rx_ab[np.ix_(slots_A, slots_B)]],
                              [Rx_ab[np.ix_(slots_A, slots_B)].T, Rw_b[np.ix_(slots_B, slots_B)]]])
        pooled = _separate_pooled_players(requested, A, 0, Rw_a) + _separate_pooled_players(requested, B, len(A), Rw_b)
        corr = _near_spd(np.clip(requested, -0.999, 0.999))
        np.fill_diagonal(requested, 1.0)

        D = np.diag(sigma)
        Sigma = (D @ corr @ D) * (randomness ** 2)
        Z = rng.multivariate_normal(mean=np.zeros(len(players_game)), cov=Sigma, size=n_samples)
        X = Z + mu
        for j, p in enumerate(players_game):
            samples = X[:, j].astype(np.float32, copy=False)
            out[p.id] = samples * CPT_MULTIPLIER if p.roster_slot_id == CPT_SLOT else samples

        if return_diagnostics:
            games.append({
                "teams": [a, b],
                "player_ids": [int(p.id) for p in players_game],
                "slots": [p.slot_label for p in players_game],
                "mu": mu.tolist(), "sigma": sigma.tolist(),
                "requested_corr": np.round(requested, 4).tolist(),
                "corr": np.round(corr, 4).tolist(),
                "sources": {a: src_a, b: src_b, "cross": src_x},
                "pooled_pairs": pooled,
            })

    # --- everyone else: independent N(mu, sigma) ---------------------------------------
    independent: list[dict] = []
    for p in players:
        if p.id in out:
            continue
        mu_i, sd_i = _mu_sigma(p, 6.0)
        samples = rng.normal(loc=mu_i, scale=sd_i * randomness, size=n_samples).astype(np.float32, copy=False)
        out[p.id] = samples * CPT_MULTIPLIER if p.roster_slot_id == CPT_SLOT else samples
        if return_diagnostics:
            reason = ("no depth-chart slot label" if p.slot_label not in SLOT_IDX
                      else "no opponent on the slate" if not any(p.team in m for m in matchups)
                      else "not placed in a game")
            independent.append({"id": int(p.id), "reason": reason, "mu": mu_i, "sigma": sd_i})

    if return_diagnostics:
        return out, {"n_samples": int(n_samples), "randomness": randomness, "method": method,
                     "games": games, "independent": independent}
    return out
