from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import json
import numpy as np
import scipy.stats as st
from sqlalchemy.orm import Session

from ivan.sports.nfl.paths import MODELS_DIR
from ivan.sports.nfl.models import NFLPlayer
from ivan.sports.nfl.refactor.team_correlation import TEAM10_SLOTS, PTS_SLOT, SLOT_IDX as _SLOT_IDX
from ivan.sports.nfl.refactor.dst import (load_dst_odds_model, game_implied_totals,
                                          team_points_from_latent, dst_from_latent)

TEAM10 = list(TEAM10_SLOTS)            # player depth slots
SLOT_IDX = {s: i for i, s in enumerate(TEAM10)}
PTS_IDX = _SLOT_IDX[PTS_SLOT]          # the team-points latent in the correlation matrices
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


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(st.rankdata(x), st.rankdata(y))[0, 1])


def summarize_sampling(samples: Dict[int, np.ndarray], diag: dict, max_pairs_reported: int = 3) -> str:
    """
    Brief check that the draws carry the dependence that was asked for: per
    game, the realised Spearman correlation between every pair of sampled
    players against the requested matrix, where each matrix came from, and
    (when the game has lines) the simulated team scores with how the QB and
    the DST move with them. Returns a few lines for printing.
    """
    from scipy.stats import spearmanr
    games = diag.get("games", [])
    if not games:
        return "sampling: no games built (every player independent)"
    lines = [f"sampling: {diag.get('n_samples')} draws, {len(games)} game(s), "
             f"{len(diag.get('independent', []))} player(s) independent, method {diag.get('method')}"]
    worst = []
    for g in games:
        ids = g["player_ids"]
        X = np.column_stack([samples[i] for i in ids])
        realised = spearmanr(X).correlation if len(ids) > 2 else np.array([[1.0, spearmanr(X[:, 0], X[:, 1]).correlation], [0, 1.0]])
        requested = np.asarray(g["requested_corr"], float)
        iu = np.triu_indices(len(ids), 1)
        # requested is Pearson of the latent normals; compare on the same scale
        req_s = (6.0 / np.pi) * np.arcsin(np.clip(requested, -1, 1) / 2.0)
        dev = np.abs(realised[iu] - req_s[iu])
        structural = set(g.get("structural_ids", []))
        if structural:
            keep = np.array([ids[i] not in structural and ids[j] not in structural for i, j in zip(*iu)])
            dev = np.where(keep, dev, 0.0)
        src = g.get("sources", {})
        a, b = g["teams"]
        lines.append(f"  {a}-{b}: {len(ids)} players, mean |requested rho| {np.abs(req_s[iu]).mean():.2f}, "
                     f"max |realised - requested| {dev.max():.2f}, "
                     f"sources {src.get(a)}/{src.get(b)}, cross {src.get('cross')}")
        lat = g.get("latent")
        if lat:
            parts = []
            for t in (a, b):
                parts.append(f"{t} {lat['implied'][t]:.1f} implied -> {lat['points_mean'][t]:.1f} +- {lat['points_sd'][t]:.1f}")
            lines.append(f"    team points: " + "; ".join(parts))
            qb = {t: None for t in (a, b)}; dst = {t: None for t in (a, b)}
            for pid, slot, team in zip(ids, g["slots"], g["player_teams"]):
                c = lat["corr"].get(str(pid)) or lat["corr"].get(pid)
                if c is None:
                    continue
                if slot == "QB" and qb[team] is None:
                    qb[team] = c["own_pts"]
                if slot == "DST" and dst[team] is None:
                    dst[team] = c["opp_pts"]
            fmt = lambda d: " ".join(f"{t} {v:+.2f}" for t, v in d.items() if v is not None)
            lines.append(f"    QB vs own points rho {fmt(qb)}; DST vs opponent points rho {fmt(dst)}")
        else:
            lines.append(f"    no game lines: DST sampled from its fitted distribution, no team-points latent")
        k = int(np.argmax(dev))
        worst.append((dev.max(), g["teams"], g["slots"][iu[0][k]], g["slots"][iu[1][k]], req_s[iu][k], realised[iu][k]))
    worst.sort(reverse=True)
    for d, teams, sa, sb, r, q in worst[:max_pairs_reported]:
        if d > 0.08:
            lines.append(f"  largest gap {teams[0]}-{teams[1]} {sa} vs {sb}: requested {r:+.2f}, realised {q:+.2f}")
    return "\n".join(lines)


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
    One correlated draw per *game*. Every offensive player and kicker is
    Normal(mu, sigma) on a latent Gaussian with a slot-structured correlation:
    a within-team matrix for each side plus a cross-team block. When the game
    has lines and the DST model is fitted, the draw also carries one latent
    per team for its points scored (implied total plus a historical residual),
    correlated with that team's slots and the opponent's from history. Each
    DST is then built inside the universe: the points-allowed bracket of the
    opponent's drawn score plus takeaway points on the DST's own latent, so a
    universe where an offense scores is the universe where the opposing DST
    does not. Without lines a DST falls back to its fitted Normal.

    Players need NFLPlayer.slot_label in TEAM10 (set by assign_slots at ingest)
    and a paired opponent on the slate; anything else is sampled independently.
    Captain rows (roster_slot_id 511) are the same draw times 1.5.

    randomness scales the spread (0 = every sample at the mean, 1 = full
    distribution). With return_diagnostics=True the second return value records
    what was actually sampled, so a run can be verified afterwards:

        {"n_samples", "randomness", "method",
         "games": [{"teams", "player_ids", "slots", "player_teams", "mu", "sigma",
                    "requested_corr", "corr", "sources", "pooled_pairs",
                    "latent": {"implied", "lines", "points_mean", "points_sd",
                               "corr": {player_id: {"own_pts", "opp_pts"}}} | None}],
         "independent": [{"id", "reason"}]}

    where `requested_corr` is the player block of the correlation assembled
    from the model and `corr` is what was sampled from after clipping and the
    SPD projection (also the player block).
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

    # The game layer needs the fitted DST model (residual pools) and the lines.
    dst_params = load_dst_odds_model(model_dir) if use_team_specific_corr else None

    # --- one multivariate normal per game ----------------------------------------------
    out: Dict[int, np.ndarray] = {}
    games: list[dict] = []
    for a, b in matchups:
        A = sort_team([p for p in by_team.get(a, []) if p.slot_label in SLOT_IDX])
        B = sort_team([p for p in by_team.get(b, []) if p.slot_label in SLOT_IDX])
        players_game = A + B
        if not players_game:
            continue
        n_p = len(players_game)

        mu, sigma = [], []
        for p in players_game:
            m, s = _mu_sigma(p, SLOT_SIGMA_DEFAULT[p.slot_label])
            mu.append(m)
            sigma.append(s)
        mu, sigma = np.array(mu, float), np.array(sigma, float)

        Rw_a, src_a = within(a)
        Rw_b, src_b = within(b)
        Rx_ab, src_x = cross(a, b)
        has_pts = Rw_a.shape[0] > PTS_IDX and Rw_b.shape[0] > PTS_IDX and Rx_ab.shape[0] > PTS_IDX
        lines = game_implied_totals(session, a, b) if (dst_params is not None and has_pts) else None
        implied = {a: lines["team_implied"], b: lines["opp_implied"]} if lines else None

        # nodes: A's players, B's players, then (with lines) A's points and B's points
        nodes = [(a, SLOT_IDX[p.slot_label]) for p in A] + [(b, SLOT_IDX[p.slot_label]) for p in B]
        if implied:
            nodes += [(a, PTS_IDX), (b, PTS_IDX)]
        n = len(nodes)
        requested = np.eye(n)
        for i in range(n):
            ti, si = nodes[i]
            for j in range(i + 1, n):
                tj, sj = nodes[j]
                if ti == tj:
                    r = (Rw_a if ti == a else Rw_b)[si, sj]
                elif ti == a:
                    r = Rx_ab[si, sj]
                else:
                    r = Rx_ab[sj, si]
                requested[i, j] = requested[j, i] = r
        pooled = _separate_pooled_players(requested, A, 0, Rw_a) + _separate_pooled_players(requested, B, len(A), Rw_b)
        corr = _near_spd(np.clip(requested, -0.999, 0.999))

        Z = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=n_samples)
        X = mu + sigma * randomness * Z[:, :n_p]

        latent = None
        if implied:
            u_pts = {a: st.norm.cdf(Z[:, n_p]), b: st.norm.cdf(Z[:, n_p + 1])}
            pts = {t: team_points_from_latent(dst_params, implied[t], u_pts[t], randomness) for t in (a, b)}
            for j, p in enumerate(players_game):
                if p.slot_label != "DST":
                    continue
                opp = b if p.team == a else a
                X[:, j] = dst_from_latent(dst_params, implied[p.team], implied[opp],
                                          u_opp_points=u_pts[opp], u_take=st.norm.cdf(Z[:, j]), randomness=randomness)
            latent = {"implied": {t: round(float(implied[t]), 2) for t in (a, b)},
                      "lines": {"total": lines["total"], "home_margin": lines["home_margin"]},
                      "points_mean": {t: round(float(pts[t].mean()), 2) for t in (a, b)},
                      "points_sd": {t: round(float(pts[t].std()), 2) for t in (a, b)},
                      "corr": {}}
            if return_diagnostics:
                for j, p in enumerate(players_game):
                    own, opp = (a, b) if p.team == a else (b, a)
                    latent["corr"][int(p.id)] = {"own_pts": round(_rank_corr(X[:, j], pts[own]), 3),
                                                 "opp_pts": round(_rank_corr(X[:, j], pts[opp]), 3)}

        for j, p in enumerate(players_game):
            samples = X[:, j].astype(np.float32, copy=False)
            out[p.id] = samples * CPT_MULTIPLIER if p.roster_slot_id == CPT_SLOT else samples

        if return_diagnostics:
            np.fill_diagonal(requested, 1.0)
            games.append({
                "teams": [a, b],
                "player_ids": [int(p.id) for p in players_game],
                "slots": [p.slot_label for p in players_game],
                "player_teams": [p.team for p in players_game],
                "mu": mu.tolist(), "sigma": sigma.tolist(),
                "requested_corr": np.round(requested[:n_p, :n_p], 4).tolist(),
                "corr": np.round(corr[:n_p, :n_p], 4).tolist(),
                "sources": {a: src_a, b: src_b, "cross": src_x},
                "pooled_pairs": pooled,
                "latent": latent,
                # rows whose dependence is structural (DST built from the opponent's score), so the
                # requested matrix is not their target: skip them in correlation error checks
                "structural_ids": [int(p.id) for p in players_game if p.slot_label == "DST"] if latent else [],
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
