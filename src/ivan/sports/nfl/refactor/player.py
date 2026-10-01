from dataclasses import dataclass, field

from ivan.sports.nfl.models import NFLPlayer, NFLStat, NFLOdds
from ivan.sports.nfl.refactor.stat import (
    MARKET_TO_STAT, StatDistribution, _get_anytime_td_prob_from_odds,
    clear_neighbor_cache, fit_stat_distribution, position_logs, sample_stat,
)
from ivan.sports.nfl.refactor.volume import VolumeModel
from ivan.sports.nfl.refactor.dst import project_dst_samples
from ivan.sports.nfl.refactor.spread import load_sd_calibration, calibrate_spread, describe_sd_calibration, spread_factor
import pandas as pd
import numpy as np
import scipy.stats as st

# DraftKings scoring weights
FPTS_WEIGHTS = {
    'passing_yards': 0.04,
    'passing_tds': 4.0,
    'passing_interceptions': -1.0,
    'rushing_yards': 0.1,
    'rushing_tds': 6.0,
    'receptions': 1.0,
    'receiving_yards': 0.1,
    'receiving_tds': 6.0,
    # Kickers: kicker_pts is the books' kicking points (3 per FG, 1 per PAT);
    # DraftKings adds 1 for a 40-49 yard FG and 2 for 50+, so fg_made carries
    # only that distance bonus, per kicker (see _kicker_fg_bonus).
    'kicker_pts': 1.0,
    'fg_made': 0.65,
}

# League-average DraftKings distance bonus per made FG (2022-2025: 3.59-3.67
# points per FG, i.e. 0.59-0.67 above the flat 3) and how many of a kicker's
# own FGs it takes to move the prior as far as the league does.
FG_BONUS_LEAGUE = 0.65
FG_BONUS_PRIOR_KICKS = 30.0


def _kicker_fg_bonus(history: list) -> float:
    """Expected DK points above 3 per made FG for this kicker, shrunk to the league."""
    made = bonus = 0.0
    for h in history:
        fg = h.fg_made or 0.0
        if not fg:
            continue
        pts = (3.0 * ((h.fg_made_0_19 or 0.0) + (h.fg_made_20_29 or 0.0) + (h.fg_made_30_39 or 0.0))
               + 4.0 * (h.fg_made_40_49 or 0.0) + 5.0 * ((h.fg_made_50_59 or 0.0) + (h.fg_made_60_ or 0.0)))
        if pts <= 0:
            continue          # distance split missing for this game
        made += fg
        bonus += pts - 3.0 * fg
    return (bonus + FG_BONUS_LEAGUE * FG_BONUS_PRIOR_KICKS) / (made + FG_BONUS_PRIOR_KICKS)


def _fpts_weights(player: NFLPlayer, history: list, stats: list[str]) -> np.ndarray:
    w = np.array([FPTS_WEIGHTS.get(s, 0.0) for s in stats], dtype=float)
    if player.position == 'K' and 'fg_made' in stats:
        w[stats.index('fg_made')] = _kicker_fg_bonus(history)
    return w

STATS = {
    'QB': ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'],
    'WR': ['receptions', 'receiving_yards', 'receiving_tds'],
    'TE': ['receptions', 'receiving_yards', 'receiving_tds'],
    'K':  ['kicker_pts', 'fg_made'],
    'DST': ['def_tds', 'def_sacks', 'def_interceptions', 'special_teams_tds', 'points_allowed'] 
}

# Roster slot IDs for slate type detection
SHOWDOWN_SLOTS = {511, 512}
CLASSIC_SLOTS = {67, 68, 69, 70, 71}

# Depth-chart slots the correlation model knows about (TEAM10 in correlated_samples):
# how many of each base position get their own slot; deeper players are pooled
# into the last one.
SLOT_CAPS = {"QB": 1, "RB": 2, "WR": 3, "TE": 2, "K": 1, "DST": 1}


def assign_slots(session) -> int:
    """
    Set NFLPlayer.slot_label (QB, RB1, RB2, WR1..WR3, TE1, TE2, K, DST) from each
    team's projected depth chart, so the correlated sampler can place every
    player in its slot-structured covariance. CPT and FLEX rows of the same
    player share a label. Returns the number of rows labelled. Must run after
    fpts_mean is populated; players without a projection sort last.
    """
    players = session.query(NFLPlayer).filter(NFLPlayer.team.isnot(None)).all()

    by_name: dict[tuple, list] = {}
    for p in players:
        base = (p.position or "").upper()
        if base not in SLOT_CAPS or not p.name:
            continue
        by_name.setdefault((p.team, base, p.name), []).append(p)

    buckets: dict[tuple, list] = {}
    for (team, base, name), rows in by_name.items():
        proj = [r.fpts_mean for r in rows if r.fpts_mean is not None]
        buckets.setdefault((team, base), []).append((max(proj) if proj else float("-inf"), name, rows))

    labelled = 0
    for (team, base), group in buckets.items():
        group.sort(key=lambda t: t[0], reverse=True)
        for depth, (_, _, rows) in enumerate(group, start=1):
            label = base if SLOT_CAPS[base] == 1 else f"{base}{min(depth, SLOT_CAPS[base])}"
            for r in rows:
                r.slot_label = label
                labelled += 1
    session.commit()
    return labelled

# Odds filtering thresholds
# Prices more extreme than these are likely data errors (American odds)
MIN_PRICE = -1000  # Exclude odds like -1000 (implies 90%+ prob, usually bad data)
MAX_PRICE = 10000  # Exclude extreme longshots that are likely errors


def _filter_outlier_odds(odds: list[NFLOdds]) -> list[NFLOdds]:
    """
    Filter out outlier odds that are likely data errors.
    
    Removes:
    - Extreme prices (e.g., -1000 which implies 90%+ probability)
    - Point outliers that are far from the median for that market
    """
    if not odds:
        return odds
    
    # First pass: filter by price
    filtered = [o for o in odds if MIN_PRICE <= o.price <= MAX_PRICE]
    
    if len(filtered) < len(odds):
        removed = len(odds) - len(filtered)
        print(f"  Filtered {removed} odds with extreme prices")
    
    # Second pass: filter point outliers within each market
    # Group by market and remove points that are >3 std from median
    from collections import defaultdict
    by_market = defaultdict(list)
    for o in filtered:
        by_market[o.market].append(o)
    
    final = []
    for market, market_odds in by_market.items():
        if len(market_odds) < 2:
            # Not enough data to detect outliers
            final.extend(market_odds)
            continue
        
        points = np.array([o.point for o in market_odds])
        median = np.median(points)
        std = np.std(points)
        
        if std == 0:
            final.extend(market_odds)
            continue
        
        for o in market_odds:
            z_score = abs(o.point - median) / std
            if z_score <= 5:
                final.append(o)
            else:
                print(f"  Filtered outlier: {o.market} point={o.point} (median={median:.1f}, z={z_score:.1f})")
    
    return final


def _is_classic_slate(players: list[NFLPlayer]) -> bool:
    """Check if slate is Classic based on roster_slot_ids present."""
    slot_ids = {p.roster_slot_id for p in players}
    return bool(slot_ids & CLASSIC_SLOTS) and not bool(slot_ids & SHOWDOWN_SLOTS)


def _dedupe_classic_players(session, players: list[NFLPlayer]) -> list[NFLPlayer]:
    """For Classic slates, keep only one player per name (remove duplicates)."""
    seen_names = set()
    keep = []
    remove = []
    
    for player in players:
        if player.name in seen_names:
            remove.append(player)
        else:
            seen_names.add(player.name)
            keep.append(player)
    
    # Delete duplicates from DB
    for player in remove:
        session.delete(player)
    session.commit()
    
    print(f"Removed {len(remove)} duplicate players for Classic slate")
    return keep


def _populate_fpts_stats(player: NFLPlayer, samples: np.ndarray) -> None:
    """
    Populate all fpts statistical fields on NFLPlayer from samples.
    """
    player.fpts_n = len(samples)
    player.fpts_mean = float(np.mean(samples))
    player.fpts_std = float(np.std(samples))
    player.fpts_min = float(np.min(samples))
    player.fpts_max = float(np.max(samples))
    player.fpts_skew = float(st.skew(samples))
    player.fpts_kurtosis = float(st.kurtosis(samples))
    
    # Percentiles
    player.fpts_p01 = float(np.percentile(samples, 1))
    player.fpts_p05 = float(np.percentile(samples, 5))
    player.fpts_p10 = float(np.percentile(samples, 10))
    player.fpts_p25 = float(np.percentile(samples, 25))
    player.fpts_p50 = float(np.percentile(samples, 50))
    player.fpts_p75 = float(np.percentile(samples, 75))
    player.fpts_p90 = float(np.percentile(samples, 90))
    player.fpts_p95 = float(np.percentile(samples, 95))
    player.fpts_p99 = float(np.percentile(samples, 99))


def _store_fit(player: NFLPlayer, samples: np.ndarray, cal: dict) -> None:
    """
    Finish a player's fit: widen the samples by the position's predictive-sd
    factor (see refactor.spread), fit the Normal the sampler draws from, and
    record both the model spread and the calibrated one.
    """
    model_sd = float(np.std(samples))
    samples = calibrate_spread(samples, player.position, cal)
    mu, sigma = st.norm.fit(samples)
    player.fpts_dist_name = 'norm'
    player.fpts_dist_params = (float(mu), float(sigma))
    _populate_fpts_stats(player, samples)
    player.fpts_std_model = model_sd


def _fit_dst_by_salary(player: NFLPlayer, rng: np.random.Generator, n_samples: int = 10000) -> np.ndarray:
    """
    Fallback when the game has no lines: DST fpts samples scaled by salary.
    Classic DST salaries run 2000-4000; showdown FLEX salaries run higher and
    saturate this, which is why the lines are preferred (see refactor.dst).
    """
    # Normalize salary to a 0-1 scale (assuming 2000-4500 range)
    salary_norm = (player.salary - 2000) / 2500
    salary_norm = np.clip(salary_norm, 0, 1)
    
    # Base mean around 7, scaled by salary (+/- 2 pts)
    base_mean = 5.0 + (salary_norm * 4.0)  # Range: 5 to 9
    
    # DST has high variance
    sigma = 5.0
    
    # Generate samples
    samples = rng.normal(base_mean, sigma, n_samples)
    
    return samples


@dataclass
class _Pending:
    """A player between the two fitting passes."""
    player: NFLPlayer
    odds_df: pd.DataFrame
    history: list
    p_td: float | None
    dists: dict[str, StatDistribution] = field(default_factory=dict)


def _has_market(odds_df: pd.DataFrame, stat: str) -> bool:
    return bool(set(odds_df['market']) & set(MARKET_TO_STAT.get(stat, ())))


def fit_fpts_distributions(session):
    """
    Fit every slate player's fantasy-point distribution.

    Two passes. The first fits each stat the books price, calibrating a KDE of
    the player's history to the posted lines, and records (salary, anytime-TD
    probability, market-implied mean) for the slate's volume model. The second
    fills the stats nobody prices from that model: salary and TD odds predict
    the mean, and a neighbor pool of same-position player-seasons with that
    role supplies the shape. Current odds always win; salary only speaks where
    the market is silent.
    """
    players = session.query(NFLPlayer).all()

    # Dedupe for Classic slates
    if _is_classic_slate(players):
        print("Detected Classic slate - removing duplicate players")
        players = _dedupe_classic_players(session, players)

    rng = np.random.default_rng(42)
    fitted = 0
    clear_neighbor_cache()
    cal = load_sd_calibration()
    print(describe_sd_calibration(cal))
    volume = VolumeModel()
    pending: list[_Pending] = []
    # per-run validation tallies: stats priced by a market vs filled from the
    # salary model, and where each K/DST projection came from
    tally = {"priced": 0, "unpriced": 0, "by_pos": {}, "dst_lines": 0, "dst_salary": 0, "k_market": 0, "k_salary": 0}

    # ---- pass 1: market-priced stats -------------------------------------
    for player in players:
        # A captain row is the FLEX row at 1.5x salary; it is copied from the
        # FLEX fit below, and must not feed the salary model.
        if player.roster_slot_id == 511:
            continue

        # DST: mean from the game lines with history-shaped residuals; salary
        # only when the lines are missing.
        if player.position == 'DST':
            projected = project_dst_samples(session, player.team, player.opponent, rng)
            if projected is not None:
                dst_samples, info = projected
                tally["dst_lines"] += 1
                print(f"DST {player.team} vs {player.opponent}: lines total={info['total']} "
                      f"team={info['team_implied']} opp={info['opp_implied']} -> mean {info['mu']} "
                      f"(bracket {info['bracket']} + takeaways {info['takeaways']})")
            else:
                dst_samples = _fit_dst_by_salary(player, rng)
                tally["dst_salary"] += 1
                print(f"DST {player.team} vs {player.opponent}: no lines or no DST model, salary fallback")
            _store_fit(player, dst_samples, cal)
            session.commit()
            fitted += 1
            continue

        stats_to_model = STATS.get(player.position)
        if not stats_to_model:
            print(f'skipping {player.name}: no stat model for position {player.position}')
            continue

        # Get player odds (only Over/Yes sides)
        odds = session.query(NFLOdds).filter(
            NFLOdds.player == player.name,
            NFLOdds.side.in_(['Over', 'Yes'])
        ).all()

        # Filter out outlier odds (bad data from some sportsbooks)
        odds = _filter_outlier_odds(odds)

        if not odds:
            print(f'skipping {player.name}: no odds found')
            continue

        odds_df = pd.DataFrame([
            {'market': o.market, 'side': o.side, 'point': o.point, 'price': o.price}
            for o in odds
        ])

        history = session.query(NFLStat).filter(
            NFLStat.player_display_name == player.name
        ).all()

        item = _Pending(player, odds_df, history, _get_anytime_td_prob_from_odds(odds_df))
        for stat in stats_to_model:
            if not _has_market(odds_df, stat):
                continue
            hv = [getattr(h, stat) for h in history if getattr(h, stat) is not None]
            sd = fit_stat_distribution(odds_df, hv, stat, session=session, position=player.position)
            if sd is None or sd.num_odds_points == 0:
                continue
            item.dists[stat] = sd
            tally["priced"] += 1
            mean = float(np.mean(sample_stat(sd, 4000, rng)))
            volume.add(player.position, stat, player.salary, item.p_td, mean)
        if player.position == 'K':
            tally["k_market" if 'kicker_pts' in item.dists else "k_salary"] += 1
        pending.append(item)

    volume.fit()
    print("Salary/TD-odds volume model (unpriced stats):")
    print(volume.describe())

    # ---- pass 2: unpriced stats, then the joint sample --------------------
    for item in pending:
        player = item.player
        stats_to_model = STATS[player.position]

        for stat in stats_to_model:
            if stat in item.dists:
                continue
            hv = [getattr(h, stat) for h in item.history if getattr(h, stat) is not None]
            target = volume.predict(player.position, stat, player.salary, item.p_td)
            sd = fit_stat_distribution(
                item.odds_df, hv, stat,
                session=session, position=player.position, target_mean=target,
            )
            if sd is not None:
                item.dists[stat] = sd
                tally["unpriced"] += 1

        stats_valid = [s for s in stats_to_model if s in item.dists]
        if not stats_valid:
            print(f"no valid stat distributions for {player.name}; skipping")
            continue
        dists_valid = [item.dists[s] for s in stats_valid]

        # Dependence between stats: the player's own history when he has
        # enough of it, otherwise every game log at his position.
        df_hist = pd.DataFrame({
            stat: [getattr(h, stat) for h in item.history if getattr(h, stat) is not None]
            for stat in stats_valid
        }).dropna() if item.history else pd.DataFrame(columns=stats_valid)
        if df_hist.shape[0] < 5:
            logs = position_logs(session, player.position)
            cols = [c for c in stats_valid if c in logs.columns]
            df_hist = logs[cols].dropna() if len(cols) == len(stats_valid) else pd.DataFrame()
        if df_hist.shape[0] < 5:
            print(f"insufficient joint history for correlation: {player.name}")
            continue

        corr = df_hist.corr(method='spearman').fillna(0.0).values

        # Sample independent marginals
        n_samples = 10000
        rng = np.random.default_rng(42)
        X = np.zeros((n_samples, len(stats_valid)), dtype=float)
        for j, sd in enumerate(dists_valid):
            X[:, j] = sample_stat(sd, n_samples, rng)

        # Impose dependence using a Gaussian copula on ranks
        L = np.linalg.cholesky(corr + 1e-6 * np.eye(corr.shape[0]))
        Z = rng.standard_normal(size=(n_samples, corr.shape[0])) @ L.T
        U_target = st.norm.cdf(Z)

        X_dep = np.zeros_like(X)
        for j in range(X.shape[1]):
            x_sorted = np.sort(X[:, j])
            order = np.argsort(U_target[:, j])
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(n_samples)
            X_dep[:, j] = x_sorted[inv_order]

        weights = _fpts_weights(player, item.history, stats_valid)
        fpts_samples = X_dep @ weights

        _store_fit(player, fpts_samples, cal)
        session.commit()
        fitted += 1
        tally["by_pos"][player.position] = tally["by_pos"].get(player.position, 0) + 1

    # A captain scores exactly 1.5x the same player's FLEX line (the sampler
    # applies the multiplier), so the CPT row carries the FLEX row's fit. The
    # salary-based DST fit would otherwise see the 1.5x captain salary.
    synced = _sync_captain_rows(session)

    # Clean up: remove players with no fpts distribution
    players_without_fpts = session.query(NFLPlayer).filter(
        NFLPlayer.fpts_dist_params.is_(None)
    ).all()

    if players_without_fpts:
        print(f"\nRemoving {len(players_without_fpts)} players with no fpts distribution:")
        for p in players_without_fpts:
            print(f"  - {p.name} ({p.position})")
            session.delete(p)
        session.commit()

    # Depth-chart slots drive the game-level correlation structure; without
    # them every player would be sampled independently.
    labelled = assign_slots(session)
    n_stats = tally["priced"] + tally["unpriced"]
    by_pos = " ".join(f"{p} {n}" for p, n in sorted(tally["by_pos"].items()))
    print(f"\nFitted {fitted} players ({by_pos}; DST {tally['dst_lines'] + tally['dst_salary']}); "
          f"{synced} captain rows copied from FLEX; {labelled} rows given depth-chart slots for correlation.")
    print(f"validation: {tally['priced']}/{n_stats} stat distributions priced by a market, "
          f"{tally['unpriced']} from the salary model | "
          f"K: {tally['k_market']} on the kicking-points market, {tally['k_salary']} salary | "
          f"DST: {tally['dst_lines']} from game lines, {tally['dst_salary']} salary fallback | "
          f"predictive sd factors " + " ".join(f"{p} x{spread_factor(cal, p):.2f}" for p in ("QB", "RB", "WR", "TE", "K", "DST")))


def _sync_captain_rows(session) -> int:
    players = session.query(NFLPlayer).all()
    flex = {p.name: p for p in players if p.roster_slot_id == 512}
    fpts_cols = [c.name for c in NFLPlayer.__table__.columns if c.name.startswith("fpts_")]
    synced = 0
    for p in players:
        src = flex.get(p.name) if p.roster_slot_id == 511 else None
        if src is None:
            continue
        for col in fpts_cols:
            setattr(p, col, getattr(src, col))
        synced += 1
    if synced:
        session.commit()
    return synced


