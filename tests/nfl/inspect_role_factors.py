#!/usr/bin/env python3
"""
Inspect role factors and projections for players to diagnose overprojection issues.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sqlalchemy import text
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, NFLOdds


def inspect_role_factors(session, min_salary=None, max_salary=None):
    """
    Show players with their:
    - Anytime TD odds (p_any)
    - Derived role factor
    - Projection components
    - Salary and value metrics
    """

    # Get all players with projections
    players = session.query(NFLPlayer).filter(
        NFLPlayer.fpts_mean.isnot(None),
        NFLPlayer.position.notin_(['K', 'DST'])
    ).all()

    if min_salary:
        players = [p for p in players if (p.salary or 0) >= min_salary]
    if max_salary:
        players = [p for p in players if (p.salary or 0) <= max_salary]

    rows = []
    for p in players:
        # Get anytime TD odds
        anytime_row = (
            session.query(NFLOdds)
            .filter(NFLOdds.player == p.name, NFLOdds.market == 'player_anytime_td')
            .order_by(NFLOdds.last_update.desc())
            .first()
        )

        if anytime_row:
            # Calculate what the role factor would be
            from ivan.sports.nfl.fpts_distribution_calculations import (
                american_to_prob,
                _poisson_mu_from_prob_at_least_one,
                ROLE_MIN, ROLE_MAX, ROLE_BETA
            )

            p_any = american_to_prob(float(anytime_row.price))
            mu_any = _poisson_mu_from_prob_at_least_one(p_any)

            # For this to work, we'd need position baseline, but let's approximate
            # Typical baseline scrim TD for positions:
            baseline_td = {
                'RB': 0.5,   # RBs average ~0.5 combined TDs
                'WR': 0.35,  # WRs average ~0.35 combined TDs
                'TE': 0.30,  # TEs average ~0.30 combined TDs
            }.get(p.position, 0.35)

            ratio = mu_any / max(baseline_td, 0.01)
            role = np.clip(ratio ** ROLE_BETA, ROLE_MIN, ROLE_MAX)
        else:
            p_any = None
            mu_any = None
            role = None

        rows.append({
            'name': p.name,
            'position': p.position,
            'team': p.team,
            'salary': p.salary or 0,
            'fpts_mean': p.fpts_mean or 0,
            'fpts_p50': getattr(p, 'fpts_p50', None) or 0,
            'value': (p.fpts_mean / max(p.salary, 1) * 1000) if p.fpts_mean and p.salary else 0,
            'p_any': p_any,
            'mu_any': mu_any,
            'role_factor': role,
        })

    df = pd.DataFrame(rows)

    # Sort by salary to see cheap players
    df = df.sort_values('salary')

    print("\n" + "="*100)
    print("ROLE FACTOR INSPECTION")
    print("="*100)
    print(f"\nTotal players: {len(df)}")
    print(f"With anytime TD odds: {df['p_any'].notna().sum()}")
    print(f"Missing anytime TD odds: {df['p_any'].isna().sum()}")

    print("\n" + "-"*100)
    print("CHEAPEST 20 PLAYERS (potential value captains)")
    print("-"*100)
    print(df.head(20).to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    print("\n" + "-"*100)
    print("TOP 20 BY VALUE (projection per $1K)")
    print("-"*100)
    print(df.nlargest(20, 'value').to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    print("\n" + "-"*100)
    print("ROLE FACTOR STATISTICS")
    print("-"*100)
    print(df[df['role_factor'].notna()]['role_factor'].describe())

    # Show problematic patterns
    print("\n" + "-"*100)
    print("POTENTIAL OVERPROJECTIONS (high role_factor, low salary)")
    print("-"*100)
    cheap_high_role = df[
        (df['salary'] <= 5000) &
        (df['role_factor'] >= 0.30) &
        (df['role_factor'].notna())
    ].sort_values('role_factor', ascending=False)

    if len(cheap_high_role) > 0:
        print(cheap_high_role.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    else:
        print("None found")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', default='sqlite:///data/nfl/nfl_history.sqlite')
    parser.add_argument('--min-salary', type=int, default=None)
    parser.add_argument('--max-salary', type=int, default=None)
    parser.add_argument('--output', default=None, help='Save to CSV')
    args = parser.parse_args()

    engine = init_db('nfl', args.db_url)
    session = get_session(engine)

    df = inspect_role_factors(session, args.min_salary, args.max_salary)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")
