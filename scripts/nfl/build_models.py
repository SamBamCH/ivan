#!/usr/bin/env python3
"""
scripts/nfl/build_models.py

Thin wrapper that delegates to the core modeling functions in
`ivan.sports.nfl.model_builder`.
"""
import argparse
from pathlib import Path

from ivan.sports.nfl.player_level_calculations import build_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build and store NFL stat models (marginals + copulas)'
    )
    parser.add_argument(
        '--db-url', required=False, default=None,
        help='Optional database URL for NFL stats (default uses data/nfl/nfl_history.sqlite)'
    )
    parser.add_argument(
        '--model-dir', default='data/nfl/models',
        help='Directory to save model files'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate diagnostic plots of data vs fitted PDFs'
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    build_models(args.db_url, model_dir, args.plot)


if __name__ == '__main__':
    main()