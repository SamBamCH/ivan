# LoL (dormant)

Two modules (`src/ivan/sports/lol/models.py`, `lineup_generation.py`, no docstrings) and seven
scripts (`scripts/lol/`: `create_db`, `ingest`, `update_stats`, `get_contest_ids`, `get_payouts`,
`generate_lineups`, `check_lineups`; none use argparse). There is no simulate or choose step. The
scripts read `config/lol.json`, which does not exist, and there is no `data/lol/` directory, so nothing
here runs today. Inferred order: create_db → ingest (expects `paths.odds`, e.g. `data/lol_odds.json`) →
update_stats → get_contest_ids → get_payouts → generate_lineups → check_lineups.
