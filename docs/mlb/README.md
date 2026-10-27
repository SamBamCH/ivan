# MLB (dormant)

Code from previous seasons, kept as is: `src/ivan/sports/mlb/` (16 modules, only `utils.py` has a
docstring) and `scripts/mlb/` (17 scripts). It predates the shared layout: no `paths.py` conventions,
`config/mlb.json` pins Windows absolute paths (`C:/Users/samba/...`) that will not resolve on Linux, and
`data/mlb/` holds only `debug/` PNGs (no database, no inputs).

Inferred run order (not documented anywhere else): `create_db.py` → `ingest.py` (projections +
`init_players`) → `get_contest_ids.py` → `get_payouts.py` → `generate_lineups.py N J` → `sim_all.py -n N`
→ `choose_all.py -o chosen.csv`. `overnight.py` ran the stats fetch, contest scrape, model builds and
contest ids (a stale `overnight.lock` sits in `scripts/mlb/`). `fit_historical_fields.py` fitted KDEs per
historical contest; `plot_historical_contest.py --csv --asof` plotted one.

Config keys: `database_url`, `paths.{projections_dir, ids, entries_csv}`, `verbose`, `lineup`
(`min_fpts_threshold` 100, pitcher / hitter dispersion, `min_salary` 48000, `max_salary` 50000,
`max_opposing_batters` 0, `stack_probabilities`), `field` (`proj_exponent` 1, `strict_player_pool`).

Shared code it uses: `sports.common.draftkings` (the Chrome session and standings download,
`get_upcoming_contest_ids`) and `sports.common.payouts.scale_payouts_correctly`. To revive it, start
by porting the stages onto the NFL/NBA module shape ([../architecture.md](../architecture.md)) and
fixing the config paths.
