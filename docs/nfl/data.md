# NFL: config, tables and files

## `config/nfl.json`

| Key | Meaning |
|---|---|
| `paths.entries_csv` | the entries file (`data/nfl/input/entries.csv`) |
| `projection.unpriced_volume` | `environment` (live) or `salary`; `IVAN_UNPRICED_VOLUME` overrides |
| `ownership.num_lineups` 1000, `ownership.num_simulations` 1 | how many sampled lineups ownership is estimated from |
| `ownership.model` | `learned` (live), `sampler_calibrated`, `sampler`; `ownership.calibration.enabled` |
| `field.ownership_calibration` {`enabled`, `iterations` 30, `damping` .7, `max_ratio` 4, `min_target_pct` .05} | rake the sampler's lineup weights to the projected ownership |
| `field.dupe_model` | `product` |
| `randomness` .5 | the generator's spread |
| `classic.lineup_generation` {`n_samples`, `n_jobs`, `salary_cap` 50000, `min_salary` 49500, `max_off_vs_dst`, `min_qb_wrte_stack`, `min_qb_skill_stack`, `min_qb_bringback`, `max_team_players` 3, `jitter` .05} | classic solver rules |
| read by code, absent from the file (defaults apply) | `sim.method` (montecarlo), `showdown.lineup_generation.*`, `solver.*` (`glpk_path`, `keep_files`, `msg`), `database_url` |

`config/replay/nfl.json`: the contest menu for replays (`contests`: name regex → entries, capped at the
contest's max; `max_fee` 5; `weights`; optional `mode` per item; `picks` written by the hub as
`selection.json`). `config/tax.json`: the tax rules and 2026 figures (verify with a CPA).

## Tables (`src/ivan/sports/nfl/models.py`)

| Table | What | Synced to the server |
|---|---|---|
| `nfl_player`, `odds`, `game_odds` | the loaded slate: pool, player props, game lines | no (working) |
| `projection_snapshot` | every ingest's projections per player, for the backtests | no (working) |
| `lineup`, `lineup_simulation_results`, `contest_sim_report` | the pool, the sim results, the sim reports | no (working) |
| `historical_contest_information`, `payouts` | DK contests and payout tables | yes |
| `nfl_stats`, `nfl_schedule`, `nfl_dst_game` | box scores, schedules with closing lines and scores, DK DST points per team-game | yes |
| `showdown_*`, `classic_contest_metadata`, `contest_field_scores`, `contest_entry_result` | scraped contests: lineups, ownership, every entry's actual points, our finishes | yes |
| `archived_slate`, `archived_blob`, `raw_artifact` | the slate archive and the raw index | yes |
| `slate_status` | the settle job's status per slate | yes |

## On disk (`data/nfl/`, `src/ivan/sports/nfl/paths.py`)

```
data/nfl/
├── nfl_history.sqlite              reference + this machine's working tables
├── replay/nfl_replay.sqlite        the replay's copy (per-worker clones .w<i>. are deleted when a replay ends)
├── input/entries.csv, input/web/   the entries file; web uploads + weights.json
├── raw/{dk,odds}/, index.jsonl     every payload as received
├── derived/contests/<id>/          lineups.parquet, scores.npy from settle
├── models/                         classic_*.pkl, dst_odds_params.json, dupe_model_v2.json, ownership_calibration.json, ownership_model.pkl, sd_calibration.json, manifest.json; asof/<cutoff>/; archive/
├── output/
│   ├── chosen_entries.*            CLI choose runs (web: output/web/chosen.*)
│   ├── projections/, sim/<stamp>/  the ingest's and the sim's artifacts
│   ├── validation/<stamp>/         validate.py reports (usage_<stamp>/ from backtest_usage)
│   ├── backtest/<stamp>/           backtest_tax reports
│   ├── backtests/<name>/           the hub's runs: run.json, selection.json, nfl/<slate>/ (replay.json + csv, rep_*/), nfl/report_*/, nfl/validation/
│   ├── replay*/                    replay roots (replay, replay_repeats, replay_artifacts, ...)
│   └── models/, chooser_ab/        the models scoreboard, compare_choosers
│       (backtests/, models/, replay*/: rep_*/, validation csvs, replay.sqlite and logs go 30 days after the
│        run's last write unless pinned; manifests, csv, reports stay: scripts/nfl/retention.py)
├── diagnostics/contest_<id>_<name>/report.json
├── jobs/                           web job logs
└── logs/nightly.log
```
