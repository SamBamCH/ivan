# NFL: the pre-lock pipeline

Five scripts, run top to bottom from the repository root; the web UI's five buttons run the same
scripts (`src/ivan/web/pipeline_nfl.py`). Each is argparse around one function in
`ivan.sports.nfl`, so the replay runs the same code in process.

| # | Step | Command | What it does |
|---|---|---|---|
| 1 | Ingest | `venv/bin/python scripts/nfl/ingest.py -p --update-odds true [--draft-group ID] [--no-ownership] [--no-snapshot]` | `ingest.fetch_slate` (DK draftables, the player props and game lines from the-odds-api; nothing is written until every fetch succeeded), `store_slate` (replaces `nfl_player`, `odds`, `game_odds`; drops OUT/IR players; archives the capture; indexes the raw payloads), `project_slate` (fits the FPTS distributions, projects and snapshots ownership, writes the projections / environment / usage artifacts). `-p` re-fetches the pool and prompts for the slate; without it only the odds refresh. `scripts/nfl/project_ownership.py` re-runs ownership alone. |
| 2 | Contests and payouts | `venv/bin/python scripts/nfl/get_contest_ids.py && venv/bin/python scripts/nfl/get_payouts.py [--days N \| --all]` | Upcoming DK contests (`historical_contest_information`) and the payout table of every contest without one. The sim needs a payout table for every contest in the entries file. |
| 3 | Generate | `venv/bin/python scripts/nfl/generate_lineups.py 10000 8 [--no-progress]` | `field_generation.generate_field`: a correlation check in the parent, worker processes drawing correlated player samples and solving one lineup per sample (GLPK ILP, showdown or classic), dedupe at the end. Seeded from `IVAN_SEED`. |
| 4 | Simulate | `venv/bin/python scripts/nfl/sim.py [--entries f.csv] [-n 1000] [-c 10000] [-d]` | `simulate_contest`: for every contest in the entries file, a field sampled from the pool with the dupe model, one correlated draw per simulation shared across the slate's contests, lineups ranked and paid from the payout table scaled to the field. Results replace earlier ones per contest; a report per contest and a sim artifact per run. [simulation.md](simulation.md). |
| 5 | Choose | `venv/bin/python scripts/nfl/choose.py [--entries f.csv] [--no-prompt] [--method greedy\|portfolio\|slate] [--cash-pct .5 --avg-roi-win .5 --diversity-lambda .1] [--bankroll 3] [--bankroll-dollars 1000] [--weights-json ...] [-o base.csv]` | Picks lineups per contest and writes the DK upload CSV plus the files the UI and the backtests read. Prompts per contest unless a weight flag, `--weights-json` or `--no-prompt` is given. [chooser.md](chooser.md). |

The **entries file** is the CSV exported from DraftKings for the contests you entered (`Entry ID,
Contest Name, Contest ID, Entry Fee, ...`), read from `config/nfl.json → paths.entries_csv`
(`data/nfl/input/entries.csv`), `--entries` or `IVAN_ENTRIES_CSV`. The web UI uploads to `data/nfl/input/web/entries.csv`.

**Outputs** land in `data/nfl/output/` (the web UI: `output/web/`):

| File | Content |
|---|---|
| `chosen_entries.showdown.<stamp>.csv` / `.classic.<stamp>.csv` | the DK upload file |
| `chosen_entries.exposures.<stamp>.csv` | player exposure across the chosen entries |
| `chosen_entries.lineups.<stamp>.csv` | one row per player per entry with the lineup's sim cash % and ROI |
| `chosen_entries.choose.<stamp>.json` | what the chooser saw per contest: every candidate's cash %, ROI, quality, map position, the picks, the greedy trace or the book |
| `chosen_entries.simroi.<stamp>.npz` | each pick's per-simulation ROI (and `roi_draw` for the luck test), kept because the sim tables are cleared weekly |
| `projections/projections.<stamp>.csv`, `environment.<stamp>.csv`, `usage.<stamp>.csv` | the ingest's artifacts |
| `sim/<stamp>/contest_<id>.npz + .json` | the sim's artifacts |
| `diagnostics/contest_<id>_<name>/report.json` | the sim report (also the `contest_sim_report` row) |

`--weights-json` takes a file or a literal keyed by contest id or name with an optional `default`:

```json
{"default": {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1},
 "193391004": {"diversity_lambda": 0.3, "method": "portfolio"}}
```

## Housekeeping between slates

`scripts/nfl/overnight.py` is the manual new-slate reset (box scores, clear lineups and sim results,
`VACUUM` when a quarter of the file is free pages, settle, sync to the server); `clear_sims.py` and
`clear_all.py` clear less. The timer job never deletes ([overnight.md](overnight.md)).

## Reproducibility

`IVAN_SEED=<int>` seeds every draw (the generator's jitter, the sim's universes, the field sampler);
`IVAN_AS_OF` pins the clock; `IVAN_RUN_DIR` redirects the artifacts. The replay driver sets all three.
