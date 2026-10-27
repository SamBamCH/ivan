# MMA

The UFC model ported 2026-10-10 from the standalone `mma_dfs` project onto Ivan's layout: one SQLite
file (`data/mma/mma_history.sqlite`), per-sport tables (`mma_player`, `mma_fight`, the lineup pool,
one sim result per lineup per contest, the DK contest and payout tables), `config/mma.json`, one script
per step in `scripts/mma/`, code in `src/ivan/sports/mma/`. Live.

## The flow

```bash
venv/bin/python scripts/mma/ingest.py                        # 1. the DK Classic MMA slate (most contests; --draft-group ID for another), the fighter pool,
                                                             #    Win odds from the-odds-api (DraftKings moneylines), contests + payouts; writes the two CSV templates
#   fill data/mma/input/fight_odds.csv (fight, u60 sec, 1st round, itd) and player_odds.csv (player, 1st round win, ITD, Win, sig strikes, takedowns)
#   from the sportsbook in American odds; a non-empty cell wins over the API
venv/bin/python scripts/mma/ingest.py                        #    again: reads the overrides; the summary says what is still blank
venv/bin/python scripts/mma/generate_lineups.py --universe   # 2. every valid lineup (six fighters, one per fight, under the cap; 50-110k on a 12-13 fight card);
                                                             #    the MIP mode `12000 4` (jittered win probability, main-event boost) is the fallback
venv/bin/python scripts/mma/sim.py                           # 3. every contest in data/mma/input/entries.csv, each against its own sampled field
venv/bin/python scripts/mma/choose.py [--roi-weight 0.8] [--diversity-weight 0.4] [--method portfolio --bankroll 3]   # 4. fills every entry row
```

Other forms: `ingest.py --players-csv <DK export> --main-events 1 --no-odds-api --no-contests`;
`sim.py --contest "mini-MAX" | --contest-id ID | --payouts p.csv --total-entries 47500 --fee 0.5 --name X [--field N] [-n N]`;
`choose.py --name <contest> --lineups 150` for one contest (`chosen_lineups.{lineups,upload,exposures}.<stamp>.csv`).
Outputs: `data/mma/output/chosen_entries.<stamp>.csv` (DK upload: your entry rows with the six F columns),
`.lineups`, `.exposures`, `field.*.csv` (each sampled field). `IVAN_SEED` makes a run reproducible.
Tests: `venv/bin/python -m pytest tests/mma` (end to end on the sample card under `tests/mma/data`).

## How it works

| Piece | Module | What |
|---|---|---|
| the slate | `ingest.py`, `market.py`, `contests.py`, `entries.py` | DK pool (`sports.common.draftkings.ingest_players`), moneylines matched to fights by name similarity (exact normalised, last name, unique shared token, `name_aliases`), hand CSVs as overrides, win probability and projection per fighter; the latest `--main-events` fights are five rounds |
| a fight | `fight.py` | Monte Carlo of one fight from its markets: finish type from the fight markets (u60, R1, ITD), winner from the fighter markets for that finish (R1 price; ITD − R1; ML − ITD for a decision), time uniform within the round, sig strikes and takedowns scaled by time fought with diminishing returns and noise, the round's finish bonus to the winner, the loser keeps half his production. `scoring.*` in the config (`sig_strike`, `time_exponent` fitted 2026-10-17 on real scores) |
| the universe / candidates | `universe.py`, `lineup_generation.py` | every valid lineup, or MIP solves (GLPK if `glpsol` is present, else CBC) |
| the field | `sample_field.py`, `standings.py`, `scripts/mma/build_dupe_model.py` | `field.size` entries sampled with replacement weighted by duplication. `field.model = fitted`: a Poisson log-linear model of copies on salary left, summed win probability and a same-fight flag (`data/mma/models/dupe_model.json`), fitted on real fields exported with `get_contest_data.py --contest-ids ... | --played` (Chrome profile; raw `data/mma/raw/dk/standings/`, derived `data/mma/derived/contests/<id>/`). `original` is the hand-set exponential the port started with. Retune 2026-10-17 on real 10-10 fields: 12,000 solves, boost 1.4, projection_share 0.35 |
| the contest | `simulate_contest.py`, `payouts.py` | payouts scaled to the field (`sports.common.payouts`), `simulation.n_sims` worlds, every entry ranked, ties pooling prizes, one result row per unique lineup |
| choose | `choose.py` | greedy: quality = roi_weight × mean ROI + (1 − roi_weight) × cash rate, minus diversity_weight × the largest fighter overlap with anything already picked; or `--method portfolio` (log utility, `bankroll` in multiples of fees) |

## Config (`config/mma.json`)

`paths` (entries, fight odds, player odds, payouts CSVs), `name_aliases`, `slate` (`game_type` Classic,
`num_5_round_fights` 1), `odds_api.bookmakers` draftkings, `scoring` (sig strike 0.8, takedown 5, loser
production share 0.5, finish bonuses u60 115 / r1 90 / r2 70 / r3 45 / r4 40 / r5 40 / decision 30,
production noise cv 0.2, time exponent 0.5), `lineup_generation` (`n_candidates` 12000, `jitter_sd` .15,
`min_salary` 48500, `salary_cap` 50000, `roster_size` 6, `main_event_boost` 1.4, `projection_share` .35),
`field` (`size` 5000, the original model's knobs, `model` fitted, `dupe_model_path`), `contest` (the
hand-described default), `simulation.n_sims` 2000, `choose` (`num_lineups` 150, `roi_weight` .8,
`diversity_weight` .4, `method` greedy, `bankroll` 3). Each block has a `_note`.

## Data (`data/mma/`)

`mma_history.sqlite`; `input/` (entries.csv, fight_odds.csv, player_odds.csv and their `.prev` / `.sample`
copies, payouts.csv, player_ids.csv); `output/`; `models/dupe_model.json` (+ dated versions);
`raw/{dk/draftables, dk/standings/<id>/}`, `index.jsonl`; `derived/contests/<id>/` (lineups.parquet,
players.csv). `diagnostics/` is defined in `paths.py` but not created yet.

## Not built

Settle, validation, replay (phase after lock in the port plan); the generator docstring still omits
`--universe`. Plan memory: standings scraper + fitted dupe model were the 2026-10-17 work.
