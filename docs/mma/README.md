# MMA

The UFC model ported 2026-10-10 from the standalone `mma_dfs` project onto Ivan's layout: one SQLite
file (`data/mma/mma_history.sqlite`), per-sport tables (`mma_player`, `mma_fight`, the lineup pool,
one sim result per lineup per contest, the DK contest and payout tables), `config/mma.json`, one script
per step in `scripts/mma/`, code in `src/ivan/sports/mma/`. Live.

## The flow

```bash
venv/bin/python scripts/mma/ingest.py                        # 1. the DK Classic MMA slate (most contests; --draft-group ID for another), the fighter pool,
                                                             #    Win odds from the-odds-api (DraftKings moneylines), contests + payouts; writes the two CSV templates
#   fill data/mma/input/player_odds.csv (player, ITD, Win, sig strikes, takedowns) from the sportsbook in American odds, and
#   fight_odds.csv (fight, itd) when the book posts a fight-level ITD price (optional: the two fighters' ITD prices stand in);
#   a non-empty cell wins over the API. (fight.model = tree wants the u60 / 1st round / 1st round win columns too)
venv/bin/python scripts/mma/ingest.py                        #    again: reads the overrides; the summary says what is still blank
venv/bin/python scripts/mma/fights.py                        #    every fight as the model simulates it, and what the pool still lacks (also the slate page)
venv/bin/python scripts/mma/generate_lineups.py --universe   # 2. every valid lineup (six fighters, one per fight, under the cap; 50-110k on a 12-13 fight card);
                                                             #    the MIP mode `12000 4` (jittered win probability, main-event boost) is the fallback
venv/bin/python scripts/mma/sim.py                           # 3. every contest in data/mma/input/entries.csv, each against its own field:
                                                             #    its own size (up to field.size) and the crowd its entries-per-user cap draws
venv/bin/python scripts/mma/choose.py [--roi-weight 0.8] [--diversity-weight 0.4] [--method portfolio --bankroll 3]   # 4. fills every entry row
```

After a card (DraftKings keeps standings about 14 days), the fields it left feed the duplication model:

```bash
venv/bin/python scripts/mma/get_contest_data.py --played     # every played contest's standings into data/mma/raw/dk/standings (Chrome, the DK login)
venv/bin/python scripts/mma/derive_contests.py               # each scraped field laid on the card it was played on (the card record ingest stored)
venv/bin/python scripts/mma/build_dupe_model.py              # refit on every card; prints the held-out test; the old file is kept beside the new
```

Other forms: `ingest.py --players-csv <DK export> --main-events 1 --no-odds-api --no-contests`;
`sim.py --contest "mini-MAX" | --contest-id ID | --payouts p.csv --total-entries 47500 --fee 0.5 --entry-cap 150 --name X [--field N] [-n N]`;
`derive_contests.py --card DG --player-odds FILE` (a card record by hand for a card ingested before 2026-11-14);
`build_dupe_model.py --report` (the model on disk);
`choose.py --name <contest> --lineups 150` for one contest (`chosen_lineups.{lineups,upload,exposures}.<stamp>.csv`).
Outputs: `data/mma/output/chosen_entries.<stamp>.csv` (DK upload: your entry rows with the six F columns),
`.lineups`, `.exposures`, `field.*.csv` (each sampled field). `IVAN_SEED` makes a run reproducible.
`fights.py [-n 100000]` reads only.
Tests: `venv/bin/python -m pytest tests/mma` (end to end on the sample card under `tests/mma/data`).

## How it works

| Piece | Module | What |
|---|---|---|
| the slate | `ingest.py`, `market.py`, `contests.py`, `entries.py` | DK pool (`sports.common.draftkings.ingest_players`), moneylines matched to fights by name similarity (exact normalised, last name, unique shared token, `name_aliases`), hand CSVs as overrides, win probability and projection per fighter; the latest `--main-events` fights are five rounds |
| a fight | `hazard.py` (`fight.model` hazard, the default) | one risk of finish per fighter: the fight's ITD price (or the two fighters' ITD prices summed) sets a constant hazard, each fighter's ITD price their share of the stoppages, the moneyline minus that share who wins a decision; finish time exponential, cut at the final bell, the round's bonus to the winner; strikes and takedowns are Gamma-Poisson counts at a rate that holds the posted line as the mean (the loser keeps `loser_production_share`). u60 and round-1 prices are not read: a flat hazard reproduces them ([notes/hazard_2026-11-07.md](notes/hazard_2026-11-07.md)). `ingest.py` prints the fitted parameters per fight (P(finish), expected minutes, round shares) |
| a fight, tree model | `fight.py` (`fight.model` tree) | the original: finish type from the fight markets (u60, R1, ITD), winner from the fighter markets for that finish (R1 price; ITD − R1; ML − ITD for a decision), time uniform within the round, sig strikes and takedowns scaled by time fought with diminishing returns (`time_exponent`) and noise. `scoring.*` in the config (`sig_strike`, `time_exponent` fitted 2026-10-17 on real scores) is shared by both models |
| each fight's results | `fight_report.py` (`scripts/mma/fights.py`, the slate page's Fights section) | `card_report`: every fight of the pool drawn 20,000 times on a fixed seed (the same fight objects the contest sim draws, `simulate_detail`), so two reads of a card agree. Per fight: the share of worlds each fighter wins by each round and by decision, the finish chance and where it came from (the fight's ITD price, the fighters' ITD prices, or the default), the mean time fought. Per fighter: win share (by finish, by decision), fantasy points mean, quartiles, 90th, mean when winning and when losing, share of worlds at 100+, points per $1,000, strikes and takedowns landed against their lines, and the distribution per 10 points on one axis for the whole card. `alerts`, worst first: an input file (`player_odds.csv`, `fight_odds.csv`) saved after the last ingest, so the pool does not hold it; fights with no finish price (the default chance applies); fighters with no strikes or takedowns line (they score the win bonus alone); fighters with no moneyline; contest sims older than the pool |
| the universe / candidates | `universe.py`, `lineup_generation.py` | every valid lineup, or MIP solves (GLPK if `glpsol` is present, else CBC) |
| the field | `sample_field.py`, `dupe_model.py`, `cards.py`, `standings.py` | one field per contest, sampled with replacement from the pool, weighted by duplication. `field.model = fitted` (`data/mma/models/dupe_model.json`): within a contest an entry lands on a lineup in proportion to exp(x · (b + z·g)), x = salary left (per $100, flat past $2,000), exact cap, summed win probability, log rank by summed win probability among the card's lineups, main-event fighters; z = log(the contest's entries per user / 20), so b is a 20-max crowd and g how each pull moves toward a single-entry or a 150-max one (`multi_entry` on the contest row; unknown reads as 20). A Gamma(`alpha`) multiplier per lineup is the lumpiness the features miss, measured on cards the fit did not see. The same weights give every candidate's predicted copies in the real field (`Pred Dupes`). Fitted on every scraped contest of every recorded card laid on every valid lineup of that card (zeros included); `build_dupe_model.py` prints the acceptance test, each card predicted by a fit on the others, real field shape against a sampled one by entry cap ([notes/contest_dupes_2026-11-14.md](notes/contest_dupes_2026-11-14.md)). The pool must be the universe (`generate_lineups.py --universe`): the rank is taken among the pool. `original` is the hand-set exponential the port started with |
| the card record | `cards.py` | every ingest of a DK draft group stores the card as the model saw it (fighter ids, salaries, fights, win probabilities, the five-round fighters) in the raw store, kind `card`; the newest is the card at lock. The hand-entered prices are overwritten by the next card, so this is what puts a scraped field back on its card |
| the contest | `simulate_contest.py`, `payouts.py` | the field is min(`field.size`, the contest's entries) entries; a bigger contest has its payouts scaled to the field (`sports.common.payouts`); `simulation.n_sims` worlds, every entry ranked, ties pooling prizes, one result row per unique lineup. With `IVAN_SEED` the worlds are shared by the contests and each contest's field is its own draw |
| choose | `choose.py` | greedy: quality = roi_weight × mean ROI + (1 − roi_weight) × cash rate, minus diversity_weight × the largest fighter overlap with anything already picked; or `--method portfolio` (log utility, `bankroll` in multiples of fees) |

## Config (`config/mma.json`)

`paths` (entries, fight odds, player odds, payouts CSVs), `name_aliases`, `slate` (`game_type` Classic,
`num_5_round_fights` 1), `odds_api.bookmakers` draftkings, `fight` (`model` hazard | tree, `first_minute_mult` 1,
`default_itd_prob` 0.5, `max_itd_prob` 0.97, `rate_cv` 0.25), `scoring` (sig strike 0.8, takedown 5, loser
production share 0.5, finish bonuses u60 115 / r1 90 / r2 70 / r3 45 / r4 40 / r5 40 / decision 30,
production noise cv 0.2, time exponent 0.5), `lineup_generation` (`n_candidates` 12000, `jitter_sd` .15,
`min_salary` 48500, `salary_cap` 50000, `roster_size` 6, `main_event_boost` 1.4, `projection_share` .35),
`field` (`size` 5000 = the largest field simulated, the original model's knobs, `model` fitted, `dupe_model_path`), `contest` (the
hand-described default: `total_entries`, `entry_fee`, `entry_cap`), `simulation.n_sims` 2000, `choose` (`num_lineups` 150, `roi_weight` .8,
`diversity_weight` .4, `method` greedy, `bankroll` 3). Each block has a `_note`.

## Data (`data/mma/`)

`mma_history.sqlite`; `input/` (entries.csv, fight_odds.csv, player_odds.csv and their `.prev` / `.sample`
copies, payouts.csv, player_ids.csv); `output/`; `models/dupe_model.json` (+ the files it replaced, by build date);
`raw/{dk/draftables, dk/standings/<id>/, card/<draft group>/}`, `index.jsonl`; `derived/contests/<id>/` (lineups.parquet,
players.csv; rebuilt by `derive_contests.py`). `diagnostics/` is defined in `paths.py` but not created yet.

## Not built

Settle, validation, replay (phase after lock in the port plan); the generator docstring still omits
`--universe`. A field bigger than `field.size` is simulated at that size (results are stored per unique
lineup per world, which a 60,000-entry field would outgrow). No ownership projection: the field model knows
a fighter only by salary, win probability and the main event, which is its largest miss
([notes/contest_dupes_2026-11-14.md](notes/contest_dupes_2026-11-14.md)).

## The server

The server's MMA slate page only reads: its **Refresh from laptop** button (`scripts/mma/feed.py pull`, `ivan.core.feed.pull_ssh`) copies the
whole card as the laptop holds it over ssh (every table, `data/mma/input`, `models`, `derived`), the same way the NBA and NFL pull
([../deployment.md](../deployment.md)). Nothing is fetched or run there.
