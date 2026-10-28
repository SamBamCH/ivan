# NFL

DraftKings NFL showdown and classic slates: ingest the slate, fit every player's fantasy-point
distribution from the betting markets (salary and game-environment models where nothing is priced),
project ownership and duplication, generate a correlated field, simulate the contests in your entries
file, and pick the lineups to upload. Code in `src/ivan/sports/nfl/`, commands in `scripts/nfl/`, the web
UI in `src/ivan/web/` ([../web/nfl-app.md](../web/nfl-app.md)).

| Page | Question it answers |
|---|---|
| [pipeline.md](pipeline.md) | The five pre-lock steps with their exact commands, inputs and outputs. |
| [projections.md](projections.md) | How a player's distribution is built: the odds model, unpriced volume, the game environment and usage layers, kickers and defenses, sd calibration, correlation. |
| [ownership-and-dupes.md](ownership-and-dupes.md) | How ownership is projected and corrected, and how many copies of a lineup the field holds. |
| [simulation.md](simulation.md) | The field sampler, the Monte Carlo contest sim (live), the conditioned closed-form sim (lab), the sim report. |
| [chooser.md](chooser.md) | greedy, portfolio and slate; the shrinkage; honest ROI by cross-fit; comparing choosers on a slate. |
| [backtest-and-validation.md](backtest-and-validation.md) | Replay, repeats, as-of models, the Backtest hub and the per-slate builds page, what retention keeps and lets go, validation per component, the models scoreboard, settle and the after-tax report. |
| [overnight.md](overnight.md) | The nightly timer, the manual overnight reset, settle, the scrapes, model builds, the sync to the server. |
| [data.md](data.md) | Config keys, tables, the data directory, the models directory. |
| [notes/](notes/) | Dated plans: the slate chooser. |

## The flow in one picture

```
DK draftables + lobby ─┐
the-odds-api props ────┼─► ingest.py -p (fetch → store → project) ─► nfl_player, odds, game_odds, an archive capture, raw payloads
game lines ────────────┘        │
                                ├─► fit FPTS distributions (odds model; salary / environment+usage for unpriced stats; K from kicking markets; DST from the lines)
                                ├─► ownership: sampler → calibration lines / learned model; projection snapshot
                                └─► artifacts: projections.<stamp>.csv, environment.<stamp>.csv, usage.<stamp>.csv
get_contest_ids.py + get_payouts.py ─► contests + payout tables
generate_lineups.py N J ─► lineup pool (correlated samples → ILP, deduped; correlation check per game)
sim.py ─► per contest: sampled field (dupe model), one shared draw per slate, lineup_simulation_results, contest_sim_report, sim artifact
choose.py ─► DK upload CSV + exposures + lineups + choose json + simroi npz
after the games: nightly.py (timer) → box scores, DST table, settle (standings scrape, derived fields, slate status)
                 validate.py, backtest hub, replay.py, backtest_tax.py
```

## Where to look when something is wrong

| Symptom | Look at |
|---|---|
| A projection is off | [projections.md](projections.md); the ingest's validation line (how many stats came from a market vs the salary model, where each K/DST came from); the Validation page's projections table split by `volume_source` |
| Ownership or dupes are off | [ownership-and-dupes.md](ownership-and-dupes.md); `/dupes`; the Validation page's ownership check |
| A contest's sim looks wrong | the Contests page Checks tab; `data/nfl/diagnostics/contest_*/report.json`; [simulation.md](simulation.md) |
| The chooser stacked one game script | [chooser.md](chooser.md); the Choose page; `compare_choosers.py` |
| A slate will not settle | [overnight.md](overnight.md): `data/nfl/logs/nightly.log`, the slate's status on `/backtest/plan` (`login_needed` means log in to DraftKings in the scraper's Chrome window) |
| The server has old reference data | `scripts/nfl/sync_to_server.sh` ([../deployment.md](../deployment.md)) |

## Status (2026-10-27)

Live, showdown and classic. The live simulation is Monte Carlo on one shared draw per slate; the
conditioned closed-form sim is kept in the lab until its count model handles deterministic lineups.
The chooser shrinks ROI by its standard error; greedy is the default, portfolio and the new slate method
(whole slate as one book in dollars, unbacktested) are selectable. Open items: the sim field's top is too
flat (0.49x), classic slates lose in every replay seed, the learned captain ownership is about 11 points
too flat at the top (a log-target fit fixes most of it, not applied), the usage layer's joint draws are
not consumed yet. Rule of the house: no live sim or model change without the backtest's field checks
passing.
