# NBA: the field, the simulation and the chooser

Three stage modules with the same shapes as the NFL ones, each wrapped by one script:

| Stage | Module | Script |
|---|---|---|
| Generate the candidate pool | `ivan.sports.nba.field.generate.generate_field` | `scripts/nba/generate_lineups.py N JOBS` |
| The field (who else is in the contest) | `ivan.sports.nba.field.model`, `public_field` | (inside the sim) |
| Simulate each contest | `ivan.sports.nba.contest.simulate.simulate_entries` | `scripts/nba/sim.py` |
| Choose | `ivan.sports.nba.contest.choose` + `contest.export.run_choose` (numerics in `ivan.core.chooser`, `ivan.core.portfolio`, `ivan.core.slate_book`) | `scripts/nba/choose.py` |

## Generate (`field/generate.py`, `field/solver.py`)

```bash
venv/bin/python scripts/nba/generate_lineups.py 10000 8 [--fresh] [--randomness 0.25] [--min-fpts 260] [--no-progress]
```

The count is split over worker processes. Each worker opens its own session, draws the slate's games
(`gamesim.draw.generate_player_samples`: the game simulation, `ivan.sports.nba.gamesim.world.World`, up to 2,000
worlds per call and resampled beyond that, every player's points in a world coming from the same simulated game) and
solves one classic lineup per jittered sample
(`field.solver.generate_single_classic_lineup`, an ILP under DK's roster PG/SG/SF/PF/C/G/F/UTIL,
salary cap, min salary, max salary per team). Duplicate compositions are removed at the end. Draws are
seeded from `IVAN_SEED` when set.

Config (`config/nba.json → generate`): `n_samples` (the desk's default count), `randomness` 0.25 (spread of
the worlds around the mean), `jitter` 0.05, `min_fpts` or `min_fpts_frac` 0.9 of the optimal,
`player_min_fpts` 15. The solver reads `max_salary` 50000, `min_salary` 49000, `max_salary_per_team` 30000.

`--fresh` (the desk's "fresh pool" box, on by default) empties the slate's lineups and their sims first
(each slate's live in its own database, `data/nba/slates/<dg>/slate.sqlite`, so no other slate's are touched);
without it a second run adds to the pool, which is how two scenarios' lineups sit side by side
before lock. The generate step writes nothing else (no field, no ownership, since 2026-10-28).

## The field (`field/model.py`, `field/public_field.py`)

The field is **its own lineups** (since 2026-10-23), simulated from the fitted two-group model
(`data/nba/models/field_two_group.json`) on the **public projection** (`pub_fpts`, the Stokastic replica),
cached by a key made from the public projection alone so an edit to our numbers cannot move it. Every
candidate in our pool is scored against it (`contest.simulate.simulate_vs_field`). `field.cap` 10000 is
the largest field size simulated; bigger contests are sampled down and the payout table scaled.

The two-group field is cached on disk by its key (`data/nba/derived/field_cache/<key>.json`) and any
smaller size is served from a bigger cached field. Only two things build it: the sim, once per run at
the largest size any contest needs (`contest.simulate.prebuild_field`, every contest slicing it; the
largest contest's field is written as `nba_player.ownership`), and the desk's **Field + ownership**
button / `scripts/nba/field_ownership.py` (`ingest.refresh_field`, `field.ownership_entries` =
3,000 by default), on demand. A refresh, a generate and a pub edit never build it (2026-10-28). A
build is an ILP per lineup: about 45 seconds for 3,000 entries, 160 for 10,000, on eight workers.

`field/model.py` is the ownership model fitted on real contests only (`scripts/nba/field_model.py
validate | fit | check`): a softmax inside each contest scaled to 800% (eight slots) over public
features (salary, position, DK FPPG, status, Rotowire starters, the props-based public projection,
the game lines, plain box-score history), with a temperature that depends on entries, fee and games.
It sees only what the public saw at lock (the archive capture the contest was matched to), never our
projections. Fitting needs derived contests from the standings scrape ([overnight.md](overnight.md)).
The DK lobby's %Drafted is **not** lineup ownership; the field's shape is per slate
(finding 2026-10-23). How far the field and its ownership are from the real fields of the season, contest by contest,
is `scripts/nba/backtest_field.py` ([validation-and-backtests.md](validation-and-backtests.md#the-field-backtest-ivansportsnbabacktestfield-scriptsnbabacktest_fieldpy)).

## Simulate (`contest/simulate.py`)

```bash
venv/bin/python scripts/nba/sim.py [--entries data/nba/input/entries.csv] [-n 1000] [-c 10000]
```

For every contest in the entries file (fetching contest and payout rows it lacks through
`contests.ensure_contest`): the field is drawn at `field_cap` entries (the two-group field: a slice of the
one field built up front for the run), the payout table scaled to it
(`ivan.sports.common.payouts.scale_payouts_correctly`), then the run's `n_sims` correlated draws score every
player on the field, lineups are ranked, ties share the pooled payout. Per unique lineup the result is
fpts / payout / ROI / rank in every simulation as `lineup_simulation_results` rows (the contest's
earlier rows are replaced). Each contest also leaves a sim artifact (`data/nba/output/sim/<stamp>/contest_<id>.npz`
+ `.json`: what the field and the players looked like, for validation) and a report (`contest_sim_report`
row + `data/nba/diagnostics/<contest name>_<id>/report.json`) with the field's shape, the payout scaling,
the score distribution and the conservation checks (the field should collect the whole prize pool).

**One draw per run.** The players' points are drawn once per sim run (`contest.simulate.slate_draw`, seed
`sim/slate` under `IVAN_SEED`) and every contest scores its field and candidates in those same worlds, so world
*w* is the same games in every contest and the slate chooser can add the contests' results world by world. The
report records it as `contest.draw = "slate"` (a contest simulated on its own would say `"contest"`, and the slate
chooser refuses it). The simulation is plain Monte Carlo (each candidate scored in each world against a sampled
field); the NFL's conditioned closed-form sim has no NBA counterpart yet.

**The draw is the game simulation** (`ivan.sports.nba.gamesim.world.World`, plan and findings in
[notes/game_sim_plan_2026-11-07.md](notes/game_sim_plan_2026-11-07.md) and [notes/game_sim_fit_2026-11-07.md](notes/game_sim_fit_2026-11-07.md)).
Every game of the slate is simulated minute by minute, n worlds at once: the script (`gamesim.script`) draws the night's
pace and each six-minute segment's possessions around the two teams' trailing paces (`data/nba/models/gamesim/teams.json`)
and the books' total and spread; the rotation (`gamesim.rotation`) turns each team's skeleton grid into every player's
coverage of every minute, with the soft pool (a player's own chance of being cut tonight, a zero-minute player's chance of
playing a few minutes: `projection.cuts`, [projections.md](projections.md#6-the-pool-who-is-in-tonights-rotation-config-depth_cut-depth_extra-pool)),
scratches and foul-outs handed to heirs by the measured flow across position groups,
fouls at each player's own rate and the coach's measured reaction, garbage time off the emergent score (closers out
while the lead holds) and overtime for the five at the buzzer; each minute the team's possessions end once each as
two-point attempts, three-point attempts, free-throw trips and turnovers at the roster's pie, split among the five on
the floor by the pool's per-minute rates times coverage (so a priced player's props drive his share), makes by each
player's trailing shooting percentages (`gamesim.profiles`) with the attempts scaled by one team factor so the expected
points are what the lines say; the misses are rebounded at the team's share and by the other five, the turnovers become
the other team's steals, the missed shots its blocks, the makes carry assists. Garbage time (the closers out while the
lead holds) is **off** by default (`config/nba.json → gamesim.garbage`): the grid's minutes already discount the blowout
nights ([notes/garbage_time_2026-11-09.md](notes/garbage_time_2026-11-09.md)). Four game-level knobs (a team's pull toward
its own expected scoring, the lead's pull-back, the pace-efficiency coupling, the endgame tie bump) are calibrated on the
World against the real spread of totals, margins, team points and the overtime rate (`scripts/nba/gamesim.py
calibrate-world --slate DG`, stored as `world_*` in `data/nba/models/gamesim/script.json`). `scripts/nba/gamesim.py
draw-world --slate DG` draws a slate and prints the moments, the reconciliation factors and every player's line against
the pool. A thousand worlds of a four-game slate take about six seconds.

**The World is also the projection.** The desk projects a team by writing its mapping (minutes, grid row, rates) to the
pool and drawing its game again (`projection.worlds.simulate_game`, 1000 worlds, under a second a game); every player's
`fpts_mean` / `fpts_std` on the pool are those worlds' means and sds, so the number on the card is the number the field,
the sim and the chooser play against ([projections.md](projections.md)). The generator draws its own World on the same
inputs (same distribution, another seed).

## Solve (`field/optimal.py`, `scripts/nba/optimal.py`)

```bash
venv/bin/python scripts/nba/optimal.py --slate DG [-n 100] [-j 4]
```

The optimal lineup in each of the World's worlds: the per-world fantasy points every team's projection store record
holds (written when the team was projected), world *i* of every game taken together as one night, the salary-capped
optimal solved in each with the generator's solver in `-j` worker processes (100 worlds in about a second), written to
`slates/<dg>/projection/optimal.json`: per player his share of the optimals and whether the mean-projection optimal
holds him. The desk's **Solve** (the actions bar beside Field + ownership, and the card's World line) runs it as a job
and the **opt** column fills when it is done; a team projected since marks it stale. It is the generator's exposure at
randomness 1 and no jitter, read before the generate runs.

## Choose (`contest/choose.py`, `contest/export.py`)

```bash
venv/bin/python scripts/nba/choose.py                                   # prompts for weights per contest
venv/bin/python scripts/nba/choose.py --no-prompt                       # config choose.* for everything: the slate book on the bankroll from our results
venv/bin/python scripts/nba/choose.py --method greedy
venv/bin/python scripts/nba/choose.py --method portfolio --bankroll 3
venv/bin/python scripts/nba/choose.py --method slate --bankroll-dollars 1000    # the slate book on a given bankroll
venv/bin/python scripts/bankroll.py                                     # the bankroll a run would read
venv/bin/python scripts/nba/choose.py --weights-json '{"default": {"cash_pct": 0.3}, "123456": {"method": "portfolio"}}'
```

Every candidate's per-world ROI row is first pulled toward the pool by its own standard error
(`chooser.shrink_rows`, empirical Bayes), so no method ranks on noise. Then:

- **greedy** (per contest): quality = `cash_pct` × cash rate + `avg_roi_win` × average ROI, minus
  `diversity_lambda` × distance between the lineups' simulated point traces, picked one at a time.
- **portfolio** (per contest): judge the book, not the lineup; each step adds the lineup that most raises the book's
  log utility summed over worlds, with the bankroll as `bankroll` × the entries' fees (default 3).
- **slate** (the default; every contest at once; `choose.choose_slate` on `ivan.core.slate_book`, shared with NFL): the whole slate
  as one book in dollars from one bankroll (`bankroll_dollars`). The candidates are (contest, lineup) pairs carrying
  fee × ROI in every world of the run's shared draw; the greedy adds the pair that most raises the mean over worlds of
  log(1 + book / bankroll), each contest filling the entries file's count, a lineup at most once per contest, then an
  exchange pass swaps picks one for one wherever that raises the utility. A lineup's vector differs per contest (its
  payout table), so a steady casher lands in the flat or single-entry contests and a tail lineup in the top-heavy ones,
  and the log spreads the book across worlds instead of stacking one script. It needs every contest's sim report to
  say `contest.draw = "slate"` (one sim run over the entries file) and refuses otherwise. Its only inputs are the
  entries file and the bankroll. With no `bankroll_dollars` given, the bankroll is read from our results: `config/bankroll.json`'s `start` plus what our entries made (won − fees on the scored contests of every sport's results ledger, on the days from `since` on; `ivan.core.bankroll`, `scripts/bankroll.py` prints it): it
  grows and shrinks with the results of every sport, a contest not yet scored moves nothing, and the choose log says
  what the figure is made of ([../nfl/chooser.md](../nfl/chooser.md#the-bankroll)). A contest's own knobs are ignored
  when the desk's **Slate book** runs it; under plain **Choose** every contest without a method of its own joins the
  book (a `bankroll_dollars` given on any of them is used, the largest when they differ). Unbacktested: see
  [../nfl/notes/slate_book_2026-11-05.md](../nfl/notes/slate_book_2026-11-05.md).

Config `choose`: `cash_pct` 0.5, `avg_roi_win` 0.5, `diversity_lambda` 0.1, `method` slate, `bankroll` 3,
`bankroll_dollars` null (read from our results; a figure pins it). `config/bankroll.json`: `start` 1000, `since` null,
`sports` null.

Outputs under `data/nba/output/`, stamped:

| File | Content |
|---|---|
| `chosen_entries.classic.<stamp>.csv` | the DK upload file: your entry rows with PG..UTIL filled as "Name (id)" |
| `chosen_entries.exposures.<stamp>.csv` | player exposure across the chosen entries |
| `chosen_entries.lineups.<stamp>.csv` | one row per player per chosen entry with the lineup's sim cash % and ROI |
| `chosen_entries.choose.<stamp>.json` | what the chooser saw per contest: counts, the book, the greedy trace, cross-fit honest expectation; for the slate method the slate book under each contest it chose (`slate`: dollar distribution, exchange pass, per-contest split, placements) |
| `chosen_entries.simroi.<stamp>.npz` | each pick's per-simulation ROI, for the luck and calibration tests |

The desk's **exposures** view reads the newest choose run that covered the loaded slate's contests
and joins every chosen player to the pool as it stands now (entries, exposure, field ownership and
leverage, projection at pick vs now, status with OUT struck in red, value, team totals). The desk's
Downloads list serves the CSVs; `/export/lineups.csv` streams the whole generated pool.

## Late swap

After lock the swap plays against the **real** field: the contest's standings export (fetched as a step of
its own, reused until the next tip), its open slots completed by the optimiser; by default one export,
the biggest contest's, completed once and sampled at every contest's size. See
[game-day.md](game-day.md) step 7 and `src/ivan/sports/nba/lateswap/swap.py`.

## Reading a sim report

`data/nba/diagnostics/<contest>/report.json`: `field` (size, sampler, how many unique lineups,
duplication), `payouts` (table, scaling), `scores` (percentiles of the field, the cash line), `checks`
(prize pool conserved, every entry paid at most once). The NFL web app's Contests page renders the
NFL equivalent; the NBA desk does not render these yet, so read the JSON or the sim job log.
