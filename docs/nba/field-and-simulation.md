# NBA: the field, the simulation and the chooser

Three stage modules with the same shapes as the NFL ones, each wrapped by one script:

| Stage | Module | Script |
|---|---|---|
| Generate the candidate pool | `ivan.sports.nba.field_generation.generate_field` | `scripts/nba/generate_lineups.py N JOBS` |
| The field (who else is in the contest) | `ivan.sports.nba.field_model`, `public_field`, `sample_field` | (inside the sim) |
| Simulate each contest | `ivan.sports.nba.simulate_contest.simulate_entries` | `scripts/nba/sim.py` |
| Choose | `ivan.sports.nba.choose` + `lineup_export.run_choose` (numerics in `ivan.core.chooser`, `ivan.core.portfolio`) | `scripts/nba/choose.py` |

## Generate (`field_generation.py`, `pre_lock/`)

```bash
venv/bin/python scripts/nba/generate_lineups.py 10000 8 [--fresh] [--randomness 0.25] [--min-fpts 260] [--no-progress]
```

The count is split over worker processes. Each worker opens its own session, draws one correlated
universe of player scores per sample (`pre_lock.player.generate_player_samples`: every player with
stored quantiles through a Gaussian copula per game with the role-pair correlation; players without
quantiles as Normals on the same copula) and solves one classic lineup per jittered sample
(`pre_lock.lineup_generation.generate_single_classic_lineup`, an ILP under DK's roster PG/SG/SF/PF/C/G/F/UTIL,
salary cap, min salary, max salary per team). Duplicate compositions are removed at the end. Draws are
seeded from `IVAN_SEED` when set.

Config (`config/nba.json → generate`): `n_samples` (the desk's default count), `randomness` 0.25 (spread of
the worlds around the mean), `jitter` 0.05, `min_fpts` or `min_fpts_frac` 0.9 of the optimal,
`player_min_fpts` 15. The solver reads `max_salary` 50000, `min_salary` 49000, `max_salary_per_team` 30000.

`--fresh` (the desk's "fresh pool" box, on by default) empties the loaded slate's lineups and their sims first
(other slates' stay; generate, the field sampler and the sim only ever read the loaded slate's, `field_generation.slate_lineups`);
without it a second run adds to the pool, which is how two scenarios' lineups sit side by side
before lock. The generate step writes nothing else (no field, no ownership, since 2026-10-28).

## The field (`field_model.py`, `public_field.py`, `sample_field.py`)

Config `field.sampler = two_group` (since 2026-10-23): the field is **its own lineups**, simulated from
the fitted two-group model (`data/nba/models/field_two_group.json`) on the **public projection**
(`pub_fpts`, the Stokastic replica), cached by a key made from the public projection alone so an edit
to our numbers cannot move it. Every candidate in our pool is scored against it. The alternative
`projection_model` is the old sampler: the pickled bin model (`field_distribution_model.pkl`, fitted on
last season's contest bins) draws entries from **our** pool by normalised projection. `field.cap`
10000 is the largest field size simulated; bigger contests are sampled down and the payout table scaled.

The two-group field is cached on disk by its key (`data/nba/derived/field_cache/<key>.json`) and any
smaller size is served from a bigger cached field. Only two things build it: the sim, once per run at
the largest size any contest needs (`simulate_contest.prebuild_field`, every contest slicing it; the
largest contest's field is written as `nba_player.ownership`), and the desk's **Field + ownership**
button / `scripts/nba/field_ownership.py` (`ingest.refresh_field`, `field.ownership_entries` =
3,000 by default), on demand. A refresh, a generate and a pub edit never build it (2026-10-28). A
build is an ILP per lineup: about 45 seconds for 3,000 entries, 160 for 10,000, on eight workers.

`field_model.py` is the ownership model fitted on real contests only (`scripts/nba/field_model.py
validate | fit | check`): a softmax inside each contest scaled to 800% (eight slots) over public
features (salary, position, DK FPPG, status, Rotowire starters, the props-based public projection,
the game lines, plain box-score history), with a temperature that depends on entries, fee and games.
It sees only what the public saw at lock (the archive capture the contest was matched to), never our
projections. Fitting needs derived contests from the standings scrape ([overnight.md](overnight.md)).
The DK lobby's %Drafted is **not** lineup ownership; the field's shape is per slate
(finding 2026-10-23).

## Simulate (`simulate_contest.py`)

```bash
venv/bin/python scripts/nba/sim.py [--entries data/nba/input/entries.csv] [-n 1000] [-c 10000]
```

For every contest in the entries file (fetching contest and payout rows it lacks through
`contests.ensure_contest`): the field is drawn at `field_cap` entries (the two-group field: a slice of the
one field built up front for the run), the payout table scaled to it
(`ivan.sports.common.payouts.scale_payouts_correctly`), then `n_sims` correlated draws score every
player on the field, lineups are ranked, ties share the pooled payout. Per unique lineup the result is
fpts / payout / ROI / rank in every simulation as `lineup_simulation_results` rows (the contest's
earlier rows are replaced). Each contest also leaves a sim artifact (`data/nba/output/sim/<stamp>/contest_<id>.npz`
+ `.json`: what the field and the players looked like, for validation) and a report (`contest_sim_report`
row + `data/nba/diagnostics/<contest name>_<id>/report.json`) with the field's shape, the payout scaling,
the score distribution and the conservation checks (the field should collect the whole prize pool).

The simulation is plain Monte Carlo (each candidate scored in each world against a sampled field);
the NFL's conditioned closed-form sim has no NBA counterpart yet.

## Choose (`choose.py`, `lineup_export.py`)

```bash
venv/bin/python scripts/nba/choose.py                                   # prompts for weights per contest
venv/bin/python scripts/nba/choose.py --no-prompt                       # config choose.* for everything
venv/bin/python scripts/nba/choose.py --method portfolio --bankroll 3
venv/bin/python scripts/nba/choose.py --weights-json '{"default": {"cash_pct": 0.3}, "123456": {"method": "portfolio"}}'
```

Every candidate's per-world ROI row is first pulled toward the pool by its own standard error
(`chooser.shrink_rows`, empirical Bayes), so neither method ranks on noise. Then per contest:

- **greedy** (default): quality = `cash_pct` × cash rate + `avg_roi_win` × average ROI, minus
  `diversity_lambda` × distance between the lineups' simulated point traces, picked one at a time.
- **portfolio**: judge the book, not the lineup; each step adds the lineup that most raises the book's
  log utility summed over worlds, with the bankroll as `bankroll` × the entries' fees (default 3).

Config `choose`: `cash_pct` 0.5, `avg_roi_win` 0.5, `diversity_lambda` 0.1, `method` greedy, `bankroll` 3.
The NFL's whole-slate "slate" method (one bankroll in dollars across contests) is not ported to NBA.

Outputs under `data/nba/output/`, stamped:

| File | Content |
|---|---|
| `chosen_entries.classic.<stamp>.csv` | the DK upload file: your entry rows with PG..UTIL filled as "Name (id)" |
| `chosen_entries.exposures.<stamp>.csv` | player exposure across the chosen entries |
| `chosen_entries.lineups.<stamp>.csv` | one row per player per chosen entry with the lineup's sim cash % and ROI |
| `chosen_entries.choose.<stamp>.json` | what the chooser saw per contest: counts, the book, the greedy trace, cross-fit honest expectation |
| `chosen_entries.simroi.<stamp>.npz` | each pick's per-simulation ROI, for the luck and calibration tests |

The desk's **exposures** view reads the newest choose run that covered the loaded slate's contests
and joins every chosen player to the pool as it stands now (entries, exposure, field ownership and
leverage, projection at pick vs now, status with OUT struck in red, value, team totals). The desk's
Downloads list serves the CSVs; `/export/lineups.csv` streams the whole generated pool.

## Late swap

After lock the swap plays against the **real** field: the contest's standings export (fetched as a step of
its own, reused until the next tip), its open slots completed by the optimiser; by default one export,
the biggest contest's, completed once and sampled at every contest's size. See
[game-day.md](game-day.md) step 7 and `src/ivan/sports/nba/post_lock/lateswap.py`.

## Reading a sim report

`data/nba/diagnostics/<contest>/report.json`: `field` (size, sampler, how many unique lineups,
duplication), `payouts` (table, scaling), `scores` (percentiles of the field, the cash line), `checks`
(prize pool conserved, every entry paid at most once). The NFL web app's Contests page renders the
NFL equivalent; the NBA desk does not render these yet, so read the JSON or the sim job log.
