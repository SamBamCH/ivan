# Process map, first path: a minutes pin to the drawn fantasy points (2026-11-20)

The first of the swim-lane maps ([../figures/minutes_pin_trace.drawio.svg](../figures/minutes_pin_trace.drawio.svg), the
plain-words version [../figures/minutes_pin_overview.drawio.svg](../figures/minutes_pin_overview.drawio.svg)). It was
drawn by reading the code on master `7a6f133` from the box on the team page to the numbers written back, with config
`projection.alloc = solve`, `gamesim.rotation = stints`, `gamesim.rotation_pool = true`, `gamesim.cuts = all`,
`gamesim.garbage = false`. Nothing was removed; this is the list of what the map turned up.

## What the path does to the number

1. The typed text becomes a number and the name becomes his NBA key; the pin is saved as one row in
   `minutes_overrides.csv`.
2. The team is mapped: the solve holds the pin (sd 0.05) and gives the others the books' implied minutes or their
   history, to 241; the depth cut keeps the coach's usual count (pins and starters always kept) and solves the kept
   players again; the grid is built to those targets and **his minutes become the grid row's sum**, whole minutes.
3. His rates are set after the minutes: a priced stat is the books' mean over his minutes, so his plan total for a
   priced stat does not move with the pin; an unpriced stat is his history rate, so it does.
4. He gets `p_out` 0 (a pin or a starter is not scratched) and a `p_cut` from his own cut history (0 for a starter;
   a pin does not zero it).
5. The plan is written to his `nba_player` row and the simulation reads it back from there.
6. In the simulation his minutes are the steering target of the stint draw (one tilt per player, solved once per
   team so the median lands on the target scaled to 240); he sits the whole night in 0.7 x `p_cut` x the DNP share of
   his cut nights of the worlds unless he is one of the five at the tip.
7. The minute loop turns coverage x rates into his stats, and the result (mean, sd, quantiles, minutes
   distribution, on-floor share per minute) is written to the pool and the team's store record for both teams.

## Computed or kept, not used on today's settings

Each is a candidate, not a decision. "Read" = established by reading the code; "measured" = checked on stored data.

| What | Where | Finding | Basis |
|---|---|---|---|
| The second, "history-only" mapping per edit | `ingest._map_team` (`ingest.py:289`), read by `desk/rows.py:193` | Under `alloc = solve` a priced stat's rate is replaced by the books' mean over the minutes whatever `market_weight` is (`engine.py:220`), and the solve does not read `market_weight`, so the second run equals the first. In the 12 most recently written store records (121 players with minutes) the two frames' minutes and plan fantasy points are identical for every player, priced or not. The "market +x% vs history" flag it feeds therefore compares the market with itself. It also builds the whole grid a second time on every edit. | measured |
| `write_plan` reads the second run's grid | `live.py:285`, `:300` | `_last_rotation` is overwritten by the history-only run before `write_plan` reads it for `rotation_cells`. Harmless while the two runs are identical; it would write the wrong grid the day they differ. | read |
| The market-rate blend for a priced stat | `engine.py:133` then `:220` | `blend_rates` runs for every player and its result is replaced for each priced stat a few lines later. The stored market *rate* and its basis minutes (`basis_minutes_for_market`, a full extra projection of every team with the older allocation) then appear to matter only for the `priced` flag. | read |
| Band and fppm pins | `web/nba/team.py:31`, `live.py:164`, `engine.py:259`, `:272` | No box on the page sends `low`, `high` or `fppm` any more. A band's only effects are a pin at its midpoint when minutes is blank and a `band_pinned` flag (`tau_from_band`'s value is dropped); an fppm value in an old file row still scales every rate. | read |
| `locks`, the non-solve allocation, `rebalance`, the `rng` argument | `engine.py:185`, `:162-175`, `:114` | `locks` is always `None` from the live path (`live.py:236`). The non-solve branches run only for `basis_minutes_for_market`. `rng` is documented as unused. | read |
| `boundary_sd`, `minutes_sd`, the grid cells after minute 0 | written `live.py:312-315`, read `world.py:236-239` | Only the plan rotation (`_plan_coverage`, `gamesim.rotation = plan`) uses them. On stints the grid's first column names the five and the row's sum is the minutes; the rest of the grid, built by `skeleton.py` (about 500 lines), is not read by the draw. `minutes_sd` is still shown on the minutes hover. | read |
| `PlayerIn.fpts_std` | `world.py:237` | Set, read nowhere. | read |
| Garbage time (`_pull_closers`), `stint_cuts`, `rotation.draw` | `world.py:481`, `:530`; `stints.py:1071`; `gamesim/rotation.py` | The garbage pull is skipped whenever stints run. `stint_cuts` is set only by `rerun_slate.py --compare stints-cuts`. `rotation.draw` is the plan rotation. All three are reachable only by changing config or by that script. | read |
| The `p_out` scratch draw | `engine.py:250`, `stints.py:1069` | 0 for every pinned row and every starter, so on a desk where every rotation row is pinned it never fires. | read |

Not checked here: whether anything outside this path (the backtest's game view, the consumer's desk, the scenarios)
depends on the pieces above. The plan rotation in particular is still the comparison arm of the rerun script.

## Next paths to map

Refresh (DK pool, props, market rates), the starter call, Solve, the field and ownership, the contest simulation
and the chooser, late swap, the nightly jobs.
