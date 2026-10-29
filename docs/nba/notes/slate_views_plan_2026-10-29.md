# Slate views: pre-lock, late swap, backtest (plan, 2026-10-29)

## The problem

The desk holds one working slate in the database (the pool, lineups, sims). Looking at another slate meant loading
it: a lobby slate re-fetched (about 10 s), a past one restored from its archive, and either way `isolate_slate`
deleted every other slate's lineups, sims and contest reports. Flipping through slates was slow, destructive, and
looked broken: the page header was built from the old slate while the load ran, so the picker still showed the old
slate after the new one loaded (fixed 2026-10-29: the page reloads when a load finishes). A past slate had no view
of its own; restoring it into the pre-lock tools answered none of the backtest questions.

## The split

A slate's view follows from its date and its first lock against now (`slate_backtest.mode_for`):

| slate | view | what it is for |
|---|---|---|
| an earlier day | **backtest**, read only | how our entries did against the sim's claim, P&L, every player against our projection, actual rotations over our grid |
| today, after the first lock | **late swap** by default (the slate view one tab away for late news) | swap the entries of the games not started |
| today before the first lock, or later | **pre-lock** (the slate view) | map rotations, news, generate, sim, choose |

The picker reads the date: a past slate opens its backtest without touching the working tables; only today's and
tomorrow's slates load. Re-working a past slate stays possible through an explicit "Restore into the desk".

## Phase 1 (built 2026-10-29, branch nba-rookies)

- `mode_for` and the default view on opening the desk; the picker loads at once on a pick, ‹ › step through slates,
  past days open `/?view=backtest&dg=N`, a confirm before a load clears the loaded slate's lineups.
- `ivan.sports.nba.slate_backtest` + the backtest view + `scripts/nba/backtest_slate.py`: P&L per contest from our
  upload files' Entry IDs and the newest standings export, rescored from the box scores (exports taken during the
  games carry live points); the chooser's claim beside it; players by team (projection at lock vs box score,
  exposure, field ownership, notes at lock); our lineups with field copies.
- First numbers: 10-27 −$39.64 on $66.50 (claimed +214%), 10-28 −$69.23 on $125 (claimed +836%).

## Phase 2 (next)

1. **Actual rotations over our grid in the backtest**: per team, the game's stints grid with our projected minutes
   beside it (the desk's past-game grid already computes "vs our projection" from the captures; it needs to run for a
   team that is not on the loaded slate).
2. **The sim's view of each entry**: per entry, the simulated ROI / cash rate at choose time against the result, and a
   calibration of the claim across slates (the chooser stores per-contest aggregates; per-entry claims need the
   lineup ids to survive `isolate_slate`, so store them with the chosen file).
3. **Contest and lineup sections** from the derived standings (`derived/contests/<id>`): field duplication, the
   field's top-owned against our projected ownership, how the field's lineups scored against ours.
4. **Everything tied to the slate**: pins, grids and notes already are (by draft group); lineups and sims are deleted
   on a switch. Scope the lineup and sim queries by draft group and let retention clean up instead, so two of today's
   slates can be worked side by side without one erasing the other.
5. **News changes since the last refresh** highlighted on the team view.
