# The defence's rebound share on the stint rotation: the gate rerun (2026-11-20)

The defence's share ([rebounds_2026-11-13.md](rebounds_2026-11-13.md)) was measured on the plan rotation. Since then the
World draws rotations stint by stint and each team's rotation per world, so the check was run again on that code before
the change went in.

**Run.** `backtest.rerun` from the first tip record of every slate that has one, 31 slates from 2026-10-23 to
2026-11-19, 300 worlds, seed 0, nothing written. Before = every miss the offence leaves credited to a defender
(`p_dreb` held at 1.0); after = the share as built. Pooled over 3,502 player-games:

| metric | bias before | bias after | MAE before | MAE after | PIT before | PIT after |
|---|---|---|---|---|---|---|
| rebounds | +0.331 | -0.097 | 1.945 | 1.859 | 0.474 | 0.518 |
| fantasy points | +0.044 | -0.522 | 7.451 | 7.420 | 0.517 | 0.533 |
| points | -0.495 | -0.496 | 4.330 | 4.328 | 0.536 | 0.535 |
| minutes | -1.309 | -1.310 | 4.998 | 4.996 | 0.566 | 0.566 |
| assists | +0.013 | +0.015 | 1.295 | 1.295 | 0.499 | 0.499 |

Steals, blocks, turnovers, attempts, the minutes by role and the game checks (71 games: possessions, total) did not
move. The rebound bias is closer to zero on 19 of the 31 slates; on the other 12 it went from a small plus to a
larger minus (the widest: +0.29 to -0.33).

**Reading.** Rebounds are better on every measure. Fantasy points are a hair better on MAE and worse on bias: the
extra rebounds had been covering the points and minutes the draw leaves short (minutes -1.3 and points -0.5 a
player-game on both sides of the change), so with the rebounds right the fantasy points run 0.5 low. The shortfall is
in the minutes and the scoring, not in this change.
