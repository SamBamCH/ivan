# The rotation drawn per team, per world (2026-11-19)

**Why.** Since the stint rotation went live (2026-11-16, `gamesim.rotation = stints`) the soft pool's cut nights were off
in it (`stint_cuts` false: they had put the bench low), so every pool player played in every world unless scratched. The
desk's entries leaned on players the field barely owned (10% of our roster slots on players under 2% owned, the field 5%),
and those were the rotation toss-ups: 23% of them scored under our drawn 10th percentile, 17% played under 10 minutes
(18 slates, 2026-10-28 to 11-18).

**What.** `gamesim.stints.sit_draw`, used by `Context.side` when `rotation_pool` is on: per team and per world, who the
coach does not use tonight. Each man sits the whole night with his own chance: `rotation_pool_scale` x `p_cut` x the DNP
share of his cut nights (`projection.cuts`: his own recent cut history by role); a fringe man (planned zero) sits
`1 - fringe_p` and, when he plays, is lifted to his fringe minutes. The five at the tip never sit. The draw is one per
team: the number sitting in a world stays within one of its expectation (systematic sampling over a fresh random order),
so the men at the end of the bench trade places from world to world instead of all playing a little in every one. A man
who plays is steered to his minutes as before, so his minutes are his minutes when he plays and the minutes of a man who
sits go to the others on the floor. No token-minute nights are added: the stint draw's spread already makes them.

**The gate.** 13 slates, 2026-11-10 to 11-18, each drawn 300 worlds from its first tip record by `backtest.rerun`'s as-of
inputs, 1,233 player-games with planned minutes; `now` = the stint rotation as live, `pool` = scale 1.0, `pool07` = 0.7.

| planned min | variant | minutes drawn / played | min MAE | sat out, drawn / real | under 8, drawn / real | fpts drawn / scored | fpts MAE | under own p10 | over own p90 |
|---|---|---|---|---|---|---|---|---|---|
| up to 12 | now | 9.0 / 8.0 | 6.66 | 18% / 36% | 47% / 56% | 8.7 / 6.8 | 6.49 | 3% | 7% |
| | pool07 | 8.1 / 8.0 | 6.60 | 33% / 36% | 52% / 56% | 7.8 / 6.8 | 6.45 | 1% | 7% |
| 12-20 | now | 15.9 / 15.8 | 6.40 | 3% / 10% | 13% / 17% | 15.8 / 13.5 | 7.97 | 19% | 8% |
| | pool | 15.6 / 15.8 | 6.49 | 13% / 10% | 19% / 17% | 15.6 / 13.5 | 8.06 | 8% | 6% |
| | pool07 | 15.8 / 15.8 | 6.41 | 10% / 10% | 17% / 17% | 15.8 / 13.5 | 8.02 | 11% | 6% |
| 20-28 | now | 23.2 / 23.8 | 5.08 | 0% / 1% | 1% / 3% | 22.7 / 22.1 | 8.11 | 14% | 11% |
| | pool07 | 23.7 / 23.8 | 5.05 | 2% / 1% | 3% / 3% | 23.1 / 22.1 | 8.14 | 14% | 10% |
| 28+ | now | 31.0 / 30.9 | 4.68 | 0% / 2% | 0% / 2% | 33.9 / 34.0 | 8.86 | 11% | 13% |
| | pool07 | 31.5 / 30.9 | 4.63 | 0% / 2% | 0% / 2% | 34.5 / 34.0 | 8.89 | 12% | 10% |
| all | now | 22.1 / 22.1 | 5.488 | 2.9% / 7.9% | 9% / 13% | 22.9 / 21.9 | 8.139 | 13.5% | 10.1% |
| | pool07 | 22.3 / 22.1 | 5.462 | 7.0% / 7.9% | 11% / 13% | 23.1 / 21.9 | 8.164 | 11.2% | 8.5% |

The old cut switch (`stint_cuts`) on the same slates: sat out 22% drawn against 10% real at 12-20 minutes, minutes 13.9
against 15.8. What `pool07` buys is the shape, not the mean: the 12-20 minute bench sits out as often as it really does
and scores under its own 10th percentile 11% of the time instead of 19%. What it costs: the players at 28+ draw 0.6
minutes more than they played (they take the sitters' minutes). The scale 0.7 was chosen on these 13 slates.

**Found, not fixed.** The bench is projected too high per minute whatever the rotation: fantasy points a minute drawn
against played, players with 8+ minutes: 1.01 against 0.86 at 12-20 planned minutes, 0.99 against 0.86 up to 12, 0.98
against 0.93 at 20-28, 1.10 against 1.09 at 28+. At 16 minutes that is 2.3 fantasy points on every bench player, and it
is the larger part of why the chooser likes them. The rates are the next thing to gate.

**Not built.** A box on the team page to type a man's chance of playing; the page already shows the chance drawn (the
minutes hover: plays in X% of the worlds).
