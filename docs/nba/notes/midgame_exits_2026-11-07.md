# Who inherits a mid-game exit (measured 2026-11-07)

Step 2 of [game_sim_plan_2026-11-07.md](game_sim_plan_2026-11-07.md). The question: when a player leaves for good
during the game (fouls out, gets hurt, is benched), who plays the minutes he would have played? The pre-game version
was settled on 2026-10-23 (position groups carry the structure, substitution pairs are diffuse); the sampler's own rule
for a scratched player's stints (`rotation.sample_worlds`: the best substitute off the floor by his curve at those
minutes) had never been checked against anything.

Code: `ivan.sports.nba.gamesim.fouls`, `scripts/nba/gamesim.py exits`.

## What was measured

Every player-game in the stints (763 games, 2025-10-21 .. 2026-11-06) where the player was not on the floor at the
buzzer, left with at least 4 minutes of regulation to play, not in garbage time and not at a blowout margin for the
minute (30 points before minute 24, 25 from 24, 18 from 30, 15 from 36, 12 from 40), and whose trailing curve (the last
10 games of his team, half-life 4) owed him at least 3 more minutes: **1,468 exits** (383 bigs, 241 guards, 844 wings),
18 of them foul-outs (most foul-outs come in the last four minutes, where there is nothing left to inherit). Each
exit owes 6.9 minutes on average. For every teammate: the minutes he played after the exit minus what his own curve
expected of him in those minutes (his *gain*).

Rules compared, each splitting the owed minutes over the teammates: **pairs** (the coach's recorded single-swap
habit for the exiting player, `nba_sub`, last 15 games), **group** (the pre-game rule: his position group first, bigs
exclusive, guards 4:1), **slot** (what `sample_worlds` does: the players off the floor at the exit by their curve in
the remaining minutes), **proportional** (usual minutes), **flow** (the matrix below, fitted on the first half of the
season, within a group by the room each player has left) and **none** (nobody gains).

## Findings

**Single exits are noise.** The gain of one teammate after one exit has a mean absolute size of 3.65 minutes and
*no rule predicts it better than predicting zero* (held-out half, 746 exits: proportional 3.68, none 3.70, group 3.71,
pairs 3.74, flow 3.77, slot 3.78). Picking the teammate who gained most: pairs 8%, flow 7%, group 6%, slot 2% (one
in 14 teammates by chance is 7%). The coach's reaction to an exit is not readable one game at a time.

**The aggregate flow is clear and matches the pre-game picture.** Share of the owed minutes that landed on each
heir group (mean over exits; the remainder went to players without a curve or nowhere, the rotation tightening):

| exiting player | to bigs | to guards | to wings | reaches a rotation player |
|---|---|---|---|---|
| big (383) | **40%** | 30% | 8% | 78% |
| guard (241) | -8% | **47%** | 36% | 74% |
| wing (844) | 6% | 24% | **46%** | 76% |

First half vs second half of the season: big to bigs 54% / 40%, guard to guards 27% / 47%, wing to wings 57% / 46%.
The direction holds; the sizes move by 10-20 points between halves, which is the noise level of this table.

**The backup takes about half of his group's share, not all of it.** For a big's exit, the big off the floor with
the most usual minutes gains 22% of the owed minutes himself, the next big 0%, the other bigs the rest of the 40%. For
a guard's exit the first two guards off the floor take 12% and 18%. For a wing's exit no single wing stands out.
So the user's hypothesis (the backup centre inherits most or all) is right in direction and wrong in size: the
minutes go to the group, led by the backup, with a fifth of them lost to the rotation.

**The current scratch rule is the worst.** `slot` (the best substitute off the floor by curve) is last on MAE in every
cut and its pick gains *less* than his curve expected (top share -13% to -17%): the player whose curve is highest in
those minutes is the one already expected to play them, so he has no room. Room, not curve height, is what decides who
can take minutes.

## What the sim does with it

Layer B's rule for a mid-game exit (foul-out, in-game injury) and for a pre-game scratch alike:

1. the owed minutes are split across position groups by the flow matrix (`fouls.DEFAULT_FLOW`, refit by `fit_flow`
   as the season grows), with the lost fifth really lost (the rotation plays short, which the balance pass turns into
   slightly longer stints for everyone);
2. inside a group by the room each player has left (48 minus the minute minus his own expected minutes), the
   recorded pair habit as a tie-breaker only;
3. `sample_worlds`' best-substitute-by-curve rule goes.

The team page shows the matrix (league) and the team's own exits this season with who gained (counts included), not
per-pair shares presented as precision they do not have.
