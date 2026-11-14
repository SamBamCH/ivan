# The strike and takedown weights, 2026-11-14

The question: after the 11-14 card's lines were ingested the projections leaned on output (strikes and
takedowns) more than real scores do. Measured on the ufcstats history on disk: 1,500 fights, 2024-01-13 to
2026-10-31, each fighter's stat line scored with DraftKings' rules (0.2 a strike, 0.2 more when significant,
5 a takedown, 5 a reversal, 10 a knockdown, 0.03 a second of control; win bonuses apart).

## What real fights pay

- Production (points without the win bonus) averages 32.6 a fighter: strikes 20.7, takedowns 5.4, control
  3.7, knockdowns 2.3, reversals 0.6. Winners 41.8, losers 23.4 (0.56 of a winner; the sim's loser share is 0.5).
- With the bonus: a decision winner 84, a finish winner 107, a decision loser 34, a finished loser 13;
  about 60 a fighter over everything. 51% of fights end inside the distance.
- Control time has no line and is not simulated. It goes with takedowns (correlation 0.72): 3.5 points of
  control per takedown landed, 1.2 points a fight for a fighter who lands none.

## The weights

`scoring.sig_strike` 0.8 and `scoring.takedown` 5 were fitted under the tree model, which shrinks a line by
the time fought; the hazard model holds the line as the mean landed, so the same weights paid 20% too much
(0.8 x strikes + 5 x takedowns on the real stat lines of the 2026 fights: 1.20 of real production).

Fitted on fighters' averages over their fights (393 fighters with four or more), the form a line takes,
production = a x strikes + b x takedowns: **a = 0.50, b = 9.1**. The miss by how often the fighter takes
people down (weights x career averages, minus real production):

| takedowns a fight | fighters | real | 0.50 / 9.1 | 0.8 / 5 | 0.63 / 5 |
|---|---|---|---|---|---|
| under 0.5 | 158 | 26.1 | -2.0 | +10.5 | +2.9 |
| 0.5 to 1.5 | 131 | 32.7 | -1.2 | +8.3 | +0.6 |
| 1.5 or more | 104 | 46.5 | +0.2 | +2.8 | -4.9 |

0.63 / 5 (one weight moved, the takedown left at DraftKings' 5) has the right total and takes five points
from every grappler to give three to every striker. 0.50 / 9.1 is level across the three groups. On single
fights (fit before 2026, tested on the 948 fighter rows of 2026) the error per fighter is 16.8 points at
0.8 / 5, 11.7 at 0.63 / 5 and 10.0 at 0.49 / 8.8.

Config from this date: `sig_strike` 0.5, `takedown` 9. On the 11-14 card the average projection goes from
67.6 to 62.1; a striker with a 70-strike line loses about 19 points, a fighter with a 3.5 takedown
line gains from the takedowns what the strikes lose.

## Not measured

- Whether the lines typed in match what fighters land: one earlier card's lines are on disk.
- The finish price is used with the book's margin in it, so finish chances and bonuses run a few percent
  high (the 11-14 fights average about 70% to finish; history is 51%).
- A fighter who controls without takedowns (clinch, top position after a knockdown) is still undercounted
  by about 1 to 2 points; a control line or a fighter-history prior would carry it.
- No contest backtest: the check is the fighter-level one above.
