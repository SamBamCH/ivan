# The minutes distribution and the soft pool (2026-11-10)

After a bad night on 2026-11-09 the question was whether the World draws each player's minutes right. Measured on the
as-of reruns of 156338 (11-08, 5 games) and 156358 (11-09, 7 games), 1000 worlds, against the box scores, by the role
the plan gave each player, before any change:

| | starters / stars | bench (planned 12-22) | deep (planned ≤ 12) |
|---|---|---|---|
| drawn minutes bias, 11-09 | 0.0 / 0.0 | **+4.3** | +2.6 |
| actual below the drawn p10 / above the p90, 11-09 | 7% / 9% | **30% / 4%** | 33% / 17% |
| fantasy points bias, 11-09 | +0.8 / −1.7 | **+6.2** | +3.4 |
| drawn sd against the sd of the error | 5.7 vs 4.5-6.1 | 5.7 vs 5.1-7.3 | 5.0 vs 7.4-7.8 |
| worlds with zero minutes | 0% | 0% | 0% |
| real 2025-26: next game under 8 minutes (DNP) given healthy | 6% (5%) / 5% (4%) | **25% (15%)** | 63% (36%) |

Three findings.

1. **The band's width was right.** 78-84% of actuals inside the drawn 10-90 band (the target 80%), so the jitter of the
   substitution times is not the problem; modelling the clock and the timeouts to place subs exactly would refine what
   already holds.
2. **The pool was hard where reality is soft.** Pool players drew 242 team minutes against a real 223 on both slates: on
   11-09, 31 players outside every pool played 243 minutes (17 a team; Javon Small 23.7, GG Jackson 17.1, Minix 16.6),
   and those minutes inflated every pool player, the bench most. On the other side a healthy bench player is cut to
   under 8 minutes one night in four (a straight DNP one in seven) and the World never drew it: Huff 19 → 0, Post
   20 → 1.9, Christie 19 → 4.8, Ware 22 → 9. A player's own recent cut history predicts it: bench players cut 0 of their
   last 10 roster games are cut again 10% of the time, 1-2 → 19%, 3-4 → 39%, 5+ → 53%; starters 4%.
3. **Each player's spread is his own.** Pritchard plays 35+ minutes in 22% of his games (sd 7.4 on a 30.3 mean) and the
   World drew 8% (sd 5.7) on a 27-minute plan.

**What changed** (`projection/cuts.py`, `data/nba/models/cuts.json` fitted by `gamesim.py fit-cuts`, the draw in
`gamesim/rotation.py`): every row but a named starter's carries `p_cut`, his own chance that the rotation does
not hold for him (by role and own-cut bin; of those nights the measured share is a DNP, the rest a token stint of about
4 minutes); every available player the plan left at zero or you dropped carries `fringe_p` and `fringe_minutes`, the chance he plays
and the minutes he takes at the end of a half, fitted on the zero-minute players of every tip record (906 rows: 39%
play, 8.4 minutes when they do; a DNP-CD regular 32% and 6); each player's substitution spread is scaled so his drawn
sd matches the sd of his own last 20 games played (`history.min_sd20`). The pool's minutes then sum to what real pools
play; the user's mapping is unchanged, the simulation stops taking it as gospel.

**After**, the same reruns:

| | 11-09 before → after | 11-08 before → after |
|---|---|---|
| fantasy MAE (the published projection's) | 8.84 → **7.52** (8.53) | 7.72 → **6.72** (7.25) |
| fantasy bias | +1.9 → −0.1 | +3.2 → +1.0 |
| bench minutes bias | +4.3 → **0.0** | +2.0 → −2.3 |
| bench below p10 / above p90 | 30% / 4% → 3% / 8% | 15% / 5% → 3% / 10% |
| bench drawn sd vs error sd | 5.7 vs 7.3 → 7.8 vs 7.2 | 5.7 vs 5.4 → 7.0 vs 5.4 |
| stars minutes bias | 0.0 → −0.2 | +2.0 → +2.2 |
| minutes in band, all | 0.80 → 0.78 | 0.80 → 0.81 |

Two slates, one of them the kind of night the change is for. The bench's downside is now in the draw, the mean is on
the real level, and the fantasy error beats what was published both nights. Open: the deep bench (planned ≤ 12, five
or six players a slate) draws 2-4 minutes low (its cut rate 43-73% applied to small plans); the 11-08 bench drew two
minutes low on a night the pools held (8% under 8 against the usual 25%); the fringe at 39% × 8.4 is fitted on
fourteen slates and will move. The reruns' "minutes by role" rows are where to watch it.
