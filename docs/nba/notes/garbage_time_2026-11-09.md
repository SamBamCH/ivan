# Garbage time in the World: off (2026-11-09)

**Finding.** `scripts/nba/gamesim.py draw-world --slate 156338 -n 1000` on a slate of 10 to 21 point spreads (MIA @ SAC
+11.5, BKN @ DAL −21, BOS @ MIL +11, UTA @ MIN −10.5, NOP @ LAC +1.5) pulled the closers in **57.5% of worlds**. Every
star drew short of his planned minutes (Giannis 33.0 → 28.8, Flagg 35.0 → 29.7, Tatum 33.0 → 30.3, Porter 30.0 → 26.4)
and the end of every bench drew long (Wagner 17.0 → 21.9, Beringer 13.0 → 17.7, Bamba 9.0 → 14.6, Conwell 13.0 → 19.6,
Poole 9.0 → 14.7), the slate's minutes unchanged (23.90 against 23.76 a player). Wagner's fantasy points went 20.1 → 28.0:
the minutes plus his share of the pie once Porter and Randle sat.

**Why it is wrong on the mean.** The grid's minutes are the books' implied minutes (priced players) and trailing
minutes (the rest): both already carry the blowout discount, since the books price a 21-point favourite's starters for
the nights they sit out the fourth quarter. The World's pull on top of that plan discounts the same nights twice. The
fit note of 2026-11-07 saw the same shape (the World +1.25 fantasy points high on the bench, 46% of worlds in garbage
time) and left it to the as-of rerun; the rerun of 156184 then read the minutes layer +1.1 high and the fantasy layer
+2.2 high, "the bench gains what garbage time takes from the closers".

**Decision.** Garbage time is luck, not a projection: the mechanism stays in `gamesim.rotation.draw` and
`gamesim.world.draw_game` but is **off by default** (`config/nba.json → gamesim.garbage: false`,
`world.garbage_setting`). A world where a lead runs away still has its closers on the floor to the buzzer; what it
loses is the correlation "bench up when the stars sit", which the first rerun could not tell from the bias. With it
off on 156338, Wagner's gap to the plan fell from +7.9 to +3.4 fantasy points, the rest of it rebounds (his 3.4 planned
boards drew 5.6: the World hands every opponent miss to somebody on the floor, and the Nets drew 50.6 team rebounds
against a plan of 34.9), which is the World's rebound mechanism and the next calibration item, not the pull.

**How to check it.** The as-of rerun's minutes layer by role (`scripts/nba/rerun_slate.py`,
[../validation-and-backtests.md](../validation-and-backtests.md)): the stars' bias with it off against on, over a
handful of slates. Turn it back on only if the stars' drawn minutes run high with it off.
