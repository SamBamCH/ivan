"""
The NBA projection, in two layers.

Layer 1, the mapping: each player's minutes (a team hands out 240, so an absence moves minutes around the roster: the
solve within position groups, then the rotation skeleton's 48-minute grid) and his per-minute rates (the books' means
over his minutes where he is priced, box-score history elsewhere, the desk's pins over both), his scratch probability.
Layer 2, the World: the game simulation (ivan.sports.nba.gamesim) drawn on the mapping, n worlds at once, read back as
every player's fantasy points, minutes and stats as a distribution. The pool's fpts_mean / fpts_std are the World's.

Modules
  market        props to market means to per-minute rates; the dispersion that turns a line into a mean
  book_minutes  the books' implied minutes
  history       trailing per-player state (minutes, rates) as of any date
  solve, groups the per-team minutes solve within position groups
  skeleton      the rotation grid: the coach's slots filled at the solve's minutes
  rotation      the grid's proposal from the stints, the drawn grids, the old minutes sampler (the game sim's check uses it)
  team_profile, team_minutes   the coach's pattern; the scratch probabilities and roles
  minutes, rates, lines        the old allocator (alloc = headroom), the rates with shrinkage and the usage bump, the game lines
  cuts          the soft pool: a player's own chance of being cut tonight, a zero-minute player's chance of playing (cuts.json)
  engine        project_team(): layer 1, one call from roster + availability + history to the mapping
  live          LiveSlate: the pool, starters, market rates and overrides in; write_plan / write_sim out
  worlds        layer 2: simulate_game, the World per game read back per player; the fpts pin
  store         the projection store: one record per team (frame = layer 1, sim = layer 2, the samples)
  record        the capture rows and the projections CSV: the record the scorecard scores
"""
