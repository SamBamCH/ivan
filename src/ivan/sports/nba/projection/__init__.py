"""
Minutes-anchored NBA projections.

A player's fantasy line is modelled as minutes times per-minute rates. Minutes
are the volume signal (the NBA analogue of the salary/TD-odds anchor used for
unpriced NFL stats): a team has 240 minutes to hand out, so when a player is
ruled out the rest of the roster absorbs his minutes and part of his usage.
Rates come from box score history (and later from priced markets).

Modules
  history   trailing per-player state (minutes, rates) as of any date
  minutes   baseline minutes, team-level allocation, manual overrides
  rates     per-minute stat rates with shrinkage and the usage bump
  fpts      Monte Carlo fantasy points (mean, sd, double-double bonuses)
  engine    project_team(): one call from roster + availability to a table
"""
