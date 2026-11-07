"""
The NBA game simulation: the slate's games simulated minute by minute, n worlds at once, so every player's fantasy points
in a world come from the same game. Plan: docs/nba/notes/game_sim_plan_2026-11-07.md; what each layer's fit and check said:
docs/nba/notes/game_sim_fit_2026-11-07.md; the running system: docs/nba/field-and-simulation.md.

possessions   events, possession ends and five-man states from the stored play-by-play (the fitting data of every layer)
fouls         who inherits a mid-game exit (the measurement) and the flow matrix the heirs follow
script        layer A: pace, possessions per segment, the script's game-level parameters and their fit
rotation      layer B: each player's coverage of every minute under scratches, fouls, garbage time and overtime
usage         layer C: shares of the possessions on the floor, the team's pie, the true-minutes check
profiles      each player's shooting profile (attempts and percentages) from the box scores
world         layer D and the World: the minute loop, the outcomes on both sides, to_player_samples, the calibration
"""
