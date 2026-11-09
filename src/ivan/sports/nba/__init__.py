"""
NBA: DraftKings classic slates, from the lobby to the upload file. One package per stage of the process, in the order
the process runs; the commands are scripts/nba/, the desk is ivan.web.nba (its model in desk/).

    models.py       the tables (SQLAlchemy), nothing else
    slates.py       a slate is a directory: data/nba/slates/<dg>/ with its own database; the current slate in context
    ingest.py       the slate pipeline: fetch, store, project, in one transaction (the desk's Refresh, scripts/nba/ingest.py)
    names.py        one spelling for every source's player names
    scoring.py      DraftKings' weights and bonuses
    paths.py        where data/nba/ is laid out;  utils.py  positions and the odds API's clock

    data_fetch/     everything that talks to the outside world: DraftKings contests and payouts, Rotowire news, nba_api
                    box scores and play-by-play, the live feed, DraftKings' standings exports, each player's first season
    data_clean/     what was fetched, parsed into the tables: box scores, the odds rows, the play-by-play, standings exports
    projection/     a player's projection: props to market means, history, the minutes solve, the rotation skeleton,
                    rates, fantasy points; live.py applies it to the slate, store.py keeps each team's frame
    gamesim/        the slate's games simulated minute by minute, n worlds at once; draw.py is the one draw the field,
                    the sim and the chooser read
    field/          the public's side: the public projection, the two-group field and its ownership, the fitted
                    ownership model; and our candidate pool (generate.py, the ILP in solver.py)
    contest/        each contest simulated against the field (simulate.py), the lineups chosen (choose.py) and
                    written as the DraftKings upload (export.py)
    lateswap/       after lock: the standings export, the in-game update, the re-optimised open slots
    backtest/       how the slate went and how right the model was: the scorecard (validation.py), a slate's results,
                    the money, the field backtest, the tip record, the as-of rerun of the game simulation
    slate/          what the desk keeps about a slate while working it: the autopilot switch, status changes, void
                    mappings, saved scenarios, notes, special considerations, job time estimates
    desk/           the Desk object the web pages render (no web imports; the scripts use it too)
"""
