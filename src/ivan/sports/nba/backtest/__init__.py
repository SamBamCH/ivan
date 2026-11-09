"""
How the slate went and how right the model was.

    validation    the scorecard: the projection at lock against the box score, per component; the pre-lock review
    slate         one past slate read only: our results, the chooser's claim, the misses (the desk's backtest view)
    results       the money, from DraftKings' final standings exports and the payout tables
    readiness     which past days are ready to backtest (box scores, final exports) and why not
    field         the field backtest: real fields against the two-group field and the ownership models
    tips          the record at each game's tip (written by the archive timer), the source of truth for a backtest
    rerun         the game simulation re-run from a tip record with the history cut at that day, scored per layer
"""
