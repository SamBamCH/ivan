"""
What was fetched, parsed into the tables and read back.

    boxscores     an nba_api box score -> nba_stats rows (DraftKings points by scoring.dk_points), name fixes
    odds          the odds API's rows -> the odds table (props, replaced by game) and game_odds (the lines); odds_stamp
    pbp           the play-by-play -> stints, fouls, substitutions, the running score; the tables read back
                  (who comes in for whom, a team's trailing rotation map)
    standings     a DraftKings standings export -> entries and the side table; a final export derived into a real
                  contest field under data/nba/derived/contests/<id>/
"""
