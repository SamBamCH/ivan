"""
Everything that talks to the outside world, one module per source. Each keeps the payload raw (ivan.core.raw) before
anything parses it.

    contests      DraftKings' lobby and contest detail: contests and payout tables (free)
    newsfeed      Rotowire's NBA news page: headlines per player, first_seen as the news clock
    boxscores     nba_api: this season's box scores into nba_stats, Rotowire's starting lineups
    pbp           nba_api play-by-play: stints (who was on the floor), fouls, substitutions
    live          NBA's public live feed: scoreboard and box scores while the games are on (the late swap)
    standings     DraftKings' standings exports: the real fields (scrape with the logged-in Chrome profile, derive)
    experience    nba_api's all-players list: each player's first season (the rookie marks)

The pool itself (DraftKings draftables, the odds API's props and game lines) is fetched by ingest.fetch_slate.
"""
