"""
Everything that talks to the outside world, one module per source. Each keeps the payload raw (ivan.core.raw) before
anything parses it.

    slate         the slate's inputs: the lobby's classic slates, DraftKings' draftables (the pool), Rotowire's lineups,
                  the odds API's props and game lines (fetch_slate -> SlateFetch)
    contests      DraftKings' lobby and contest detail: contests and payout tables (free); the stored contests looked up
    newsfeed      Rotowire's NBA news page: headlines per player, first_seen as the news clock
    boxscores     nba_api: this season's box scores into nba_stats, Rotowire's starting lineups
    pbp           nba_api play-by-play fetched per game and processed into the tables (the parsing is data_clean.pbp)
    live          NBA's public live feed: scoreboard and box scores while the games are on (the late swap)
    standings     DraftKings' standings exports scraped through the logged-in Chrome profile (the parsing is data_clean.standings)
    experience    nba_api's all-players list: each player's first season (the rookie marks)

"""
