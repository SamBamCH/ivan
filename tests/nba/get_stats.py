from datetime import datetime
from sportsipy.nba.boxscore import Boxscores

games = Boxscores(datetime(2018, 1, 1), datetime(2018, 1, 5))

print(games.games)