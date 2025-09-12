from pathlib import Path

from sqlalchemy import select

from ivan.sports.common.odds import get_odds, parse_odds 
from ivan.sports.nfl.models import NFLOdds, NFLPlayer
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.fpts_distribution_calculations import generate_and_fit_all


engine = init_db("nfl", None)
session = get_session(engine)


api_key = '58699e4fe47856c9182db84884772536' 
relevant_teams = set(session.scalars(select(NFLPlayer.team).distinct()))

odds = get_odds("americanfootball_nfl", api_key, list(relevant_teams))
df_odds = parse_odds(odds)

records = df_odds.to_dict(orient='records') 
session.bulk_insert_mappings(NFLOdds, records)
session.commit()

model_dir = 'data/nfl/models'
generate_and_fit_all(session, Path(model_dir), n_samples=10000, verbose=True, delete_players_without_odds=True, save_fpts_sample=True)#TODO: add a plot arg to plot the actual distribution of sampled values against the fitted model. 