from ivan.sports.mlb.stat_fetcher import fetch_and_store
from ivan.sports.mlb.historical_contests import fit_slate_kdes
from ivan.sports.mlb.build_models import build_models

# 1) fetch_boxscores for last night's games
fetch_and_store(days_back=1)
# 2) fit_historical_fields
fit_slate_kdes(contest_dir=r"I:\data\mlb\historical_contests")
# 3) build_models
build_models()
