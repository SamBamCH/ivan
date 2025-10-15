import nflreadpy as nfl

data = nfl.load_pbp(seasons=[2024]).to_pandas()

data.to_csv('data/nfl/pbp_2024.csv', index=False)