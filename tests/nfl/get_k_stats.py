import nfl_data_py as nfl

data = nfl.import_pbp_data([2024])

data.to_csv('data/nfl/pbp_2024.csv', index=False)