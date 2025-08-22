from ivan.database.session import init_db, get_session
from ivan.sports.nfl.correlated_samples import generate_correlated_samples
import os
import json
import pandas as pd

here   = os.path.dirname(__file__)
cfg_fp = os.path.abspath(os.path.join(here, "../../config/nfl.json"))
cfg    = json.load(open(cfg_fp, encoding="utf-8"))

engine = init_db('nfl')
session = get_session(engine)

df = pd.read_csv(
    cfg["paths"]["entries_csv"],
    engine="python",
    usecols=["Contest ID", "Contest Name"],
    dtype={"Contest ID": str, "Contest Name": str}
)

# Trim whitespace
df["Contest ID"]   = df["Contest ID"].str.strip()
df["Contest Name"] = df["Contest Name"].str.strip()

# Drop rows missing either column
df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()

# Convert Contest ID → int
df["Contest ID"] = df["Contest ID"].astype(int)

# 3) Only keep unique (Contest ID, Contest Name) pairs
df = df.sort_values("Contest ID")
unique_df = df.drop_duplicates(subset=["Contest Name"], keep="last")

samples = generate_correlated_samples(session)

