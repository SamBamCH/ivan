# scripts/mlb/ingest.py
import sys
from ivan.sports.mlb.ingest import ingest_projections
from ivan.sports.mlb.init_players import init_players

if __name__ == "__main__":
    batch_ts = ingest_projections()
    init_players(batch_ts)