#!/usr/bin/env python
# scripts/mlb/create_db.py
import os
from ivan.database.session import init_db, Base
import ivan.sports.mlb.models

def main():
    # point at data/mlb/mlb_history.sqlite
    engine = init_db("mlb")
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    main()
