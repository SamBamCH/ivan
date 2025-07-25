# scripts/mlb/overnight.py
#!/usr/bin/env python3
import os, sys



# ——— 2) Ensure repo root on PYTHONPATH —————————————————————
ROOT = os.path.abspath(os.path.join(__file__, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ——— 3) Orchestrate the 3 steps —————————————————————————
from ivan.sports.mlb.stat_fetcher        import fetch_and_store
from ivan.sports.mlb.historical_contests import get_contest_data
from ivan.sports.mlb.build_models        import build_models

def main():
    # 1) last night’s box scores
    fetch_and_store(days_back=1)

    # 2) parse & model contest data
    get_contest_data()

    # 3) rebuild all models
    build_models()

if __name__ == "__main__":
    main()