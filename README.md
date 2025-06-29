# IVAN – DFS Brain

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [MLB Database Initialization](#mlb-database-initialization)  
- [Fetching Historical MLB Data](#fetching-historical-mlb-data)  
- [Configuration](#configuration)  
- [Running an MLB Simulation](#running-an-mlb-simulation)   
- [Adding a New Sport](#adding-a-new-sport)  


This repo powers **IVAN**, your multi-sport DFS simulation engine.  
Below is the directory structure and a brief description of each component.

```text
ivan/
├── src/
│   └── ivan/                        # Everything under src/ivan is clearly part of the “Ivan” library. You’ll import it in your scripts like: "from ivan.core.player import BasePlayer"
│       ├── core/                    # sport-agnostic interfaces & utilities
│       │   ├── player.py            # BasePlayer & abstract methods
│       │   ├── lineup.py            # BaseLineup & abstract methods
│       │   ├── constraints.py       # BaseConstraints interface
│       │   └── utils.py             # config loader, logging, parsing gametime
│       │
│       ├── database/                # shared DB session factory
│       │   └── session.py           # init_db(sport) → SQLAlchemy Session
│       │
│       └── sports/                  # one sub-package per sport
│           └── mlb/                 # MLB implementation
│               ├── models.py        # ORM classes: MLBStat, MLBLineup, MLBSimResult
│               ├── fetcher.py       # get_raw_boxscores() via MLB-StatsAPI #TODO: schedule
│               └── ingest.py        # reads in all player data from csv files to DB
│
├── scripts/                         # thin, user-facing runners
│   └── mlb/
│       ├── create_db.py             # python scripts/mlb/create_db.py
│       ├── fetch_boxscores.py       # python scripts/mlb/fetch_boxscores.py --start …
│       └── ingest.py                # python scripts/mlb/ingest.py 
│
├── config/                          # per-sport JSON settings
│   └── mlb.json
│
├── data/                            # runtime data files (SQLite DBs, outputs)
│   └── mlb/
│       └── mlb_history.sqlite
│
├── pyproject.toml / setup.py        # build, package & dependency config
└── README.md                        # this file
```

## Prerequisites

- **Python** 3.8+  
- A POSIX‑style shell (macOS/Linux) or PowerShell/WSL (Windows)  
- **Git** and a virtual‑environment tool (`venv`, `pipenv`, `conda`, etc.)  
- [`statsapi`](https://pypi.org/project/statsapi/) for MLB box scores  
- [`SQLAlchemy`](https://www.sqlalchemy.org/) for ORM + SQLite  

## Installation of IVAN

Follow these three steps to get IVAN up and running:

1. **Clone the repository (don't have one yet)**  
    ```bash 
    git clone https://github.com/your-username/ivan-dfs-brain.git
    cd ivan-dfs-brain
    ```
2. **Create & activate a virual environment**
    ```bash
    python -m venv .venv

    #macOS / Linux
    source .venv/bin/activate

    #Windows PowerShell
    .\.venv\Scripts\Activate.ps1
    ```
3. **Install IVAN in editable mode (not sure what this does yet)**
    ```bash
    pip install -e .
    ```

## MLB Database Initialization

Before fetching any games or running simulations, you must initialize the per‑sport SQLite database and create the table schema:

1. **Using the installed CLI**  
   ```bash
   ivan-create-db
   ```
2. **Directly via the python script**
    ```bash
    python scripts/mlb/create_db.py
    ```




