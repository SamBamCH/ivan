# ivan

A modular, multi-sport analytics and simulation framework built in Python.

## 🚀 Overview
`ivan` is designed to support robust sports analytics pipelines for multiple sports with shared core utilities, database integration, statistical modeling, and lineup sampling. Its modular structure makes it easy to add new sports, customize regressors and KDE fitting, and scale simulations.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SamBamCH/ivan.git
   cd ivan
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🗂️ Directory Structure

```
ivan/                            # top-level Python package
├── __init__.py
├── core/                        # Generic utilities and config
│   ├── __init__.py
│   ├── config.py                # Configuration loader and settings
│   └── utils.py                 # File I/O, logging, helpers
│
├── database/                    # Database and session management
│   ├── __init__.py
│   └── session.py               # init_db(), get_session()
│
├── sports/                      # Per-sport implementations
│   ├── __init__.py
│   ├── common/                  # Shared across all sports
│   │   ├── __init__.py
│   │   ├── kdes.py              # Gaussian KDE & histogram utilities
│   │   ├── io.py                # data in/out
│   │   └── regressors.py        # Bin regressors & fallback models
│   │
│   ├── mlb/                     # MLB-specific functionality
│   │   ├── __init__.py
│   │   ├── constraints.py       # lp model constraints
│   │   ├── models.py            # SQLAlchemy models for MLB
│   │   ├── sampling.py          # lineup sampling logic
│   │   ├── selection.py         # lineup selection & dedupe logic
│   │   ├── diagnostics.py       # plotting and histogram checks
│   │   └── build_models.py      # batch KDE & regressor training
│   │
│   └── nfl/                     # NFL-specific functionality (stub)
│       ├── __init__.py
│       ├── models.py
│       ├── sampling.py
│       ├── selection.py
│       └── diagnostics.py
│
├── scripts/                     # Command-line entrypoints
│   ├── mlb_analyze.py           # Analyze MLB pipelines
│   ├── mlb_simulate.py          # Run MLB lineup simulations
│   └── nfl_analyze.py           # Stub for NFL analysis
│
└── tests/                       # Pytest test suite
    ├── core/
    ├── database/
    └── sports/
        ├── mlb/
        └── nfl/
```

## 🔧 Usage

- **Database setup:**
  ```python
  from ivan.database.session import init_db
  init_db()
  ```

- **Fit models:**
  ```bash
  python -m scripts.mlb_analyze
  ```

- **Simulate lineups:**
  ```bash
  python -m scripts.mlb_simulate --contest-id 123456789
  ```

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/awesome-sport`
3. Commit your changes and push: `git push origin feature/awesome-sport`
4. Open a pull request.

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

