"""
The NFL screens (ivan.web.main mounts `app` at /nfl): core.py the app, the desk and the report pages; pipeline.py the steps as
the UI runs them (scripts/nfl/*); backtest_hub.py, backtest_view.py, models_view.py, validate_view.py, dupes_view.py,
plan_view.py the readers behind the backtest, models, validation, dupes and plan pages. Shared helpers stay in ivan.web
(frame, auth, jobs, charts).
"""
from __future__ import annotations

from .core import app, sport, KEY, LABEL, VIEWS      # noqa: F401
