"""
The NBA desk: the slate page, the team page, the pipeline, the late swap, the exports (ivan.web.main mounts `app` at /nba).
One module per concern; the routes register on the shared app when the package is imported:

    core.py       the FastAPI app, one Desk per slate bound per request (desk, LOCK), the job runner, consumer / limits, render helpers, sport()
    pages.py      the shell, the slate page (read only), the players view, the team page, the board
    team.py       the team page's edit routes (pins, starters, the grid, scenarios, notes, late news)
    pipeline.py   Refresh, Snapshot, the chain, generate / sim / choose / field jobs, news + autopilot switches, entries, slate switch
    lateswap.py   the post-lock view
    exposures.py  the exposures views
    exports.py    CSV exports and downloads
    knobs.py      the pipeline's knobs; edits.py the typed edits; outputs.py what the pipeline wrote
"""
from __future__ import annotations

from .core import app, desk, sport, runner, LOCK, CONSUMER, LIMITED, LIMITS, KEY, LABEL, VIEWS, LOADING, DESKS, desk_for, bound, slate_url, _loaded_slate, _render, _here      # noqa: F401
from .knobs import pipeline_form, PIPELINE_DEFAULTS, clamp_form, write_desk_config                                            # noqa: F401
from . import edits, outputs, exposures, pipeline, lateswap, pages, team, exports                                              # noqa: F401  (the routes register here)
from .core import slate as _slate_router

app.include_router(_slate_router)                                   # every slate's actions, under /<day>/<draft group>/, once every module has registered its routes
