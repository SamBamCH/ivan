# The web UI

Two FastAPI apps, no build step, Jinja templates and a vendored htmx, in `src/ivan/web/`:

| App | Module | Serve | What it is |
|---|---|---|---|
| **NFL app** | `ivan.web.app` + `pipeline_nfl.py`, `jobs.py`, `*_view.py`, `backtest_hub.py` | `scripts/nfl/serve.py` (port 8000; the server's `ivan-web` unit on the Tailscale IP port 80) | the five NFL pre-lock steps as forms with a job log, and pages that read what the pipeline wrote: Contests, Choose, Exposures, Downloads, Jobs, Validation, Models, Dupes, Backtest (runs + plan + tax). [nfl-app.md](nfl-app.md) |
| **NBA desk** | `ivan.web.nba_desk` | `scripts/nba/serve.py` (port 8001) or mounted at `/nba` in the NFL app | the slate desk: team cards, rotation grids, pins, Publish, the pipeline panel, exposures, late swap, CSV exports, a consumer mode. [nba-desk.md](nba-desk.md) |

Shared pieces: `ivan.web.jobs.JobRunner` (one job at a time, `nice`d, logs under
`data/<sport>/jobs/`), `ivan.web.auth` (a password on a desk), `ivan.web.charts` (dependency-free SVG).
Both apps shell out to the sport's scripts for the heavy steps, so the job log is the script's output
and anything the UI does can be reproduced from a terminal.

Access: the NFL app has no login and is bound to a LAN or Tailscale address. The desk has a password
and is the only path the Cloudflare tunnel forwards (`deploy/cloudflared`, `app._public_only_the_desk`).
Deployment is in [../deployment.md](../deployment.md).

## Direction: one Ivan UI

The two apps grew separately: the NFL app is a pipeline runner with report pages, the desk is a
slate workspace. The intended end state is **one UI** where you pick the **sport**, then the **date**,
then the **slate**, and get the same shape of workspace for each sport: load the slate, work the
projections (the desk's cards and grids, or the NFL equivalents), run the pipeline, read the sim
reports, choose, download, late swap, and look back at past slates (validation, backtests) from the
same place. The desk's slate picker (day, then slate; lobby or archive) is the pattern to generalise;
the NFL app's report pages are the pieces each sport needs under a slate. The NBA plan's phase 3 (a
sport descriptor the web app mounts) is the hook that was never built, see
[../nba/notes/plan_2026-27.md](../nba/notes/plan_2026-27.md).

What that implies for new work: keep every page keyed by (sport, slate) rather than "the current
slate"; keep the per-sport logic in `ivan.sports.<sport>` with the web layer as thin forms over
scripts; write new report pages so they read run artifacts (`ivan.core.artifacts`) rather than live
tables, so a past slate renders the same as tonight's.
