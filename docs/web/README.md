# The web UI

One site for every sport: pick the **sport**, then the **date**, then the **slate**, then what to look at.
`scripts/serve.py` runs `ivan.web.main:app` (FastAPI + Jinja + a vendored htmx, no build step); the
server's `ivan-web` unit runs the same on the Tailscale IP, port 80 ([../deployment.md](../deployment.md)).

```
/                        the sports: each with its loaded slate, its running job, how many days of slates it knows
/<sport>/                the sport's days (today and the days ahead first, then the past), the loaded slate's day open
/<sport>/<day>/          that day's slates: lock, label, where it came from (lobby, archive, working), what is on file
/<sport>/<day>/<key>/    the slate; ?view= one of the sport's views (the dropdown after the slate)
/<sport>/...             the sport's own actions, partials, exports and tools (below)
/login /logout           the one password (ivan.web.auth)
```

The bar on every page is the frame (`ivan.web.frame`, `templates/_frame.html`): **Ivan › sport › day ›
slate › view**, each a dropdown that goes where it says, ‹ › through the slates in time order, the sport's
tools on the right, sign out. The page always shows the slate in the bar: a **day** opens on its main slate
(`/<sport>/<day>/` redirects to the held slate of that day, else the one with the most games, else its only
one, `Sport.main_slate`; `?pick` shows the day's list instead, as does a day whose main slate cannot be told),
and a slate's page shows that slate, fetching it first when the sport has to (the NBA desk loads a lobby slate
it does not hold when its page is opened, [nba-desk.md](nba-desk.md)).

| Piece | Module | What |
|---|---|---|
| main | `ivan.web.main` | builds the app: `/static`, the login, the home page, one sub-application per sport mounted at `/<sport>`. A sport whose module fails to import (a missing dependency, no database) is listed with the error and the rest of the site runs. `IVAN_SPORTS` or `--sports nba,nfl` mounts a subset. |
| the sports | `ivan.web.sports` | `Sport`: key, label, the slate views, the tools, `days(now)` (the slate index the picker and the dropdowns read), `summary(now)` (the home tile). `SlateInfo` / `Day` are what the index returns. `REGISTRY` is what main mounted. |
| the frame | `ivan.web.frame` | the bar's context (`ctx`), the picker pages installed at every sport's root (`install_picker`), the shared Jinja environment |
| NBA | `ivan.web.nba` (package; model in `ivan.sports.nba.desk`) | the desk: pre-lock, players, post-lock (late swap), backtest, the team page. [nba-desk.md](nba-desk.md) |
| NFL | `ivan.web.nfl` (package: `core.py`, `pipeline.py`, `backtest_hub.py`, `*_view.py`) | the desk (the NBA desk's look: games strip, actions, contests, the open game's correlation check, the pool), sim report, choose, exposures, backtest (the slate's builds); tools: the backtest runs hub, models, validation, dupes, jobs. [nfl-app.md](nfl-app.md) |
| MMA | `ivan.web.mma` (package: `core.py`) | the slot only: the days and cards (the DraftKings lobby and the stored contests), a slate card (on the server with a **Refresh from laptop** button, `scripts/mma/feed.py pull`), a backtest view that says what settle has not built. The pipeline runs from `scripts/mma/` ([../mma/README.md](../mma/README.md)). |

Shared pieces: `ivan.web.jobs.JobRunner` (one job at a time per sport, `nice`d, logs under
`data/<sport>/jobs/`), `ivan.web.auth` (the password), `ivan.web.charts` (dependency-free SVG); one package per sport
(`ivan.web.nba`, `ivan.web.nfl`, `ivan.web.mma`, each with a `core.py` that builds its app and `sport()`),
`static/ivan.css` (the frame, the home and picker pages), `static/nfl_desk.css` (the NFL desk). Every sport keeps its own database, job runner
and working slate; flipping between sports is navigation, so an NFL slate and an NBA slate can be worked
in the same process. Switching slates within a sport is still a load (the desk keeps the replaced slate's
lineups, sims and edits). Every button is a script in `scripts/<sport>/`; the job log is the script's output.

## The slate index

Each sport's `days(now)` lists what it knows, by the day the slate locks:

- **NBA**: the DraftKings lobby's classic draft groups (today, tomorrow), every slate with an archive
  capture, every held slate (`Lobby.slate_days`), each with its game count. A lobby slate not held carries
  a **load** form (the picker's button; its page starts the same load by itself).
- **NFL**: the lobby's showdown and classic draft groups, the archive's ingest captures with the settle
  job's status (`slate_status`), the slates any backtest run replayed (`run_index`), and the working slate
  (the pool in `nfl_player`, keyed by the capture archived at its lock).
- **MMA**: the lobby's MMA slates and every draft group whose contests `scripts/mma/ingest.py` stored.

## Login and public access

One password for the whole site: `IVAN_PASSWORD_HASH` in `.env` (`scripts/password.py --set`; plain
`IVAN_PASSWORD` is the fallback; the pre-2026-10-30 `IVAN_DESK_*` names still work). With none set the
site is open, so bind it to localhost or the tailnet. Through the Cloudflare tunnel, or with `IVAN_PUBLIC=1`,
the site refuses to serve without a password; the tunnel forwards the whole site. The cookie is signed with
the secret in `data/.desk_secret`; guesses are throttled per address and globally.

## Trying a branch before it is merged

A worktree has no venv, `.env` or `data/`. Run it on another port with the main checkout's interpreter and
databases (read-only URLs keep it from writing), and `PYTHONPATH=src` so the worktree's code wins over the
editable install; link the data folders the pages read:

```bash
cd ~/ivan/.claude/worktrees/<branch>
mkdir -p data/nba data/nfl && for d in raw output input derived models; do ln -s ~/ivan/data/nba/$d data/nba/$d; ln -s ~/ivan/data/nfl/$d data/nfl/$d; done
IVAN_NBA_DB_URL='sqlite:///file:/home/sam-pratt/ivan/data/nba/nba_history.sqlite?mode=ro&uri=true' \
IVAN_NFL_DB_URL='sqlite:///file:/home/sam-pratt/ivan/data/nfl/nfl_history.sqlite?mode=ro&uri=true' \
PYTHONPATH=src ~/ivan/venv/bin/python scripts/serve.py --port 8002      # http://127.0.0.1:8002/
```

With writable URLs instead, Refresh, Publish, Generate and Sim write the shared databases: on a slate day
look and click but leave those to the live site. Tests: `venv/bin/python -m pytest tests/web`.
