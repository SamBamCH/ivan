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
slate › view**, each going where it says, ‹ › through the slates in time order, the sport's
tools on the right, sign out. The **day is a calendar**: the bar's day button opens one month at a time, a day with
slates is a link (its number is how many), the open day is filled, today dashed, a day held on this machine ringed, and a
**past day is coloured by its readiness for the backtest** (`Sport.day_status`, NBA: `ivan.sports.nba.backtest.readiness`,
`scripts/nba/readiness.py`): **green** when every input is on file (the box scores of every team of its slates and, where
we played, a final standings export of every contest of ours), **red** when a fetch failed and the day cannot be
backtested until it is fixed (a standings scrape recorded a failure or a needed login, `slates/<dg>/results/scrape.json`;
box scores still missing 36 hours after lock), **no colour** when nothing was attempted yet. The picker's day list carries
the same colour as a left edge. The page always shows the slate in the bar: a **day** opens on its main slate
(`/<sport>/<day>/` redirects to the held slate of that day, else the one with the most games, else its only
one, `Sport.main_slate`; `?pick` shows the day's list instead, as does a day whose main slate cannot be told),
and a slate's page shows that slate, fetching it first when the sport has to (the NBA desk loads a lobby slate
it does not hold when its page is opened, [nba-desk.md](nba-desk.md)).

A sport may put **actions** on its picker (`Sport.actions`, POST buttons under the heading) and show its sport-level
**job** there (`Sport.job`: name, status, the log's tail; the page refreshes every 5 s while it runs). A sport may also
give the bar a **job strip** (`Sport.job_strip`: an address under its root answering one line of HTML on its running
job, any slate's; the bar polls it every 4 s while a job runs, every 20 s otherwise, on every page of the sport), so a
job started on one page is in sight on the next; NBA answers at `/nba/jobstrip`. NBA (the laptop
only): **Fetch upcoming slates** = `scripts/nba/fetch_upcoming.py` as a job: every classic slate the lobby lists (today,
tomorrow) that the desk does not hold is loaded into its own directory, then every contest of those slates and each one's
payout table are stored (free; a few minutes the first time, one request per payout table).

| Piece | Module | What |
|---|---|---|
| main | `ivan.web.main` | builds the app: `/static`, the login, the home page, one sub-application per sport mounted at `/<sport>`. A sport whose module fails to import (a missing dependency, no database) is listed with the error and the rest of the site runs. `IVAN_SPORTS` or `--sports nba,nfl` mounts a subset. |
| the sports | `ivan.web.sports` | `Sport`: key, label, the slate views, the tools, `days(now)` (the slate index the picker and the calendar read), `summary(now)` (the home tile), `ledger` (the money), `day_status` (a past day's readiness), `actions` and `job` (the picker's buttons and running job), `job_strip` (the bar's one-line job read). `SlateInfo` / `Day` are what the index returns. `REGISTRY` is what main mounted. |
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
  capture, every held slate (`Lobby.slate_days`), each labelled lock time · game count · DK's slate title when it has
  one ("Early Only", "Turbo"; the main slate has none). A lobby slate not held carries
  a **load** form, which its page submits by itself when opened (the picker shows no load button: **open** is the load).
- **NFL**: the lobby's showdown and classic draft groups, the archive's ingest captures with the settle
  job's status (`slate_status`), the slates any backtest run replayed (`run_index`), and the working slate
  (the pool in `nfl_player`, keyed by the capture archived at its lock).
- **MMA**: the lobby's MMA slates and every draft group whose contests `scripts/mma/ingest.py` stored.

## The money

A sport may give the frame a `ledger(now)` (`Sport.ledger`): `{"overall": agg, "days": {iso day: agg}, "source"}` with
fees, won, net, ROI, entries, cashed, contests, how many slates are not scored, and the tax's two sums (`gross_win`: the
payout over the fee of every entry that beat its fee; `gross_loss`: what every other entry lost of its fee). Every sport
builds it with `ivan.core.pnl.roll_up` from one row per contest. It shows in three places, all on a machine that has
the results (a consumer desk, the server, leaves `Sport.ledger` None and shows no money anywhere):

- the **picker**: a strip under the sport's heading (every slate on file), each day's row and the chosen day's heading;
- the sport's **home tile**: net, ROI, fees, entries cashed, days;
- the **home page's strip** above the tiles (`ivan.web.main.money` = `ivan.core.pnl.combined`; terminal
  `scripts/pnl.py [--year] [--sports] [--cap] [--rate] [--basis]`): this calendar year over every sport, each sport's line, and
  the US federal tax on the total (`ivan.core.tax.compute_tax`, one return for every sport). The first row of the tax
  table is the filing in `config/tax.json` (filing status, standard deduction, rate, the 90% loss cap), and its
  after-tax figure is the headline; the rows under it change one thing each: itemizing under the 90% cap, itemizing at
  100%, the standard deduction (no losses deducted), net DFS, and the other count of winnings and losses. Losses
  count only when itemizing, and from 2026 only 90% of them: with the configured `other_itemized` of 0 the standard
  deduction is the larger, so the configured row taxes the winnings in full. An estimate, not tax advice.

What counts as winnings and losses is `wager_basis` in `config/tax.json` (`ivan.core.tax.wagering_totals`; `scripts/pnl.py --basis`):

- `entry` (the default): each entry is one wager. An entry that beats its fee adds its payout over the fee to the
  winnings; any other entry adds what it did not get back of its fee to the losses. Nothing is netted between entries.
- `fees`: every payout is winnings and every entry fee a loss, so the cap (and the standard deduction) also reaches
  the fees of the winning entries. The reading that never understates the tax.

Both give the same profit. Which one a DFS return should use is not settled by anything this project can cite (the IRS
treats a DFS entry fee as a wager, CCA 202042015, and its public guidance says winnings are reported in full and
losses separately, without saying whether a winning wager's own stake comes off): the page shows both, and the choice
belongs to a CPA. The NFL after-tax backtest counts per entry (its spec).

**DraftKings' entry history** (`ivan.core.dk_history`, `scripts/dk_history.py`): the account's own export of every
entry with its fee and winnings (draftkings.com → My Contests → History → Download Entry History), imported with
`scripts/dk_history.py import [PATH] [--since DAY | --all]` (the newest `draftkings-contest-entry-history*.csv` in
`~/Downloads`, entries since January 1 of this year, into `data/dk/entry_history.csv` by entry id; a later export
replaces). Where imported it is the money of every contest it lists, laid over each sport's ledger
(`dk_history.merge`): a contest both have takes the history's money; a contest only the history has (its standings
were never scraped) becomes a row of its slate where the sport knows the contest's draft group, else of a slate of
its day (`dk-<day>`); an unscored contest of a day before the export that the history does not list is dropped (we
did not enter it). A sport with no results code gets its ledger from the history alone, on the home tile when the
sport is mounted (MMA) and as a line of the home strip either way. `scripts/dk_history.py` with no argument prints
what is stored per year and sport. Import a fresh export whenever the standings of a slate were missed.

Where each sport's ledger comes from:

- **NBA** (`ivan.sports.nba.backtest.results`, `scripts/nba/results.py`): DraftKings' final standings exports and the
  payout tables, our entries by our DraftKings user name; cached per slate, complete slates never recomputed. The
  slate's backtest view shows the same slate's numbers ([nba-desk.md](nba-desk.md#backtest-view-backtest-dgdraft-group-_nba_backtesthtml)).
- **NFL** (`ivan.sports.nfl.results`, `scripts/nfl/results.py`): the finish DraftKings gave each of our entries in the
  scraped standings (`contest_entry_result`, written by the settle job's scrape) against the entry fees; read from the
  database on every page, no cache. The slate's backtest page leads with the same slate's numbers ([nfl-app.md](nfl-app.md)).
- **MMA** and any sport without a workspace: DraftKings' entry history alone, once imported.

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
