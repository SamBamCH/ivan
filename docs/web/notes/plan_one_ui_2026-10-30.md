# One Ivan UI: sport, then date, then slate (plan, 2026-10-30)

Written 2026-10-30 at the start of the `one-ui` branch. The page that describes the result is
[../README.md](../README.md); this note is the plan and the decisions, and is not updated.

## Why

Two web apps grew separately: the NFL app (a pipeline runner with report pages, `scripts/nfl/serve.py`,
no login) and the NBA desk (a slate workspace, `scripts/nba/serve.py`, a password, mounted at `/nba`
inside the NFL app on the server). The server already ran one process serving both, so "one UI" was
never about the serve script: it was about the frame around the screens (where am I, which sport,
which slate) and about every page being keyed by (sport, slate) rather than "the current slate".

## Decisions (Sam, 2026-10-30)

- Flow: pick the **sport**, then the **date**, then the **slate**; after that the sports differ in
  which screens they have (MMA has no late swap), but every sport has a **backtest** view in the
  dropdown.
- **One password** for the whole site, not one per sport (`IVAN_PASSWORD_HASH`; the old desk names
  still work).
- The Cloudflare tunnel forwards the **whole site** behind that password (the server's tunnel config
  is Sam's to widen; the app no longer answers 404 to non-`/nba` paths from the tunnel).
- **MMA**: a slot only for now (dates, the picker, a slate card, a backtest placeholder); its
  pipeline stays in the terminal.
- An **NFL past slate** opens on its backtest (the slate's builds page from the retention branch) and
  its report pages read the artifacts tied to the slate's contests, while the working tables stay the
  one live slate, as the NBA desk does.
- One loaded slate per sport, as before: flipping between sports is navigation, flipping between
  slates within a sport is still a load (the desk keeps the replaced slate's lineups, sims and edits).

## Shape

- `scripts/serve.py` -> `ivan.web.main:app`. Main holds the login, `/static`, the home page and mounts
  one sub-application per sport at `/<sport>`; a sport whose module fails to import is listed with
  the error and the rest of the site runs.
- `ivan.web.sports.Sport`: the descriptor each sport module returns from `sport()`: key, label, the
  slate views, the tools, `days(now)` (its slate index), `summary(now)` (the home tile), the app.
- `ivan.web.frame`: the bar every page shares (Ivan › sport › day › slate › view as dropdowns, the
  arrows through the slates, the tools, sign out) and the picker page installed at every sport's root.
- URLs: `/`, `/<sport>/`, `/<sport>/<day>/`, `/<sport>/<day>/<key>/?view=`. The sports' action routes,
  partials and exports keep their flat paths under the sport root; every action redirects to the
  loaded slate's page (`/nba/desk/` and `/nfl/pipeline` are the flat aliases).
- The NBA desk's picker became the frame's (its `slate_days` rows feed it); the NFL app gained a slate
  index (lobby + archive captures + settle status + the runs' slates + the working tables).

## Order of work

1. Shell: main, frame, home, picker, one password, the unit and the tunnel template, docs.
2. Sport descriptors: the desk and the NFL app as mounted routers with slate pages.
3. NFL past slates through artifacts (projections file, sim reports and chooser runs by contest id).
4. MMA slot.

Worked in the `one-ui` worktree on top of `nfl-retention` (its slate builds page is the NFL slate
backtest); `master` stays the live checkout.
