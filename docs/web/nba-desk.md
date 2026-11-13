# The NBA desk

`src/ivan/web/nba/` (one module per concern: `core.py` the app and the Desk object, `pages.py`, `team.py`, `pipeline.py`, `lateswap.py`, `exposures.py`, `exports.py`, `rerun.py`, `backtest.py`; the desk's model is `ivan.sports.nba.desk`, no web imports; FastAPI + Jinja + htmx, templates `src/ivan/web/templates/nba_desk.html`
and `_nba_*.html`). One page to manage a slate in the last minutes before lock, plus the pipeline
panel, the exposures and the late swap. Mounted at `/nba` inside the one Ivan UI (`scripts/serve.py`,
[README.md](README.md)); the login is the site's. Every link carries the mount prefix (`root`). The flow of
a day is [../nba/game-day.md](../nba/game-day.md); this page is what is on the screen.

## Header

The frame's bar (**Ivan › NBA › day › slate › view**, [README.md](README.md)) with the desk's own items after it.

- **Slate**: the desk shows the slate in the bar, always, and holds as many slates as you open (one directory each,
  [../nba/data.md](../nba/data.md)). A slate reads **lock time · games · DK's slate title** ("19:00 · 8 games · Early Only";
  the main slate of the night has no title: `Lobby.slate_label`, the title from the lobby's `ContestStartTimeSuffix`, kept
  in the held slate's `slate.json` as `title` once the desk loads it). `/nba/` lists the days, `/nba/<day>/` opens the day's **main slate** (the held
  one, else the one with the most games, else its only one; `/nba/<day>/?pick` lists the day's classic draft groups
  instead: from the DK lobby for today and tomorrow, from the archive for past days, every held slate marked **held**),
  `/nba/<day>/<draft group>/` one slate. In the bar the day dropdown goes to the day's main slate, the slate dropdown
  and the ‹ › arrows to any slate in time order, and the page shows what its address names:
  - a slate of **today or later** that this machine does not hold **loads when its page is opened** (`begin_switch`):
    the producer fetches it from DraftKings into its own directory and projects it; a consumer asks the laptop for it
    over ssh (`publish nba <dg>`: the laptop fetches the slate if it does not hold it, publishes its bundle, and the
    consumer copies it; `scripts/nba/feed.py pull --slate DG`). Nothing else is touched: flipping back to a held slate
    is instant. While it loads the bar keeps the slate you picked, the header reads "loading slate N…" (or "asking the
    laptop for slate N…") and the board shows the progress: a spinner, the seconds so far and the last lines the load
    logged (fetching DraftKings, Rotowire, storing, projecting each team, the news), refreshed every 2 s; when it is done
    the whole page reloads on the new slate.
    When that slate's jobs run the page says so and offers the load as a button (`POST /slate`; the picker shows no load
    button of its own since opening the slate is the load).
  - a **held slate** opens on the view its lock picks: **pre-lock** before its first lock, **post-lock** (the late
    swap) after it, **backtest** once its day has passed (`backtest.slate.mode_for`). A refresh of the slate running
    meanwhile is a job like any other: "refreshing the pool…" in the bar, a line on the job strip, and its log in the
    slate page's terminal (below); its pages keep reading, its edits answer "wait", and the page reloads when the
    refresh is done.
  - a slate of an **earlier day** not held opens its backtest (below), read only: nothing is loaded or deleted. Its
    **Restore into the desk** button loads it from its archive capture into its own directory when it is to be re-worked.
  Every action of a slate hangs off its address (`/nba/<day>/<dg>/refresh`, `.../team/<TEAM>/pin`, `.../pipeline/sim`,
  ...), so two slates' pages serve side by side and a job on one never refuses a refresh of another; `/nba/desk/` goes
  to today's main held slate.
- **View** dropdown: **pre-lock** (the slate view below, `view=games`), **post-lock** (the late swap,
  `view=lateswap`), **backtest**. With no view in the address the slate's lock picks one (above). The old
  `view=players` and `view=exposures` links open pre-lock: the players table, with the exposures in it, is a section of
  pre-lock and post-lock.
  A team's page (below) sits one level under the slate: **Ivan › NBA › day › slate › TEAM**.
- **locks in / locked**: the slate's countdown. Next to it, **refreshed HH:MM (desk | autopilot | script)**: when this
  slate's pool was last refreshed and by what (`slate.json`); on a consumer one stamp, **projections HH:MM**: when the slate's
  projection was last written, by the laptop's refresh as pulled or by a team re-figured here by an edit (the hover has the
  laptop's refresh time and the pull time); a consumer shows no other refresh times.
- **autopilot** box (producer only; `scripts/nba/autopilot.py on|off|status`): the capture tick refreshes,
  fetches props at the offsets and re-projects every 15 minutes while no desk job runs, so the server's Refresh finds a fresh pool; the
  last tick's time shows next to it ("held" when it stood down, "failed" in red). See
  [../nba/game-day.md](../nba/game-day.md#2-refresh-through-the-day). On a consumer the same spot shows
  the laptop's state instead: autopilot on/off, who published last (desk, autopilot, nightly) and when.
- **The job strip** (the frame's bar, every page of the sport: `GET /nba/jobstrip`, `Sport.job_strip`): one thin line on
  the sport's running job, any slate's (its slate as a link, the job's name, status, seconds so far, the estimate's
  seconds left and the last line it logged), read every 4 s while one runs and every 20 s otherwise, so a job started
  on one page stays in sight on the next; a job that ended stays on the strip for 90 s (green done, red failed). The
  slate page's terminal has the log. **Everything this server runs is a job of the one runner**
  (`ivan.web.jobs.JobRunner`, records and logs under `data/nba/jobs/`): the scripts as subprocesses, queued one at a
  time, and a slate's refresh, pull from the laptop, restore and first load as in-process jobs (`refresh`, `pull`,
  `restore`, `load`: `JobRunner.begin` / `log` / `end`, started by `pipeline._begin`, closed by `_finish`), which run
  at once in their own thread and have no cancel. So the strip and the terminal are the two places a job is read, and
  there is no other progress display on a held slate.
- **A− fit A+**: the page size. **fit** (the default) scales the whole page so the team grid with its columns fills
  the window's width; A− / A+ set a size by hand. Remembered in this browser only.

## The slate view (pre-lock), top to bottom

The slate page **reads**. Nothing on it edits a projection: a team's minutes, pins, starter calls and late news are on
its own page (below), and every edit there writes the team into the pool at once, so this page, the sim, the chooser
and the other desk all show the same numbers. You open a team only to map its minutes (the games strip's chips say
which: grey unmapped, yellow an open question, red changed since mapped) or when news breaks.

**The page has no banners.** What runs and what happened is read in two places: the job strip on the bar and the
**terminal** under the actions (`_nba_pipeline_status.html`): the running job's status and log on top, and under it
the desk's messages, newest last, each with its time (what a refresh found, the news it read, a **heads up** before a
run, a refusal such as "already refreshing the pool"; `Desk.flash.log`, the last 8 of 60 kept). The team page still
shows the answer to an edit next to the edit (the card's own line).

Under the actions, when it applies, a line reads **this pool is from HH:MM**: another held slate of the same day was refreshed
after this one and its pool carries DraftKings statuses this one does not (`Desk.sibling_changes`: the players of this slate
whose status differs on the most recently refreshed sibling, downgrades first, with a **Refresh DK** button). 2026-11-06:
the Turbo slate was last refreshed at 13:10 and kept Barnes questionable with 31 minutes while the main slate, refreshed at
18:40, had him OUT, and the Turbo lineups carried him at full exposure. The same players carry the flag on their team's
card ("OUT on the 18:00 slate's pool, refreshed 18:40 (here Q): refresh this slate", a bad alert when it rules him out).

1. **Games strip** (`_nba_games.html`, also at the top of every team page): one table above the actions, fixed column widths and row heights: each tip time heads four
   columns (away, home, spread as "DET -7.5" or "PK" for a pick'em, total), and row N holds every tip's Nth game
   (the strip scrolls sideways when the tips outgrow the row). Each team is a chip coloured by its
   map state (green mapped, yellow an open question, red changed since mapped, grey unmapped; a dot for news or for
   what the last refresh moved); a chip is a **link to the team's page**; a started game is faded. Its first slot
   is the **mapped** summary: teams mapped of the slate, the open questions, "good to go".
2. **Actions** (`_nba_pipeline.html`, re-rendered by every `/pipeline/<step>`), the steps of a slate in order:
   1 **Refresh DK** (free) / **Lines** (every game's spread and total, about 3 credits) / **Props** (about 12 credits a
   game plus the lines; confirm dialog); on a consumer a single **Refresh from laptop** pulls the laptop's pool in the
   background (the board shows "refreshing from the laptop…" and reloads when it is in). Every refresh
   reads the news too; the public field and ownership are not rebuilt by it.
   2 **Field + ownership** (`scripts/nba/field_ownership.py`, about 45 s). When it is done the page reloads on the new
   ownership (every job reloads the page when it is done: the status poll that finds a job finished answers `HX-Refresh`;
   a failed job keeps the panel and its log on the screen).
   Beside it **the chain**: `pool HH:MM › snapshot › field › lineups › sim › picks` (`_chain_info`), the
   pool's time being its newest write (any edit, refresh or `project.py`) and each stage the time of its last run on
   this desk (the job runner; the snapshot from the capture table): green when built after the pool's last write,
   **amber** when the pool moved since (run it again), **red** when the pool moved 10+ minutes ago and the stage never
   followed, grey when never run here. The late swap appears once it has run. The **snapshot** is not a button: Generate,
   Sim and the row take it themselves before they start (`ingest.capture_pool`: the capture rows the scorecard scores,
   source "ours", and the projections CSV, from the projection store, without projecting anything; values still typed
   into a grid are applied first), so the record is what the run was built on. From a terminal, `scripts/nba/snapshot.py`.
   3 **Generate**, 4 **Sim**, 5 **Choose**, and **3–5 in a row**: jobs (below).
   Under Generate's and Sim's knobs a **time estimate** re-reads as you type (`/pipeline/estimate?step=…`,
   `ivan.sports.nba.slate.job_time`, `scripts/nba/job_time.py estimate`): "≈ 4 min, done 18:52 · lock 19:00", from this machine's
   past runs of that step (seconds per unit of work: lineups per worker for generate, contests × sims × field cap for the
   sim and the late swap; a conservative default until a step has 4 runs), green when even a slow run leaves 5 minutes
   before lock, amber when only the expected run does, red when the run would end inside those 5 minutes or after lock.
   A red or amber run still starts; the terminal says **heads up** and why (nothing is refused). The late swap's deadline is
   the next tip. While a job runs its status line shows the estimate and the seconds left; the record it leaves carries
   the estimate and the work units, so the next fit learns from it (`scripts/nba/job_time.py fit` prints the model).
   **Knobs**: resting the pointer on Generate or Sim opens its knobs under the button (`_nba_knobs.html`; they stay open
   while the pointer is in them or a field has focus): generate's fresh pool, lineups, jobs, min salary, max salary per
   team, randomness, min fpts frac, player min fpts; sim's sims and field cap. A button sends only its own knobs
   (3–5 in a row sends both); the desk keeps every knob it was last sent, seeded from `config/nba.json`, and the jobs
   read them through `data/nba/jobs/desk_config.json`. Choose has no panel knobs: they are set per contest in the
   contests section, a blank one is the desk default (the slate book). **Slate book** beside Choose (its one knob under
   it: the bankroll in dollars, blank = read from our results, `scripts/bankroll.py`; a figure in
   `config/nba.json → choose.bankroll_dollars` or typed here pins it) chooses the whole slate as one book from the
   entries file alone, the per-contest knobs unused (`scripts/nba/choose.py --method slate
   [--bankroll-dollars B]`, [../nba/field-and-simulation.md](../nba/field-and-simulation.md#choose-contestchoosepy-contestexportpy)).
   6 **Late swap** opens the late swap view.
   Under the steps (the producer only; a consumer has the header's one stamp): when each input was last fetched (props, DraftKings, starters, projected; amber when old), when
   the news was read, the slate's lineups and sim rows; then **the terminal** (the job's status and log, the desk's
   messages), the exports and the latest picks.
3. **Contests** (folded once the entries are on file): **Upload entries CSV** (the DraftKings entries export, saved as
   `data/nba/input/entries.csv`; the page reloads), every contest of the file with its fee, field and our entries,
   checked against the loaded slate (`historical_contest_information`): green the loaded slate, red another slate's
   (the sim skips those contests and says so in its log, `contest.simulate.simulate_entries`), grey not stored yet
   (checked when the sim starts, one DraftKings lookup). Each contest's **choose** cell sets the chooser's knobs for
   that contest alone (method slate / greedy / portfolio, cash, roi, diversity, bankroll, the slate's bankroll in dollars; blank = the desk default from `config/nba.json`, shown greyed, the bankroll's blank = read from our results). They save
   as they are typed (`/contest/<id>/weights` = `scripts/nba/contest_weights.py set <id> knob=value`) into
   `data/nba/input/contest_weights.json`, and Choose passes that file as `--weights-json`, so a contest's own knobs
   win over the defaults.
4. **News** (below): the slate's questions and outs; a name opens the team's page, where the call is made.
5. **Players** (below): every player of the slate with his row and his exposure; loads after the board.

Hover popups (a projection's stat lines, a player's last ten box-score games with an `S` in the `st` column for a game he started and a red `DNP` row for a game he dressed and did not play) open after a short pause on the cell, beside it
rather than over the rows below, and close as soon as the pointer leaves the cell; the cell and its row are marked
while one is open.

## The team page (`/nba/<day>/<slate>/<TEAM>/`)

A held slate's team on a page of its own, reached from a chip in the games strip, a name in the news table or
the players table, or an old `/desk/?open=TEAM` link. The bar reads **Ivan › NBA › day › slate › TEAM**; **‹ slate** and
the bar's slate go back; **the other side** opens the opponent's page; the **games strip** heads the page as on the
slate page, the page's own chip outlined, so any other team is one click away. The page shell renders at once and the card
(`GET /team/<TEAM>`, `_nba_side.html`) loads into it, as the board does on the slate page. Tonight's game heads the
card: tip, countdown, the line ("DET -7.5 · O/U 228"; "no line" until Refresh lines or Refresh props fetches them).

**The market rates follow the books.** On each request (at most every 20 s) the desk looks whether the odds table moved
since this slate's market rates were built (a props fetch, for this slate or another of the day: the table is shared); when it
has, the rates are rebuilt past the status cutoffs and the teams whose means moved re-projected (`Desk.follow_odds` =
`ingest.ensure_market_rates`, about a second a team; the card says who). A publish does the same before it projects. Before 2026-11-07 the rates were a snapshot taken at the slate's own
refresh, and a slate projected the night on means a later fetch had replaced in the table.

**Every edit on the page writes the team into the slate's pool at once** (`Desk.reproject` = `ingest.project_team`: the
team's mapping, then its game drawn again in the World, both teams' `nba_player` rows and store records, about a second
and a half): a pin, a starter call, a painted row, a scenario, a late-news call. The opponent's card moves with it (its
rebounds, steals and blocks come off this team's misses and turnovers). Leaving the page loses nothing and there is nothing to publish afterwards. The page stays open while the
slate's jobs run and while the slate refreshes (edits wait for the refresh; a job never blocks a page). The card's alert score
(bad 10, warn 3, info 1) colours its frame:

| Level | Fires when |
|---|---|
| bad | a rotation player is OUT or scratched ("out, frees N minutes"); a fresher sibling slate of the day has him OUT ("OUT on the 18:00 slate's pool, refreshed 18:40 (here Q): refresh this slate"; another status difference is a warn); dnp-prone, deep bench projected like a rotation player, impossible band, distribution suspect |
| (flag) | **cut-prone: N% of nights under 8 min**: a 10+ minute row whose own cut rate is 30% or more; the World draws those nights; a named starter is exempt |
| warn | a new starter or a benched regular; the market disagrees with history ("market ..."); a questionable rotation player; new team / above recent maximum / market implies other minutes / no history for a 10+ minute player; a 24+ minute player left with no minutes in the rotation; no Rotowire lineup or an unconfirmed one inside 75 minutes of tip |
| info | minutes far from history for a 12+ minute player; the available roster's history minutes far from 241; rotation depth notes; unpriced rotation players |

The flags come from `ivan.sports.nba.backtest.validation.review_team`, the same ones `scripts/nba/validate.py review` prints.

### The news table

On the slate page under the contests (`_nba_news.html`, `Desk.news_table` =
`ivan.sports.nba.data_fetch.newsfeed.slate_news`; terminal: `scripts/nba/newsfeed.py slate`). Its heading carries the counts, the
**auto every 5 min** box (reads Rotowire's news every 5 minutes while the desk runs; every Refresh reads it too) and
when the news was last read. One table, one row per player whose night is news, in three groups:

1. **Questions**: DK status Q / GTD / D / P, not ruled out. Columns: player (bold when in the expected lineup;
   the name opens his team's page, where the call is made), team, opponent and tip, status, salary, usual minutes (the history state),
   the model's minutes now with your call ("no call" in amber until a pin or a starter call answers it), the
   latest headline on file.
2. **Out tonight, played last game**: DK OUT / O / IR / SUSP and on the floor in his team's last box score. These are
   the outs the rotation has not absorbed yet: the column shows the last game's date and his minutes in it.
3. **Already out** (folded; **show** opens them): OUT and out of the team's last game too, or never in a box score.

Each list is ordered earliest tip first, then the most usual minutes. The headline is the newest `nba_news`
row for the player **since his team's last game** (after 21:00 on the day it played, so post-game reports count and
what was said before that game does not: every game is a new decision; `newsfeed.since_last_game`; within 7 days at
most), with its source link (hover for the text), its age, the count of such items and **new** when it arrived after
the team was mapped (within 3 hours when the team is unmapped). "No news on file" means nothing since the last game:
Refresh reads the news. The team's last game is the newest box score on file, so until the nightly brings last night's
game the "last game" is the one before it (and that night's pre-game headlines still show).

### The card

Under the heading, the **World line**: the game as the simulation drew it on this grid (how many worlds, the factor on
this side's attempts against the line, the drawn total and margin against the books', the team's drawn fantasy points
against the plan's), the state of the optimal solve (when, how many worlds, stale when a team was projected since) and
the **Solve** button ([../nba/projections.md](../nba/projections.md), the optimal share).

Rows: starters first, then the players who are out, then the bench, minutes descending; the available players the plan
left at zero or you dropped sit as blank rows with `plays 39% · 8 min`, the World's fringe (the + beside a dropped name restores him). Per player:
status, Rotowire starter mark, history minutes (and the last games), the plan's minutes with the World's drawn band,
the seven stats as planned, **fpts** (the World's) and **pub** (the public projection the field plays to), DK FPPG,
salary, value, the optimal share, the rate source (market or history), the ownership, the market-implied minutes for
priced players, the flags.

The grid's columns, right of the 48 minute cells, are the two layers side by side. Layer 1, what you tell the
simulation: **min** (the row's minutes, a pin box; hover for this season's games with a DNP row in red, the drawn
minutes with their 10-90 band, his cut chance, and his own minutes sd), then one column per stat, **pts reb ast 3pm stl blk tov** (the plan for the game: the
row's minutes × his per-minute rate; green when the rate is the books', amber when the plan sits 15%+ from the books'
mean). Layer 2, what came out: **fpts** (the World's mean ± sd over the worlds the game was drawn in; hover for each
stat as planned and as drawn, the books' line and mean, the 10-50-90 of his night, the plan's total), **sim** (the
number the field, the sim and the lineups use: the World's unless you type a number here), **val**, **opt** (the share of
the solved worlds whose optimal lineup holds him, bold when the mean-projection optimal does; grey once a team was
projected after the solve, a dash before the first Solve), **pub** and its **val**, **own**. The foot row adds each
column up for the team (fpts: the World's, the plan's in the hover; opt: the worlds solved).

Every stat cell is a pin box and a hover. **Hover**: the books' lines for this game, one row per book and main line, the lowest line first
(when the book posted it, the line, the over and under prices) and the alternates condensed to one row per book (the
lines within 3 of the consensus line with the over price; alternates price overs only), headed by ours, the books' mean
and the minutes it was priced over. The lines are fetched the first time the cell is hovered (`GET
/team/<TEAM>/props/<player>/<stat>`, `_nba_props.html`; about a second the first time after the odds moved, then at once),
so the card itself and Refigure stay quick. A line the book posted before the player was last downgraded (to an out status, or out
of the starting five) is struck through and the header says how many are set aside and when he moved: those lines priced his
old role and the projection does not read them; an upgrade never sets lines aside and one back out of a downgrade counts them
again (`changes.mark_status_changes` at the refresh, `touch_status` on a hand starter call; [../nba/projections.md](../nba/projections.md),
Market rates). **Type a total** (9 rebounds) and that stat is his for the game: its per-minute rate
becomes total / minutes and the other stats keep the model's, so a pin says "the books are wrong on his boards" without
touching his points. Blank restores the model's. Turnovers are never priced: that column is history alone.

Inputs on the row:

- **minutes** pins (0 scratches); a **stat** total per column; **fpts** (sim) pin; **pub** pin; blank unpins. Saved to
  `minutes_overrides.csv` at once (columns `minutes, fppm, min_low, min_high, starter, fpts, pub, points, rebounds,
  assists, fg3m, steals, blocks, turnovers`); the team re-projects and is written to the pool (`/team/<TEAM>/pin`;
  **Refigure** and the pipeline buttons send every typed box the same way). The older **fppm** pin (every rate scaled
  together) is still read from the file and noted in the fpts hover; the stat boxes override it stat by stat.
- **starter** toggle (`/starter`): the hand call; a call that benches him sets his props posted before the click aside, one
  that starts him counts them again (the market rates are rebuilt before the team re-projects; `ingest.refresh_market_rates`).
- **Refigure** (`/refigure`): re-run the team's projection now with the current pins (the mapping, then the game).
- **Solve** (`/solve`, also the actions bar's Solve = `scripts/nba/optimal.py`): the optimal lineup in each of the
  World's worlds over the whole slate, a few seconds as a job; the **opt** column fills when it is done.
- **Clear** (`/clear`): drop the team's pins and grid.

**Since the last refresh** (top of the card): what the last Refresh moved for the team (`ivan.sports.nba.slate.changes`,
returned by `ingest.run_ingest` and printed by `ingest.py`; a consumer's Refresh diffs its pull): statuses, players
into or out of Rotowire's five and a confirmed lineup, players added to or dropped from the pool, projections that
moved 3+ points, new headlines. The team's chip in the games strip gets a blue dot with the list on hover.

**Notes** (`/team/<TEAM>/note` = `scripts/nba/notes.py set|clear|list`, `ivan.sports.nba.slate.notes`): the pencil beside a
name on tonight's grid asks for a note on that player (blank clears it); it turns amber when he has one, hovering shows
it, and the card lists the team's notes with a clear button. A note belongs to the slate it was written on (draft group
and day): it shows only while that slate is loaded, a past day restored from its archive included, and stays until
cleared. The slate's scorecard lists the notes that stood at lock beside how each player went
([../nba/validation-and-backtests.md](../nba/validation-and-backtests.md)). Each machine keeps its own (the table is not in the feed).

**Special considerations** (above the grids; `Desk.considerations` = `ivan.sports.nba.slate.considerations`, terminal
`scripts/nba/considerations.py [--team XXX]`): what is unusual about tonight's game, warnings first. Rest (back to back,
3 games in 4 nights, 3+ days off; the opponent's back to back), a regular starter (started most games, 20+ usual
minutes) OUT or questionable, tonight's five against the last game's (who is new in it, who dropped out; a rookie
starting), a 15+ minute player back after sitting out the last game(s), a player whose last game was for another
team, a spread of 10+ (blowout risk) or no line yet, the opponent's regular starters out. From the files on hand: the
box scores lag until the nightly brings last night's game, so a back to back can read as a day of rest more before it
runs.

The **rotation grid** (`_nba_rotation.html`): players × 48 minutes. Paint cells to put a player on the
floor; the grid autosaves every stroke (`/rotation`). Row lock holds a row exactly; **Re-fill** fills the
unlocked rows around the locks (`/rotation/propose` with `source=propose`, or a past game's grid with
`game_id`); **Hold all** locks everything as drawn; **Clear locks**; **Map** adds a player the cut left off
(`/rotation/add`), **drop** removes one, **restore** undoes. A player the mapping counted on ruled OUT (a DK status on a
refresh, a scratch call), or a five that comes in differently from the five the grid was drawn for (Rotowire's lineup with
the hand starter calls, a full five against a full five), voids the mapping on its own (`ivan.sports.nba.slate.mapping`,
`Desk.void_if_stale` before every re-projection, `ingest.void_stale_mappings` inside the refresh; `scripts/nba/mapping.py`):
the grid and the team's positive minutes pins go, the model's proposal for the new pool and five shows, the card says so and
reads unmapped; the old mapping is kept as a scenario assuming what it assumed ("Johnson in", "Hauser starting"), so it comes
back on its own if the news reverses. A five that comes in as mapped changes nothing. The refresh note and the team's dot
carry it ("mapping void: the lineup came in differently (-Hauser +Pritchard)"). Spot colours mark the slot each player
inherited (the lanes under the grid name each spot's holders; there is no separate key); fouls are shown on
historical views (a digit in a cell is that player's n-th foul in that minute). The row sums are the minutes projection; the foot row adds them up under `min` and, under `proj`, adds up the team's projected fantasy points beside its trailing average.
Rookies carry a purple **R** badge with their games played this season (`R4`) and a purple edge on the name, on
tonight's grid and on a past game's alike (`Desk.rookie_of`). A rookie is a player whose first NBA season is this one
in nba_api's all-players list (`ivan.sports.nba.data_fetch.experience`, fetched by the nightly or `scripts/nba/experience.py
fetch`), matched by name without suffixes, then by last name on his team; a name the list lacks is a rookie when no
NBA game is on file before this season.
The past-game dropdown lists the team's games with stints, this season then last: each option reads the date, the opponent, the final from the team's side (`W 112-105`) and how many of that night's five had not started the team's previous game (`2 new starters`; the oldest game in the list is compared with the game before it; blank when either five is unknown) (`projection.rotation.game_scores` / `starter_changes`). The default is the latest game.
**Shaded games** are the ones that are no picture of the usual rotation, so the games worth reading for tonight stand out as the unshaded ones (`Desk.game_context` = `projection.rotation.game_context`; terminal `scripts/nba/game_context.py [--team XXX] [--last N]`): **amber = not close** (six or more minutes of garbage time by the stints, or a final margin of 20 or more), **red = an injury** (a regular did not play: he had averaged 24+ minutes over at least three of the team's five games before it, `absent_regulars`; or a regular left early and played under half his season's minutes), **orange = both**. The option's line says which ("blowout +33", "no Cunningham, Duren", "Duren left early"), the picked game's reason is written beside the dropdown, and the same shading is on every player's recent-games hover (the minutes cell here, the hist cell of the players table; `_nba_recent.html`), each line with the team's final margin and the reason; a player's own absence does not shade his own line. Overtime is named and not shaded. The thresholds are constants in `projection/rotation.py` (`BLOWOUT_MARGIN`, `GARBAGE_SHADE_MIN`, `REGULAR_MIN`, `REGULAR_LOOKBACK`, `REGULAR_PLAYED`, `GONE_EARLY_SHARE`). On the game's grid those starters carry a blue **NEW** badge (`new_starters`). The panel has no header line: the dropdown names the game.
**vs our grid** (a box in the past game's bar) lays what we projected for that game over what was played, as the backtest
view does under each team: our grid for that day is the one drawn on the desk that day (`data/nba/days/<day>/edits/rotations/`),
else the archive's grid rows at that day's lock (`backtest.slate.grid_for_day`); blue cells were on the floor in both,
orange played but not in ours, hatched in ours but sat, a player we projected who did not play is a DNP row, the minutes
column reads played / ours, and the bar gives the share of the 240 player-minutes on the floor in both and where ours came
from (`Desk.overlay_ours`, `/rotation/propose` with `overlay=1`; the box stays on across the dropdown's games).
The past game's grid sits right under tonight's, always open, with no header of its own (the dropdown names the game).
It has two small rows above the minutes: each team's running score at the end of every
3 minutes, the team's own row green while it leads and red while it trails; the team, home/away and the final sit at the
right end of the rows, past the minutes columns,
and the score at each minute on hover (`Desk.score_strip` = `pbp.game_score`, the stored play-by-play, no fetch;
overtime folds into the last cell like the grid; no rows when the game's feed is not on disk).

The **late news form** (`/team/<TEAM>/news`): player, call (out / starts / bench / back), then
(sim + choose, late swap, the call only). One request: the call, the team re-projected and written to the pool, the
jobs queued; the page reloads in about a second. **Scenarios**: save the team's edits under a name, activate,
delete (`/scenario/*`); the one whose assumptions hold activates itself when the facts move.

Each card lists its open questions with the latest news on file (headline, source link, first seen).

## Players (view `players`)

The whole pool as one sortable table, read only, by ownership: team, position, salary, status, minutes,
proj, value, ownership, flags. A name or a team opens the team's page. The fpts filter applies.

## Backtest (view `backtest`, `/?view=backtest&dg=<draft group>`, `_nba_backtest.html`)

A past slate, read only (`ivan.sports.nba.backtest.slate`, terminal `scripts/nba/backtest_slate.py --draft-group N |
--date YYYY-MM-DD`; the panels' routes are `ivan.web.nba.backtest`). Built in about four seconds from what is on disk and
kept for two minutes so the panels open at once; the desk's working tables are not touched. Top to bottom:

- **The head**: the day, the slate, its lock, and **Restore into the desk** (`POST /slate`, source archive): loads the
  slate's archive capture (the pool as it stood at lock) into the slate's own directory and opens its desk, so a past slate
  can be worked again like a live one (a what-if, the chain re-run by hand). Nothing on the backtest page changes; it is
  only needed when a past slate is to be re-worked.
- **The readiness strip** (`_nba_bt_fetch.html`, `GET /backtest/fetch`; a held slate only): the state the calendar colours
  the day with (`ivan.sports.nba.backtest.readiness.slate_status`: **ready** when every team's box score and, where we
  played, every contest's final export are on file; **failed** when a fetch failed; **pending** otherwise) with the reason,
  and on the laptop two buttons for what is missing, each a job: **Fetch box scores** (`POST /backtest/fetch/boxscores` =
  `scripts/nba/nightly.py --max-games 40`: the season's box scores not yet stored, this day's among them, then the
  scorecards and the play-by-play; off when every team is on file) and **Fetch standings** (`POST /backtest/fetch/standings`
  = `scripts/nba/standings.py --slate N scrape`: the final export of every contest of ours on this slate and the largest
  public ones, through the logged-in Chrome; off when every contest is scored, and on a desk without the login). The strip
  polls while the job runs and the page reloads on the new data when it is done. The morning after a slate, with the box
  scores and the standings not yet in, this is where they are fetched from.
- **Tiles**: net and ROI, winnings on fees, entries cashed, the sim's claim (the chooser's ROI for the same entries,
  fee-weighted; red when it claimed far more than happened). The money (the tiles, the contests and their panels) shows
  **on the laptop only**: a consumer desk (the server) gets a line saying so and keeps the players. The same numbers,
  summed per day and over the season, are on the NBA picker and the Ivan home tile ([README.md](README.md#the-money));
  `ivan.sports.nba.backtest.results` makes them for all three.
- **Contests**: fee, field, our entries found, best rank, cashed, the cash line, won, net, ROI, the claim, the share of
  the field duplicated, where the points came from. Our entries are the ones under our DraftKings user name plus the Entry
  IDs in the upload files the desk wrote (`chosen_entries.classic.*.csv`, `lateswap.*.csv`). A contest is scored only from
  a **final** standings export (every game over, no LOCKED slot, `standings.export_is_final`): DraftKings' own points,
  ranks from them, ties sharing the payouts of the places they cover (the payout table). A contest whose newest export is
  the late swap's mid-slate fetch (every not-yet-tipped player reads LOCKED, ours included) or that has no export is listed
  **unscored**, greyed, with what it cost: it is not in the tiles, and the line under them says how many and the fees until
  Fetch standings brings the final export. Nothing is rescored from the box scores (2026-11-04: eleven such contests scored
  on one to four visible players read as a $28.86 loss on a slate that made $194.82; the rescore of final exports also
  differed from DraftKings' points by $46 across the slate).
  **A row opens the contest's panel** (`GET /backtest/contest/<id>`, `_nba_bt_contest.html`,
  `ivan.sports.nba.backtest.slate.contest_detail`): the contest-level data only, player performance being the table
  below. Its tiles (net and ROI, cashed and best rank, the chooser's claim with its time and method, the **payout curve**:
  what 1st, 2nd, 3rd, 5th, 10th ... and the last paid place got), the places paid, the cash line, the winning score, the
  duplication; **our entries** in it, best first (rank, points, payout, copies of the lineup in this field, the sim's ROI for
  the lineup at choose time: chooser runs from 2026-10-29 keep it per lineup, matched through the chooser's upload file, so
  a late-swapped entry shows its claim at choose time, and the lineup); **the field**: every player this field owned, by
  ownership (DraftKings' %drafted in this contest), our projected ownership at the first tip (red when the field had 10+
  points more than we projected), our share of our entries in this contest, **lev** = the field's ownership minus ours (red
  when the field was 10+ points heavier on him than we were, green the other way), points projected and scored.
- **Current code** (`_nba_rerun.html`, `ivan.web.nba.rerun`; terminal `scripts/nba/rerun_slate.py --slate N [-n 1000]
  [--tip HHMM]`): what the game simulation on this checkout says about the slate, drawn from its **tip record** (the
  pool, grid rows, props and lines as the desk had them at the first tip; [../nba/data.md](../nba/data.md)) with the
  history cut at that day, against what happened. **Rerun with the current code** queues the script as a job with the
  sims count typed in the box (100 to 5,000; a few seconds per 500 sims per game) and, with several tip records, the tip
  to rerun from; the block polls while it runs and shows the newest record on file (its commit and time). Then: the
  **layered scorecard** (possessions, points and margin, minutes, attempts, makes and outcomes, fantasy points: the
  drawn mean's MAE and bias, the share of actuals inside the drawn 10-90 band, the mean percentile of the actual, and the
  same error for what was published at the tip or for the line), the **games** (line → drawn → actual for the total and
  margin, possessions, garbage time and overtime shares, and the factor by which each team's attempts were scaled to meet
  the line: below 1 means the pool projected more points than the line allowed), and **per team every player**: minutes
  and fantasy points as published → drawn → actual, the chosen stat's drawn mean and actual, and the drawn
  **distribution** of that stat as bars (the 10-90 band shaded, the actual's bin dark with a red mark) with the
  percentile the actual fell at (red under 10 or over 90). The stat select (fantasy points, minutes, points, rebounds,
  assists, threes, steals, blocks, turnovers, attempts, free throws) re-renders the block. Records live under
  `data/nba/slates/<dg>/rerun/<stamp>-<commit>/`; a slate archived before 2026-10-23 has no tip record and the button says so.
- **Players**: a row of **team chips** (projected → scored fantasy points, the grid agreement), then **every player of
  the slate in one sortable table** (`_nba_bt_rows.html`, `flat_table`), the most points scored first: team, position,
  salary, the lineup at lock (S confirmed starter, s expected, b bench), minutes and fantasy points projected (our last
  capture before lock) → played, the miss, the value as played (points per $1k), our share of our entries, the field's
  ownership (DraftKings' %drafted in the largest contest), our projected ownership at the first tip, **lev** (the field's
  ownership minus ours), the desk's note that stood at lock. DNP rows grey, priced rows green. A chip or a team cell opens
  the **team's panel** (`GET /backtest/team/<TEAM>`, `_nba_bt_team.html`) above the table: the team's totals, **the game
  as played over our grid** (the grid drawn on the desk, `slates/<dg>/edits/rotations/<TEAM>.json`, else the grid rows
  the archive kept at the tip, `rotation_cells`, kept from 2026-10-29 by `archive_at_lock.py`; each minute marked on the
  floor in both, played but not in ours, or in ours and sat; the share of the 240 player-minutes on the floor in both) and
  the team's rows with the same columns. The same overlay is on a live slate's team page, per past game (the last-game
  panel's **vs our grid** box).

The plan this is phase 1 of: [../nba/notes/slate_views_plan_2026-10-29.md](../nba/notes/slate_views_plan_2026-10-29.md).

## Jobs (`/pipeline/<step>`)

The actions bar's Generate, Sim, Choose, 3–5 in a row and Field + ownership post to `/pipeline/<step>` with the Config
tile's knobs; the late swap view's form posts its own (`ls_*`). Generate, sim, late swap and the row take a snapshot first
(the only snapshot there is; `scripts/nba/snapshot.py` from a terminal).
Generate writes no ownership; the sim builds the public field once, at the largest size any contest in the run
needs, every contest slices it, and the largest contest's field becomes the ownership column.
**Field + ownership** (`/pipeline/field` = `scripts/nba/field_ownership.py`, about 45s) builds the field on
the stored public projection and writes ownership from it, on demand; nothing else does on a slate day. The desk warns
(a line in the terminal) when a team's mapping is incomplete. The scripts run one at a time in `ivan.web.jobs.JobRunner`
(a slate's refresh is a job of the same runner, in the server's own process); status at
`/pipeline/status` (polled every 2 s while a job runs; the poll that finds the job done reloads the whole page, whatever
the job was, so every section shows what it wrote; a pool-writing job, field, sim or solve, has the desk reload the pool
first; a failed job leaves its log on the screen), cancel at `/pipeline/cancel/<job id>`, logs
under `data/nba/jobs/`. The entries upload (`/entries`, the contests section) replaces the slate's
`data/nba/slates/<dg>/input/entries.csv`.

## Players (a section of pre-lock and post-lock: `GET /players`, `_nba_players.html`, `exposures._exposures_view`)

One table: every player's row and his exposure. Every player of the slate, held by a lineup or
not, with his row (team, opponent, position, salary, starter call, projected minutes with the books' implied minutes,
history with the recent-games hover, fppm, projection ± sd with the stat lines on hover, the public projection, value,
ownership, opt, the review's flags) and his **exposure** in one set of lineups: **in** (lineups holding him), **exp**
(share of the set), **lev** (exposure less ownership) and, for the chosen entries, **@pick** and **Δ** (our projection
when he was chosen and what moved since). The **exposure in** switch picks the set: the candidate pool, the sampled
field of every simmed contest or of one, the chosen entries of every contest or of one. A player the set does not hold
reads 0. The set is remembered on the slate's desk (until the server restarts); with none picked it is the chosen
entries when there are picks, else the pool. Rows: the set's players by exposure, then the rest by ownership; a header
click sorts; a name opens his team's page.

Filters: text, team, status, a minimum exposure, **in the set only**, and the page's **show ≥ N fpts** floor (the same
one the team cards use, remembered in the browser). A player the set holds is shown whatever his projection: a pick
now projected low or OUT (struck in red) is the one to see.

**histogram** (folded under the switch; its line carries the mean, p10, median and p90): the set's lineups by projected
total, ownership sum or salary used, with the bins knob, and over a sampled field the dotted shape of the real fields.

The section loads after the board and its switches reload only the section; a job that ends reloads the page, so it
shows what the job wrote. While the slate refreshes it says so and comes back with the page. `GET /exposures` is the
same section (its address before the two tables were one).

## Late swap (post-lock, view `lateswap`, `_nba_lateswap.html`)

Under the buttons, **Upload entries CSV** (the same `/entries` upload as the contests section, with the file's time and
entry count): after lock, the DraftKings entries export with the lineups as they stand, which is what the swap edits. The
file is the slate's own (`slates/<dg>/input/entries.csv`): a new slate starts with none. A contest in it that the stored
contest table puts on another draft group (an older slate's export) is listed apart in red and not swapped (the run logs
"contest of slate N, not this slate; skipped"); the view says to upload this slate's export.

Two buttons. **Fetch standings** (`/lateswap/fetch`, job `fetch_standings` = `scripts/nba/fetch_standings.py`;
producer only) downloads every entries contest's standings export through the logged-in Chrome profile
and keeps it under raw `dk/standings/<contest id>/`: press it a minute after a tip. **Run late swap**
(`/pipeline/lateswap` = `scripts/nba/lateswap.py`) with: candidates (distinct lineups per entry; an entry's
"cands" count is lower only when no more fit its open slots or the time limit `lateswap.unique_seconds` ran
out), field cap, sims; **standings**
(`ls_source`: "stored while current" = `--source auto`, the stored export while no game has tipped since it
was taken, a download otherwise; "DraftKings live"; "stored export"); **field** (`ls_field`: "one for all
contests" = `--field shared`, the biggest contest's export completed once and sampled per contest; "each
contest's own"); live scoring (`ls_live`). Each contest's heading shows its export's time, green while
current, amber "before the last tip" once a game has tipped since. No lineup is written twice in a contest: an entry
moved off a lineup another entry holds, without a gain of its own, carries "unique"; one that could not move (its
original repeats another's and all eight are locked) carries "duplicate" in red.

Below, the report of the newest `lateswap.<stamp>.json`: per contest and entry, the original and the swapped
lineup with the expected payout of each, and the exposure table before and after. The upload box
(`/lateswap/standings`; the only way on a consumer or limited desk, which runs `--source raw`) stores
standings CSVs exported by hand under the same raw path.

## Exports and downloads

- `/export/projections.csv`: the pool's projection (name, team, position, salary, minutes, proj, sd, pub,
  ownership, status), which is also what the desk shows and what Generate reads (every edit writes the pool), so the
  file, the page and the lineups agree; sent with `Cache-Control: no-store`. The filename's stamp is the pool's
  projection time. `scripts/nba/consumer_check.py` compares the file with the producer's pool.
- `/export/lineups.csv`: the slate's generated pool, one row per lineup, PG..UTIL as names, or DraftKings ids with `?ids=1`;
  when the slate has no lineups the file is one explanatory line, never another slate's lineups (each slate's live in its
  own database). The filename carries the slate id.
- `/download/<name>`: files from the slate's `output/` (the chooser's CSVs, the late swap CSV).

## Consumer mode (the server)

Set by `IVAN_FEED_SSH` in `.env` (the laptop's tailnet IP as ssh reaches it; the older `IVAN_FEED_URL` still works). The
desk never fetches DraftKings, the odds API or Rotowire. Every 30 minutes `ivan-nba-pull.timer` pulls every slate the
laptop published (`scripts/nba/feed.py pull`); a slate page's one data button, **Refresh from laptop** (`/refresh` =
`scripts/nba/feed.py pull --slate DG`, `feed.pull_ssh`), pulls that slate now, in the background: the laptop publishes
on request (each held slate's pool with every edit written into it, its pins and grids, its projection store, the shared
props, lines and headlines), the server copies it over ssh and applies every bundle that moved, each into its own slate
directory here. Picking a slate the laptop does not hold has the laptop fetch it first. Only a consumer (a machine with
`IVAN_FEED_SSH` or `IVAN_FEED_URL` set, `slates.is_consumer`) reads a published layer at all; the laptop ignores one it may
find under a day (`slates.published_pins`, `published_grid`). The laptop's pins and grids
arrive as the slate's `published/` layer and show as green placeholders; the consumer's own edits (amber) sit on top in
its own files, a blank box falls back to the published value, Clear locks drops only its own layer, and a pull
re-projects only the teams it edited. The card says "published" and counts "N of your own". No props, lines or news
buttons, no news auto box: the headlines come with the pool. When nothing changed on the laptop, Refresh says so. The
header reads "projections HH:MM" (the pool's newest projection time; the hover has the laptop's refresh and the pull time from the slate's record)
(`slate.json`) and the pull state (`data/feed-state/nba.json`, `feed.status`).

To try to break a consumer desk without touching his: `deploy/sandbox_server.sh up` runs a second consumer
on the server, fed by the same directory his Refresh fills, reached over an ssh tunnel at `http://127.0.0.1:8002/nba/`
(see [../deployment.md](../deployment.md#a-sandbox-consumer-on-the-server)). To test this desk from the
producer without a browser on the other machine:
`scripts/nba/consumer_check.py` (see [../deployment.md](../deployment.md)) logs in with a cookie minted on
the server, walks the pages and the CSV exports through the public hostname or the tailnet, presses
Refresh and compares what the desk shows with the bundle the laptop published.

## Trying a branch's desk before it is merged

A worktree has no venv, `.env` or `data/`. Run the site on another port with the main checkout's interpreter and
database, and `PYTHONPATH=src` so the worktree's code wins over the editable install ([README.md](README.md) has
the read-only form):

```bash
cd ~/ivan/.claude/worktrees/<branch>
IVAN_NBA_DB_URL=sqlite:////home/sam-pratt/ivan/data/nba/nba_history.sqlite PYTHONPATH=src \
  ~/ivan/venv/bin/python scripts/serve.py --sports nba --port 8002      # http://127.0.0.1:8002/nba/
```

Views that read the raw archive (a past game's score strip reads the stored play-by-play) need the main checkout's:
link it once with `mkdir -p data/nba && ln -s ~/ivan/data/nba/raw data/nba/raw` in the worktree. The backtest view
reads our upload files: link `data/nba/output` the same way. Raw files and outputs the branch's desk then writes land
in the main folders, as the live desk's would. The slate directories (`data/nba/slates`) stay the worktree's own, so the
backtest's overlay shows grids drawn on the live desk only after the merge.

It shows the live slate. Its pins, grids, jobs and autopilot switch live under the worktree's own `data/`, so the
live desk's edits are not in it and nothing typed here reaches the live desk. **Refresh, every team-page edit,
Generate and Sim write the shared database** (the pool, the projections, the lineups), so on a slate day look but
leave those to the live desk, or copy the sqlite file (5 GB, `sqlite3 … ".backup …"`) into the worktree's `data/nba/`
and run without `IVAN_NBA_DB_URL` for a free hand (the branch of 2026-11-03 was built that way). A Refresh also reads the news, which only adds headlines.

## Login and public access

The site's one password (`ivan.web.auth`, [README.md](README.md#login-and-public-access)): `IVAN_PASSWORD_HASH`
in `.env`, the login page at `/login`. On the server the whole site is also on the internet through the
Cloudflare tunnel (`deploy/setup_cloudflare_tunnel.sh`), which refuses to serve without a password. The tunnel
can cache a `.csv` URL at the edge: the CSV routes send `no-store` headers (fix of 2026-10-26); a bypass-cache
rule for the hostname is the belt and braces.
