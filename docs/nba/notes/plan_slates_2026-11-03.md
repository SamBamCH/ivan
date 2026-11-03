# Plan 2026-11-03: everything keyed by slate

Decision (the user, 2026-11-03): the desk shows the slate in the bar, always; flipping between slates must be
possible; the team screen must stay usable while things run; everything pre-lock (pool, projections, props, lines,
news, edits) is stored by slate for backtesting; the server pulls every published slate every 30 minutes and can
fetch one slate by hand, showing when the laptop last refreshed it and when the server last pulled it.

## Why the one pool cannot do it

Each machine held exactly one slate: `nba_player` (unique on name) was replaced whole on every load, the projection
store, grids, pins and minutes edits were keyed by team ("the current slate") and filed into a folder per draft group
when the slate moved, the Desk was a singleton around that pool, and its busy flag hid every page while the pool was
rewritten. About 60 places in 26 files read the player table as "the slate".

## The design

**A slate is a directory.** `data/nba/slates/<draft group>/`:

```
slate.json        draft_group, lock, day, label, games, created_at, refreshed_at, refreshed_by, published_at | pulled_at
slate.sqlite      the slate's own tables (below)
edits/            this machine's minutes_overrides.csv, rotations/<TEAM>.json, scenarios/<TEAM>/
published/        a consumer's base layer: the laptop's edits as pulled (minutes_overrides.csv, rotations/)
projection/       the projection store, <TEAM>.pkl (mirrored to a consumer)
input/            entries.csv, contest_weights.json (the slate's contests and knobs)
output/           lineups, chosen entries, exposures, late swap files, sim artifacts
```

**Two databases, one session.** The slate's engine opens `slate.sqlite` as *main* and attaches
`nba_history.sqlite` as `hist`. SQLite resolves an unqualified table name in main first, then the attached
databases, so every existing query works unchanged: `nba_player` is the slate's, `nba_stats` is the shared history.
The slate tables: `nba_player`, `starting_lineups`, `nba_market_rate`, `nba_player_note`, `lineup`, `lineup_pool`,
`lineup_simulation_results`, `contest_sim_report`, `historical_contest_information`, `payouts`,
`classic_contest_metadata`, `projection_snapshot`. Everything else stays in the history database: box scores and
stints, `odds` and `game_odds` (facts about games, shared by the day's slates), `nba_news` (Rotowire headlines by id),
the archive captures, the projection captures. `ensure_schema` creates history tables in the history file only and
`ensure_slate_schema` the slate tables in the slate file only; creating a history table in a slate file would shadow
the real one, so the two lists are explicit.

**The slate in context.** `ivan.sports.nba.slates` holds the layout (`paths(dg)`), the engine per slate
(`engine(dg)`, cached), the index of held slates (`held()`, from the directories and their `slate.json`) and the
*current slate*: a context variable a script sets once from `--slate` and the web sets per request from the address.
Every path that used to be a constant (`MINUTES_OVERRIDES_CSV`, `ROTATIONS_DIR`, `STORE_DIR`, `OUTPUT_DIR`, ...) is a
function of the current slate. Library code that is called with no slate in context fails loudly.

**Pre-lock history by slate.** The archive captures already record the pool, props and lines at every refresh
(`archived_slate` by slate key). Added: a projection capture (source `ours`, note `refresh`) at every refresh, so the
projection as it stood at each refresh is kept, not only at Snapshot; `nba_slate_news(draft_group_id, news_id,
seen_at)` links every headline a refresh brought to the slate it was working, so a slate's pre-lock news is a join.
Slate directories are never deleted by the code.

**Desks.** One `Desk` per slate in the web process, made on first use, each with its own lock and busy flag. A
refresh of slate A touches nothing of slate B. A refresh blocks only the writes of its own slate (edits answer "wait");
reads render from the frames the desk holds. Jobs carry their slate; a job on one slate does not refuse a refresh of
another. Opening a lobby slate the machine does not hold ingests it into a new directory; nothing is displaced.

**The autopilot** refreshes every held slate that has not locked (today's and tomorrow's), props at the offsets per
slate; `active_slate.json` is gone. The archive-at-lock timer records every held slate's tips.

**The feed.** The producer publishes a bundle per slate (`data/feed/nba/slates/<dg>/`: the feed tables of
`slate.sqlite` as a SQL dump, the edits as the published layer, the projection store, `slate.json`) beside the history
bundles (live: `odds`, `game_odds`, `nba_news`, `nba_slate_news`; reference as before). The consumer rsyncs the whole
feed and applies every slate whose stamp changed into its own slate directory (creating it when new, keeping its own
lineups, sims and edits), recording `pulled_at`. `publish nba <dg>` publishes that one slate, ingesting it first when
the laptop does not hold it. The server's `ivan-nba-pull.timer` pulls everything every 30 minutes; the slate page's
**Refresh from laptop** pulls that slate; the header shows "laptop refreshed HH:MM (desk|autopilot) · pulled HH:MM".

**Addresses.** Pages stay `/nba/<day>/<dg>/` and `/nba/<day>/<dg>/<TEAM>/`. Every action moves under the slate:
`/nba/<day>/<dg>/team/<TEAM>/pin`, `/nba/<day>/<dg>/refresh`, ... (the templates' `root` is the slate's root). The flat
`/nba/desk/` goes to today's main held slate.

**Migration.** `scripts/nba/migrate_slates.py` builds the directory of the held slate from the one pool and of every
slate that has lineups, contests or filed edits, moves their rows and files, and renames the moved tables in the
history database to `_migrated_*` so a stale reader fails loudly. Run once on the laptop and once on the server.

## Order of work

1. `slates.py`, the table split, the engine with the attached history, the per-slate paths.
2. ingest, projection store, overrides, rotations, scenarios, exports on the current slate; the news link table and
   the projection capture at refresh; the slate's `slate.json`.
3. the migration script, run on the sandbox copy.
4. every script takes `--slate` (default: the only open slate today).
5. autopilot, archive-at-lock, nightly over the held slates.
6. the feed per slate, the 30-minute pull timer, `pulled_at` / `refreshed_at` in the header.
7. the desks per slate, the routes under the slate, the picker over held slates, reads during a refresh.
8. tests, docs, deploy, migrate both machines.
