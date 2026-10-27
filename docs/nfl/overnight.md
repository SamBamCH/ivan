# NFL: the nightly job, settle and the producer's chores

## The timer (`deploy/ivan-nightly.timer`, producer only)

```bash
deploy/install_nightly_timer.sh [--run-now]        # a user unit rendered with this checkout's path
systemctl --user list-timers ivan-nightly.timer
tail -f data/nfl/logs/nightly.log
```

06:30, 12:30 and 18:30 (+ up to 5 minutes), `Persistent=true`: a slot missed while the laptop slept runs
at wake. It runs `scripts/nfl/nightly.py --days 13`: nflverse schedules and box scores, the DST table,
then `settle()` over the last 13 days (DraftKings purges contest data after 14) with `IVAN_DK_USERNAME`
and the desktop's display passed through for the scrape; `--no-scrape` and `--headless` exist. It
**deletes nothing**. When a slate stays waiting, login_needed or no_contests it raises a desktop
notification (`--no-notify` to silence). An expired DraftKings login marks the slate `login_needed`:
log in once in the scraper's Chrome window (it waits up to ten minutes) and the next run continues.

## The manual reset (`scripts/nfl/overnight.py`)

The new-slate reset, not for the timer: the same fetches, then deletes `lineup` and
`lineup_simulation_results`, runs `VACUUM` when more than a quarter of the file is free pages (the file
once reached 18 GB around 600 MB of data), runs `settle.py --days 3` (`--no-scrape` unless
`IVAN_DK_USERNAME` and a display are set), and `sync_to_server.sh $IVAN_SYNC_HOST` when that is set.

## Producer-only work

Needs the full `requirements.txt` and, for the scrapes, the logged-in Chrome profile
`chrome-selenium/dk-profile`.

| Script | Purpose |
|---|---|
| `scripts/nfl/settle.py` | settle slates by hand ([backtest-and-validation.md](backtest-and-validation.md)) |
| `scripts/nfl/get_contest_data.py [--days-back 7] [--min-entries 5000] [--contest-ids ...] [--username]` | scrape finished contests (lineups, ownership, payouts) into the historical tables; raw under `data/nfl/raw/dk/standings/` |
| `scripts/nfl/fetch_actuals.py` | schedule, box scores and the DST table only |
| `scripts/nfl/build_models.py [--cutoff YYYY-MM-DD] [--only dst_table,dst,classic,dupe,sd,ownership_calibration,ownership_model] [--sd-since]` | every model the pipeline reads → `data/nfl/models/` (live) or `models/asof/<cutoff>/`; one function per model in `model_build.py` |
| `scripts/nfl/build_dupe_model.py backtest\|build`, `build_ownership_calibration.py`, `build_ownership_model.py backtest\|build` | the individual fits |
| `scripts/nfl/backfill_archive.py --since --until [--odds]` | captures for slates that predate the archive (odds-api history, billed) |
| `scripts/nfl/sync_to_server.sh [host]` | reference tables + models to the server ([../deployment.md](../deployment.md)) |

Memory of the house: no live sim or model change without the backtest's field checks passing;
exposures are a symptom, not a target; revert when unsure.
