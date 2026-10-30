# Ivan documentation

Start with the [root README](../README.md) for the one-page install and run. Everything else is here.

## The shared system

| Page | What |
|---|---|
| [architecture.md](architecture.md) | repository layout, `ivan.core` (config, paths, raw, archive, artifacts, clock, seed, replay, validation, feed, chooser numerics), the database, config and `.env` |
| [data_framework.md](data_framework.md) | the design principle: store the raw inputs and outcomes of every slate, derive everything else; the backtest as a result plus a confidence |
| [deployment.md](deployment.md) | two machines (producer laptop, consumer server), the feed, the hardened `ivan-web` unit, Tailscale and the Cloudflare tunnel, `deploy/push.sh`, the user timers |
| [web/](web/README.md) | the one web UI (sport, then date, then slate): the frame, the slate index, the login; the [NFL screens](web/nfl-app.md) and the [NBA desk](web/nba-desk.md) |

## Per sport

Each sport folder has the same shape: a README with the flow and where to look, then one page per
area (setup, game day, projections, field and simulation, overnight jobs, validation, data), and a
`notes/` folder of dated design notes and plans.

| Sport | Status | Folder |
|---|---|---|
| NBA | live, 2026-27 season | [nba/](nba/README.md) |
| NFL | live, showdown and classic | [nfl/](nfl/README.md) |
| MMA | live, CSV-driven odds model | [mma/](mma/README.md) |
| MLB | dormant | [mlb/](mlb/README.md) |
| LoL | dormant | [lol/](lol/README.md) |

## Conventions for these docs

- A page describes the code as it is, with the date of any measurement it quotes. Plans and findings
  go under `notes/` with the date in the filename or the first line.
- Commands are written to run from the repository root with `venv/bin/python`; every path in the
  project is resolved against the repository root, so the current directory never matters.
- When code moves, the page that names it moves with it in the same change.
