# The NFL web app

`src/ivan/web/app.py` with `pipeline_nfl.py` (the five steps as the UI sees them), `jobs.py` (one job at a
time), `backtest_hub.py` and the `*_view.py` readers. Served by `scripts/nfl/serve.py` (port 8000 on the
laptop; the server's `ivan-web` unit). No login: bind it to a LAN or Tailscale address. Uploads go to
`data/nfl/input/web/`, outputs to `data/nfl/output/web/`, job logs to `data/nfl/jobs/`. Everything the UI
runs is a script from `scripts/nfl/`, so the job log is the script's output.

## Getting in (for a user on the server)

The site is reachable through Tailscale only. Accept the invite, install Tailscale, connect, open
**http://ivan.prattflix.com**. No Tailscale, no site.

## The front page: the five steps

Upload your DraftKings entries export first (**Your entries file**; the page lists the contests and
entry counts; re-upload when your entries change). Then **Run** each step top to bottom; the job box
shows the live log and queues behind a running job.

1. **Ingest the slate**: pick the slate (showdown and classic draft groups from DraftKings), Run.
   Players, salaries, player odds, fitted point distributions, projected ownership. Re-run near lock
   for fresh odds. (`ingest.py -p --draft-group ID --update-odds`)
2. **Fetch contests & payouts** (`get_contest_ids.py`, `get_payouts.py`).
3. **Generate the field**: lineups (10,000 default) and parallel workers (`generate_lineups.py N J`).
4. **Simulate your contests**: simulations and field cap; the slow step; replaces earlier results
   (`sim.py -n 1000 -c 10000 --entries data/nfl/input/web/entries.csv`).
5. **Choose lineups**: one row of weights per contest in the entries file (cash %, ROI, diversity,
   method greedy / portfolio / slate, bankroll, bankroll $), a "set every contest to" row, remembered in
   `input/web/weights.json` (`choose.py --weights-json ...`).

**Reset** buttons clear the slate's lineups or sim results. The page's slate list and status refresh
through htmx partials (`/partials/status|job|slates`).

## Pages

| Page | Reads | Shows |
|---|---|---|
| **Players** `/players` (+ `.csv`) | `nfl_player` | the ingested pool with projections and ownership |
| **Contests** `/contests` | `contest_sim_report`, `diagnostics/*/report.json` | one report per simulated contest: Checks (payouts conserved, ties, rake, sampled scores vs projections, correlation), Field (duplication, top lineups, exposure vs projected ownership), Payouts (real vs scaled table), Simulation (cash line, winning score, best candidates), Correlation (target vs sampled per game), and with the conditioned sim the worlds' team points |
| **Choose** `/choose` | `chosen_entries.choose.<stamp>.json` | per contest: candidates on the cash % vs ROI plane with the weight direction, the fpts-trace map, the greedy walk, the picks; the book summary for portfolio; "the slate as one book" tiles and per-contest table for the slate method; what to root for from the worlds |
| **Exposures** `/exposures` | `chosen_entries.lineups/exposures.*.csv` | the chosen lineups per contest (players, salary, projection, team split, stack labels, sim cash % and ROI) and player exposure by slot |
| **Downloads** `/downloads` | `output/web/` | the DraftKings upload file and the other CSVs of each run |
| **Jobs** `/jobs`, `/jobs/{id}` | `data/nfl/jobs/` | every job's log; cancel |
| **Validation** `/validate` | `output/validation/<stamp>/` | projected vs actual per component (projections, environment, usage coverage, ownership, field, simulator, chooser, our entries) with pass / warn / fail rows; a form submits `validate.py` as a job |
| **Models** `/models` | `output/models/` | the per-component scoreboard over code iterations: a fixed benchmark window replayed per commit with `replay.py --through STAGE`, repeat noise bands; a form runs it |
| **Dupes** `/dupes` | `dupe_analysis` | the dupe model's predicted copies against the copies real fields held, by predicted copies, captain, ownership decile, team split, stack, salary left; residuals from projected and from actual ownership |
| **Backtest** `/backtest`, `/backtest/{name}` | `output/backtests/<name>/` | the run hub: Data (what the window lacks and what each fetch costs), Contests (tick what to enter), Parameters (repeats, workers, lineups, sims, field cap, seed), Results (sim claim vs actual, after tax, repeats distribution, assumption checks, per slate and contest). `/backtest/tax` the after-tax report, `/backtest/plan` the readiness per slate |

## If something goes wrong

A red **failed** badge means the script stopped; the log says why (usually a step run before the one
above it, or an entries file for a slate not yet ingested). **Cancel** stops the running job; a re-run
fixes anything half done. Deploying code (`deploy/push.sh`) restarts the server process and drops its
running job.
