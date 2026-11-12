# The NFL screens

`src/ivan/web/nfl/` (`core.py` the app and pages, `pipeline.py` the steps as the UI sees them, `backtest_hub.py` and
the `*_view.py` readers; `ivan.web.jobs` runs one job at a time), mounted at `/nfl` inside the one Ivan UI
(`scripts/serve.py`, [README.md](README.md)); the login is the site's. Uploads go to `data/nfl/input/web/`,
outputs to `data/nfl/output/web/`, job logs to `data/nfl/jobs/`. Everything the UI runs is a script from
`scripts/nfl/`, so the job log is the script's output.

## Getting in (for a user on the server)

Through Tailscale: accept the invite, install Tailscale, connect, open **http://ivan.prattflix.com** and
sign in with the site's password. The Cloudflare tunnel (`nba.prattflix.com`) reaches the same site.

## The slate

`/nfl/` lists the days with slates, `/nfl/<day>/` opens the day's main slate (the working one, else its only one;
`?pick` lists the day's slates, as does a day with several and none working), `/nfl/<day>/<key>/` one slate with the view
dropdown: **desk**, **sim report**, **choose**, **exposures**, **backtest**.

- The **working slate** (the one in the database's working tables, marked *loaded*) opens on the desk. The flat addresses
  (`/nfl/desk`, `/nfl/contests`, ...) are the same pages for the working slate; `/nfl/pipeline` redirects to the desk.
- A **lobby slate** not yet ingested opens on the desk with Ingest's slate set to it; ingesting makes it the working slate
  (the old one's lineups and sims go with the clear buttons).
- A **past slate** opens on its **backtest**. First the **Results** card (`_nfl_results.html`;
  `ivan.sports.nfl.results.slate_results`; `scripts/nfl/results.py --slate <key>`): what our entries made on the slate
  as DraftKings ranked and paid them (net and ROI, won on fees, entries cashed), then one row per contest (fee, field,
  ours, best rank, best score, cashed, won, net, ROI); a contest row opens its panel under the table
  (`GET /nfl/results/contest/{id}`, `_nfl_results_contest.html`, `scripts/nfl/results.py --contest <id>`): our entries
  best first with rank, tie, points, payout, net, our copies of the lineup and the lineup. A contest the chooser entered
  that was never scraped is listed as unscored with its fees, out of the totals. The laptop only: a consumer (the
  server) shows no money. Then the slate's builds (every run that replayed it, with the commit,
  the parameters and the result; `scripts/nfl/slate_runs.py <key>`, [../nfl/backtest-and-validation.md](../nfl/backtest-and-validation.md))
  and what the archive holds for it. Its other views read what the pipeline left on disk for it, tied to the
  slate through DraftKings' contest list (`historical_contest_information`): **sim report** the sim reports of its
  contests (a later sim of the same contest replaces its report); **choose** and **exposures** the chooser runs that chose
  for its contests, the web runs and the terminal's (`output/chosen_entries.*`). `?view=players` and `?view=downloads` still
  answer (the projections artifact written at its ingest, the web runs' files). A notice at the top says which slate the
  page is about and what was found.

## The desk (`templates/nfl_desk.html`, `_nfl_game.html`, `_nfl_job.html`, `_nfl_pair.html`, `static/nfl_desk.css`)

The NBA desk's look: full width, one page, top to bottom. The bar carries the working slate's countdown ("locks in 2h 10m"
or "locked") and its mode.

1. **Games strip**: a tile per game, earliest first: kickoff, the two teams (the favourite underlined), the line
   ("GB -3.5 · 47 · 21.8/25.2" = favourite and spread, total, implied totals), the games behind each team's correlation
   matrix and "new QB" when tonight's QB has fewer than 8 of them, "drawn" once a sim drew the game. The tile is coloured
   by the game's correlation check (below): green clean, amber something to look at, red the last draw is off. A tile
   opens its game's card (`/desk/game?game=CAR-GB`). The first slot sums it up ("1/1 game clean").
2. **Actions**, the steps in order; each button submits its script as a job (`/run/<step>`, one at a time, queued behind a
   running one). Resting the pointer on **Ingest**, **Generate** or **Sim** opens its knobs under the button (they stay
   open while the pointer is in them or a field has focus):
   0 **Refresh from laptop** (the server only, `IVAN_FEED_SSH` set): the laptop's reference tables and model pickles
   over ssh (`scripts/nfl/feed.py pull`, `ivan.core.feed.pull_ssh`, [../deployment.md](../deployment.md)); this
   machine's slate, lineups and sims are untouched. A job like the others, so its log shows below.
   1 **Ingest**: the slate (showdown and classic draft groups from DraftKings), re-fetch players, update odds
   (`ingest.py -p --draft-group ID --update-odds`). Re-run near lock for fresh odds.
   2 **Contests + payouts** (`get_contest_ids.py`, `get_payouts.py`).
   3 **Generate**: lineups (10,000 default), parallel workers (`generate_lineups.py N J`).
   4 **Sim**: simulations, field cap, duplicates in classic (`sim.py -n 1000 -c 10000 --entries data/nfl/input/web/entries.csv`);
   replaces earlier results.
   5 **Choose**: every contest of the entries file with its own knobs from the contests section
   (`choose.py --weights-json data/nfl/input/web/weights.json`); a contest with no method of its own is chosen by the
   default, the slate book, on the bankroll read from our results (`scripts/bankroll.py`). **Slate book** beside it
   (resting the pointer opens its one knob, the bankroll in dollars: blank = read from our results, a figure holds for
   that run alone): the whole slate as one book from the entries file alone, the per-contest knobs unused
   (`choose.py --method slate [--bankroll-dollars B]`, [../nfl/chooser.md](../nfl/chooser.md#the-bankroll)).
   **clear sims** / **clear lineups + sims** (`clear_sims.py`, `clear_all.py`; refused while a job runs).
   Under them: when the working slate was last ingested (amber after 3 hours), when the entries file was uploaded, the
   pool's rows and teams, how many players carry ownership, the lineups, the sim rows and contests. Then the job: its
   status, cancel, the log (polls `/desk/job` every 2 s while it runs), and the latest picks (the DraftKings upload CSV,
   links to choose, exposures and every file).
3. **Contests** (folded once the entries are on file and all belong to the working slate): **Upload entries CSV** (the
   DraftKings entries export, saved as `data/nfl/input/web/entries.csv`, a stamped copy beside it), then every contest
   of the file: fee and field (`historical_contest_information`, after step 2), our entries, its slate against the working
   one (red: another slate's), the payout tiers on file, the sim (the report's check counts, linking to the sim report, and
   the sim rows), and the **choose** knobs for that contest alone: method (slate, greedy, portfolio), cash, roi, div,
   bankroll; blank = the default (slate), shown greyed. They save as they are typed (`/contest/<id>/weights` =
   `scripts/nfl/contest_weights.py set <id> knob=value`) into `weights.json`, which Choose passes; choose.py fills a
   contest's missing knobs from the file's `default`.
4. **The game card**: the correlation check of the open game (`ivan.sports.nfl.correlation_check`, terminal
   `scripts/nfl/correlations.py --game CAR-GB`; built in about 3 s and kept until the pool or a sim report changes).
   - Header: kickoff, line and implied totals, a pill per team with the games behind its matrix and tonight's QB's share
     of them, whether the cross matrix is the two teams' history or the league's, and the last draw (the newest sim report
     of a contest on the game: its draws, largest error against the target and the sampling noise).
   - The alerts: a team on the league matrix (under 10 games), a new QB, pairs whose own history is beyond 2 se of what
     the sim uses, DST pairs whose draw is more than 0.12 from history, draws more than 3 se off target.
   - **The pairs table**: per team with itself (QB × WR1-3, TE1, RB1, PTS; RB1 × PTS; WR1 × WR2; RB1 × RB2; RB1 and QB × DST;
     K and DST × PTS) and across the game (QB × QB, QB × WR1, WR1 × WR1, RB1 × RB1, QB and RB1 × DST, DST × DST, PTS × PTS, in
     both directions). Columns: the players in those slots tonight, the league's Spearman, the team's own history and its
     count, the same with tonight's QB only, **used** (own shrunk 300 games toward the league), all of them on one −1..+1
     axis (league grey tick, own blue dot ±1 se, used black diamond, drawn orange ring), the players' target Pearson,
     what the sim drew, the flags. A DST pair says "built": the DST is built from the opponent's drawn score plus takeaways,
     so its draw is judged against history, not a target.
   - A row opens **the pair's games** beside the table (`/desk/pair`, `correlations.py --pair TEAM A B [--opp]`): one point
     per team-game on file (oldest faint, tonight's QB's games blue), with the Spearman over all, with tonight's QB, and over
     the last 16. The card opens on its first pair.
   - **The players' matrix**: everyone in the game in depth order, target above the diagonal, drawn below (after a sim), ×100.
   - **Players in the game**: depth slot, salary, the mean and sd the sampler uses, ownership; players drawn independently
     (no slot, no opponent) are listed under it.
5. **Players**: the whole pool, one row per player (a showdown's captain row folded in as cpt salary and cpt ownership),
   sortable: team, opponent, position, depth slot, salary, projection, sd, p10/p50/p90, value per $1000, ownership,
   status, where the volume came from. **show ≥ N fpts** hides the rows below N here and in the game card (remembered in
   this browser). **CSV** = `/players.csv`.

## Pages

| Page | Reads | Shows |
|---|---|---|
| **Players** `/players` (+ `.csv`) | `nfl_player` | the ingested pool with projections and ownership (the desk's Players section is the same pool) |
| **Sim report** `/contests` | `contest_sim_report`, `diagnostics/*/report.json` | one report per simulated contest: Checks (payouts conserved, ties, rake, sampled scores vs projections, correlation), Field (duplication, top lineups, exposure vs projected ownership), Payouts (real vs scaled table), Simulation (cash line, winning score, best candidates), Correlation (target vs sampled per game), and with the conditioned sim the worlds' team points |
| **Choose** `/choose` | `chosen_entries.choose.<stamp>.json` | per contest: candidates on the cash % vs ROI plane with the weight direction, the fpts-trace map, the greedy walk, the picks; the book summary for portfolio; for the slate method "the slate as one book": the dollar tiles, the exchange pass, distinct and reused lineups, the per-contest table (entries, fee, mean net, mean ROI, cash, lineups also played elsewhere, honest ROI) and the lineups played in more than one contest; what to root for from the worlds |
| **Exposures** `/exposures` | `chosen_entries.lineups/exposures.*.csv` | the chosen lineups per contest (players, salary, projection, team split, stack labels, sim cash % and ROI) and player exposure by slot |
| **Downloads** `/downloads` | `output/web/` | the DraftKings upload file and the other CSVs of each run |
| **Jobs** `/jobs`, `/jobs/{id}` | `data/nfl/jobs/` | every job's log; cancel |
| **Validation** `/validate` | `output/validation/<stamp>/` | projected vs actual per component (projections, environment, usage coverage, ownership, field, simulator, chooser, our entries) with pass / warn / fail rows; a form submits `validate.py` as a job |
| **Models** `/models` | `output/models/` | the per-component scoreboard over code iterations: a fixed benchmark window replayed per commit with `replay.py --through STAGE`, repeat noise bands; a form runs it |
| **Dupes** `/dupes` | `dupe_analysis` | the dupe model's predicted copies against the copies real fields held, by predicted copies, captain, ownership decile, team split, stack, salary left; residuals from projected and from actual ownership |
| **Backtest** `/backtest`, `/backtest/{name}` | `output/backtests/<name>/` | the run hub: Data (what the window lacks and what each fetch costs), Contests (tick what to enter), Parameters (repeats, workers, lineups, sims, field cap, seed), Results (sim claim vs actual, after tax, repeats distribution, assumption checks, per slate and contest; **Pin** / **Unpin** = `scripts/nfl/retention.py --pin/--unpin NAME`, keeps the run's heavy files past the 30-day sweep). `/backtest/tax` the after-tax report, `/backtest/plan` the readiness per slate |
| **Slate** `/backtest/slate/{key}` | every run root, `contest_entry_result` | the slate's Results card (`scripts/nfl/results.py --slate KEY`, the laptop only), then its builds = `scripts/nfl/slate_runs.py KEY`: what the archive holds for it, then every run that replayed it (hub runs, Models runs, lab replays) with the commit, stage, parameters, models version, chooser, and the slate's fees / sim claim / actual / mean over repeats; linked from the hub's slate rows and landing page (`slate_runs.py --list`) |

## If something goes wrong

A red **failed** badge means the script stopped; the log says why (usually a step run before the one
above it, or an entries file for a slate not yet ingested). **Cancel** stops the running job; a re-run
fixes anything half done. Deploying code (`deploy/push.sh`) restarts the server process and drops its
running job.
