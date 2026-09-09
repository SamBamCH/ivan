# Using the Ivan web UI

## Getting in

The site isn't on the public internet; it's reachable through Tailscale, a private network
between your devices and the server. One-time setup:

1. Accept the Tailscale invite Sam sends you (email) and install the Tailscale app on the
   computer you'll use (tailscale.com/download). Sign in with the same account you accepted with.
2. With Tailscale connected, open **http://ivan.prattflix.com** in your browser. Bookmark it.

If the page won't load, check that Tailscale shows "Connected". That's the whole security model:
no Tailscale, no site.

## Around the page

Everything is on one page, and the steps go top to bottom.

## Before you start: your entries file

1. Enter your contests on DraftKings as usual (any lineup will do — it gets replaced).
2. Export your entries as a CSV from DraftKings. It's the same file you would use to upload
   lineups in bulk; the first columns are `Entry ID, Contest Name, Contest ID, Entry Fee`.
3. On the Ivan page, under **Your entries file**, choose the CSV and click **Upload**.
   The page lists the contests it found and how many entries are in each.

If you add or remove entries on DraftKings later, export and upload again.

## The five steps

Click **Run** on each, wait for the job box to say **done**, then move to the next one.
The job box shows the live log; if another job is still running, yours waits in line.

1. **Ingest the slate** — pick the slate from the list (it loads from DraftKings; only
   Showdown and Classic slates are shown), then Run. This pulls the players and salaries and the
   betting odds for each player, turns those into a projected range of fantasy points, and
   projects how popular each player will be (ownership). Run this first. Run it again close
   to lock if you want the latest odds; ownership refreshes with it.
2. **Fetch contests & payouts** — looks up the contests on DraftKings and how they pay out.
   Needed before simulating.
3. **Generate the field** — builds a big pool of possible lineups. 10,000 is a good default.
   "Parallel workers" is how many CPU cores to use; more is faster.
4. **Simulate your contests** — needs your entries file. Simulates each contest thousands of
   times and scores every candidate lineup. This is the slow step.
5. **Choose lineups** — needs your entries file. Picks the lineups to submit, one per entry.
   The three weights say what you care about:
   - *Cash %* — how often the lineup cashes.
   - *ROI* — how much it wins when it wins.
   - *Diversity* — higher spreads your entries across more different lineups.

## After step 5

- **Exposures** (top menu) shows how often each player appears across your chosen lineups,
  grouped by CPT/FLEX (showdown) or position (classic).
- **Downloads** has the file to give back to DraftKings — the one labelled *DraftKings upload*.
  Upload it through DraftKings' lineup upload and check that the entries look right.
- You can re-run step 5 with different weights as many times as you like; each run makes a new
  set of files.

## If something goes wrong

- A red **failed** badge means the script stopped; the log in the job box says why. The
  most common reasons are running a step before the one above it, or an entries file for a
  slate that hasn't been ingested yet.
- **Cancel** stops the running job. Nothing is left half-done that a re-run won't fix.
- The **Jobs** page keeps the log of everything that has run.
