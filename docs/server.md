# Running the server

The server is the *consumer* in a two-machine setup: it runs the web UI and the five
pre-lock steps on its own SQLite file, and receives history and models from the *producer*
laptop. It never scrapes contest results or trains models, so it needs no Chrome, no
DraftKings login, and only the thin `requirements-server.txt`.

## First-time setup

On the server:

```bash
sudo apt install python3-venv sqlite3 rsync
git clone git@github.com:SamBamCH/ivan.git ~/ivan
~/ivan/deploy/bootstrap_server.sh --service     # venv, deps, dirs, empty DB, systemd unit
nano ~/ivan/.env                                # ODDS_API_KEY=...  (read fresh for every job; no restart needed)
sudo systemctl restart ivan-web
sudo ufw allow 8000/tcp                         # Ubuntu's firewall drops the port by default
```

Symptom of a missed `ufw` step: the UI answers on the server's own `curl localhost:8000`
but browsers on the LAN time out (not "connection refused").

`bootstrap_server.sh` is safe to re-run after a `git pull`. Without `--service` it only
prepares the checkout; start the UI by hand with `venv/bin/python scripts/nfl/serve.py`.

From the producer laptop, push the reference data once:

```bash
scripts/nfl/sync_to_server.sh server           # any ssh host alias works
```

The UI is then at `http://<server>:8000`. It has no login — reach it over the LAN or
Tailscale, not the public internet.

## Keeping it in sync

The laptop's `overnight.py` pushes automatically when `IVAN_SYNC_HOST=server` is in its
`.env`. What moves:

- **Reference tables** — everything except `nfl_player`, `odds`, `projection_snapshot`,
  `lineup`, `lineup_simulation_results` (and MLB tables), with their explicit indexes.
  Dumped from a `VACUUM INTO` snapshot, gzipped (~24 MB), and loaded on the server by
  `scripts/nfl/import_reference.sh` with drop-and-replace per table.
- **`data/nfl/models/`** — rsync with `--delete` (minus `archive/`), so unchanged pickles cost
  nothing and retrained ones replace the old.

The server's five working tables are never touched, so a push during someone's slate is
safe. If the server is mid-write, the import waits up to 30 s for the lock and then fails
loudly; re-run the sync.

## Sizing

The pipeline is CPU-bound and forks: `config/nfl.json` sets `sampling.n_jobs` and
`simulation.n_jobs` (8 on the laptop), and step 3 takes its own worker count from the form.
Match those to the server's cores. The two classic models are ~280 MB each and are loaded
per worker; 16 GB of RAM is comfortable. Jobs run under `nice -n 10`, so other services
(e.g. Jellyfin) stay responsive without stopping anything.

## Where things are

| | |
|---|---|
| `~/ivan/data/nfl/nfl_history.sqlite` | the server's database |
| `~/ivan/data/nfl/input/web/` | uploaded entries files (`entries.csv` is the current one) |
| `~/ivan/data/nfl/output/web/` | chosen-entries and exposures CSVs served on the Downloads page |
| `~/ivan/data/nfl/jobs/` | one `.log` and `.json` per job |
| `~/ivan/data/nfl/sync/` | the last received reference bundle |
| `journalctl -u ivan-web -f` | server log |

## Updating code

```bash
cd ~/ivan && git pull && deploy/bootstrap_server.sh && sudo systemctl restart ivan-web
```

`requirements-server.txt` is generated, not hand-edited. After changing imports on the
pre-lock path or retraining models with a new library, regenerate it on the laptop with
`venv/bin/python deploy/derive_server_requirements.py` (it imports every pre-lock module
*and* unpickles every model file, because scikit-learn only shows up at load time), commit,
and re-run the bootstrap on the server.
