# Running the server

The server is the *consumer* in a two-machine setup: it runs the web UI and the five
pre-lock steps on its own SQLite file, and receives history and models from the *producer*
laptop. It never scrapes contest results or trains models, so it needs no Chrome, no
DraftKings login, and only the thin `requirements-server.txt`.

## Access model

The UI is reachable only over Tailscale. Port 8000 is allowed in `ufw` on the `tailscale0`
interface alone, so nothing is exposed to the internet and there is no login to maintain:
being on the tailnet *is* the login. Give someone access by inviting them to the tailnet
(Tailscale admin console → Users → Invite; the free plan covers three users), and remove it by
removing them there.

The service runs as the unprivileged user `ivan` (no `sudo`, no `docker`) under a sandboxed
systemd unit: the filesystem is read-only to it except `data/` and its cache, no new
privileges, private `/tmp`, empty capability set. A bug in the app or in a model pickle can
touch this app's own state and nothing else on the box.

## First-time setup

On the server, as your admin user:

```bash
sudo apt install python3-venv sqlite3 rsync
git clone git@github.com:SamBamCH/ivan.git ~/ivan
sudo ~/ivan/deploy/harden_server.sh      # creates user ivan, moves the checkout to /opt/ivan/app,
                                          # builds the venv, installs the sandboxed unit, ufw rule
sudo -u ivan nano /opt/ivan/app/.env      # ODDS_API_KEY=...  (read fresh for every job; no restart needed)
```

From the laptop, add an ssh alias for the service user (same key as your admin user) and push
the reference data:

```
# ~/.ssh/config
Host ivan-server
    HostName <server LAN or Tailscale IP>
    User ivan
    IdentityFile ~/.ssh/id_ed25519
```

```bash
scripts/nfl/sync_to_server.sh ivan-server
```

The app binds to the server's Tailscale IP on port 80 (`install_unit.sh` detects it), so it is
unreachable from the LAN or the internet by construction. Names:

- `http://ivan.prattflix.com` — a public `A` record (DNS only) pointing at the Tailscale IP. Resolves
  everywhere, routes only on the tailnet; the URL to give people. On the home LAN, Pi-hole's
  `*.prattflix.com` wildcard must be overridden with a local record for this name.
- `https://pratt-server.tail6f8004.ts.net` — `sudo tailscale serve --bg http://<tailscale ip>:80`
  after enabling HTTPS certificates in the Tailscale admin console. Needs MagicDNS on the client.

`tailscale serve` only answers for the node's own MagicDNS names, which is why the custom domain
goes straight to the app instead.

## Keeping it in sync

The laptop's `overnight.py` pushes automatically when `IVAN_SYNC_HOST=ivan-server` is in its
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
| `/opt/ivan/app/data/nfl/nfl_history.sqlite` | the server's database |
| `/opt/ivan/app/data/nfl/input/web/` | uploaded entries files (`entries.csv` is the current one) |
| `/opt/ivan/app/data/nfl/output/web/` | chosen-entries and exposures CSVs served on the Downloads page |
| `/opt/ivan/app/data/nfl/jobs/` | one `.log` and `.json` per job |
| `/opt/ivan/app/data/nfl/sync/` | the last received reference bundle |
| `journalctl -u ivan-web -f` | server log |

## Updating code

From the laptop, push the checkout and bounce the service (the unit restarts on its own):

```bash
rsync -az --delete --exclude /venv --exclude /data --exclude /.env ./ ivan-server:/opt/ivan/app/
ssh ivan-server 'cd /opt/ivan/app && deploy/bootstrap_server.sh && fuser -k -KILL 8000/tcp'
```

`requirements-server.txt` is generated, not hand-edited. After changing imports on the
pre-lock path or retraining models with a new library, regenerate it on the laptop with
`venv/bin/python deploy/derive_server_requirements.py` (it imports every pre-lock module
*and* unpickles every model file, because scikit-learn only shows up at load time), commit,
and re-run the bootstrap on the server.
