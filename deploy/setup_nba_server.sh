#!/usr/bin/env bash
# One-time NBA setup on the server, run as the service user inside the checkout after deploy/push.sh:
#     deploy/setup_nba_server.sh [SEED_DIR]
# It creates the data directories and the database schema, seeds the database from feed bundles in
# SEED_DIR (default data/feed-seed: copy data/feed/nba/* there from the machine that captured so far),
# installs the user timers (capture every 15 min, nightly, the feed server on the tailnet) and publishes
# the first feed. The desk itself is served by ivan-web at /nba (ivan.web.app mounts it).
#
# Needs in .env: ODDS_API_KEY (props), IVAN_FEED_TOKEN (the feed), IVAN_DESK_PASSWORD (the login).
# User timers keep running without a login session only with lingering on, once, by an admin:
#     sudo loginctl enable-linger <service user>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
miss=0
for k in ODDS_API_KEY IVAN_FEED_TOKEN IVAN_DESK_PASSWORD; do
  grep -q "^$k=." .env 2>/dev/null || { echo "!! $k is missing in $ROOT/.env"; miss=1; }
done
[[ $miss -eq 0 ]] || exit 1
mkdir -p data/nba/{input/rotations,output,models,jobs,logs,diagnostics/scorecard,derived,raw} data/feed
venv/bin/python - <<'PY'
from ivan.database.session import init_db
from ivan.sports.nba.ingest import ensure_schema
ensure_schema(init_db("nba")); print("nba database ready")
PY
SEED="${1:-data/feed-seed}"
if [[ -f "$SEED/live.sql.gz" ]]; then
  venv/bin/python scripts/nba/feed.py apply "$SEED"
else
  echo "no seed bundles in $SEED (optional: copy data/feed/nba/* from the previous producer)"
fi
if ! loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q yes; then
  echo "!! lingering is off: the timers stop when this login ends. An admin runs: sudo loginctl enable-linger $(id -un)"
fi
deploy/install_nba_timers.sh
deploy/install_nba_timers.sh --feed
venv/bin/python scripts/nba/feed.py publish
# ivan-web mounts the desk at import time, so it must restart once the database exists (Restart=always brings it back)
OLD=$(systemctl show ivan-web -p MainPID --value 2>/dev/null || echo 0)
[[ "$OLD" != "0" ]] && kill "$OLD" && sleep 4
TS="$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}')"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://$TS/nba/login" || true)
[[ "$code" == "200" ]] && echo "desk login page answers at http://$TS/nba/login" || echo "!! desk not answering at http://$TS/nba/login (status $code); check: journalctl -u ivan-web -n 50"
echo "done. desk: http://$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}')/nba/   feed: http://$(tailscale ip -4 2>/dev/null):8790/nba/manifest.json"
