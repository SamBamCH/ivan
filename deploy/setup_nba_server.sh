#!/usr/bin/env bash
# One-time NBA setup on the server (the consumer), run as the service user inside the checkout after deploy/push.sh:
#     deploy/setup_nba_server.sh
# The laptop fetches everything from DK and the odds API once, publishes the feed and pushes it here
# (IVAN_FEED_PUSH on the laptop -> this checkout's data/feed). This script creates the data directories and the
# database schema, applies whatever has been pushed, and installs the pull timer (every 5 minutes, from this
# machine's own data/feed). The desk is served by ivan-web at /nba (ivan.web.app mounts it).
#
# Needs in .env: IVAN_DESK_PASSWORD (the login) and IVAN_FEED_URL=file:///opt/ivan/app/data/feed.
# User timers keep running without a login session only with lingering on, once, by an admin:
#     sudo loginctl enable-linger <service user>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
grep -q "^IVAN_DESK_PASSWORD=." .env 2>/dev/null || { echo "!! IVAN_DESK_PASSWORD is missing in $ROOT/.env"; exit 1; }
grep -q "^IVAN_FEED_URL=" .env 2>/dev/null || printf 'IVAN_FEED_URL=file://%s/data/feed\n' "$ROOT" >> .env
mkdir -p data/nba/{input/rotations,output,models,jobs,logs,diagnostics/scorecard,derived,raw} data/feed/nba data/feed-state
venv/bin/python - <<'PY'
from ivan.database.session import init_db
from ivan.sports.nba.ingest import ensure_schema
ensure_schema(init_db("nba")); print("nba database ready")
PY
if [[ -f data/feed/nba/manifest.json ]]; then
  venv/bin/python scripts/nba/feed.py pull --no-project
else
  echo "nothing pushed yet: on the laptop set IVAN_FEED_PUSH=<host>:$ROOT/data/feed and run scripts/nba/feed.py push"
fi
if ! loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q yes; then
  echo "!! lingering is off: the timers stop when this login ends. An admin runs: sudo loginctl enable-linger $(id -un)"
fi
deploy/install_nba_timers.sh --pull
# ivan-web mounts the desk at import time, so it must restart once the database exists (Restart=always brings it back)
OLD=$(systemctl show ivan-web -p MainPID --value 2>/dev/null || echo 0)
[[ "$OLD" != "0" ]] && kill "$OLD" && sleep 4
TS="$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}')"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://$TS/nba/login" || true)
[[ "$code" == "200" ]] && echo "desk login page answers at http://$TS/nba/login" || echo "!! desk not answering at http://$TS/nba/login (status $code); check: journalctl -u ivan-web -n 50"
echo "done. desk: http://$TS/nba/"
