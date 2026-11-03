#!/usr/bin/env bash
# One-time NBA setup on the server (the consumer), run as the service user inside the checkout after deploy/push.sh:
#     deploy/setup_nba_server.sh
# The server pulls from the laptop over ssh when its desk's Refresh is pressed (ivan.core.feed.pull_ssh): one ssh command
# asks the laptop to publish its live bundle, rsync copies the laptop's data/feed here, and the pull applies it. Nothing is
# pushed from the laptop and no timer pulls. This script creates the data directories and the database schema, makes sure
# this user has an ssh key (printed, to bind on the laptop with deploy/install_feed_key.sh), tries the pull, and restarts
# ivan-web (it mounts the desk at import time). The desk is served by ivan-web at /nba (ivan.web.main mounts it).
#
# Needs in .env: IVAN_PASSWORD_HASH (the site's login, scripts/password.py) and IVAN_FEED_SSH=<user>@<the laptop on the tailnet>.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
grep -qE "^IVAN(_DESK)?_PASSWORD(_HASH)?=." .env 2>/dev/null || { echo "!! IVAN_PASSWORD_HASH is missing in $ROOT/.env (scripts/password.py --set)"; exit 1; }
if ! grep -q "^IVAN_FEED_SSH=" .env 2>/dev/null; then
  echo "!! IVAN_FEED_SSH is missing in $ROOT/.env: the laptop as ssh reaches it, e.g. IVAN_FEED_SSH=sam-pratt@main-thinkpad"; exit 1
fi
HOST="$(grep '^IVAN_FEED_SSH=' .env | cut -d= -f2- | tr -d '"'"'" )"
mkdir -p data/nba/{input/rotations,output,models,jobs,logs,diagnostics/scorecard,derived,raw} data/feed/nba data/feed-state
venv/bin/python - <<'PY'
from ivan.database.session import init_db
from ivan.sports.nba.ingest import ensure_schema
ensure_schema(init_db("nba")); print("nba database ready")
PY
if [[ ! -f "$HOME/.ssh/id_ed25519.pub" ]]; then
  ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -C "ivan-feed@$(hostname)"
fi
echo "this machine's key for the laptop (bind it there with deploy/install_feed_key.sh '<this line>'):"
cat "$HOME/.ssh/id_ed25519.pub"
if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$HOST" publish nba >/dev/null 2>&1; then
  venv/bin/python scripts/nba/feed.py pull --no-project | tail -3
else
  echo "!! $HOST does not answer 'publish' yet: run install_feed_key.sh on the laptop with the key above (and sudo systemctl enable --now ssh there), then scripts/nba/feed.py pull here"
fi
# the consumer runs no NBA timers of its own (the laptop's capture, archive and nightly timers are the producer's)
systemctl --user disable --now ivan-nba-pull.timer ivan-nba-capture.timer ivan-nba-nightly.timer 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/ivan-nba-pull.service" "$HOME/.config/systemd/user/ivan-nba-pull.timer"; systemctl --user daemon-reload 2>/dev/null || true
# ivan-web mounts the desk at import time, so it must restart once the database exists (Restart=always brings it back)
OLD=$(systemctl show ivan-web -p MainPID --value 2>/dev/null || echo 0)
[[ "$OLD" != "0" ]] && kill "$OLD" && sleep 4
TS="$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}')"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://$TS/nba/login" || true)
[[ "$code" == "200" ]] && echo "desk login page answers at http://$TS/nba/login" || echo "!! desk not answering at http://$TS/nba/login (status $code); check: journalctl -u ivan-web -n 50"
echo "done. desk: http://$TS/nba/  (its Refresh pulls from $HOST)"
