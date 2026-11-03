#!/usr/bin/env bash
# Push this checkout to the server and restart the web UI on the new code.
#
#   deploy/push.sh [ssh-host]        (default host: ivan-server)
#
# 1. rsync the working tree (not git) into the server's checkout, leaving its
#    venv, data/ and .env alone
# 2. re-run the bootstrap there (idempotent: deps, dirs, any new tables)
# 3. restart: the hardened unit hides the listening socket's owner from
#    fuser/ss, so `fuser -k 80/tcp` silently does nothing as user ivan. Kill
#    the unit's main pid instead; Restart=always brings it back.
set -euo pipefail

HOST="${1:-ivan-server}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${IVAN_REMOTE_ROOT:-/opt/ivan/app}"

echo "== rsync -> $HOST:$REMOTE_ROOT"
# .git and .claude stay home: the server runs the tree, not the repository (and a worktree's .git is a file, which would replace the directory there)
rsync -az --delete --info=stats1 --exclude /venv --exclude /data --exclude /.env --exclude /.git --exclude /.claude "$ROOT/" "$HOST:$REMOTE_ROOT/"

echo "== bootstrap + restart on $HOST"
ssh "$HOST" bash -s "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
cd "$1"
deploy/bootstrap_server.sh >/dev/null
OLD=$(systemctl show ivan-web -p MainPID --value)
[[ "$OLD" != "0" ]] && kill "$OLD"
for _ in $(seq 1 30); do
  sleep 1
  NEW=$(systemctl show ivan-web -p MainPID --value)
  [[ "$NEW" != "0" && "$NEW" != "$OLD" ]] && break
done
URL=$(systemctl show ivan-web -p ExecStart --value | grep -o -- '--host [^ ]* --port [^ ]*' | awk '{print "http://" $2 ":" $4 "/"}')
echo "service: $(systemctl is-active ivan-web), pid $OLD -> $NEW, started $(systemctl show ivan-web -p ExecMainStartTimestamp --value)"
# the site's root sends an anonymous visitor to /login (303), so either answer means the UI is up; the app imports every
# sport at start, so give it up to 30 s
for i in $(seq 1 10); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$URL" || true)
  [[ "$code" == "200" || "$code" == "303" ]] && { echo "UI answers $code at $URL"; exit 0; }
  sleep 3
done
echo "!! UI not answering at $URL (last status: $code); check: journalctl -u ivan-web -n 50" >&2
exit 1
REMOTE
