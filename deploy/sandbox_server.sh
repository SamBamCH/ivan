#!/usr/bin/env bash
# A sandbox consumer on the server: a second copy of the app in consumer mode, on its own port with its own data, fed by
# the SAME directory the laptop pushes to (/opt/ivan/app/data/feed). Everything the brother's desk does happens here the
# same way (the rsync push, the pull from disk, the server's code, the /nba mount, the consumer limits) except the
# Cloudflare tunnel and his login: you reach it over an ssh tunnel with the password "sandbox". Break it freely; his desk
# and his data are untouched.
#
#   deploy/sandbox_server.sh up [host]       rsync this tree there, set up data + .env, pull once, (re)start it
#   deploy/sandbox_server.sh status [host]   is it running, what did it last pull
#   deploy/sandbox_server.sh pull [host]     pull the feed into the sandbox right now (its own pull loop does this every 5 minutes, like his timer)
#   deploy/sandbox_server.sh reset [host]    wipe the sandbox's own data (its edits, lineups, sims) and start over
#   deploy/sandbox_server.sh down [host]     stop it
#
# Then, on the laptop:   ssh -f -N -L 8002:127.0.0.1:8002 ivan-server    and open   http://127.0.0.1:8002/nba/
# Check it the way consumer_check.py checks his desk:   scripts/nba/consumer_check.py --sandbox --refresh
set -euo pipefail
ACTION="${1:-up}"
HOST="${2:-ivan-server}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SB="${IVAN_SANDBOX_ROOT:-/opt/ivan/sandbox/app}"
PORT="${IVAN_SANDBOX_PORT:-8002}"
FEED="${IVAN_SANDBOX_FEED:-file:///opt/ivan/app/data/feed}"

remote() { ssh -o BatchMode=yes "$HOST" bash -s "$SB" "$PORT" "$FEED" "$@"; }

case "$ACTION" in
  up)
    echo "== rsync -> $HOST:$SB"
    ssh -o BatchMode=yes "$HOST" "mkdir -p '$SB'"
    rsync -az --delete --info=stats1 --exclude /venv --exclude /data --exclude /.env --exclude /.git --exclude /.claude "$ROOT/" "$HOST:$SB/" | tail -1
    echo "== set up + start on $HOST"
    remote <<'REMOTE'
set -euo pipefail
SB="$1"; PORT="$2"; FEED="$3"; cd "$SB"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
[[ -e venv ]] || ln -s /opt/ivan/app/venv venv            # the same dependencies; PYTHONPATH below makes THIS tree's code win over the editable install
mkdir -p data/nba/{input/rotations,output,models,jobs,logs,diagnostics/scorecard,derived,raw} data/feed-state \
         data/nfl/{input/web,output/web,models,jobs,sync,diagnostics}
[[ -f .env ]] || printf 'IVAN_FEED_URL=%s\nIVAN_DESK_PASSWORD=sandbox\nIVAN_DESK_LIMITS=1\n' "$FEED" > .env
export PYTHONPATH="$SB/src"
venv/bin/python scripts/nfl/create_db.py >/dev/null
venv/bin/python - <<'PY'
from ivan.database.session import init_db
from ivan.sports.nba.ingest import ensure_schema
ensure_schema(init_db("nba")); print("nba database ready")
PY
venv/bin/python scripts/nba/feed.py pull --no-project 2>&1 | tail -2
systemctl --user stop ivan-sandbox.service ivan-sandbox-pull.service 2>/dev/null || true
systemd-run --user --unit ivan-sandbox --collect -p WorkingDirectory="$SB" -E PYTHONPATH="$SB/src" -E PYTHONUNBUFFERED=1 \
  "$SB/venv/bin/python" "$SB/scripts/nfl/serve.py" --host 127.0.0.1 --port "$PORT" >/dev/null
# his 5-minute pull timer, as a loop: the sandbox picks up a push on its own and shows "new data · load" like his desk does
systemd-run --user --unit ivan-sandbox-pull --collect -p WorkingDirectory="$SB" -E PYTHONPATH="$SB/src" -E PYTHONUNBUFFERED=1 \
  "$SB/venv/bin/python" "$SB/scripts/nba/feed.py" pull --loop 5 >/dev/null
code=000
for _ in $(seq 1 25); do
  sleep 1
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/nba/login" || true)
  [[ "$code" == "200" ]] && break
done
echo "sandbox: $(systemctl --user is-active ivan-sandbox.service) · /nba/login answers $code on 127.0.0.1:$PORT"
[[ "$code" == "200" ]] || { journalctl --user -u ivan-sandbox -n 20 --no-pager; exit 1; }
REMOTE
    echo
    echo "tunnel + desk (the password is: sandbox):"
    echo "  ssh -f -N -L $PORT:127.0.0.1:$PORT $HOST"
    echo "  http://127.0.0.1:$PORT/nba/"
    echo "his Refresh = the sandbox's Refresh (pulls what your laptop pushed); his 5-minute timer = the sandbox's pull loop (deploy/sandbox_server.sh pull forces one now)"
    echo "check it: scripts/nba/consumer_check.py --sandbox --refresh"
    ;;
  status)
    remote <<'REMOTE'
SB="$1"; PORT="$2"; cd "$SB" 2>/dev/null || { echo "no sandbox at $SB"; exit 1; }
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
echo "sandbox: $(systemctl --user is-active ivan-sandbox.service 2>/dev/null || echo inactive) on 127.0.0.1:$PORT · pull loop: $(systemctl --user is-active ivan-sandbox-pull.service 2>/dev/null || echo inactive)"
[[ -f data/feed-state/nba.json ]] && python3 -c "import json; s=json.load(open('data/feed-state/nba.json')); print('pulled live', (s.get('live') or {}).get('stamp'), 'at', (s.get('live') or {}).get('pulled_at'), '· last check', s.get('checked_at'))"
journalctl --user -u ivan-sandbox -n 5 --no-pager 2>/dev/null | cut -c1-160
REMOTE
    ;;
  pull)
    remote <<'REMOTE'
SB="$1"; cd "$SB"; PYTHONPATH="$SB/src" venv/bin/python scripts/nba/feed.py pull
REMOTE
    ;;
  reset)
    remote <<'REMOTE'
SB="$1"; cd "$SB"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
systemctl --user stop ivan-sandbox.service ivan-sandbox-pull.service 2>/dev/null || true
rm -rf data/nba data/feed-state; echo "sandbox data wiped; run: deploy/sandbox_server.sh up"
REMOTE
    ;;
  down)
    remote <<'REMOTE'
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
systemctl --user stop ivan-sandbox-pull.service 2>/dev/null || true
systemctl --user stop ivan-sandbox.service 2>/dev/null && echo "sandbox stopped" || echo "sandbox was not running"
REMOTE
    ;;
  *)
    echo "usage: deploy/sandbox_server.sh up|status|pull|reset|down [host]" >&2; exit 2
    ;;
esac
