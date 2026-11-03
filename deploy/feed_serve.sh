#!/usr/bin/env bash
# The producer's side of every sport's feed pull (ivan.core.feed.pull_ssh): the forced command of the consumer's ssh key on
# the laptop (deploy/install_feed_key.sh writes the authorized_keys line). The key can do two things and nothing else:
#   publish <sport>             write the sport's feed bundle fresh (scripts/<sport>/feed.py publish --quiet): for the NBA
#                               the live bundle (the pool as it stands, the published edits, the projection store), for the
#                               NFL and MMA the reference bundle (tables + models / inputs)
#   rsync --server --sender ... read this checkout's data/feed through rrsync, read only (what the consumer's rsync asks for)
# Anything else is refused. No shell, no pty, no forwarding (the authorized_keys options).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmd="${SSH_ORIGINAL_COMMAND:-}"
case "$cmd" in
  "publish")
    cd "$ROOT"; exec "$ROOT/venv/bin/python" "$ROOT/scripts/nba/feed.py" publish --which live --quiet ;;
  "publish nba")
    cd "$ROOT"; exec "$ROOT/venv/bin/python" "$ROOT/scripts/nba/feed.py" publish --which live --quiet ;;
  "publish nfl"|"publish mma")
    sport="${cmd#publish }"
    cd "$ROOT"; exec "$ROOT/venv/bin/python" "$ROOT/scripts/$sport/feed.py" publish --quiet ;;
  "rsync --server --sender "*)
    exec /usr/bin/rrsync -ro "$ROOT/data/feed" ;;
  *)
    echo "feed_serve: refused: ${cmd:-(no command)}" >&2; exit 1 ;;
esac
