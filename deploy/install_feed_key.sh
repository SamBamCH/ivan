#!/usr/bin/env bash
# On the producer (the laptop): let one consumer pull the NBA feed over ssh (ivan.core.feed.pull_ssh).
#   deploy/install_feed_key.sh "ssh-ed25519 AAAA... ivan@pratt-server"      the consumer's public key (deploy/setup_nba_server.sh prints it)
# Appends one line to ~/.ssh/authorized_keys that binds the key to deploy/feed_serve.sh (publish the live bundle, or
# read data/feed through rrsync; nothing else), with no shell, pty or forwarding. Idempotent: a key already bound is left alone.
# sshd must be running on this machine (sudo apt install openssh-server) and reachable from the consumer (the tailnet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${1:-}"
[[ -n "$KEY" ]] || { echo "usage: $0 '<the consumer's public key line>'"; exit 2; }
[[ -x /usr/bin/rrsync ]] || { echo "!! /usr/bin/rrsync is missing (apt install rsync)"; exit 1; }
mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"; touch "$HOME/.ssh/authorized_keys"; chmod 600 "$HOME/.ssh/authorized_keys"
KEYBODY="$(echo "$KEY" | awk '{print $1" "$2}')"
LINE="command=\"$ROOT/deploy/feed_serve.sh\",no-agent-forwarding,no-port-forwarding,no-pty,no-X11-forwarding,no-user-rc $KEY"
if grep -qF "$KEYBODY" "$HOME/.ssh/authorized_keys"; then
  echo "that key is already in ~/.ssh/authorized_keys (leave it, or remove its line and run this again to rebind it)"
else
  echo "$LINE" >> "$HOME/.ssh/authorized_keys"; echo "bound to $ROOT/deploy/feed_serve.sh: publish + read-only rsync of data/feed"
fi
chmod +x "$ROOT/deploy/feed_serve.sh"
if ! systemctl is-active --quiet ssh 2>/dev/null && ! systemctl is-active --quiet sshd 2>/dev/null; then
  echo "!! sshd is not running here: sudo apt install openssh-server (installs and starts it)"
fi
echo "on the consumer: IVAN_FEED_SSH=$(id -un)@$(hostname) in .env, then scripts/nba/feed.py pull"
