#!/usr/bin/env bash
# Render and (re)install the systemd unit for the checkout this script lives in. Needs root.
#   sudo /opt/ivan/app/deploy/install_unit.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SVC="$(stat -c %U "$ROOT")"
HOME_DIR="$(getent passwd "$SVC" | cut -d: -f6)"
# Bind to the Tailscale IP on port 80 when Tailscale is up; else 0.0.0.0:8000 (LAN/dev box).
HOST="${IVAN_WEB_HOST:-$(tailscale ip -4 2>/dev/null || true)}"
PORT="${IVAN_WEB_PORT:-80}"
[[ -n "$HOST" ]] || { HOST=0.0.0.0; PORT="${IVAN_WEB_PORT:-8000}"; }
sed "s|__ROOT__|$ROOT|g; s|__USER__|$SVC|g; s|__HOME__|$HOME_DIR|g; s|__HOST__|$HOST|g; s|__PORT__|$PORT|g" \
  "$ROOT/deploy/ivan-web.service" > /etc/systemd/system/ivan-web.service
echo "ivan-web listens on $HOST:$PORT"
systemctl daemon-reload
systemctl enable --now ivan-web >/dev/null
systemctl restart ivan-web
sleep 3
systemctl --no-pager --lines=0 status ivan-web | sed -n 3p
