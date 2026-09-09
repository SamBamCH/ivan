#!/usr/bin/env bash
# Run ONCE as root on the server. Moves the UI to a dedicated unprivileged user and a
# sandboxed systemd unit, and restricts port 8000 to the Tailscale interface.
#
#   sudo ~/ivan/deploy/harden_server.sh            # from the existing ~/ivan checkout
#
# Afterwards the checkout lives at /opt/ivan/app, owned by user `ivan`, and deploys/syncs
# from the laptop go over ssh as `ivan` (same authorized key as the admin user).
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

ADMIN="${SUDO_USER:-sam-pratt}"
SRC="${1:-/home/$ADMIN/ivan}"
SVC=ivan
HOME_DIR=/opt/ivan
DST=$HOME_DIR/app

# 1) unprivileged service user: shell only so ssh deploys work; no sudo, no docker
if ! id "$SVC" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "$HOME_DIR" --shell /bin/bash "$SVC"
  echo "created user $SVC"
fi
install -d -m 700 -o "$SVC" -g "$SVC" "$HOME_DIR/.ssh" "$HOME_DIR/.cache"
install -m 600 -o "$SVC" -g "$SVC" "/home/$ADMIN/.ssh/authorized_keys" "$HOME_DIR/.ssh/authorized_keys"

# 2) move the checkout (code, venv, data, .env) and hand it over
systemctl stop ivan-web 2>/dev/null || true
if [[ ! -d "$DST" ]]; then
  [[ -d "$SRC" ]] || { echo "no checkout at $SRC" >&2; exit 1; }
  mv "$SRC" "$DST"
  echo "moved $SRC -> $DST"
fi
chown -R "$SVC:$SVC" "$HOME_DIR"
chmod 750 "$HOME_DIR" "$DST"
[[ -f "$DST/.env" ]] && chmod 600 "$DST/.env"

# 3) the venv has absolute paths baked in; rebuild it as the service user
sudo -u "$SVC" -H bash -c "cd '$DST' && rm -rf venv && deploy/bootstrap_server.sh"

# 4) sandboxed unit
sed "s|__ROOT__|$DST|g; s|__USER__|$SVC|g; s|__HOME__|$HOME_DIR|g" "$DST/deploy/ivan-web.service" \
  > /etc/systemd/system/ivan-web.service
systemctl daemon-reload
systemctl enable --now ivan-web
sleep 3
systemctl --no-pager --lines=0 status ivan-web | head -3

# 5) port 8000 only over Tailscale
if command -v ufw >/dev/null; then
  ufw delete allow 8000/tcp >/dev/null 2>&1 || true
  ufw allow in on tailscale0 to any port 8000 proto tcp comment 'ivan web UI (tailnet only)' >/dev/null
  echo "ufw: 8000 allowed on tailscale0 only"
fi

echo
echo "done. From the laptop: ssh $SVC@$(hostname -I | awk '{print $1}') ; UI: http://$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}'):8000"
