#!/usr/bin/env bash
# Install the resource ceiling for ivan-web (deploy/ivan-web-limits.conf) as a systemd drop-in and restart the unit.
#   sudo /opt/ivan/app/deploy/install_limits.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /etc/systemd/system/ivan-web.service.d
install -m 0644 "$HERE/ivan-web-limits.conf" /etc/systemd/system/ivan-web.service.d/limits.conf
systemctl daemon-reload
systemctl restart ivan-web
systemctl show ivan-web -p MemoryHigh -p MemoryMax -p TasksMax -p CPUWeight
echo "ivan-web: $(systemctl is-active ivan-web)"
