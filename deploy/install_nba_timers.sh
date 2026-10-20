#!/usr/bin/env bash
# Install (or re-install) the NBA capture tick and nightly job as systemd *user* timers for this checkout.
#   deploy/install_nba_timers.sh            install + enable + show the next runs
#   deploy/install_nba_timers.sh --run-now  also run one capture tick right away
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS" "$ROOT/data/nba/logs"
for unit in ivan-nba-capture ivan-nba-nightly; do
  sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/$unit.service" > "$UNITS/$unit.service"
  cp "$ROOT/deploy/$unit.timer" "$UNITS/$unit.timer"
done
systemctl --user daemon-reload
systemctl --user enable --now ivan-nba-capture.timer ivan-nba-nightly.timer >/dev/null
[[ "${1:-}" == "--run-now" ]] && systemctl --user start ivan-nba-capture.service
systemctl --user list-timers --no-pager ivan-nba-capture.timer ivan-nba-nightly.timer
echo "logs: $ROOT/data/nba/logs/capture.log, $ROOT/data/nba/logs/nightly.log"
