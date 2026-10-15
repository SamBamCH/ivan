#!/usr/bin/env bash
# Install (or re-install) the nightly data job as a systemd *user* timer for the checkout this
# script lives in. No root: the job runs in your desktop session, which is what the DraftKings
# scraper's Chrome needs.
#   deploy/install_nightly_timer.sh            install + enable + show the next runs
#   deploy/install_nightly_timer.sh --run-now  also run the job once right away
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS" "$ROOT/data/nfl/logs"
sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/ivan-nightly.service" > "$UNITS/ivan-nightly.service"
cp "$ROOT/deploy/ivan-nightly.timer" "$UNITS/ivan-nightly.timer"
systemctl --user daemon-reload
systemctl --user enable --now ivan-nightly.timer >/dev/null
[[ "${1:-}" == "--run-now" ]] && systemctl --user start ivan-nightly.service
systemctl --user list-timers --no-pager ivan-nightly.timer
echo "log: $ROOT/data/nfl/logs/nightly.log"
