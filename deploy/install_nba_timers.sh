#!/usr/bin/env bash
# Install (or re-install) the NBA capture tick and nightly job as systemd *user* timers for this checkout.
#   deploy/install_nba_timers.sh            install + enable + show the next runs
#   deploy/install_nba_timers.sh --run-now  also run one capture tick right away
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS" "$ROOT/data/nba/logs"
# the capture tick (ivan-nba-capture) is no longer enabled: the desk's Refresh is the only thing that fetches and rewrites the
# pool (2026-10-23); the archive job records the projections and pool at every game's tip without touching anything live
for unit in ivan-nba-archive ivan-nba-nightly ivan-nba-capture; do
  sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/$unit.service" > "$UNITS/$unit.service"
  cp "$ROOT/deploy/$unit.timer" "$UNITS/$unit.timer"
done
systemctl --user daemon-reload
systemctl --user enable --now ivan-nba-archive.timer ivan-nba-nightly.timer >/dev/null
systemctl --user disable --now ivan-nba-capture.timer 2>/dev/null || true
if [[ "${1:-}" == "--pull" ]]; then
  # a consumer: pull the feed instead of capturing (the capture + nightly timers are disabled here)
  for u in ivan-nba-pull.service ivan-nba-pull.timer; do sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/$u" > "$UNITS/$u"; done
  systemctl --user daemon-reload
  systemctl --user disable --now ivan-nba-capture.timer ivan-nba-nightly.timer 2>/dev/null || true
  systemctl --user enable --now ivan-nba-pull.timer
  systemctl --user list-timers --no-pager | grep -E "ivan-nba" || true
  exit 0
fi
if [[ "${1:-}" == "--feed" ]]; then
  sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/ivan-feed.service" > "$UNITS/ivan-feed.service"
  systemctl --user daemon-reload
  systemctl --user enable --now ivan-feed.service
  systemctl --user --no-pager status ivan-feed.service | head -5
fi
[[ "${1:-}" == "--run-now" ]] && systemctl --user start ivan-nba-capture.service
systemctl --user list-timers --no-pager ivan-nba-archive.timer ivan-nba-nightly.timer
echo "logs: $ROOT/data/nba/logs/archive.log, $ROOT/data/nba/logs/nightly.log"
