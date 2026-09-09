#!/usr/bin/env bash
# One-time (and safely re-runnable) setup of a server that runs the NFL web UI.
#
#   git clone git@github.com:SamBamCH/ivan.git ~/ivan
#   ~/ivan/deploy/bootstrap_server.sh            # venv, thin deps, dirs, empty DB
#   ~/ivan/deploy/bootstrap_server.sh --service  # ...and install + start the systemd unit
#
# Afterwards, from the machine that owns the models and history:
#   scripts/nfl/sync_to_server.sh <ssh host>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for tool in python3 sqlite3 rsync; do
  command -v "$tool" >/dev/null || { echo "missing: $tool (apt install $tool)" >&2; exit 1; }
done

[[ -d venv ]] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements-server.txt
venv/bin/pip install -q -e .

mkdir -p data/nfl/{input/web,output/web,models,jobs,sync,diagnostics}
if [[ ! -f .env ]]; then
  printf 'ODDS_API_KEY=\n' > .env
  echo "!! Put your the-odds-api key in $ROOT/.env (ODDS_API_KEY=...)"
fi

venv/bin/python scripts/nfl/create_db.py
echo "database: $ROOT/data/nfl/nfl_history.sqlite"

if [[ "${1:-}" == "--service" ]]; then
  sed "s|__ROOT__|$ROOT|g; s|__USER__|$USER|g" deploy/ivan-web.service | sudo tee /etc/systemd/system/ivan-web.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now ivan-web
  sleep 1; sudo systemctl --no-pager --lines=0 status ivan-web | head -3
fi

echo "UI: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8000"
