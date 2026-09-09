#!/usr/bin/env bash
# Push reference data (historical tables + model pickles) from this machine to
# the server. Never touches the server's per-slate working tables (players,
# odds, lineups, sim results), so it is safe to run while someone is mid-slate
# there.
#
# Usage:  scripts/nfl/sync_to_server.sh [ssh-host]      (default host: server)
# Env:    IVAN_NFL_DB      source sqlite file   (default: data/nfl/nfl_history.sqlite)
#         IVAN_REMOTE_ROOT checkout on the host (default: ivan, relative to remote $HOME)
#         BUNDLE_ONLY=1    build the bundle and stop; print its path (for testing)
set -euo pipefail

HOST="${1:-server}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DB="${IVAN_NFL_DB:-$ROOT/data/nfl/nfl_history.sqlite}"
MODELS="$ROOT/data/nfl/models"
REMOTE_ROOT="${IVAN_REMOTE_ROOT:-ivan}"
STAGE="$(mktemp -d)"
[[ "${BUNDLE_ONLY:-0}" == "1" ]] || trap 'rm -rf "$STAGE"' EXIT

# Per-slate working tables: owned by whichever machine is running the slate.
WORKING="'nfl_player','odds','projection_snapshot','lineup','lineup_simulation_results'"

# 1) consistent snapshot, safe even if another process is mid-write
sqlite3 "$DB" "VACUUM INTO '$STAGE/snapshot.sqlite'"

# 2) dump every table that is not working-set, MLB, or sqlite-internal
TABLES=$(sqlite3 "$STAGE/snapshot.sqlite" "SELECT name FROM sqlite_master
  WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'mlb_%'
    AND name NOT IN ($WORKING) ORDER BY name;" | tr '\n' ' ')
# .dump filters on object *name*, so explicit indexes must be listed too
# (auto-indexes from PK/UNIQUE constraints come with the table).
INDEXES=$(sqlite3 "$STAGE/snapshot.sqlite" "SELECT name FROM sqlite_master
  WHERE type='index' AND sql IS NOT NULL
    AND tbl_name IN ($(echo $TABLES | sed "s/ *$//; s/ /','/g; s/^/'/; s/$/'/")) ORDER BY name;" | tr '\n' ' ')
echo "reference tables: $TABLES"
echo "indexes: $INDEXES"
sqlite3 "$STAGE/snapshot.sqlite" ".dump $TABLES $INDEXES" | gzip -1 > "$STAGE/reference.sql.gz"
rm -f "$STAGE/snapshot.sqlite"
echo "bundle: $(du -h "$STAGE/reference.sql.gz" | cut -f1)  $STAGE/reference.sql.gz"
[[ "${BUNDLE_ONLY:-0}" == "1" ]] && exit 0

# 3) ship bundle + models (rsync delta transfer: unchanged pickles cost nothing)
ssh "$HOST" "mkdir -p '$REMOTE_ROOT/data/nfl/sync' '$REMOTE_ROOT/data/nfl/models'"
rsync -az --info=stats1 "$STAGE/reference.sql.gz" "$HOST:$REMOTE_ROOT/data/nfl/sync/"
rsync -az --info=stats1 --delete --exclude 'archive/' "$MODELS/" "$HOST:$REMOTE_ROOT/data/nfl/models/"

# 4) load on the server
ssh "$HOST" "cd '$REMOTE_ROOT' && scripts/nfl/import_reference.sh data/nfl/sync/reference.sql.gz"
