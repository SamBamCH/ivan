#!/usr/bin/env bash
# Replace this machine's reference tables with a bundle from sync_to_server.sh.
# Working tables (current slate, lineups, sim results) are untouched.
#
# Usage:  scripts/nfl/import_reference.sh data/nfl/sync/reference.sql.gz
# Env:    IVAN_NFL_DB  target sqlite file (default: data/nfl/nfl_history.sqlite)
set -euo pipefail

BUNDLE="${1:?usage: import_reference.sh <reference.sql.gz>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DB="${IVAN_NFL_DB:-$ROOT/data/nfl/nfl_history.sqlite}"
mkdir -p "$(dirname "$DB")"

# Tables in the bundle are exactly the CREATE TABLE statements it carries.
TABLES=$(zcat "$BUNDLE" | grep -oE '^CREATE TABLE (IF NOT EXISTS )?"?[A-Za-z0-9_]+"?' \
         | awk '{print $NF}' | tr -d '"' | tr '\n' ' ')
[[ -n "$TABLES" ]] || { echo "no CREATE TABLE statements in $BUNDLE" >&2; exit 1; }

{
  echo "PRAGMA foreign_keys=OFF;"
  for t in $TABLES; do echo "DROP TABLE IF EXISTS \"$t\";"; done
  zcat "$BUNDLE"
} | sqlite3 -cmd ".timeout 30000" "$DB"

echo "imported: $TABLES"
