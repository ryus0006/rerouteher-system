#!/bin/bash

# Stop local development environment for rerouteher-system

# Run from the project root regardless of where the script is called from
cd "$(dirname "$0")/.."

echo "Stopping local development environment..."

# Stop containers and remove volumes so the next start re-imports fresh reference data.
# Pass --keep-data to preserve the database volume across restarts.
if [ "$1" = "--keep-data" ]; then
    docker compose down --remove-orphans
    echo "Local environment stopped (database preserved)."
else
    docker compose down -v --remove-orphans
    echo "Local environment stopped (database reset)."
fi
