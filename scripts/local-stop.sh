#!/bin/bash

# Stop the local development environment for rerouteher-system.
#
# Brings down the Docker containers started by ./scripts/local-start.sh (db, and api
# in the default Docker mode). If you used --local, the API is a host uvicorn process,
# not a container, so stop it separately with Ctrl-C in its terminal.
#
# Usage:
#   ./scripts/local-stop.sh              reset the database (removes the volume)
#   ./scripts/local-stop.sh --keep-data  preserve the database across restarts

# Run from the project root regardless of where the script is called from.
cd "$(dirname "$0")/.."

echo "Stopping local development environment..."

# Stop containers and remove volumes so the next start re-imports fresh reference data
# and re-runs the account_store migration. Pass --keep-data to preserve the volume.
if [ "$1" = "--keep-data" ]; then
    docker compose down --remove-orphans
    echo "Local environment stopped (database preserved)."
else
    docker compose down -v --remove-orphans
    echo "Local environment stopped (database reset)."
fi

echo "If you ran with --local, remember to Ctrl-C the uvicorn API in its terminal (it is a host process, not a container)."
