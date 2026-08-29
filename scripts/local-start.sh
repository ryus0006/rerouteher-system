#!/bin/bash

# Start local development environment for rerouteher-system (pgvector DB + FastAPI API)

set -e

# Run from the project root regardless of where the script is called from
cd "$(dirname "$0")/.."

echo "Starting local development environment..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Host port for Postgres (override if 5432 is taken: DB_PORT=5433 ./scripts/local-start.sh)
DB_PORT="${DB_PORT:-5432}"
export DB_PORT

# Build and start the pgvector database and the API
# First run pulls the model layers (torch, MiniLM, spaCy) and can take a few minutes.
echo "Starting pgvector PostgreSQL and API (building images if needed)..."
docker compose up -d --build

# Wait for the database healthcheck (it also runs the reference import + pgvector migration on first init)
echo "Waiting for the database to become healthy (import + pgvector migration on first init)..."
DB_CID="$(docker compose ps -q db)"
for i in $(seq 1 40); do
    STATUS="$(docker inspect --format '{{.State.Health.Status}}' "$DB_CID" 2>/dev/null || echo unknown)"
    if [ "$STATUS" = "healthy" ]; then
        break
    fi
    if [ "$i" = "40" ]; then
        echo "Database did not become healthy. Check logs:"
        docker compose logs db | tail -40
        exit 1
    fi
    sleep 3
done

# Wait for the API health endpoint
echo "Waiting for the API..."
for i in $(seq 1 40); do
    if curl -sf "http://localhost:8080/api/health" > /dev/null 2>&1; then
        break
    fi
    if [ "$i" = "40" ]; then
        echo "API did not respond on /api/health. Check logs:"
        docker compose logs api | tail -40
        exit 1
    fi
    sleep 3
done

echo ""
echo "=========================================="
echo "Local environment is ready!"
echo "=========================================="
echo ""
echo "API:        http://localhost:8080"
echo "API docs:   http://localhost:8080/docs"
echo "Health:     http://localhost:8080/api/health"
echo ""
echo "PostgreSQL: localhost:${DB_PORT}"
echo "  Database: rerouteher"
echo "  Schema:   rerouteher"
echo "  Username: postgres"
echo "  Password: postgres"
echo ""
echo "To stop:      ./scripts/local-stop.sh"
echo ""

# Tee the API container output to var/local-app.log so the log is inspectable on disk.
# Truncated on each start so the log only covers the current session. Streaming in the
# foreground here keeps the app log attached (Ctrl-C detaches; containers keep running).
LOG_DIR="$(dirname "$0")/../var"
LOG_FILE="${LOG_DIR}/local-app.log"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"
echo "App log: $LOG_FILE"
echo "(Ctrl-C stops following the log; the containers keep running. Use ./scripts/local-stop.sh to stop them.)"
echo ""

docker compose logs -f api 2>&1 | tee "$LOG_FILE"
