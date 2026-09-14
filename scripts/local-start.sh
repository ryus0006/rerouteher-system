#!/bin/bash

# Start the local development environment for rerouteher-system.
#
# Default:  db + API both in Docker (deps live in the image: torch, models, spaCy,
#           pinned scikit-learn, HF offline). Nothing is installed on your host.
# --local:  db in Docker + API via local uvicorn with --reload (fast, hot-reload),
#           but it uses your HOST Python.
#
# Both auto-select host ports (the default if free, else +1): DB 5432->5433,
# API 8080->8081. The chosen API URL is printed as VITE_API_BASE_URL for the
# frontend / e2e. Stop with ./scripts/local-stop.sh

set -e

# Run from the project root regardless of where the script is called from.
cd "$(dirname "$0")/.."

if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Exit 0 when nothing is listening on the port (i.e. it is free).
port_free() {
    python3 -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1', int(sys.argv[1]))); s.close(); sys.exit(0 if r != 0 else 1)" "$1"
}

# The default if free, otherwise the default + 1.
pick_port() {
    if port_free "$1"; then echo "$1"; else echo "$(( $1 + 1 ))"; fi
}

wait_for_db() {
    echo "Waiting for the database to become healthy (first init loads the 333MB dump; this can take several minutes)..."
    DB_CID="$(docker compose ps -q db)"
    for i in $(seq 1 200); do
        STATUS="$(docker inspect --format '{{.State.Health.Status}}' "$DB_CID" 2>/dev/null || echo unknown)"
        if [ "$STATUS" = "healthy" ]; then return 0; fi
        if [ "$i" = "200" ]; then
            echo "Database did not become healthy. Check logs:"
            docker compose logs db | tail -40
            exit 1
        fi
        sleep 3
    done
}

DB_PORT="$(pick_port 5432)"
API_PORT="$(pick_port 8080)"
export DB_PORT
export API_PORT

if [ "$1" = "--local" ]; then
    # ---- db in Docker + API via local uvicorn (uses host Python) ----
    DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:${DB_PORT}/rerouteher"

    echo "Starting pgvector PostgreSQL on host port ${DB_PORT}..."
    docker compose up -d db
    wait_for_db

    echo ""
    echo "=========================================="
    echo "DB:  localhost:${DB_PORT} (database rerouteher)"
    echo "API: http://localhost:${API_PORT}  (local uvicorn, host Python)"
    echo ""
    echo "For the frontend / e2e, use:"
    echo "  VITE_API_BASE_URL=http://localhost:${API_PORT}"
    echo "=========================================="
    echo ""
    echo "Starting API (uvicorn) on port ${API_PORT}. Ctrl-C stops the API; the DB keeps running."
    echo "Stop the DB with: ./scripts/local-stop.sh"
    echo ""

    DATABASE_URL="$DATABASE_URL" exec python3 -m uvicorn app.main:app --port "$API_PORT" --reload
fi

# ---- Default: db + API both in Docker ----
echo "Starting pgvector PostgreSQL and API in Docker (building the API image if needed)..."
echo "First build pulls the model layers (torch, MiniLM, spaCy) and can take a few minutes."
docker compose up -d --build

wait_for_db

echo "Waiting for the API on port ${API_PORT}..."
for i in $(seq 1 40); do
    if curl -sf "http://localhost:${API_PORT}/api/health" > /dev/null 2>&1; then break; fi
    if [ "$i" = "40" ]; then
        echo "API did not respond on /api/health. Check logs:"
        docker compose logs api | tail -40
        exit 1
    fi
    sleep 3
done

echo ""
echo "=========================================="
echo "Local environment is ready (Docker)!"
echo "API:        http://localhost:${API_PORT}  (docs at /docs, health at /api/health)"
echo "PostgreSQL: localhost:${DB_PORT} (database rerouteher, user/pass postgres)"
echo ""
echo "For the frontend / e2e, use:"
echo "  VITE_API_BASE_URL=http://localhost:${API_PORT}"
echo "=========================================="
echo ""
echo "To stop: ./scripts/local-stop.sh"

LOG_DIR="./var"
LOG_FILE="${LOG_DIR}/local-app.log"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"
echo "App log: $LOG_FILE (Ctrl-C detaches; containers keep running)"
echo ""
docker compose logs -f api 2>&1 | tee "$LOG_FILE"
