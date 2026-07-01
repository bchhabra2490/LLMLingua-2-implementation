#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-5001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Python not found. Create a venv at $REPO_ROOT/.venv first."
  exit 1
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm ci --prefix "$ROOT_DIR/frontend"
fi

cleanup() {
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting backend on http://localhost:$BACKEND_PORT"
(
  cd "$ROOT_DIR/backend"
  PORT="$BACKEND_PORT" "$PYTHON" app.py
) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:$FRONTEND_PORT"
(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo
echo "App ready:"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Backend:  http://localhost:$BACKEND_PORT"
echo
echo "Press Ctrl+C to stop both."

wait "$BACKEND_PID" "$FRONTEND_PID"
