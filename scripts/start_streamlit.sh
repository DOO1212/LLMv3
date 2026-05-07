#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/LLM"
APP_FILE="$ROOT_DIR/app.py"
STREAMLIT_BIN="$ROOT_DIR/.venv/bin/streamlit"

cd "$ROOT_DIR"

exec "$STREAMLIT_BIN" run "$APP_FILE" \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  --browser.gatherUsageStats false
