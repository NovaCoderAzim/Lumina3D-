#!/usr/bin/env bash
# Runs the Lumina3D backend in full mock mode (no GPU/COLMAP/weights).
set -euo pipefail

export USE_MOCK_VISION=true
export USE_MOCK_RECONSTRUCTION=true
export USE_MOCK_AI=true
export USE_MOCK_ANALYTICS=true
export PYTHONUTF8=1

cd "$(dirname "$0")/.."
exec python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
