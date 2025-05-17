#!/usr/bin/env bash
set -euo pipefail
set -x

# Validate end-to-end flows under load

echo "📄 Starting load test in background (timeout 5m)"
timeout 5m bash scripts/test-load.sh &
LOAD_PID=$!
echo "🔍 Load test PID: $LOAD_PID"

# Wait a bit to ramp up load
sleep 10

echo "📄 Running E2E tests under load"
pytest tests/test_ui_*.py tests/test_streaming_ws.py -q

# Wait for load test to complete
echo "📄 Waiting for load test to finish"
wait $LOAD_PID

echo "✅ E2E validation under load passed"
