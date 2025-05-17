#!/usr/bin/env bash
set -euo pipefail

# Automated load test using k6

echo "📄 Running load test (k6)"
k6 run scripts/load_test.js

echo "✅ Load test completed"
