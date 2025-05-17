#!/bin/bash
# Test the chart service locally

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if we have the necessary data
if [ ! -d "build/cache/btcusd" ]; then
  echo "Missing data directory: build/cache/btcusd"
  echo "Please run scripts/csv_to_parquet.py first to generate the necessary data."
  exit 1
fi

# Start the chart service
echo "Starting chart service on port 8010..."
python -m wave.chart_service &
CHART_PID=$!

# Give it a moment to start
sleep 2

# Test the health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8010/health)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
  echo "✅ Health check passed"
else
  echo "❌ Health check failed"
  kill $CHART_PID
  exit 1
fi

# Test the bars endpoint
echo "Testing bars endpoint..."
BARS_RESPONSE=$(curl -s "http://localhost:8010/bars?symbol=btcusd&tf=1m&window=10")
if [[ $BARS_RESPONSE == *"btcusd"* ]]; then
  echo "✅ Bars endpoint passed"
else
  echo "❌ Bars endpoint failed"
  kill $CHART_PID
  exit 1
fi

# Open the chart in the browser
echo "Opening chart in browser..."
open "http://localhost:8010/chart?symbol=btcusd&tf=1m&window=60"

echo "Test script complete. Press Ctrl+C to stop the chart service."
wait $CHART_PID
