#!/bin/bash
# TrafficPlant Video Localizer - Start Script
# This script is called by Vast.ai's template

set -e

LOG_DIR="/var/log/portal"
mkdir -p "$LOG_DIR"

echo "=== TrafficPlant Video Localizer Starting ===" | tee "$LOG_DIR/localizer.log"

# Start model server in background, logging to file
cd /app
python3 -u handler_vast.py 2>&1 | tee -a "$LOG_DIR/localizer.log" &
MODEL_PID=$!

echo "Model server started (PID: $MODEL_PID)" | tee -a "$LOG_DIR/localizer.log"

# Wait for model server to be ready
echo "Waiting for model server..." | tee -a "$LOG_DIR/localizer.log"
for i in {1..60}; do
    if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
        echo "Video Localizer ready" | tee -a "$LOG_DIR/localizer.log"
        break
    fi
    sleep 2
done

# Run PyWorker (this blocks)
echo "Starting PyWorker..." | tee -a "$LOG_DIR/localizer.log"
python3 /app/worker.py
