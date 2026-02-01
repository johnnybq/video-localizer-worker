#!/bin/bash
# ==============================================================================
# Video Localizer - Vast.ai Provisioning Script
# ==============================================================================
set -e

echo "=== Video Localizer Provisioning ==="
echo "Started at: $(date)"

APP_DIR="/workspace/video-localizer"
mkdir -p "$APP_DIR"

# Install system deps
echo "=== Installing system deps ==="
apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

# Download files via curl (git has issues on Vast.ai)
echo "=== Downloading code ==="
cd "$APP_DIR"
BASE_URL="https://raw.githubusercontent.com/johnnybq/video-localizer-worker/main"
curl -sSLO "$BASE_URL/handler.py"
curl -sSLO "$BASE_URL/handler_vast.py"
curl -sSLO "$BASE_URL/requirements.minimal.txt"

# Install Python deps
echo "=== Installing Python dependencies ==="
pip install --no-cache-dir -r requirements.minimal.txt aiohttp

# Start server
echo "=== Starting Video Localizer Server ==="
exec python3 -u handler_vast.py
