#!/bin/bash
# ==============================================================================
# Video Localizer - Vast.ai Provisioning Script
# ==============================================================================
# This script runs on instance startup to set up the environment.
# Uses Vast.ai base image, pulls code from GitHub, caches deps on volume.
# ==============================================================================

set -e

echo "=== Video Localizer Provisioning ==="
echo "Started at: $(date)"

# Paths
WORKSPACE="/workspace"
APP_DIR="${WORKSPACE}/video-localizer"
CACHE_DIR="${WORKSPACE}/.cache"
MODELS_DIR="${WORKSPACE}/models"

# Create directories
mkdir -p "${CACHE_DIR}" "${MODELS_DIR}" "${APP_DIR}"

# Set up environment
export HF_HOME="${MODELS_DIR}/huggingface"
export TORCH_HOME="${MODELS_DIR}/torch"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export XDG_CACHE_HOME="${CACHE_DIR}"

# Install system dependencies
echo "=== Installing system deps ==="
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone or update code from GitHub (PUBLIC REPO)
echo "=== Fetching code from GitHub ==="
REPO_URL="https://github.com/johnnybq/video-localizer-worker.git"
BRANCH="${GITHUB_BRANCH:-main}"

if [ -d "${APP_DIR}/.git" ]; then
    echo "Updating existing repo..."
    cd "${APP_DIR}"
    git fetch origin
    git reset --hard "origin/${BRANCH}"
else
    echo "Cloning repo..."
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"

# Install Python dependencies (cached on volume)
echo "=== Installing Python dependencies ==="
REQUIREMENTS_HASH=$(md5sum requirements.minimal.txt | cut -d' ' -f1)
INSTALLED_HASH_FILE="${CACHE_DIR}/requirements_hash"

if [ -f "${INSTALLED_HASH_FILE}" ] && [ "$(cat ${INSTALLED_HASH_FILE})" = "${REQUIREMENTS_HASH}" ]; then
    echo "Dependencies already installed (hash match)"
else
    echo "Installing dependencies..."
    pip install --cache-dir="${PIP_CACHE_DIR}" -r requirements.minimal.txt
    pip install --cache-dir="${PIP_CACHE_DIR}" aiohttp
    echo "${REQUIREMENTS_HASH}" > "${INSTALLED_HASH_FILE}"
fi

# Start the server
echo "=== Starting Video Localizer Server ==="
echo "Port: 8080"

exec python -u handler_vast.py
