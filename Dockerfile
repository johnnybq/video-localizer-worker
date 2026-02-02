# Video Localizer - Vast.ai Serverless with PyWorker
FROM vastai/pytorch:2.10.0-cuda-13.0.2-py312-24.04

# Install system deps + python symlink (needed for some package builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy all files
COPY handler.py handler_vast.py worker.py start-server.sh requirements.minimal.txt ./

# Install Python deps
# Fix NumPy version first (vastai base has NumPy 2.x which breaks pandas build)
# Then install pandas binary wheel, then other deps
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir -r requirements.minimal.txt aiohttp

# Make start script executable
RUN chmod +x start-server.sh

# Create log directory
RUN mkdir -p /var/log/portal

# Environment
ENV HF_HOME=/workspace/models/huggingface
ENV TORCH_HOME=/workspace/models/torch

EXPOSE 8080

# Use our start script (Vast.ai will NOT override this with ENTRYPOINT [])
ENTRYPOINT []
CMD ["/app/start-server.sh"]
