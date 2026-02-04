# Video Localizer - Vast.ai Pull-Based Worker
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
COPY handler.py handler_vast.py requirements.minimal.txt ./

# Install Python deps
# Fix NumPy version first (vastai base has NumPy 2.x which breaks pandas build)
# Then install pandas binary wheel, then other deps
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir -r requirements.minimal.txt

# Create log directory
RUN mkdir -p /var/log

# Environment
ENV HF_HOME=/workspace/models/huggingface
ENV TORCH_HOME=/workspace/models/torch

# Pull-based: worker polls backend, no port needed
ENTRYPOINT []
CMD ["python3", "-u", "handler_vast.py"]
