# Video Localizer - Vast.ai Serverless
FROM vastai/pytorch:2.10.0-cuda-13.0.2-py312-24.04

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy handler files
COPY handler.py handler_vast.py requirements.minimal.txt ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.minimal.txt aiohttp

# Environment
ENV HF_HOME=/workspace/models/huggingface
ENV TORCH_HOME=/workspace/models/torch

EXPOSE 8080

CMD ["python3", "-u", "handler_vast.py"]
