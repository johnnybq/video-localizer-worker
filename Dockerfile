# ==============================================================================
# TrafficPlant Video Localizer - Vast.ai Pull-Based Worker
# ==============================================================================
# Base: PyTorch 2.10.0, CUDA 13.0, Python 3.12
# Includes: torch, torchvision 0.25.0, torchaudio, transformers, diffusers,
#           accelerate, xformers
# ==============================================================================
FROM vastai/pytorch:2.10.0-cuda-13.0.2-py312-24.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && git lfs install

WORKDIR /app

# ==============================================================================
# Phase 1: Python dependencies from requirements
# ==============================================================================
COPY requirements.minimal.txt ./

# Fix NumPy version first (vastai base has NumPy 2.x which breaks some builds)
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir -r requirements.minimal.txt

# ==============================================================================
# Phase 2: VideoPainter — custom diffusers fork + model checkpoints
# ==============================================================================
# VideoPainter requires a modified diffusers with CogVideoXI2VDualInpaintAnyLPipeline
RUN git clone --depth 1 https://github.com/TencentARC/VideoPainter.git /opt/videopainter && \
    cd /opt/videopainter/diffusers && \
    pip install --no-cache-dir -e .

# Download VideoPainter checkpoints (context encoder + LoRA adapter)
# Models are cached in /workspace/models for persistence across runs
RUN mkdir -p /workspace/models/videopainter && \
    git clone --depth 1 https://huggingface.co/TencentARC/VideoPainter /workspace/models/videopainter/checkpoints

# CogVideoX-5b-I2V base model will be downloaded on first run via HF cache
# (too large for Docker image — ~20GB)

# ==============================================================================
# Phase 3: Application code
# ==============================================================================
COPY handler.py handler_vast.py ./

# ==============================================================================
# Phase 4: Verify critical imports (fail build if broken)
# ==============================================================================
RUN python3 -c "\
import sys; \
print(f'Python {sys.version}'); \
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}'); \
from paddleocr import PaddleOCR; print('paddleocr OK'); \
from faster_whisper import WhisperModel; print('faster-whisper OK'); \
import decord; print('decord OK'); \
from sam2.sam2_video_predictor import SAM2VideoPredictor; print('sam2 OK'); \
import demucs.api; print('demucs OK'); \
import kornia; print('kornia OK'); \
from peft import PeftModel; print('peft OK'); \
import pyiqa; print('pyiqa OK'); \
print('All critical imports verified.') \
"

# Create log directory
RUN mkdir -p /var/log

# Environment
ENV HF_HOME=/workspace/models/huggingface
ENV TORCH_HOME=/workspace/models/torch
ENV VIDEOPAINTER_ROOT=/opt/videopainter
ENV VIDEOPAINTER_CKPT=/workspace/models/videopainter/checkpoints

# Pull-based: worker polls backend, no port needed
ENTRYPOINT []
CMD ["python3", "-u", "handler_vast.py"]
