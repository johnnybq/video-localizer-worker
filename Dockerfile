# ==============================================================================
# TrafficPlant Video Localizer - FINAL IMAGE
# ==============================================================================
# Lightweight image that extends base with application code only.
# Build time: ~3-5 minutes (upgrades transformers + copies handler files)
#
# Requires: johnnybq/video-localizer-base:v2 (with VideoPainter + custom diffusers)
# ==============================================================================
FROM johnnybq/video-localizer-base:v2

LABEL maintainer="johnnybq"
LABEL description="TrafficPlant Video Localizer worker for Vast.ai"

WORKDIR /app

# ==============================================================================
# Upgrade transformers for DeepSeek-OCR-2 (base image has 4.42.2 for VideoPainter)
# DeepSeek-OCR-2 requires >=4.46.3 for model format support.
# VideoPainter compat: FLAX_WEIGHTS_NAME shim is applied at runtime in handler.py
# ==============================================================================
RUN pip install --no-cache-dir "transformers>=4.47" "addict" "easydict" && \
    rm -rf /root/.cache/pip /tmp/*

# ==============================================================================
# Application code only
# ==============================================================================
COPY handler.py handler_vast.py ./

# Pull-based: worker polls backend, no port needed
ENTRYPOINT []
CMD ["python3", "-u", "handler_vast.py"]
