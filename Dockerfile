# ==============================================================================
# TrafficPlant Video Localizer - FINAL IMAGE
# ==============================================================================
# Lightweight image that extends base with application code only.
# Build time: ~2-3 minutes (just copies handler files)
#
# Requires: johnnybq/video-localizer-base:v2 (with VideoPainter + custom diffusers)
# ==============================================================================
FROM johnnybq/video-localizer-base:v2

LABEL maintainer="johnnybq"
LABEL description="TrafficPlant Video Localizer worker for Vast.ai"

WORKDIR /app

# ==============================================================================
# Application code only — base image has all dependencies
# ==============================================================================
COPY handler.py handler_vast.py ./

# Skip sanity check — imports verified in base, avoid slow torch init
# RUN python3 -c "import handler; print('handler.py loaded OK')"

# Pull-based: worker polls backend, no port needed
ENTRYPOINT []
CMD ["python3", "-u", "handler_vast.py"]
