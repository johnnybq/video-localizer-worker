# ==============================================================================
# TrafficPlant Video Localizer - FINAL IMAGE
# ==============================================================================
# Lightweight image that extends base with application code only.
# Build time: ~2-3 minutes (just copies handler files)
#
# Requires: johnnybq/video-localizer-base:v1 (built separately)
# ==============================================================================
FROM johnnybq/video-localizer-base:v1

LABEL maintainer="johnnybq"
LABEL description="TrafficPlant Video Localizer worker for Vast.ai"

WORKDIR /app

# ==============================================================================
# Application code only — base image has all dependencies
# ==============================================================================
COPY handler.py handler_vast.py ./

# Quick sanity check (imports already verified in base)
RUN python3 -c "import handler; print('handler.py loaded OK')"

# Pull-based: worker polls backend, no port needed
ENTRYPOINT []
CMD ["python3", "-u", "handler_vast.py"]
