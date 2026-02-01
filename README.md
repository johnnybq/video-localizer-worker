# Video Localizer Worker

Vast.ai serverless worker for video localization pipeline.

## Features
- Speech transcription (Whisper)
- Voice cloning
- Lip-sync
- Video processing

## Deployment

Use with Vast.ai Serverless:

```bash
# In Vast.ai Template On-start Script:
curl -sSL https://raw.githubusercontent.com/johnnybq/video-localizer-worker/main/provision.sh | bash
```

## Requirements
- 80GB+ VRAM GPU (A100, H100, H200)
- CUDA 12+
