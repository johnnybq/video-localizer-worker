"""
TrafficPlant Video Localizer - Vast.ai PyWorker Handler
=========================================================
Implements Vast.ai PyWorker EndpointHandler for serverless video localization.
"""

import os
import sys
import time
import json
import uuid
import logging
import dataclasses
from typing import Dict, Any, Optional, Type, Union

from aiohttp import web, ClientResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclasses.dataclass
class LocalizationPayload:
    """Input payload for video localization."""
    video_url: str
    source_language: str = "auto"
    target_language: str = "en"
    voice_clone: bool = True
    lipsync: bool = True
    lipsync_quality: str = "high"
    upscale: bool = False
    face_enhance: bool = True
    quality_threshold: float = 0.6
    stages: Optional[list] = None
    callback_url: Optional[str] = None
    translated_text: Optional[str] = None

    @classmethod
    def for_test(cls) -> "LocalizationPayload":
        """Create a test payload for benchmarking."""
        return cls(
            video_url="https://pub-c025ef96f40e47aab26156a1874f64bc.r2.dev/test/sample.mp4",
            stages=["detect_text", "transcribe"]
        )


# =============================================================================
# Try to import PyWorker (Vast.ai SDK)
# =============================================================================

try:
    from lib.backend import Backend, LogAction
    from lib.data_types import EndpointHandler, JsonDataException
    from lib.server import start_server
    PYWORKER_AVAILABLE = True
except ImportError:
    log.warning("PyWorker not available, using standalone mode")
    PYWORKER_AVAILABLE = False

    # Stub classes for standalone mode
    class EndpointHandler:
        pass

    class JsonDataException(Exception):
        pass


# =============================================================================
# Import Handler Logic
# =============================================================================

from handler import handler as run_localization, get_model_manager


# =============================================================================
# PyWorker Endpoint Handler
# =============================================================================

if PYWORKER_AVAILABLE:

    @dataclasses.dataclass
    class LocalizationHandler(EndpointHandler[LocalizationPayload]):
        """PyWorker handler for video localization endpoint."""

        benchmark_runs: int = 1
        benchmark_words: int = 100

        @property
        def endpoint(self) -> str:
            return "/localize"

        @property
        def healthcheck_endpoint(self) -> Optional[str]:
            return "http://0.0.0.0:8080/health"

        @classmethod
        def payload_cls(cls) -> Type[LocalizationPayload]:
            return LocalizationPayload

        def generate_payload_json(self, payload: LocalizationPayload) -> Dict[str, Any]:
            """Convert payload to JSON for model API."""
            return dataclasses.asdict(payload)

        def make_benchmark_payload(self) -> LocalizationPayload:
            """Create payload for performance benchmarking."""
            return LocalizationPayload.for_test()

        async def generate_client_response(
            self, client_request: web.Request, model_response: ClientResponse
        ) -> Union[web.Response, web.StreamResponse]:
            """Handle response from model server."""
            if model_response.status == 200:
                data = await model_response.json()
                return web.json_response(data=data)
            else:
                error_text = await model_response.text()
                return web.json_response(
                    {"error": error_text},
                    status=model_response.status
                )


# =============================================================================
# Standalone HTTP Server (when PyWorker not available)
# =============================================================================

async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    import torch
    return web.json_response({
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


async def handle_localize(request: web.Request) -> web.Response:
    """Handle localization request."""
    try:
        data = await request.json()
        payload = LocalizationPayload(**data)
    except Exception as e:
        return web.json_response({"error": f"Invalid request: {e}"}, status=400)

    job_id = str(uuid.uuid4())[:8]

    # Run localization
    job_input = {
        "id": job_id,
        "input": dataclasses.asdict(payload)
    }

    try:
        result = run_localization(job_input)
        return web.json_response({"job_id": job_id, **result})
    except Exception as e:
        log.error(f"Localization failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


def run_standalone_server():
    """Run standalone HTTP server (no PyWorker)."""
    log.info("Starting standalone HTTP server...")

    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_post("/localize", handle_localize)
    app.router.add_get("/ping", lambda _: web.Response(text="pong"))

    port = int(os.environ.get("PORT", "8080"))
    web.run_app(app, host="0.0.0.0", port=port)


# =============================================================================
# Model Server (for PyWorker backend)
# =============================================================================

async def model_server_handler(request: web.Request) -> web.Response:
    """Internal model server that processes localization requests."""
    try:
        data = await request.json()

        job_id = str(uuid.uuid4())[:8]
        job_input = {"id": job_id, "input": data}

        result = run_localization(job_input)
        return web.json_response({"job_id": job_id, **result})

    except Exception as e:
        log.error(f"Model server error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def model_health_handler(request: web.Request) -> web.Response:
    """Model server health check."""
    import torch
    return web.json_response({
        "status": "ok",
        "gpu": torch.cuda.is_available()
    })


def run_model_server():
    """Run internal model server (port 8080)."""
    log.info("Starting model server on port 8080...")

    app = web.Application()
    app.router.add_post("/localize", model_server_handler)
    app.router.add_get("/health", model_health_handler)

    web.run_app(app, host="0.0.0.0", port=8080)


# =============================================================================
# PyWorker Backend Setup
# =============================================================================

def run_pyworker_server():
    """Run PyWorker backend server."""
    log.info("Starting PyWorker backend...")

    backend = Backend(
        model_server_url="http://0.0.0.0:8080",
        model_log_file=os.environ.get("MODEL_LOG", "/var/log/model.log"),
        allow_parallel_requests=False,  # GPU can only handle one at a time
        benchmark_handler=LocalizationHandler(benchmark_runs=1),
        log_actions=[
            (LogAction.ModelLoaded, "Video Localizer ready"),
            (LogAction.ModelError, "Localizer error"),
            (LogAction.Info, "Processing video"),
        ]
    )

    routes = [
        web.post("/localize", backend.create_handler(LocalizationHandler())),
        web.get("/health", lambda _: web.Response(text="ok")),
        web.get("/ping", lambda _: web.Response(text="pong")),
    ]

    start_server(backend, routes)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import multiprocessing

    mode = os.environ.get("VAST_MODE", "standalone")

    if mode == "pyworker" and PYWORKER_AVAILABLE:
        # Run both model server and PyWorker backend
        model_proc = multiprocessing.Process(target=run_model_server)
        model_proc.start()

        time.sleep(5)  # Wait for model server to start

        run_pyworker_server()
    else:
        # Standalone mode - simple HTTP server
        run_standalone_server()
