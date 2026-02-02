"""
TrafficPlant Video Localizer - Vast.ai PyWorker
================================================
Configures the PyWorker to proxy requests to our model server.
"""

import os

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

# --- Model Server Configuration ---
# Our handler runs on port 8080 inside the container
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 8080
MODEL_LOG_FILE = "/var/log/portal/localizer.log"

# --- Log Patterns ---
# These patterns tell PyWorker when our model server is ready/errored
MODEL_LOAD_LOG_MSG = [
    "Video Localizer ready",
    "Starting standalone HTTP server",
    "Serving on http://0.0.0.0:8080",
]

MODEL_ERROR_LOG_MSGS = [
    "Traceback (most recent call last):",
    "RuntimeError:",
    "CUDA out of memory",
    "ModuleNotFoundError:",
]

MODEL_INFO_LOG_MSGS = [
    "Loading model",
    "Processing video",
    "Transcribing",
]


# --- Benchmark Generator ---
def localize_benchmark_generator() -> dict:
    """Generate a minimal benchmark payload for /localize endpoint."""
    return {
        "video_url": "https://pub-c025ef96f40e47aab26156a1874f64bc.r2.dev/campaigns/1/source/posts/3630685768335345688.mp4",
        "stages": ["transcribe"],  # Only transcribe for benchmark (fast)
        "source_language": "auto",
        "target_language": "en",
    }


# --- Workload Calculator ---
def localize_workload(payload: dict) -> float:
    """Estimate workload based on requested stages."""
    stages = payload.get("stages", ["transcribe", "translate", "tts", "lipsync"])

    # Base cost per stage
    stage_costs = {
        "preprocess": 10,
        "detect_text": 20,
        "create_mask": 30,
        "inpaint": 100,
        "transcribe": 50,
        "translate": 10,
        "tts": 40,
        "lipsync": 80,
        "enhance": 20,
        "upscale": 30,
        "assemble": 10,
    }

    total = sum(stage_costs.get(s, 10) for s in stages)
    return float(total)


# --- Worker Configuration ---
worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,

    handlers=[
        # Main localization endpoint with benchmark
        HandlerConfig(
            route="/localize",
            allow_parallel_requests=False,  # GPU processes one video at a time
            max_queue_time=300.0,  # 5 min queue timeout
            workload_calculator=localize_workload,
            benchmark_config=BenchmarkConfig(
                generator=localize_benchmark_generator,
                runs=1,  # Single benchmark run (video processing is slow)
                concurrency=1,
            ),
        ),

        # Health check endpoint (no benchmark)
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            workload_calculator=lambda _: 1.0,
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

# Run the worker
if __name__ == "__main__":
    Worker(worker_config).run()
