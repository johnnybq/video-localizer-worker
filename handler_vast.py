"""
TrafficPlant Video Localizer - Pull-Based Worker
=================================================
GPU worker that polls the backend for tasks instead of listening on a port.

Architecture:
  1. Worker starts on Vast.ai GPU instance
  2. Polls POST /api/sota/worker/poll for pending tasks
  3. Processes task using handler.py ML pipeline
  4. Reports result via POST /api/sota/worker/result
  5. Repeats until no tasks for IDLE_SHUTDOWN_SECS

No inbound port required — all connections are outbound from worker.
"""

import os
import sys
import time
import uuid
import logging
import traceback
from typing import Dict, Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

BACKEND_URL = os.environ.get("BACKEND_URL", "http://147.45.170.61:8081")
POLL_URL = f"{BACKEND_URL}/api/sota/worker/poll"
RESULT_URL = f"{BACKEND_URL}/api/sota/worker/result"
PROGRESS_URL = f"{BACKEND_URL}/api/sota/worker/progress"

WORKER_ID = os.environ.get("WORKER_ID", f"vast-{uuid.uuid4().hex[:8]}")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))         # seconds between polls
IDLE_SHUTDOWN_SECS = int(os.environ.get("IDLE_SHUTDOWN", "3600"))  # shutdown after N idle seconds (1 hour default)

# =============================================================================
# Model Preloading
# =============================================================================

def preload_models():
    """
    Preload large models before accepting tasks.
    CogVideoX-5b-I2V is ~20GB and must be downloaded on first run.
    """
    import os

    # CogVideoX for VideoPainter
    hf_home = os.environ.get("HF_HOME", "/workspace/models/huggingface")
    cogvideo_path = os.path.join(hf_home, "hub", "models--THUDM--CogVideoX-5b-I2V")
    cogvideo_alt_path = os.path.join(hf_home, "THUDM/CogVideoX-5b-I2V")

    if os.path.exists(cogvideo_path) or os.path.exists(cogvideo_alt_path):
        log.info(f"✓ CogVideoX-5b-I2V already cached")
    else:
        log.info("=" * 60)
        log.info("Preloading CogVideoX-5b-I2V (~20GB)...")
        log.info("This is required for VideoPainter inpainting.")
        log.info("=" * 60)

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                "THUDM/CogVideoX-5b-I2V",
                cache_dir=hf_home,
                resume_download=True,
            )
            log.info("✓ CogVideoX-5b-I2V preloaded successfully!")
        except Exception as e:
            log.warning(f"⚠ CogVideoX preload failed: {e}")
            log.warning("VideoPainter may fail, ProPainter fallback will be used.")


# =============================================================================
# Import Handler Logic
# =============================================================================

from handler import handler as run_localization, get_model_manager, set_progress_callback


# =============================================================================
# Task Validation
# =============================================================================

def validate_task(task: Dict[str, Any]) -> tuple:
    """
    Validate task structure before processing.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not isinstance(task, dict):
        return False, f"Task must be dict, got {type(task).__name__}"

    if "task_id" not in task:
        return False, "Missing required field: task_id"

    if "payload" not in task:
        return False, "Missing required field: payload"

    payload = task.get("payload", {})
    if not isinstance(payload, dict):
        return False, f"payload must be dict, got {type(payload).__name__}"

    # Required payload fields
    required_fields = ["video_url", "target_language"]
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return False, f"Missing required payload fields: {', '.join(missing)}"

    # Validate video_url is not empty
    if not payload.get("video_url"):
        return False, "payload.video_url is empty"

    # Validate target_language is not empty
    if not payload.get("target_language"):
        return False, "payload.target_language is empty"

    return True, ""


# =============================================================================
# GPU Info
# =============================================================================

def get_gpu_name() -> str:
    """Get GPU model name (for worker identification)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


# =============================================================================
# Poll → Process → Report loop
# =============================================================================

def poll_for_task() -> Dict[str, Any]:
    """
    Poll backend for a pending GPU task.

    Returns:
        {"status": "task", "task_id": ..., "payload": {...}} or
        {"status": "idle"}
    """
    try:
        resp = requests.post(
            POLL_URL,
            json={"worker_id": WORKER_ID, "gpu_name": get_gpu_name()},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError as e:
        log.warning(f"Poll connection error: {e}")
        return {"status": "error", "error": str(e)}
    except Exception as e:
        log.error(f"Poll failed: {e}")
        return {"status": "error", "error": str(e)}


def report_progress(task_id: str, stage: str, elapsed: float, errors: list):
    """Report per-stage progress to backend."""
    try:
        resp = requests.post(
            PROGRESS_URL,
            json={
                "task_id": task_id,
                "worker_id": WORKER_ID,
                "stage": stage,
                "elapsed": elapsed,
                "errors": errors,
            },
            timeout=10,
        )
        log.info(f"Progress: task={task_id} stage={stage} elapsed={elapsed:.1f}s")
    except Exception as e:
        log.warning(f"Failed to report progress for {task_id}/{stage}: {e}")


def report_result(task_id: str, status: str, result: dict):
    """Report task result back to backend."""
    try:
        resp = requests.post(
            RESULT_URL,
            json={
                "task_id": task_id,
                "worker_id": WORKER_ID,
                "status": status,
                "result": result,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        log.info(f"Result reported: task={task_id} status={status} -> {data}")
    except Exception as e:
        log.error(f"Failed to report result for task {task_id}: {e}")


def process_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single GPU task using the ML pipeline.

    Args:
        task: {"task_id": ..., "job_id": ..., "geo": ..., "payload": {...}}

    Returns:
        Result dict from handler.
    """
    # Validate task structure before processing
    is_valid, error_msg = validate_task(task)
    if not is_valid:
        log.error(f"Task validation failed: {error_msg}")
        log.error(f"Task data: {task}")
        return {
            "status": "error",
            "error": f"Task validation failed: {error_msg}",
            "validation_error": True,
        }

    task_id = task["task_id"]
    payload = task["payload"]
    geo = task.get("geo", "?")

    log.info(f"{'=' * 60}")
    log.info(f"Processing task {task_id} | geo={geo} | job={task.get('job_id')}")
    log.info(f"  video: {payload.get('video_url', '?')[:80]}")
    log.info(f"  target_language: {payload.get('target_language')}")
    log.info(f"  voice_clone: {payload.get('voice_clone')}")
    log.info(f"  lipsync: {payload.get('lipsync')}")
    log.info(f"{'=' * 60}")

    # Register per-stage progress callback for this task
    def _on_stage(stage_name, elapsed_secs, errors):
        report_progress(task_id, stage_name, elapsed_secs, [str(e) for e in errors])

    set_progress_callback(_on_stage)

    # Build job_input in the format handler() expects
    job_input = {
        "id": task_id,
        "input": payload,
    }

    start = time.time()
    result = run_localization(job_input)
    elapsed = time.time() - start

    log.info(f"Task {task_id} finished in {elapsed:.1f}s — status: {result.get('status')}")
    return result


# =============================================================================
# Main Loop
# =============================================================================

def main():
    log.info(f"{'=' * 60}")
    log.info(f"TrafficPlant GPU Worker (Pull-Based)")
    log.info(f"  worker_id:    {WORKER_ID}")
    log.info(f"  backend:      {BACKEND_URL}")
    log.info(f"  poll_interval: {POLL_INTERVAL}s")
    log.info(f"  idle_shutdown: {IDLE_SHUTDOWN_SECS}s")
    log.info(f"  gpu:          {get_gpu_name()}")
    log.info(f"{'=' * 60}")

    idle_since = time.time()
    tasks_completed = 0

    while True:
        poll_result = poll_for_task()

        if poll_result.get("status") == "task":
            # Reset idle timer
            idle_since = time.time()

            task_id = poll_result["task_id"]
            try:
                result = process_task(poll_result)
                status = "success" if result.get("status") == "success" else "error"
                report_result(task_id, status, result)
                tasks_completed += 1
            except Exception as e:
                log.error(f"Task {task_id} crashed: {e}")
                log.error(traceback.format_exc())
                report_result(task_id, "error", {"error": str(e), "traceback": traceback.format_exc()})

            # Immediately poll again (might be more tasks)
            continue

        elif poll_result.get("status") == "idle":
            idle_elapsed = time.time() - idle_since
            if idle_elapsed > IDLE_SHUTDOWN_SECS:
                log.info(
                    f"No tasks for {IDLE_SHUTDOWN_SECS}s — shutting down. "
                    f"Completed {tasks_completed} task(s) this session."
                )
                break

        elif poll_result.get("status") == "error":
            # Backend unreachable — back off longer
            log.warning(f"Backend unreachable, waiting {POLL_INTERVAL * 3}s...")
            time.sleep(POLL_INTERVAL * 3)
            continue

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    preload_models()
    main()
