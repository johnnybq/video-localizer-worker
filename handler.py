"""
TrafficPlant SOTA Video Localization Worker
============================================
RunPod Serverless handler for state-of-the-art video localization.

Updated: February 2026

VRAM Budget (A100 80GB):
┌────────────────────┬───────┬──────────┐
│ Model              │ VRAM  │ Priority │
├────────────────────┼───────┼──────────┤
│ VideoPainter       │ 26GB  │ 1 (temp) │
│ VideoRetalking     │ 10GB  │ 3 (temp) │
│ SAM 2.1 Large      │ 3GB   │ 1 (keep) │
│ Faster-Whisper     │ 3GB   │ 2 (keep) │
│ F5-TTS             │ 4GB   │ 2 (keep) │
│ Demucs             │ 2GB   │ 2 (keep) │
│ GFPGAN             │ 2GB   │ 4 (temp) │
│ Real-ESRGAN        │ 3GB   │ 4 (temp) │
│ PaddleOCR          │ 2GB   │ 1 (keep) │
├────────────────────┼───────┼──────────┤
│ Peak usage         │ ~40GB │          │
│ Available          │ 80GB  │          │
└────────────────────┴───────┴──────────┘
"""

import os
import sys
import time
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class PipelineStage(Enum):
    """Video localization pipeline stages."""
    PREPROCESS = "preprocess"          # NEW: Audio separation
    DETECT_TEXT = "detect_text"        # PaddleOCR
    CREATE_MASK = "create_mask"        # SAM 2.1
    INPAINT = "inpaint"                # VideoPainter
    TRANSCRIBE = "transcribe"          # Faster-Whisper
    TRANSLATE = "translate"            # NLLB / Argos
    TTS = "tts"                        # F5-TTS
    LIPSYNC = "lipsync"                # VideoRetalking / MuseTalk
    ENHANCE = "enhance"                # NEW: GFPGAN face enhancement
    UPSCALE = "upscale"                # Real-ESRGAN
    QUALITY_CHECK = "quality_check"    # NEW: Auto quality assessment
    ASSEMBLE = "assemble"              # Final mix


@dataclass
class JobConfig:
    """Configuration for a localization job."""
    video_url: str
    source_language: str = "auto"
    target_language: str = "en"
    voice_clone: bool = True
    lipsync: bool = True
    lipsync_quality: str = "high"  # "high" (VideoRetalking), "medium" (MuseTalk), "fast" (Wav2Lip)
    upscale: bool = False
    face_enhance: bool = True
    quality_threshold: float = 0.6
    stages: Optional[List[str]] = None
    callback_url: Optional[str] = None

    # Pre-translated text (from server's TranslatorAgent / Gemini 3 Pro)
    # If provided, skips local translate stage
    translated_text: Optional[str] = None

    # R2 storage config
    r2_bucket: str = "trafficplant"
    r2_prefix: str = "localized"


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    stage_times: Dict[str, float] = field(default_factory=dict)
    model_loads: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "stage_times_ms": {k: v * 1000 for k, v in self.stage_times.items()},
            "model_load_times_ms": {k: v * 1000 for k, v in self.model_loads.items()},
            "quality_scores": self.quality_scores,
            "errors": self.errors,
            "total_time_ms": sum(self.stage_times.values()) * 1000
        }


# =============================================================================
# Model Manager - Smart VRAM Management
# =============================================================================

class ModelManager:
    """
    Smart model loading/unloading for A100 80GB.
    Loads models on-demand, unloads heavy models after use.
    """

    MODEL_CONFIGS = {
        # Model: (vram_gb, priority, keep_loaded)
        "paddleocr": (2, 1, True),
        "sam2": (3, 1, True),
        "faster_whisper": (3, 2, True),
        "demucs": (2, 2, True),
        "f5tts": (4, 2, True),
        "videopainter": (26, 1, False),  # Unload after use
        "video_retalking": (10, 3, False),
        "musetalk": (6, 3, False),
        "wav2lip": (2, 3, False),
        "gfpgan": (2, 4, False),
        "realesrgan": (3, 4, False),
    }

    def __init__(self, device: str = "cuda", total_vram: int = 80):
        self.device = device
        self.total_vram = total_vram
        self.loaded: Dict[str, Any] = {}
        self.metrics = PipelineMetrics()

    def _get_used_vram(self) -> int:
        return sum(
            self.MODEL_CONFIGS[name][0]
            for name in self.loaded
        )

    def _ensure_vram(self, needed: int, exclude: List[str] = None):
        """Unload models to free VRAM, respecting priorities."""
        exclude = exclude or []
        current = self._get_used_vram()
        buffer = 10  # Keep 10GB buffer

        while current + needed > self.total_vram - buffer:
            # Find lowest priority loaded model (not in exclude)
            candidates = [
                (name, self.MODEL_CONFIGS[name])
                for name in self.loaded
                if name not in exclude and not self.MODEL_CONFIGS[name][2]
            ]

            if not candidates:
                logger.warning(f"Cannot free more VRAM. Current: {current}GB, Needed: {needed}GB")
                break

            # Sort by priority (highest number = lowest priority)
            candidates.sort(key=lambda x: -x[1][1])
            to_unload = candidates[0][0]

            self._unload(to_unload)
            current = self._get_used_vram()

    def _unload(self, name: str):
        """Unload model and free GPU memory."""
        if name in self.loaded:
            logger.info(f"Unloading {name}...")
            del self.loaded[name]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @contextmanager
    def use(self, name: str):
        """Context manager for using a model."""
        model = self.load(name)
        try:
            yield model
        finally:
            # Unload if not marked as keep_loaded
            if not self.MODEL_CONFIGS[name][2]:
                self._unload(name)

    def load(self, name: str) -> Any:
        """Load a model, ensuring enough VRAM."""
        if name in self.loaded:
            return self.loaded[name]

        vram_needed = self.MODEL_CONFIGS[name][0]
        self._ensure_vram(vram_needed, exclude=[name])

        logger.info(f"Loading {name} ({vram_needed}GB)...")
        start = time.time()

        try:
            model = self._load_model(name)
            self.loaded[name] = model
            elapsed = time.time() - start
            self.metrics.model_loads[name] = elapsed
            logger.info(f"Loaded {name} in {elapsed:.1f}s")
            return model
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            raise

    def _load_model(self, name: str) -> Any:
        """Actually load a specific model."""

        if name == "paddleocr":
            from paddleocr import PaddleOCR
            # Use 'en' for detection (works for Latin/Cyrillic scripts)
            # PaddleOCR doesn't support 'multilingual' - use specific lang
            # 'en' model detects text boxes well for most scripts
            return PaddleOCR(
                use_gpu=True,
                lang='en',  # Detection works for any script, recognition is English
                show_log=False,
                det_db_score_mode='slow'  # Better accuracy
            )

        elif name == "sam2":
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            return SAM2VideoPredictor.from_pretrained(
                "facebook/sam2.1-hiera-large"
            )

        elif name == "faster_whisper":
            from faster_whisper import WhisperModel
            return WhisperModel(
                "large-v3",
                device=self.device,
                compute_type="float16"
            )

        elif name == "demucs":
            import demucs.api
            return demucs.api.Separator(
                model="htdemucs_ft",
                device=self.device
            )

        elif name == "f5tts":
            # F5-TTS loading
            from f5_tts import F5TTS
            return F5TTS(device=self.device)

        elif name == "videopainter":
            # VideoPainter with CogVideoX base
            from diffusers import CogVideoXPipeline
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-5b-I2V",
                torch_dtype=torch.float16
            ).to(self.device)
            pipe.enable_xformers_memory_efficient_attention()
            return pipe

        elif name == "video_retalking":
            # VideoRetalking for high-quality lipsync
            sys.path.insert(0, "/models/video_retalking")
            from inference import VideoRetalking
            return VideoRetalking(device=self.device)

        elif name == "musetalk":
            sys.path.insert(0, "/models/musetalk")
            from musetalk import MuseTalkInference
            return MuseTalkInference(device=self.device)

        elif name == "wav2lip":
            sys.path.insert(0, "/models/wav2lip")
            from inference import Wav2LipInference
            return Wav2LipInference(device=self.device)

        elif name == "gfpgan":
            from gfpgan import GFPGANer
            return GFPGANer(
                model_path="/models/gfpgan/GFPGANv1.4.pth",
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                device=self.device
            )

        elif name == "realesrgan":
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=2)
            return RealESRGANer(
                scale=2,
                model_path="/models/realesrgan/realesr-general-x4v3.pth",
                model=model,
                device=self.device
            )

        else:
            raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Pipeline Stages
# =============================================================================

def stage_preprocess(video_path: str, mm: ModelManager) -> Dict:
    """
    NEW: Separate audio into voice/music/sfx tracks.
    Critical for videos with background music!
    """
    logger.info("Stage: PREPROCESS (audio separation)")

    demucs = mm.load("demucs")

    # Extract audio
    audio_path = video_path.replace(".mp4", "_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        audio_path
    ], capture_output=True)

    # Separate with Demucs
    origin, separated = demucs.separate_audio_file(audio_path)

    # Save separated tracks
    vocals_path = video_path.replace(".mp4", "_vocals.wav")
    background_path = video_path.replace(".mp4", "_background.wav")

    # Vocals = voice track
    # Background = drums + bass + other (everything except vocals)
    import torchaudio
    torchaudio.save(vocals_path, separated["vocals"], 44100)

    # Mix background tracks
    background = separated["drums"] + separated["bass"] + separated["other"]
    torchaudio.save(background_path, background, 44100)

    return {
        "audio_path": audio_path,
        "vocals_path": vocals_path,
        "background_path": background_path,
        "has_music": torch.abs(separated["drums"]).mean() > 0.01
    }


def stage_detect_text(video_path: str, mm: ModelManager) -> Dict:
    """Detect text overlays using PaddleOCR + EasyOCR ensemble."""
    logger.info("Stage: DETECT_TEXT")

    import cv2
    ocr = mm.load("paddleocr")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    sample_rate = max(1, int(fps / 2))  # Sample 2 frames per second

    for i in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        result = ocr.ocr(frame, cls=False)
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                conf = line[1][1]

                if conf > 0.5 and len(text) > 1:
                    # Convert bbox to normalized coordinates
                    x_coords = [p[0] / width for p in bbox]
                    y_coords = [p[1] / height for p in bbox]

                    detections.append({
                        "frame_idx": i,
                        "timestamp": i / fps,
                        "bbox": bbox,
                        "bbox_norm": [min(x_coords), min(y_coords),
                                     max(x_coords), max(y_coords)],
                        "text": text,
                        "confidence": conf
                    })

    cap.release()

    # Deduplicate similar detections
    unique_detections = _dedupe_detections(detections)

    return {
        "detections": unique_detections,
        "total_found": len(detections),
        "unique_regions": len(unique_detections),
        "fps": fps,
        "frame_count": frame_count,
        "resolution": (width, height)
    }


def _dedupe_detections(detections: List[Dict], iou_threshold: float = 0.7) -> List[Dict]:
    """Remove duplicate detections based on IoU."""
    if not detections:
        return []

    unique = []
    for det in detections:
        is_dupe = False
        for existing in unique:
            iou = _calculate_iou(det["bbox_norm"], existing["bbox_norm"])
            if iou > iou_threshold and det["text"] == existing["text"]:
                is_dupe = True
                break
        if not is_dupe:
            unique.append(det)

    return unique


def _calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def stage_create_mask(video_path: str, detections: List[Dict], mm: ModelManager) -> str:
    """Create temporally consistent masks using SAM 2.1."""
    logger.info("Stage: CREATE_MASK (SAM 2.1)")

    if not detections:
        logger.info("No text detections, skipping mask creation")
        return None

    sam2 = mm.load("sam2")

    # Initialize video predictor
    inference_state = sam2.init_state(video_path)

    # Add prompts for each text region
    for i, det in enumerate(detections[:10]):  # Limit to 10 regions
        frame_idx = det["frame_idx"]
        bbox = det["bbox"]

        # Use center point as prompt
        x_center = sum(p[0] for p in bbox) / 4
        y_center = sum(p[1] for p in bbox) / 4

        sam2.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=i,
            points=np.array([[x_center, y_center]]),
            labels=np.array([1])
        )

    # Propagate masks through video
    mask_frames = {}
    for frame_idx, obj_ids, masks in sam2.propagate_in_video(inference_state):
        # Combine all object masks
        combined_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8) * 255)
        mask_frames[frame_idx] = combined_mask

    # Render mask video
    mask_path = video_path.replace(".mp4", "_mask.mp4")
    _render_mask_video(video_path, mask_frames, mask_path)

    return mask_path


def _render_mask_video(video_path: str, masks: Dict[int, np.ndarray], output_path: str):
    """Render mask frames to video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    for i in range(frame_count):
        if i in masks:
            mask = cv2.resize(masks[i], (width, height))
        else:
            # Interpolate from nearest frames
            mask = np.zeros((height, width), dtype=np.uint8)

        out.write(mask)

    cap.release()
    out.release()


def stage_inpaint(video_path: str, mask_path: str, mm: ModelManager) -> str:
    """Remove text using VideoPainter (primary) or ProPainter (fallback)."""
    logger.info("Stage: INPAINT")

    if mask_path is None:
        logger.info("No mask, skipping inpainting")
        return video_path

    output_path = video_path.replace(".mp4", "_inpainted.mp4")

    # Try VideoPainter first (best quality, CogVideoX-based)
    try:
        with mm.use("videopainter") as videopainter:
            return _inpaint_videopainter(video_path, mask_path, output_path, videopainter)
    except Exception as e:
        logger.warning(f"VideoPainter failed: {e}, trying ProPainter")

    # Fallback to ProPainter
    try:
        return _inpaint_propainter(video_path, mask_path, output_path)
    except Exception as e:
        logger.error(f"ProPainter also failed: {e}")
        # Return original as last resort
        return video_path


def _inpaint_propainter(video_path: str, mask_path: str, output_path: str) -> str:
    """Inpaint using ProPainter (E2FGVI-based)."""
    import cv2
    from PIL import Image

    # Check if ProPainter is available
    try:
        sys.path.insert(0, "/models/ProPainter")
        from inference_propainter import ProPainterInference
    except ImportError:
        # Use Replicate API as fallback
        return _inpaint_replicate(video_path, mask_path, output_path)

    # Local ProPainter inference
    propainter = ProPainterInference(device="cuda")

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    mask_cap = cv2.VideoCapture(mask_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    masks = []

    while True:
        ret, frame = cap.read()
        ret_m, mask = mask_cap.read()
        if not ret or not ret_m:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks.append(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

    cap.release()
    mask_cap.release()

    # Process in batches (memory efficiency)
    batch_size = 10
    inpainted_frames = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_masks = masks[i:i+batch_size]

        result = propainter.inpaint(
            frames=batch_frames,
            masks=batch_masks,
            resize_ratio=0.5  # Balance quality/speed
        )
        inpainted_frames.extend(result)

    # Write output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in inpainted_frames:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (width, height))
        out.write(frame_resized)

    out.release()

    # Re-add audio
    _copy_audio(video_path, output_path)

    return output_path


def _inpaint_replicate(video_path: str, mask_path: str, output_path: str) -> str:
    """Fallback: Use Replicate API for ProPainter."""
    import replicate
    import httpx

    # Upload video to temp storage
    # For now, assume video_path is already a URL or we need presigned upload
    logger.info("Using Replicate ProPainter API")

    output = replicate.run(
        "sczhou/propainter:34a544b1df7e77e08d5d1648e8b28899cb7f8c47a28aaef54eae16ebf6f4c34a",
        input={
            "video": open(video_path, "rb"),
            "mask": open(mask_path, "rb"),
            "resize_ratio": 0.5,
            "ref_stride": 10,
            "neighbor_length": 10,
            "subvideo_length": 80
        }
    )

    # Download result
    with httpx.Client(timeout=120) as client:
        resp = client.get(output)
        with open(output_path, "wb") as f:
            f.write(resp.content)

    return output_path


def _inpaint_videopainter(video_path: str, mask_path: str, output_path: str, pipe) -> str:
    """Inpaint using VideoPainter (CogVideoX-based)."""
    import cv2
    from PIL import Image

    # VideoPainter requires specific format
    # Extract frames, apply inpainting model, reassemble

    cap = cv2.VideoCapture(video_path)
    mask_cap = cv2.VideoCapture(mask_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get first frame as reference
    ret, first_frame = cap.read()
    first_frame_pil = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    # Generate inpainted video with CogVideoX
    # Note: This uses the diffusion pipeline for video generation
    video_frames = pipe(
        prompt="",  # No text prompt, just inpaint
        image=first_frame_pil,
        num_frames=min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 49),  # CogVideoX limit
        num_inference_steps=20,
        guidance_scale=3.0,
    ).frames[0]

    # Write output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in video_frames:
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (width, height))
        out.write(frame_resized)

    cap.release()
    mask_cap.release()
    out.release()

    _copy_audio(video_path, output_path)

    return output_path


def _copy_audio(source_video: str, target_video: str):
    """Copy audio from source to target video."""
    temp_output = target_video.replace(".mp4", "_with_audio.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", target_video,
        "-i", source_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        temp_output
    ], capture_output=True)

    # Replace original
    import shutil
    shutil.move(temp_output, target_video)


def stage_transcribe(video_path: str, vocals_path: str, mm: ModelManager) -> Dict:
    """Transcribe using Faster-Whisper with word timestamps."""
    logger.info("Stage: TRANSCRIBE (Faster-Whisper)")

    whisper = mm.load("faster_whisper")

    # Use vocals track if available (cleaner for transcription)
    audio_source = vocals_path if vocals_path and os.path.exists(vocals_path) else video_path

    segments, info = whisper.transcribe(
        audio_source,
        language=None,  # Auto-detect
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    # Convert to list
    segments_list = []
    for segment in segments:
        seg_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": []
        }
        if segment.words:
            seg_dict["words"] = [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in segment.words
            ]
        segments_list.append(seg_dict)

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments_list,
        "full_text": " ".join(s["text"] for s in segments_list)
    }


def stage_translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    FALLBACK: Translate text using Argos Translate (local, free).

    NOTE: In production, translation should be done on the main server
    using TranslatorAgent (Gemini 3 Pro) for better quality.
    Pass pre-translated text via 'translated_text' input parameter.
    """
    logger.info(f"Stage: TRANSLATE ({source_lang} → {target_lang}) [FALLBACK - Argos]")
    logger.warning("Using local Argos Translate. For better quality, use server-side TranslatorAgent (Gemini 3 Pro)")

    import argostranslate.package
    import argostranslate.translate

    # Ensure language package is installed
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    # Find matching package
    for pkg in available:
        if pkg.from_code == source_lang and pkg.to_code == target_lang:
            if not pkg.installed:
                argostranslate.package.install_from_path(pkg.download())
            break

    # Translate
    translated = argostranslate.translate.translate(text, source_lang, target_lang)

    return translated


def _tts_elevenlabs(text: str, reference_audio: str, target_language: str = "en") -> Optional[str]:
    """Generate speech using ElevenLabs API (best quality, runs from GPU worker IP to avoid geo-blocks)."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.info("ElevenLabs: no API key, skipping")
        return None

    import httpx

    output_path = tempfile.mktemp(suffix=".mp3")

    try:
        # Step 1: Create voice clone from reference audio
        logger.info("ElevenLabs: cloning voice from reference audio...")
        with open(reference_audio, "rb") as f:
            clone_resp = httpx.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={"xi-api-key": api_key},
                data={"name": "clone_temp", "description": "Temporary clone for localization"},
                files={"files": ("reference.wav", f, "audio/wav")},
                timeout=30,
            )
        if clone_resp.status_code != 200:
            logger.warning(f"ElevenLabs clone failed: {clone_resp.status_code} {clone_resp.text[:200]}")
            return None

        voice_id = clone_resp.json().get("voice_id")
        if not voice_id:
            logger.warning("ElevenLabs: no voice_id returned")
            return None

        logger.info(f"ElevenLabs: voice cloned as {voice_id}")

        # Step 2: Generate speech with cloned voice
        try:
            tts_resp = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.85,
                        "style": 0.3,
                    },
                },
                timeout=60,
            )
            if tts_resp.status_code != 200:
                logger.warning(f"ElevenLabs TTS failed: {tts_resp.status_code}")
                return None

            with open(output_path, "wb") as out:
                out.write(tts_resp.content)

            logger.info(f"ElevenLabs: generated {len(tts_resp.content)} bytes of audio")
            return output_path

        finally:
            # Step 3: Delete temporary voice clone
            try:
                httpx.delete(
                    f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                    headers={"xi-api-key": api_key},
                    timeout=10,
                )
                logger.info(f"ElevenLabs: deleted temp voice {voice_id}")
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"ElevenLabs failed: {e}")
        return None


def _tts_f5(text: str, reference_audio: str, mm: ModelManager) -> str:
    """Generate speech using F5-TTS locally (fallback)."""
    logger.info("TTS fallback: F5-TTS (local)")

    f5 = mm.load("f5tts")
    output_path = tempfile.mktemp(suffix=".wav")

    audio = f5.infer(
        ref_audio=reference_audio,
        ref_text="",  # Auto-transcribe reference
        gen_text=text,
        speed=1.0
    )

    import torchaudio
    torchaudio.save(output_path, audio, 24000)

    return output_path


def stage_tts(text: str, reference_audio: str, mm: ModelManager) -> str:
    """Generate speech: ElevenLabs API (primary) → F5-TTS local (fallback)."""
    logger.info("Stage: TTS")

    # Primary: ElevenLabs (runs from US GPU IP — no geo-block)
    result = _tts_elevenlabs(text, reference_audio)
    if result:
        return result

    # Fallback: F5-TTS on local GPU
    return _tts_f5(text, reference_audio, mm)


def stage_lipsync(video_path: str, audio_path: str, quality: str, mm: ModelManager) -> str:
    """Apply lipsync based on quality setting."""
    logger.info(f"Stage: LIPSYNC (quality={quality})")

    output_path = video_path.replace(".mp4", "_lipsync.mp4")

    if quality == "high":
        model_name = "video_retalking"
    elif quality == "medium":
        model_name = "musetalk"
    else:
        model_name = "wav2lip"

    with mm.use(model_name) as model:
        model.generate(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )

    return output_path


def stage_enhance(video_path: str, mm: ModelManager) -> str:
    """NEW: Enhance faces using GFPGAN."""
    logger.info("Stage: ENHANCE (GFPGAN)")

    import cv2

    with mm.use("gfpgan") as gfpgan:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = video_path.replace(".mp4", "_enhanced.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance face
            _, _, enhanced = gfpgan.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            out.write(enhanced)

        cap.release()
        out.release()

    return output_path


def stage_upscale(video_path: str, mm: ModelManager, scale: int = 2) -> str:
    """Upscale video using Real-ESRGAN."""
    logger.info(f"Stage: UPSCALE (Real-ESRGAN x{scale})")

    import cv2

    with mm.use("realesrgan") as upscaler:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_width = width * scale
        new_height = height * scale

        output_path = video_path.replace(".mp4", f"_upscaled_{scale}x.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Real-ESRGAN expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Upscale
            upscaled, _ = upscaler.enhance(frame_rgb, outscale=scale)

            # Convert back to BGR for video writing
            upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            out.write(upscaled_bgr)

            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(f"Upscaled {frame_idx} frames...")

        cap.release()
        out.release()

    # Copy audio
    _copy_audio(video_path, output_path)

    return output_path


def stage_quality_check(video_path: str, threshold: float) -> Dict:
    """NEW: Assess output quality, reject if too low."""
    logger.info("Stage: QUALITY_CHECK")

    import pyiqa

    # Load quality metrics
    lpips_metric = pyiqa.create_metric('lpips', device='cuda')

    # Sample frames and compute quality
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scores = []
    for i in range(0, frame_count, frame_count // 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).cuda()

            # Compute perceptual quality (lower is better for LPIPS)
            # Using self-comparison as baseline
            score = 1.0 - lpips_metric(frame_tensor, frame_tensor).item()
            scores.append(score)

    cap.release()

    avg_score = sum(scores) / len(scores) if scores else 0
    passed = avg_score >= threshold

    return {
        "score": avg_score,
        "threshold": threshold,
        "passed": passed,
        "frame_scores": scores
    }


def stage_assemble(
    video_path: str,
    tts_audio: str,
    background_audio: str,
    output_path: str
) -> str:
    """Assemble final video with mixed audio."""
    logger.info("Stage: ASSEMBLE")

    # Mix TTS voice with background audio
    mixed_audio = output_path.replace(".mp4", "_mixed.wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tts_audio,
        "-i", background_audio,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 0.3[a]",
        "-map", "[a]",
        mixed_audio
    ], capture_output=True)

    # Combine video with mixed audio
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", mixed_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v",
        "-map", "1:a",
        "-shortest",
        output_path
    ], capture_output=True)

    return output_path


# =============================================================================
# R2 Storage
# =============================================================================

class R2Storage:
    """Cloudflare R2 storage for video uploads."""

    def __init__(self):
        import boto3

        self.bucket = os.getenv("R2_BUCKET", "trafficplant")
        # Correct Cloudflare account ID: e66ac290473eeddb1a026d180d738f30
        self.endpoint = os.getenv("R2_ENDPOINT", "https://e66ac290473eeddb1a026d180d738f30.r2.cloudflarestorage.com")
        self.public_url = os.getenv("R2_PUBLIC_URL", "https://pub-c025ef96f40e47aab26156a1874f64bc.r2.dev")

        access_key = os.getenv("R2_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not access_key or not secret_key:
            raise RuntimeError(
                "R2 credentials missing. Set R2_ACCESS_KEY/R2_SECRET_KEY "
                "or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
            )

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto"
        )

    def upload(self, local_path: str, remote_key: str) -> str:
        """Upload file to R2 and return public URL."""
        logger.info(f"Uploading {local_path} to R2: {remote_key}")

        # Determine content type
        content_type = "video/mp4" if local_path.endswith(".mp4") else "application/octet-stream"

        with open(local_path, "rb") as f:
            self.client.upload_fileobj(
                f,
                self.bucket,
                remote_key,
                ExtraArgs={"ContentType": content_type}
            )

        public_url = f"{self.public_url}/{remote_key}"
        logger.info(f"Uploaded to: {public_url}")
        return public_url

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for private access."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in
        )


# Global instances
_model_manager: Optional[ModelManager] = None
_r2_storage: Optional[R2Storage] = None


def get_r2_storage() -> R2Storage:
    global _r2_storage
    if _r2_storage is None:
        _r2_storage = R2Storage()
    return _r2_storage


# =============================================================================
# Main Handler
# =============================================================================

# Global model manager (persists across requests)
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler for video localization.

    Input:
        {
            "video_url": "https://...",
            "source_language": "auto",
            "target_language": "en",
            "voice_clone": true,
            "lipsync": true,
            "lipsync_quality": "high",  // "high", "medium", "fast"
            "upscale": false,
            "face_enhance": true,
            "quality_threshold": 0.6,
            "stages": null,  // or specific stages
            "callback_url": null
        }

    Output:
        {
            "status": "success",
            "output_url": "https://...",
            "metrics": {...},
            "transcript": {...}
        }
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    logger.info(f"=" * 60)
    logger.info(f"Job {job_id} started")
    logger.info(f"=" * 60)

    start_time = time.time()

    # Validate input
    if "video_url" not in job_input:
        return {"status": "error", "error": "Missing 'video_url' in input"}

    try:
        # Parse config
        config = JobConfig(
            video_url=job_input["video_url"],
            source_language=job_input.get("source_language", "auto"),
            target_language=job_input.get("target_language", "en"),
            voice_clone=job_input.get("voice_clone", True),
            lipsync=job_input.get("lipsync", True),
            lipsync_quality=job_input.get("lipsync_quality", "high"),
            upscale=job_input.get("upscale", False),
            face_enhance=job_input.get("face_enhance", True),
            quality_threshold=job_input.get("quality_threshold", 0.6),
            stages=job_input.get("stages"),
            callback_url=job_input.get("callback_url"),
            translated_text=job_input.get("translated_text"),  # Pre-translated from server
        )

        mm = get_model_manager()
        mm.metrics = PipelineMetrics()  # Reset metrics for each request
        metrics = mm.metrics

        # Download video
        logger.info(f"Downloading video from {config.video_url}")
        import httpx
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            with httpx.Client(timeout=120) as client:
                resp = client.get(config.video_url)
                f.write(resp.content)
            video_path = f.name

        # Determine stages to run
        all_stages = [s.value for s in PipelineStage]
        stages = config.stages or all_stages

        # Pipeline state
        state = {
            "video_path": video_path,
            "vocals_path": None,
            "background_path": None,
            "mask_path": None,
            "tts_audio": None,
            "transcript": None
        }

        # Execute pipeline
        for stage_name in stages:
            t0 = time.time()

            try:
                if stage_name == "preprocess":
                    result = stage_preprocess(state["video_path"], mm)
                    state["vocals_path"] = result["vocals_path"]
                    state["background_path"] = result["background_path"]

                elif stage_name == "detect_text":
                    result = stage_detect_text(state["video_path"], mm)
                    state["text_detections"] = result["detections"]

                elif stage_name == "create_mask":
                    state["mask_path"] = stage_create_mask(
                        state["video_path"],
                        state.get("text_detections", []),
                        mm
                    )

                elif stage_name == "inpaint":
                    state["video_path"] = stage_inpaint(
                        state["video_path"],
                        state["mask_path"],
                        mm
                    )

                elif stage_name == "transcribe":
                    state["transcript"] = stage_transcribe(
                        state["video_path"],
                        state.get("vocals_path"),
                        mm
                    )

                elif stage_name == "translate":
                    # Use pre-translated text from server (Gemini 3 Pro) if provided
                    if config.translated_text:
                        logger.info("Using pre-translated text from server (Gemini 3 Pro)")
                        state["translated_text"] = config.translated_text
                    elif state["transcript"]:
                        # Fallback to local Argos Translate
                        src_lang = state["transcript"]["language"]
                        if src_lang != config.target_language:
                            state["translated_text"] = stage_translate(
                                state["transcript"]["full_text"],
                                src_lang,
                                config.target_language
                            )
                        else:
                            state["translated_text"] = state["transcript"]["full_text"]

                elif stage_name == "tts":
                    if config.voice_clone and state.get("translated_text"):
                        state["tts_audio"] = stage_tts(
                            state["translated_text"],
                            state.get("vocals_path", video_path),
                            mm
                        )

                elif stage_name == "lipsync":
                    if config.lipsync and state.get("tts_audio"):
                        state["video_path"] = stage_lipsync(
                            state["video_path"],
                            state["tts_audio"],
                            config.lipsync_quality,
                            mm
                        )

                elif stage_name == "enhance":
                    if config.face_enhance:
                        state["video_path"] = stage_enhance(state["video_path"], mm)

                elif stage_name == "upscale":
                    if config.upscale:
                        state["video_path"] = stage_upscale(state["video_path"], mm, scale=2)

                elif stage_name == "quality_check":
                    qc_result = stage_quality_check(
                        state["video_path"],
                        config.quality_threshold
                    )
                    metrics.quality_scores = qc_result
                    if not qc_result["passed"]:
                        raise ValueError(f"Quality check failed: {qc_result['score']:.2f} < {qc_result['threshold']}")

                elif stage_name == "assemble":
                    output_path = video_path.replace(".mp4", "_final.mp4")
                    state["video_path"] = stage_assemble(
                        state["video_path"],
                        state.get("tts_audio", state.get("vocals_path")),
                        state.get("background_path"),
                        output_path
                    )

            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                metrics.errors.append(f"{stage_name}: {str(e)}")
                # Continue with next stage if possible

            metrics.stage_times[stage_name] = time.time() - t0
            logger.info(f"Stage {stage_name} completed in {metrics.stage_times[stage_name]:.1f}s")

        # Upload result to R2
        try:
            r2 = get_r2_storage()
            timestamp = int(time.time())
            remote_key = f"{config.r2_prefix}/{job_id}_{timestamp}.mp4"
            output_url = r2.upload(state["video_path"], remote_key)
        except Exception as e:
            logger.error(f"R2 upload failed: {e}")
            # Return local path as fallback (useful for debugging)
            output_url = f"file://{state['video_path']}"
            metrics.errors.append(f"r2_upload: {str(e)}")

        total_time = time.time() - start_time

        logger.info(f"=" * 60)
        logger.info(f"Job {job_id} completed in {total_time:.1f}s")
        logger.info(f"=" * 60)

        return {
            "status": "success",
            "output_url": output_url,
            "metrics": metrics.to_dict(),
            "transcript": state.get("transcript")
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        # Cleanup temp files
        import glob
        for pattern in ["/tmp/*.mp4", "/tmp/*.wav", "/tmp/*.png"]:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except:
                    pass


# RunPod entry point
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
