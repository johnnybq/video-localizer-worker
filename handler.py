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
import shutil
import tempfile
import subprocess
import types
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# =============================================================================
# Monkey-patch: torchvision.transforms.functional_tensor
# Removed in torchvision 0.20+, but gfpgan/basicsr still import it.
# Must be applied BEFORE any gfpgan/basicsr import.
# =============================================================================
try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale  # noqa
except ImportError:
    from torchvision.transforms.functional import rgb_to_grayscale
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft

import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Optional progress callback: called after each pipeline stage completes.
# Signature: callback(stage_name: str, elapsed_secs: float, errors: list)
# Set by handler_vast.py for per-stage reporting to backend.
_progress_callback = None


def set_progress_callback(fn):
    """Register a progress callback for per-stage reporting."""
    global _progress_callback
    _progress_callback = fn


# =============================================================================
# Configuration
# =============================================================================

class PipelineStage(Enum):
    """Video localization pipeline stages."""
    PREPROCESS = "preprocess"          # Audio separation (demucs)
    DETECT_TEXT = "detect_text"        # PaddleOCR
    CREATE_MASK = "create_mask"        # SAM 2.1
    INPAINT = "inpaint"                # VideoPainter (TencentARC)
    RENDER_TEXT = "render_text"        # Draw translated text overlays (ffmpeg)
    TRANSCRIBE = "transcribe"          # Faster-Whisper
    TRANSLATE = "translate"            # Server-side (Gemini 3 Pro)
    TTS = "tts"                        # ElevenLabs → F5-TTS
    LIPSYNC = "lipsync"                # VideoRetalking / MuseTalk
    ENHANCE = "enhance"                # GFPGAN face enhancement
    UPSCALE = "upscale"                # Real-ESRGAN
    QUALITY_CHECK = "quality_check"    # Auto quality assessment
    ASSEMBLE = "assemble"              # Final mix


# Stages that MUST succeed for the job to produce a valid localized video.
# If any of these fail, the job returns status="error" instead of "success".
CRITICAL_STAGES = {"transcribe", "translate", "tts", "assemble"}


# =============================================================================
# Language Code Normalization
# =============================================================================

LANGUAGE_CODE_MAP = {
    "en": ["en", "en-US", "en-GB", "en-AU"],
    "pt": ["pt", "pt-BR", "pt-PT"],
    "es": ["es", "es-MX", "es-ES", "es-AR"],
    "ru": ["ru", "ru-RU"],
    "de": ["de", "de-DE", "de-AT", "de-CH"],
    "fr": ["fr", "fr-FR", "fr-CA"],
    "zh": ["zh", "zh-CN", "zh-TW", "zh-HK"],
    "ja": ["ja", "ja-JP"],
    "ko": ["ko", "ko-KR"],
    "ar": ["ar", "ar-SA"],
    "hi": ["hi", "hi-IN"],
    "it": ["it", "it-IT"],
    "nl": ["nl", "nl-NL"],
    "pl": ["pl", "pl-PL"],
    "tr": ["tr", "tr-TR"],
    "uk": ["uk", "uk-UA"],
    "vi": ["vi", "vi-VN"],
    "th": ["th", "th-TH"],
    "id": ["id", "id-ID"],
}

# Build reverse lookup: locale → base language
_GEO_TO_BASE = {}
for base, geos in LANGUAGE_CODE_MAP.items():
    for geo in geos:
        _GEO_TO_BASE[geo.lower()] = base


def normalize_language_code(code: str) -> str:
    """
    Normalize a language code to its base form.
    Examples: 'pt-BR' → 'pt', 'en-US' → 'en', 'ru' → 'ru'
    """
    if not code:
        return ""
    code_lower = code.lower().strip()
    return _GEO_TO_BASE.get(code_lower, code_lower.split("-")[0])


def are_languages_same(lang1: str, lang2: str) -> bool:
    """
    Check if two language codes refer to the same base language.
    Examples: are_languages_same('pt', 'pt-BR') → True
              are_languages_same('en-US', 'en-GB') → True
              are_languages_same('pt', 'es') → False
    """
    if not lang1 or not lang2:
        return False
    return normalize_language_code(lang1) == normalize_language_code(lang2)


def diagnose_videopainter() -> Dict[str, Any]:
    """
    Diagnose VideoPainter dependencies at startup.
    Returns dict with status of each component.
    """
    results = {
        "videopainter_root": False,
        "videopainter_checkpoints": False,
        "videopainter_branch": False,  # LoRA adapter (CRITICAL!)
        "cogvideox_model": False,
        "custom_diffusers_pipeline": False,
        "errors": []
    }

    # Check VIDEOPAINTER_ROOT (custom diffusers code)
    vp_root = os.environ.get("VIDEOPAINTER_ROOT", "/opt/videopainter")
    if os.path.exists(vp_root):
        results["videopainter_root"] = True
        logger.info(f"✓ VIDEOPAINTER_ROOT exists: {vp_root}")
    else:
        results["errors"].append(f"VIDEOPAINTER_ROOT not found: {vp_root}")
        logger.warning(f"✗ VIDEOPAINTER_ROOT not found: {vp_root}")

    # Check checkpoints directory
    vp_ckpt = os.environ.get("VIDEOPAINTER_CKPT", "/workspace/models/videopainter/checkpoints")
    if os.path.exists(vp_ckpt):
        results["videopainter_checkpoints"] = True
        logger.info(f"✓ VIDEOPAINTER_CKPT exists: {vp_ckpt}")
        # List contents for debugging
        try:
            ckpt_contents = os.listdir(vp_ckpt)
            logger.info(f"  Contents: {ckpt_contents}")
        except Exception as e:
            logger.warning(f"  Could not list contents: {e}")
    else:
        results["errors"].append(f"VIDEOPAINTER_CKPT not found: {vp_ckpt}")
        logger.warning(f"✗ VIDEOPAINTER_CKPT not found: {vp_ckpt}")

    # Check branch directory (LoRA adapter) - CRITICAL for inpainting quality!
    # HF repo structure: checkpoints/VideoPainter/checkpoints/branch
    branch_path = os.path.join(vp_ckpt, "VideoPainter", "checkpoints", "branch")
    if os.path.exists(branch_path):
        results["videopainter_branch"] = True
        logger.info(f"✓ VideoPainter branch (LoRA) exists: {branch_path}")
        # Check for adapter files
        try:
            branch_contents = os.listdir(branch_path)
            has_adapter = any("adapter" in f.lower() or "lora" in f.lower() or f.endswith(".safetensors") for f in branch_contents)
            logger.info(f"  Branch contents: {branch_contents[:10]}")
            if has_adapter:
                logger.info(f"  ✓ LoRA adapter files found")
            else:
                logger.warning(f"  ⚠ No obvious adapter files found, but directory exists")
        except Exception as e:
            logger.warning(f"  Could not list branch contents: {e}")
    else:
        results["errors"].append(f"CRITICAL: VideoPainter branch (LoRA) not found at {branch_path}")
        logger.error(f"✗ CRITICAL: VideoPainter branch (LoRA) not found: {branch_path}")

    # Check CogVideoX model (may need to download)
    hf_home = os.environ.get("HF_HOME", "/workspace/models/huggingface")
    cogvideo_path = os.path.join(hf_home, "THUDM/CogVideoX-5b-I2V")
    hub_path = os.path.join(hf_home, "hub", "models--THUDM--CogVideoX-5b-I2V")
    if os.path.exists(cogvideo_path) or os.path.exists(hub_path):
        results["cogvideox_model"] = True
        logger.info(f"✓ CogVideoX-5b-I2V model cached")
    else:
        results["errors"].append("CogVideoX-5b-I2V not cached (will download on first use, ~20GB)")
        logger.warning(f"✗ CogVideoX-5b-I2V not cached - will need to download (~20GB)")

    # Check custom diffusers pipeline import
    try:
        if results["videopainter_root"]:
            import sys
            sys.path.insert(0, vp_root)
        from diffusers.pipelines.cogvideo import (
            CogVideoXI2VInpaintAnyLPipeline,
        )
        results["custom_diffusers_pipeline"] = True
        logger.info("✓ Custom diffusers pipeline (CogVideoXI2VInpaintAnyLPipeline) available")
    except ImportError as e:
        results["errors"].append(f"Custom diffusers pipeline import failed: {e}")
        logger.warning(f"✗ Custom diffusers pipeline import failed: {e}")

    return results


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

    # Pre-translated on-screen text overlays (from server)
    # Each dict: {text, translated_text, appears_at, disappears_at, position, font_style, ...}
    translated_overlays: Optional[List[Dict]] = None

    # Original subtitle style from Gemini manifest (font, color, background, etc.)
    subtitle_style: Optional[Dict] = None

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
            from demucs_infer.pretrained import get_model
            model = get_model("htdemucs_ft")
            model.eval()
            model.to(self.device)
            return model

        elif name == "f5tts":
            # F5-TTS loading (v1.x API)
            from f5_tts.api import F5TTS
            return F5TTS(device=self.device)

        elif name == "videopainter":
            # VideoPainter (TencentARC) — proper mask-guided video inpainting
            # Uses custom CogVideoXI2VInpaintAnyLPipeline from their diffusers fork
            vp_root = os.environ.get("VIDEOPAINTER_ROOT", "/opt/videopainter")
            vp_ckpt = os.environ.get("VIDEOPAINTER_CKPT", "/workspace/models/videopainter/checkpoints")

            logger.info(f"VideoPainter: VIDEOPAINTER_ROOT={vp_root}")
            logger.info(f"VideoPainter: VIDEOPAINTER_CKPT={vp_ckpt}")

            # Add VideoPainter's custom diffusers to path
            sys.path.insert(0, vp_root)

            from diffusers import CogVideoXTransformer3DModel
            from diffusers.pipelines.cogvideo import (
                CogVideoXI2VInpaintAnyLPipeline,
            )

            model_path = os.path.join(os.environ.get("HF_HOME", "/workspace/models/huggingface"),
                                       "THUDM/CogVideoX-5b-I2V")
            # Download base model if not cached
            if not os.path.exists(model_path):
                logger.info(f"VideoPainter: CogVideoX model not cached, will download from HuggingFace")
                model_path = "THUDM/CogVideoX-5b-I2V"
            else:
                logger.info(f"VideoPainter: Using cached CogVideoX model at {model_path}")

            # HF repo structure: checkpoints/VideoPainter/checkpoints/branch
            branch_path = os.path.join(vp_ckpt, "VideoPainter", "checkpoints", "branch")

            # Check if branch (LoRA adapter) exists
            if os.path.exists(branch_path):
                logger.info(f"VideoPainter: ✓ Branch (LoRA) found at {branch_path}")
                # List contents for debugging
                try:
                    branch_contents = os.listdir(branch_path)
                    logger.info(f"VideoPainter: Branch contents: {branch_contents[:5]}...")
                except Exception as e:
                    logger.warning(f"VideoPainter: Could not list branch contents: {e}")
            else:
                logger.error(f"VideoPainter: ✗ Branch NOT found at {branch_path}")
                logger.error(f"VideoPainter: Contents of {vp_ckpt}: {os.listdir(vp_ckpt) if os.path.exists(vp_ckpt) else 'DIR NOT FOUND'}")
                raise FileNotFoundError(
                    f"VideoPainter branch (LoRA adapter) not found at {branch_path}. "
                    f"Check VIDEOPAINTER_CKPT env var and ensure HF clone completed."
                )

            logger.info("VideoPainter: Loading CogVideoXTransformer3DModel...")
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer", torch_dtype=torch.bfloat16
            )

            logger.info("VideoPainter: Loading CogVideoXI2VInpaintAnyLPipeline with branch...")
            pipe = CogVideoXI2VInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch_path,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            ).to(self.device)

            pipe.enable_xformers_memory_efficient_attention()
            logger.info("VideoPainter: ✓ Pipeline loaded successfully with xformers")
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

    demucs_model = mm.load("demucs")

    # Extract audio
    audio_path = video_path.replace(".mp4", "_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        audio_path
    ], capture_output=True)

    # Separate with demucs-infer
    import torchaudio
    from demucs_infer.apply import apply_model

    wav, sr = torchaudio.load(audio_path)
    wav = wav.unsqueeze(0)  # (1, channels, samples)

    with torch.no_grad():
        sources = apply_model(demucs_model, wav, device=mm.device)
    # sources shape: (1, num_sources, channels, samples)
    # htdemucs_ft sources: drums, bass, other, vocals (index order from model.sources)
    source_names = demucs_model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    separated = {name: sources[0, i] for i, name in enumerate(source_names)}

    # Save separated tracks
    vocals_path = video_path.replace(".mp4", "_vocals.wav")
    background_path = video_path.replace(".mp4", "_background.wav")

    # Vocals = voice track
    torchaudio.save(vocals_path, separated["vocals"].cpu(), sr)

    # Background = drums + bass + other (everything except vocals)
    background = separated["drums"] + separated["bass"] + separated["other"]
    torchaudio.save(background_path, background.cpu(), sr)

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

    # Sort detections by confidence (best first) and filter low-confidence
    sorted_dets = sorted(
        [d for d in detections if d.get("confidence", 0.5) >= 0.3],
        key=lambda d: d.get("confidence", 0.5),
        reverse=True
    )
    logger.info(f"CREATE_MASK: {len(detections)} total, {len(sorted_dets)} after confidence filter")

    # Add prompts for each text region (top 15 by confidence)
    for i, det in enumerate(sorted_dets[:15]):
        frame_idx = det["frame_idx"]
        bbox = det["bbox"]  # List of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        confidence = det.get("confidence", 0.5)

        # Convert polygon bbox to axis-aligned box [x_min, y_min, x_max, y_max]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        # Use float32 dtype to match SAM2's internal BFloat16 (avoids dtype mismatch)
        box = np.array([min(x_coords), min(y_coords), max(x_coords), max(y_coords)], dtype=np.float32)

        # Use BOX prompt (much better for text regions than center point)
        # SAM2 box format: [x1, y1, x2, y2] as np.array
        try:
            sam2.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=i,
                box=box  # Use box prompt instead of single point
            )
            logger.debug(f"CREATE_MASK: Added box prompt #{i} conf={confidence:.2f} box={box.tolist()}")
        except Exception as e:
            # Fallback to center point if box prompt fails
            logger.warning(f"CREATE_MASK: Box prompt failed, using center point: {e}")
            x_center = sum(p[0] for p in bbox) / 4
            y_center = sum(p[1] for p in bbox) / 4
            sam2.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=i,
                points=np.array([[x_center, y_center]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32)
            )

    # Propagate masks through video
    mask_frames = {}
    for frame_idx, obj_ids, masks in sam2.propagate_in_video(inference_state):
        # masks: torch.Tensor (num_objects, H, W) or (num_objects, 1, H, W)
        masks_np = masks.cpu().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(1)  # (N, 1, H, W) → (N, H, W)
        combined_mask = np.zeros(masks_np.shape[1:], dtype=np.uint8)
        for mask in masks_np:
            combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8) * 255)
        mask_frames[frame_idx] = combined_mask

    # Render mask video
    mask_path = video_path.replace(".mp4", "_mask.mp4")
    _render_mask_video(video_path, mask_frames, mask_path)

    return mask_path


def _render_mask_video(video_path: str, masks: Dict[int, np.ndarray], output_path: str):
    """Render mask frames to video with nearest-neighbor interpolation."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Pre-compute sorted frame indices for nearest-neighbor lookup
    mask_frames = sorted(masks.keys())
    logger.info(f"MASK_VIDEO: Rendering {frame_count} frames, {len(mask_frames)} keyframes")

    for i in range(frame_count):
        if i in masks:
            mask = cv2.resize(masks[i], (width, height))
        elif mask_frames:
            # Nearest-neighbor interpolation (not zeros!)
            # Find closest keyframe
            nearest_idx = min(mask_frames, key=lambda x: abs(x - i))
            # Only use if within reasonable distance (30 frames = ~1 sec)
            if abs(nearest_idx - i) <= 30:
                mask = cv2.resize(masks[nearest_idx], (width, height))
            else:
                # Too far from any keyframe - no text expected here
                mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        out.write(mask)

    cap.release()
    out.release()
    logger.info(f"MASK_VIDEO: Saved to {output_path}")


def stage_inpaint(video_path: str, mask_path: str, mm: ModelManager, errors: Optional[List[str]] = None) -> Tuple[str, bool]:
    """
    Remove text using VideoPainter (primary) or ProPainter (fallback).

    Args:
        errors: Optional list to append error messages (for metrics tracking)

    Returns:
        Tuple of (video_path, inpaint_succeeded)
        - inpaint_succeeded=True: Text was removed, eraser plate optional
        - inpaint_succeeded=False: Text NOT removed, eraser plate REQUIRED
    """
    logger.info("Stage: INPAINT")

    if mask_path is None:
        logger.info("No mask, skipping inpainting")
        return video_path, True  # No text to remove = "success"

    output_path = video_path.replace(".mp4", "_inpainted.mp4")
    all_errors = []

    # Try VideoPainter first (best quality, CogVideoX-based)
    try:
        logger.info("INPAINT: Attempting VideoPainter (CogVideoX-based)...")
        with mm.use("videopainter") as videopainter:
            result = _inpaint_videopainter(video_path, mask_path, output_path, videopainter)
            logger.info("INPAINT: VideoPainter succeeded!")
            return result, True
    except Exception as e:
        import traceback
        err_msg = f"VideoPainter failed: {e}"
        logger.warning(err_msg)
        logger.debug(f"VideoPainter traceback:\n{traceback.format_exc()}")
        all_errors.append(err_msg)

    # Fallback to ProPainter
    try:
        logger.info("INPAINT: Attempting ProPainter fallback...")
        result = _inpaint_propainter(video_path, mask_path, output_path)
        logger.info("INPAINT: ProPainter succeeded!")
        return result, True
    except Exception as e:
        import traceback
        err_msg = f"ProPainter failed: {e}"
        logger.error(err_msg)
        logger.debug(f"ProPainter traceback:\n{traceback.format_exc()}")
        all_errors.append(err_msg)

    # All methods failed - eraser plate is now REQUIRED
    combined_error = f"inpaint: ALL methods failed - {'; '.join(all_errors)}"
    logger.error(combined_error)
    logger.warning("INPAINT: Falling back to ERASER-PLATE-ONLY mode (original text NOT removed)")
    if errors is not None:
        errors.append(combined_error)

    # Return original but signal that inpaint failed
    return video_path, False


def _inpaint_propainter(video_path: str, mask_path: str, output_path: str) -> str:
    """
    Inpaint using ProPainter (E2FGVI-based) via command-line interface.
    ProPainter is a script, not a Python module, so we call it via subprocess.
    """
    import cv2

    propainter_dir = "/models/ProPainter"
    propainter_script = os.path.join(propainter_dir, "inference_propainter.py")

    # Check if ProPainter is installed
    if not os.path.exists(propainter_script):
        logger.warning(f"ProPainter not found at {propainter_script}, falling back to Replicate")
        return _inpaint_replicate(video_path, mask_path, output_path)

    # ProPainter expects a directory with frames and masks, not video files
    # Create temp directories for frames
    video_frames_dir = tempfile.mkdtemp(prefix="propainter_video_")
    mask_frames_dir = tempfile.mkdtemp(prefix="propainter_mask_")
    result_dir = tempfile.mkdtemp(prefix="propainter_result_")

    try:
        # Extract video frames
        logger.info(f"ProPainter: Extracting video frames to {video_frames_dir}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(video_frames_dir, f"{frame_idx:05d}.png"), frame)
            frame_idx += 1
        cap.release()
        logger.info(f"ProPainter: Extracted {frame_idx} video frames")

        # Extract mask frames
        logger.info(f"ProPainter: Extracting mask frames to {mask_frames_dir}")
        mask_cap = cv2.VideoCapture(mask_path)
        mask_idx = 0

        while True:
            ret, mask = mask_cap.read()
            if not ret:
                break
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(mask_frames_dir, f"{mask_idx:05d}.png"), mask)
            mask_idx += 1
        mask_cap.release()

        # Pad masks if shorter than video
        while mask_idx < frame_idx:
            last_mask = os.path.join(mask_frames_dir, f"{mask_idx-1:05d}.png")
            new_mask = os.path.join(mask_frames_dir, f"{mask_idx:05d}.png")
            shutil.copy(last_mask, new_mask)
            mask_idx += 1

        logger.info(f"ProPainter: {mask_idx} mask frames ready")

        # Run ProPainter
        cmd = [
            "python", propainter_script,
            "--video", video_frames_dir,
            "--mask", mask_frames_dir,
            "--output", result_dir,
            "--resize_ratio", "0.5",  # Memory optimization
            "--ref_stride", "10",
            "--neighbor_length", "10",
            "--subvideo_length", "80",
            "--fp16",  # Use half precision
        ]

        logger.info(f"ProPainter: Running {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=propainter_dir,
            capture_output=True,
            timeout=600,  # 10 min timeout
        )

        if result.returncode != 0:
            logger.error(f"ProPainter failed: {result.stderr.decode()[:500]}")
            raise RuntimeError(f"ProPainter subprocess failed: {result.stderr.decode()[:200]}")

        # Find output video in result_dir
        result_files = [f for f in os.listdir(result_dir) if f.endswith(('.mp4', '.avi'))]
        if not result_files:
            # ProPainter outputs frames, need to reassemble
            logger.info("ProPainter: Reassembling frames to video")
            result_frames = sorted([f for f in os.listdir(result_dir) if f.endswith('.png')])
            if not result_frames:
                raise RuntimeError("ProPainter produced no output frames")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for fname in result_frames:
                frame = cv2.imread(os.path.join(result_dir, fname))
                frame_resized = cv2.resize(frame, (width, height))
                out.write(frame_resized)
            out.release()
        else:
            # Copy result video
            shutil.copy(os.path.join(result_dir, result_files[0]), output_path)

        # Re-add original audio
        _copy_audio(video_path, output_path)

        logger.info(f"ProPainter: Success! Output saved to {output_path}")
        return output_path

    finally:
        # Cleanup temp directories
        for d in [video_frames_dir, mask_frames_dir, result_dir]:
            try:
                shutil.rmtree(d)
            except Exception:
                pass


def _inpaint_replicate(video_path: str, mask_path: str, output_path: str) -> str:
    """Fallback: Use Replicate API for ProPainter."""
    import replicate
    import httpx
    import base64

    logger.info("Using Replicate ProPainter API")

    # Replicate API expects URLs or base64 data URIs, NOT file handles
    def file_to_data_uri(path: str, mime: str) -> str:
        """Convert local file to base64 data URI for Replicate API."""
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    # Convert video and mask files to data URIs
    video_uri = file_to_data_uri(video_path, "video/mp4")
    mask_uri = file_to_data_uri(mask_path, "video/mp4")

    logger.info(f"Replicate: video data URI length: {len(video_uri)}, mask data URI length: {len(mask_uri)}")

    output = replicate.run(
        "sczhou/propainter:34a544b1df7e77e08d5d1648e8b28899cb7f8c47a28aaef54eae16ebf6f4c34a",
        input={
            "video": video_uri,
            "mask": mask_uri,
            "resize_ratio": 0.5,
            "ref_stride": 10,
            "neighbor_length": 10,
            "subvideo_length": 80
        }
    )

    # Download result — output can be string URL or list
    if isinstance(output, str):
        result_url = output
    elif isinstance(output, list) and len(output) > 0:
        result_url = str(output[0])
    else:
        result_url = str(output)

    logger.info(f"Replicate: downloading result from {result_url[:100]}...")

    with httpx.Client(timeout=120) as client:
        resp = client.get(result_url)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)

    logger.info(f"Replicate: saved {len(resp.content)} bytes to {output_path}")
    return output_path


def _inpaint_videopainter(video_path: str, mask_path: str, output_path: str, pipe) -> str:
    """Inpaint using VideoPainter (TencentARC) — mask-guided CogVideoX inpainting."""
    import cv2
    from PIL import Image

    # Extract video frames as PIL images
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply mask to frame (black out inpaint regions)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_frames.append(pil)
    cap.release()

    # Extract mask frames as binary PIL images
    mask_cap = cv2.VideoCapture(mask_path)
    mask_frames = []
    while True:
        ret, frame = mask_cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mask_pil = Image.fromarray(gray)
        mask_frames.append(mask_pil)
    mask_cap.release()

    # Pad mask_frames to match video length
    while len(mask_frames) < len(video_frames):
        mask_frames.append(mask_frames[-1] if mask_frames else Image.new("L", (width, height), 0))

    # Create masked video frames (black out inpaint regions)
    masked_frames = []
    for vf, mf in zip(video_frames, mask_frames):
        v_np = np.array(vf)
        m_np = np.array(mf.resize(vf.size))
        # Zero out masked regions
        v_np[m_np > 128] = 0
        masked_frames.append(Image.fromarray(v_np))

    # Limit to 49 frames (CogVideoX constraint), process in chunks for longer videos
    max_frames = 49
    all_output_frames = []

    for chunk_start in range(0, len(video_frames), max_frames):
        chunk_end = min(chunk_start + max_frames, len(video_frames))
        chunk_masked = masked_frames[chunk_start:chunk_end]
        chunk_masks = mask_frames[chunk_start:chunk_end]

        inpaint_out = pipe(
            prompt="",
            image=chunk_masked[0],
            num_videos_per_prompt=1,
            num_inference_steps=20,
            num_frames=len(chunk_masked),
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(42),
            video=chunk_masked,
            masks=chunk_masks,
            strength=1.0,
            output_type="np",
        )

        chunk_frames = inpaint_out.frames[0] if hasattr(inpaint_out, 'frames') else inpaint_out[0]
        all_output_frames.extend(chunk_frames)

    # Write output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in all_output_frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
        frame_resized = cv2.resize(frame_bgr, (width, height))
        out.write(frame_resized)

    out.release()

    # Copy original audio to inpainted video
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


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Simple Levenshtein similarity ratio (0.0 - 1.0)."""
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    # Simple edit distance
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    distance = dp[len1][len2]
    return 1.0 - (distance / max(len1, len2))


def _find_overlay_bbox(
    overlay: Dict,
    text_detections: List[Dict],
    video_width: int,
    video_height: int,
    appears_at: float = 0.0,
    disappears_at: float = 0.0,
    video_fps: float = 30.0,
) -> Tuple[int, int, int, int]:
    """
    Find pixel bounding box for an overlay based on OCR detections.

    SOTA ASS Logic 3.0 — Smart matching:
    1. TIME FILTERING: Only consider detections within overlay's time window
    2. Fuzzy text match (Levenshtein > 60%) → use that exact bbox
    3. Zone filtering with thresholds:
       - top:    bbox entirely in y < 0.40
       - bottom: bbox entirely in y > 0.60
       - middle: bbox overlaps 0.40-0.60
    4. Cluster nearby detections (vertical + horizontal proximity)
    5. Reduced minimum sizes (25% width, not 50%)

    Returns (x, y, w, h) in pixels.
    """
    position = overlay.get("position", "top")
    original_text = overlay.get("text", "")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: Time filtering — only consider detections in overlay's time window
    # ═══════════════════════════════════════════════════════════════════════
    if appears_at > 0 or disappears_at > 0:
        # Convert time window to frame range (with margin)
        margin_sec = 1.0  # 1 second margin
        start_frame = max(0, int((appears_at - margin_sec) * video_fps))
        end_frame = int((disappears_at + margin_sec) * video_fps)

        time_filtered = [
            d for d in text_detections
            if start_frame <= d.get("frame_idx", 0) <= end_frame
        ]
        logger.info(
            f"RENDER_TEXT bbox: TIME FILTER {appears_at:.1f}s-{disappears_at:.1f}s "
            f"(frames {start_frame}-{end_frame}): {len(text_detections)} → {len(time_filtered)} detections"
        )
        text_detections = time_filtered

    logger.info(f"RENDER_TEXT bbox: position='{position}', detections={len(text_detections)}, text='{original_text[:40]}...'")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Fuzzy text matching (WITH zone filtering to prevent center false positives)
    # ═══════════════════════════════════════════════════════════════════════
    if original_text and len(original_text) > 5:
        best_match = None
        best_ratio = 0.0
        for d in text_detections:
            det_text = d.get("text", "")
            if not det_text:
                continue

            # ZONE FILTER: Only consider detections in the correct zone!
            bbox = d.get("bbox_norm", [0, 0, 1, 1])
            y_top = bbox[1]
            y_bottom = bbox[3]

            in_correct_zone = False
            if position == "top" and y_bottom < 0.45:  # Allow slightly more tolerance
                in_correct_zone = True
            elif position == "bottom" and y_top > 0.55:
                in_correct_zone = True
            elif position in ("middle", "center") and y_top < 0.65 and y_bottom > 0.35:
                in_correct_zone = True

            if not in_correct_zone:
                continue  # Skip detections outside the expected zone

            ratio = _levenshtein_ratio(original_text, det_text)
            # Also check if detection is substring of overlay or vice versa
            if original_text.lower() in det_text.lower() or det_text.lower() in original_text.lower():
                ratio = max(ratio, 0.7)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = d

        if best_match and best_ratio >= 0.6:
            bbox = best_match.get("bbox_norm", [0, 0, 1, 1])
            logger.info(f"RENDER_TEXT: Fuzzy match found! ratio={best_ratio:.2f} text='{best_match.get('text', '')[:30]}'")

            # Small padding around matched text (3% relative)
            pad_x = (bbox[2] - bbox[0]) * 0.15  # 15% of bbox width
            pad_y = (bbox[3] - bbox[1]) * 0.2   # 20% of bbox height

            x_min = max(0, bbox[0] - pad_x)
            y_min = max(0, bbox[1] - pad_y)
            x_max = min(1.0, bbox[2] + pad_x)
            y_max = min(1.0, bbox[3] + pad_y)

            x = int(x_min * video_width)
            y = int(y_min * video_height)
            w = int((x_max - x_min) * video_width)
            h = int((y_max - y_min) * video_height)

            # Minimum height for readability (4% of video)
            min_h = int(video_height * 0.04)
            if h < min_h:
                h = min_h

            logger.info(f"RENDER_TEXT: FUZZY MATCH bbox: ({x}, {y}, {w}x{h})")
            return (x, y, w, h)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Zone filtering (bbox y_min/y_max range, not just center)
    # ═══════════════════════════════════════════════════════════════════════
    # Zone thresholds: top < 0.40, middle 0.40-0.60, bottom > 0.60
    filtered = []
    for d in text_detections:
        bbox = d.get("bbox_norm", [0, 0, 1, 1])
        y_top = bbox[1]     # Top edge of detection
        y_bottom = bbox[3]  # Bottom edge of detection

        # Check if bbox is ENTIRELY within zone (not just center)
        if position == "top" and y_bottom < 0.40:  # Entire bbox in top 40%
            filtered.append(d)
        elif position == "bottom" and y_top > 0.60:  # Entire bbox in bottom 40%
            filtered.append(d)
        elif position in ("middle", "center"):
            # Any overlap with middle zone (0.40-0.60)
            if y_top < 0.60 and y_bottom > 0.40:
                filtered.append(d)

    logger.info(f"RENDER_TEXT: Zone-filtered {len(filtered)} detections for '{position}'")

    if filtered:
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Cluster nearby detections (vertical + horizontal proximity)
        # ═══════════════════════════════════════════════════════════════════
        # Sort by confidence (if available) and take best cluster
        filtered.sort(key=lambda d: d.get("confidence", 0.5), reverse=True)

        # Start with highest-confidence detection
        main_bbox = filtered[0].get("bbox_norm", [0, 0, 1, 1])
        cluster_bboxes = [main_bbox]

        # Add nearby detections (both vertical AND horizontal proximity)
        for d in filtered[1:]:
            bbox = d.get("bbox_norm", [0, 0, 1, 1])

            # Calculate cluster center
            cluster_x_center = sum(b[0] + b[2] for b in cluster_bboxes) / (2 * len(cluster_bboxes))
            cluster_y_center = sum(b[1] + b[3] for b in cluster_bboxes) / (2 * len(cluster_bboxes))

            det_x_center = (bbox[0] + bbox[2]) / 2
            det_y_center = (bbox[1] + bbox[3]) / 2

            # Check BOTH vertical AND horizontal proximity
            # Vertical: within 12% of video height
            # Horizontal: within 40% of video width (text blocks are often wide)
            vert_close = abs(det_y_center - cluster_y_center) < 0.12
            horiz_close = abs(det_x_center - cluster_x_center) < 0.40

            if vert_close and horiz_close:
                cluster_bboxes.append(bbox)

        logger.info(f"RENDER_TEXT: Clustered {len(cluster_bboxes)} nearby detections (of {len(filtered)} filtered)")

        # Union of clustered bboxes
        x_min = min(b[0] for b in cluster_bboxes)
        y_min = min(b[1] for b in cluster_bboxes)
        x_max = max(b[2] for b in cluster_bboxes)
        y_max = max(b[3] for b in cluster_bboxes)

        # Relative padding (5% of bbox dimensions)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        pad_x = bbox_w * 0.08
        pad_y = bbox_h * 0.15

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(1.0, x_max + pad_x)
        y_max = min(1.0, y_max + pad_y)

        # Convert to pixels
        x = int(x_min * video_width)
        y = int(y_min * video_height)
        w = int((x_max - x_min) * video_width)
        h = int((y_max - y_min) * video_height)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Reduced minimum sizes (25% width, 4% height)
        # ═══════════════════════════════════════════════════════════════════
        min_w = int(video_width * 0.25)  # Reduced from 50%
        min_h = int(video_height * 0.04)  # Reduced from 6%

        if w < min_w:
            expand = (min_w - w) // 2
            x = max(0, x - expand)
            w = min_w
        if h < min_h:
            h = min_h

        # Safety: max 90% width to avoid edge issues
        max_w = int(video_width * 0.90)
        if w > max_w:
            x = int((video_width - max_w) / 2)
            w = max_w

        # Keep within screen bounds
        if x + w > video_width:
            x = video_width - w
        if y + h > video_height:
            y = video_height - h
        x = max(0, x)
        y = max(0, y)

        logger.info(f"RENDER_TEXT: CLUSTERED bbox for '{position}': ({x}, {y}, {w}x{h})")
        return (x, y, w, h)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Fallback for top/bottom (skip middle/center entirely)
    # ═══════════════════════════════════════════════════════════════════════
    if position in ("middle", "center"):
        logger.warning(
            f"RENDER_TEXT: No OCR in zone '{position}' — skipping (would obscure content)"
        )
        return (0, 0, 1, 1)  # Signal to skip this overlay

    logger.warning(f"RENDER_TEXT: No detections, using position fallback for '{original_text[:30]}...'")

    # Narrower fallback box (70% width instead of 94%)
    margin = int(video_width * 0.15)
    box_w = int(video_width * 0.70)
    box_h = int(video_height * 0.08)

    if position == "top":
        y = int(video_height * 0.03)
    else:  # bottom
        y = int(video_height * 0.85)

    return (margin, y, box_w, box_h)


def _estimate_text_height(text: str, font_size: int, box_width: int) -> int:
    """Estimate how many lines the text will wrap to and total height needed."""
    # Approximate: ~0.6 * font_size per character width (monospace estimate)
    chars_per_line = max(1, int(box_width / (font_size * 0.52)))
    words = text.split()
    lines = 1
    current_line_len = 0
    for word in words:
        if current_line_len + len(word) + 1 > chars_per_line and current_line_len > 0:
            lines += 1
            current_line_len = len(word)
        else:
            current_line_len += len(word) + 1
    return int(lines * font_size * 1.3)  # 1.3 line spacing


def _wrap_text_for_ass(text: str, max_chars_per_line: int = 25) -> str:
    """
    Word-wrap text for ASS subtitles.

    Uses \\N for hard line breaks in ASS format.
    Default to ~25 chars per line (optimal for mobile viewing).
    """
    if max_chars_per_line <= 0:
        max_chars_per_line = 25

    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if current_len + word_len + 1 > max_chars_per_line and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = word_len
        else:
            current_line.append(word)
            current_len += word_len + (1 if current_len > 0 else 0)

    if current_line:
        lines.append(" ".join(current_line))

    # ASS uses \N for hard line breaks
    return "\\N".join(lines)


def _hex_to_ass_color(hex_color: str, alpha: float = 0.0) -> str:
    """
    Convert hex color (#RRGGBB) to ASS format (&HAABBGGRR).

    ASS uses BGR order and alpha at the start.
    alpha: 0.0 = fully opaque, 1.0 = fully transparent
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    else:
        r, g, b = 255, 255, 255  # default white

    # Convert alpha (0-1 transparent scale) to ASS (0-255 where 0=opaque)
    a = int(alpha * 255)

    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def _generate_ass_subtitles(
    translated_overlays: List[Dict],
    text_detections: List[Dict],
    video_width: int,
    video_height: int,
    video_duration: float,
    video_fps: float,
    subtitle_style: Optional[Dict] = None,
    inpaint_succeeded: bool = True,
) -> str:
    """
    Generate ASS subtitle file with SOTA Logic 3.0.

    Strategy: "Eraser Plate + Text" — two events per overlay:
    1. Layer 0: Dark plate covering original text (eraser)
    2. Layer 1: White translated text on top

    This GUARANTEES original text is covered, even if Inpaint fails.
    """
    # Font size: ~3.2% of video height for mobile readability
    font_size = int(video_height * 0.032)

    # Eraser plate color: FULLY OPAQUE BLACK when inpaint failed, dark gray otherwise
    # ASS format: &HAABBGGRR where AA=alpha (00=opaque, FF=transparent)
    if inpaint_succeeded:
        # Inpaint worked — eraser plate is just a subtle background
        eraser_color = "&H00202020"  # Dark gray, slightly transparent feel
    else:
        # Inpaint FAILED — eraser plate MUST cover original text completely
        eraser_color = "&H00000000"  # Pure black, fully opaque
        logger.info("ASS: Using FULLY OPAQUE BLACK eraser plates (inpaint failed)")

    # ASS Header with two styles:
    # - EraserPlate: solid background to cover original text
    # - TranslatedText: white text with thin outline
    ass_content = f"""[Script Info]
Title: TrafficPlant SOTA Subtitles v3
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: EraserPlate,Arial,20,{eraser_color},{eraser_color},{eraser_color},{eraser_color},0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: TranslatedText,Noto Sans,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1.5,1,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Style breakdown:
    # EraserPlate: solid color plate, BorderStyle=1, no outline/shadow
    # TranslatedText: PrimaryColour=&H00FFFFFF (white BGR), Bold=0, Outline=1.5, Shadow=1

    for i, overlay in enumerate(translated_overlays):
        translated_text = overlay.get("translated_text", "")
        if not translated_text:
            continue

        appears_at = overlay.get("appears_at", 0.0)
        disappears_at = overlay.get("disappears_at", 0.0)
        position = overlay.get("position", "top")

        # Only extend to video end if Gemini didn't provide disappears_at (0 or missing)
        # DO NOT extend short overlays — they should disappear when the original text does
        if disappears_at <= 0:
            disappears_at = video_duration
        # Safety: ensure at least 2s display time if times are nonsensical
        if disappears_at <= appears_at:
            disappears_at = appears_at + 2.0

        # Get bounding box from OCR (zone + time filtered)
        x, y, box_w, box_h = _find_overlay_bbox(
            overlay, text_detections, video_width, video_height,
            appears_at=appears_at, disappears_at=disappears_at, video_fps=video_fps
        )

        # Format times
        start_time = _format_ass_time(appears_at)
        end_time = _format_ass_time(disappears_at)

        # ═══════════════════════════════════════════════════════════════
        # Skip entire overlay if no valid bbox (middle/center zone without OCR)
        # ═══════════════════════════════════════════════════════════════
        # Minimal bbox (0,0,1,1) means "skip this overlay entirely"
        if x == 0 and y == 0 and box_w == 1 and box_h == 1:
            logger.info(
                f"RENDER_TEXT ASS [{i}]: skipping overlay for middle/center zone "
                f"(no OCR detections, would obscure content)"
            )
            continue

        # ═══════════════════════════════════════════════════════════════
        # EVENT 1: ERASER PLATE (Layer 0)
        # ═══════════════════════════════════════════════════════════════
        # Draw solid rectangle using ASS vector drawing commands
        # Relative padding: 8% of bbox width, 15% of bbox height (scales with resolution)
        pad_x = max(8, int(box_w * 0.08))  # min 8px to avoid tiny plates
        pad_y = max(6, int(box_h * 0.15))  # min 6px
        plate_x1 = max(0, x - pad_x)
        plate_y1 = max(0, y - pad_y)
        plate_x2 = min(video_width, x + box_w + pad_x)
        plate_y2 = min(video_height, y + box_h + pad_y)

        # ASS drawing: m = move, l = line
        # Rectangle: move to top-left, line to each corner
        plate_drawing = (
            f"m {plate_x1} {plate_y1} "
            f"l {plate_x2} {plate_y1} "
            f"l {plate_x2} {plate_y2} "
            f"l {plate_x1} {plate_y2}"
        )

        # {\p1} enables drawing mode, {\c&HBBGGRR&} sets fill color
        # Use eraser_color (black when inpaint failed, gray otherwise)
        plate_event = (
            f"Dialogue: 0,{start_time},{end_time},EraserPlate,,0,0,0,,"
            f"{{\\p1\\c{eraser_color}&\\1a&H00&}}{plate_drawing}"
        )
        ass_content += plate_event + "\n"

        # ═══════════════════════════════════════════════════════════════
        # EVENT 2: TRANSLATED TEXT (Layer 1)
        # ═══════════════════════════════════════════════════════════════
        # Word-wrap text (~28 chars for mobile)
        wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=28)

        # Escape special ASS characters
        safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

        # Position: center of plate
        cx = (plate_x1 + plate_x2) // 2
        cy = (plate_y1 + plate_y2) // 2

        # {\an5} = center alignment, {\pos(x,y)} = absolute position
        # {\fad(200,200)} = 200ms fade in/out
        text_event = (
            f"Dialogue: 1,{start_time},{end_time},TranslatedText,,0,0,0,,"
            f"{{\\an5\\pos({cx},{cy})\\fad(200,200)}}{safe_text}"
        )
        ass_content += text_event + "\n"

        logger.info(
            f"RENDER_TEXT ASS [{i}]: position='{position}' "
            f"plate=({plate_x1},{plate_y1})-({plate_x2},{plate_y2}) "
            f"text_center=({cx},{cy}) "
            f"time={start_time}-{end_time}"
        )

    return ass_content


def stage_render_text(
    video_path: str,
    translated_overlays: List[Dict],
    text_detections: List[Dict],
    resolution: Tuple[int, int],
    fps: float,
    subtitle_style: Optional[Dict] = None,
    inpaint_succeeded: bool = True,
) -> str:
    """
    Render translated text overlays onto video using ASS subtitles.

    ASS (Advanced SubStation Alpha) provides:
    - Precise positioning with {\pos(x,y)}
    - Opaque background boxes with BorderStyle=3
    - Fade-in/fade-out animations with {\fad(200,200)}
    - Smart word-wrapping (~25 chars per line for mobile)
    - Safe zone awareness (avoid TikTok UI elements)

    When inpaint_succeeded=False, eraser plates use fully opaque black
    to ensure original text is covered.

    Zero VRAM — pure CPU/ffmpeg.
    """
    mode = "ERASER-PLATE-REQUIRED" if not inpaint_succeeded else "normal"
    logger.info(f"Stage: RENDER_TEXT ASS ({len(translated_overlays)} overlays, mode={mode})")

    if not translated_overlays:
        logger.info("No overlays to render, skipping")
        return video_path

    video_width, video_height = resolution
    output_path = video_path.replace(".mp4", "_textrendered.mp4")

    # Get video duration to ensure overlays extend to end
    import cv2 as cv2_render
    cap = cv2_render.VideoCapture(video_path)
    frame_count = int(cap.get(cv2_render.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2_render.CAP_PROP_FPS) or fps
    video_duration = frame_count / video_fps if video_fps > 0 else 10.0
    cap.release()
    logger.info(f"RENDER_TEXT: Video duration = {video_duration:.1f}s, resolution = {video_width}x{video_height}")

    # Generate ASS subtitle content
    ass_content = _generate_ass_subtitles(
        translated_overlays=translated_overlays,
        text_detections=text_detections,
        video_width=video_width,
        video_height=video_height,
        video_duration=video_duration,
        video_fps=video_fps,
        subtitle_style=subtitle_style,
        inpaint_succeeded=inpaint_succeeded,
    )

    # Write ASS file to temp location
    ass_path = video_path.replace(".mp4", "_subs.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    logger.info(f"RENDER_TEXT: Generated ASS file: {ass_path}")

    # Log ASS content for debugging (first 1000 chars)
    logger.info(f"RENDER_TEXT: ASS preview:\n{ass_content[:1000]}")

    # Use ffmpeg with ASS filter
    # Note: ASS filter path needs escaping for Windows-style paths or special chars
    # We use the simple form since we're on Linux
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]

    logger.info(f"RENDER_TEXT: Running ffmpeg with ASS filter")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error(f"RENDER_TEXT ffmpeg failed (rc={result.returncode}): {result.stderr[-500:]}")
        # Non-critical — return original video
        return video_path

    if not os.path.exists(output_path):
        logger.error("RENDER_TEXT: output file not created")
        return video_path

    # Clean up temp ASS file
    try:
        os.remove(ass_path)
    except Exception:
        pass

    logger.info(f"RENDER_TEXT: Successfully rendered {len(translated_overlays)} overlay(s) via ASS")
    return output_path


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
    Translation is done SERVER-SIDE via Gemini 3 Pro (TranslatorAgent).
    The translated text must be passed via 'translated_text' in the job payload.
    This function should never be called directly — if it is, it means
    the server didn't provide translated text, which is a pipeline error.
    """
    raise RuntimeError(
        f"No translated text provided by server. "
        f"Translation must be done server-side via TranslatorAgent (Gemini 3 Pro). "
        f"Source: {source_lang}, Target: {target_lang}, Text length: {len(text)}"
    )


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
            # Build TTS payload with language_code for proper pronunciation
            tts_payload = {
                "text": text,
                "model_id": "eleven_v3",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.85,
                    "style": 0.3,
                },
            }
            # Add language_code for eleven_v3 (supports 70+ languages)
            lang_code = normalize_language_code(target_language)
            if lang_code:
                tts_payload["language_code"] = lang_code
                logger.info(f"ElevenLabs: using language_code={lang_code} (from {target_language})")

            tts_resp = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=tts_payload,
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

    wav, sr, _ = f5.infer(
        ref_file=reference_audio,
        ref_text="",  # Auto-transcribe reference
        gen_text=text,
        file_wave=output_path,
        seed=None,
    )

    return output_path


def stage_tts(text: str, reference_audio: str, mm: ModelManager, target_language: str = "en") -> str:
    """Generate speech: ElevenLabs API (primary) → F5-TTS local (fallback)."""
    logger.info(f"Stage: TTS (target_language={target_language})")

    # Primary: ElevenLabs (runs from US GPU IP — no geo-block)
    result = _tts_elevenlabs(text, reference_audio, target_language=target_language)
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
    """Assess output quality using no-reference metric (MUSIQ)."""
    logger.info("Stage: QUALITY_CHECK (MUSIQ no-reference)")

    import pyiqa
    import cv2

    # MUSIQ: no-reference image quality metric (0-100 scale)
    try:
        metric = pyiqa.create_metric('musiq', device='cuda')
    except Exception:
        # Fallback to NIQE if MUSIQ unavailable
        metric = pyiqa.create_metric('niqe', device='cuda')
        logger.info("Using NIQE metric (MUSIQ unavailable)")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = max(1, frame_count // 10)

    scores = []
    for i in range(0, frame_count, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).cuda()

            score = metric(frame_tensor).item()
            # MUSIQ returns 0-100, normalize to 0-1
            normalized = score / 100.0 if score > 1.0 else score
            scores.append(normalized)

    cap.release()

    avg_score = sum(scores) / len(scores) if scores else 0
    passed = avg_score >= threshold

    logger.info(f"Quality: {avg_score:.3f} (threshold: {threshold}, {'PASS' if passed else 'FAIL'})")

    return {
        "score": avg_score,
        "threshold": threshold,
        "passed": passed,
        "frame_scores": scores
    }


def stage_assemble(
    video_path: str,
    tts_audio: Optional[str],
    background_audio: Optional[str],
    output_path: str
) -> str:
    """Assemble final video with mixed audio."""
    logger.info("Stage: ASSEMBLE")

    if not tts_audio:
        raise RuntimeError("No TTS audio available — cannot assemble localized video")

    if background_audio and os.path.exists(background_audio):
        # Mix TTS voice with background audio
        mixed_audio = output_path.replace(".mp4", "_mixed.wav")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tts_audio,
            "-i", background_audio,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 0.3[a]",
            "-map", "[a]",
            mixed_audio
        ], capture_output=True, check=False)
        audio_to_use = mixed_audio if os.path.exists(mixed_audio) else tts_audio
    else:
        # No background audio — use TTS audio directly
        audio_to_use = tts_audio

    # Combine video with audio
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_to_use,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v",
        "-map", "1:a",
        "-shortest",
        output_path
    ], capture_output=True, check=False)

    if not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg assemble failed — output not created: {output_path}")

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

    # Run VideoPainter diagnostics on first job (lazy init)
    diag = get_videopainter_diagnostics()
    if diag and diag.get("errors"):
        logger.warning(f"VideoPainter issues: {diag['errors']}")

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
            translated_overlays=job_input.get("translated_overlays"),  # Pre-translated text overlays
            subtitle_style=job_input.get("subtitle_style"),  # Original subtitle style from manifest
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
        critical_failure = None  # Set if a CRITICAL stage fails

        for stage_name in stages:
            if critical_failure:
                break  # Stop pipeline on critical failure

            t0 = time.time()

            try:
                if stage_name == "preprocess":
                    # Graceful: extract raw audio if demucs fails
                    try:
                        result = stage_preprocess(state["video_path"], mm)
                        state["vocals_path"] = result["vocals_path"]
                        state["background_path"] = result["background_path"]
                    except Exception as e:
                        logger.warning(f"Audio separation failed: {e}, extracting raw audio")
                        raw_audio = state["video_path"].replace(".mp4", "_raw_audio.wav")
                        subprocess.run([
                            "ffmpeg", "-y", "-i", state["video_path"],
                            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                            raw_audio
                        ], capture_output=True, check=False)
                        if os.path.exists(raw_audio):
                            state["vocals_path"] = raw_audio
                        metrics.errors.append(f"preprocess: {e} (fallback to raw audio)")

                elif stage_name == "detect_text":
                    result = stage_detect_text(state["video_path"], mm)
                    state["text_detections"] = result["detections"]
                    state["text_resolution"] = result["resolution"]  # (width, height)
                    state["text_fps"] = result["fps"]

                elif stage_name == "create_mask":
                    state["mask_path"] = stage_create_mask(
                        state["video_path"],
                        state.get("text_detections", []),
                        mm
                    )

                elif stage_name == "inpaint":
                    video_path, inpaint_ok = stage_inpaint(
                        state["video_path"],
                        state["mask_path"],
                        mm,
                        errors=metrics.errors  # Track inpaint failures
                    )
                    state["video_path"] = video_path
                    state["inpaint_succeeded"] = inpaint_ok

                elif stage_name == "render_text":
                    if config.translated_overlays:
                        state["video_path"] = stage_render_text(
                            state["video_path"],
                            config.translated_overlays,
                            state.get("text_detections", []),
                            state.get("text_resolution", (1920, 1080)),
                            state.get("text_fps", 30.0),
                            config.subtitle_style,
                            inpaint_succeeded=state.get("inpaint_succeeded", True),
                        )
                    else:
                        logger.info("RENDER_TEXT: No translated overlays provided, skipping")

                elif stage_name == "transcribe":
                    state["transcript"] = stage_transcribe(
                        state["video_path"],
                        state.get("vocals_path"),
                        mm
                    )

                elif stage_name == "translate":
                    # Translation MUST be provided by server (Gemini 3 Pro)
                    if config.translated_text:
                        logger.info("Using pre-translated text from server (Gemini 3 Pro)")
                        state["translated_text"] = config.translated_text
                    elif state.get("transcript"):
                        src_lang = state["transcript"]["language"]
                        # Use proper language comparison (handles pt == pt-BR, en == en-US, etc.)
                        if are_languages_same(src_lang, config.target_language):
                            logger.info(f"Source and target languages are same family: {src_lang} ≈ {config.target_language}")
                            state["translated_text"] = state["transcript"]["full_text"]
                        else:
                            # No server translation and languages differ — critical error
                            raise RuntimeError(
                                f"Server must provide translated_text for {src_lang}→{config.target_language}. "
                                f"Local translation removed — use TranslatorAgent (Gemini 3 Pro)."
                            )
                    else:
                        raise RuntimeError("No transcript available for translation")

                elif stage_name == "tts":
                    if config.voice_clone and state.get("translated_text"):
                        state["tts_audio"] = stage_tts(
                            state["translated_text"],
                            state.get("vocals_path") or video_path,
                            mm,
                            target_language=config.target_language,
                        )
                    elif not state.get("translated_text"):
                        raise RuntimeError("No translated text available for TTS")

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
                        logger.warning(f"Quality check failed: {qc_result['score']:.2f} < {qc_result['threshold']}")

                elif stage_name == "assemble":
                    output_path = video_path.replace(".mp4", "_final.mp4")
                    state["video_path"] = stage_assemble(
                        state["video_path"],
                        state.get("tts_audio"),
                        state.get("background_path"),
                        output_path
                    )

            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                metrics.errors.append(f"{stage_name}: {str(e)}")

                if stage_name in CRITICAL_STAGES:
                    critical_failure = f"Critical stage '{stage_name}' failed: {e}"
                    logger.error(f"CRITICAL FAILURE: {critical_failure}")
                # Optional stages: continue

            metrics.stage_times[stage_name] = time.time() - t0
            logger.info(f"Stage {stage_name} {'FAILED' if stage_name in [e.split(':')[0] for e in metrics.errors] else 'OK'} in {metrics.stage_times[stage_name]:.1f}s")

            # Fire progress callback if registered
            if _progress_callback:
                try:
                    _progress_callback(stage_name, metrics.stage_times[stage_name], metrics.errors)
                except Exception:
                    pass  # Never let callback failures break the pipeline

        # If critical stage failed — return error WITHOUT uploading
        if critical_failure:
            return {
                "status": "error",
                "error": critical_failure,
                "metrics": metrics.to_dict(),
                "transcript": state.get("transcript"),
            }

        # Upload result to R2
        try:
            r2 = get_r2_storage()
            timestamp = int(time.time())
            remote_key = f"{config.r2_prefix}/{job_id}_{timestamp}.mp4"
            output_url = r2.upload(state["video_path"], remote_key)
        except Exception as e:
            logger.error(f"R2 upload failed: {e}")
            output_url = f"file://{state['video_path']}"
            metrics.errors.append(f"r2_upload: {str(e)}")

        total_time = time.time() - start_time

        logger.info(f"=" * 60)
        logger.info(f"Job {job_id} completed in {total_time:.1f}s")
        logger.info(f"Errors: {len(metrics.errors)} (non-critical)")
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
# VideoPainter diagnostics - run lazily on first job, not at import time
# (import-time diagnostics break Docker build sanity check)
_videopainter_diag = None


def get_videopainter_diagnostics() -> Dict[str, Any]:
    """Get cached VideoPainter diagnostics, running them on first call."""
    global _videopainter_diag
    if _videopainter_diag is None:
        try:
            _videopainter_diag = diagnose_videopainter()
            if _videopainter_diag["errors"]:
                logger.warning(f"VideoPainter diagnostics: {len(_videopainter_diag['errors'])} issues found")
            else:
                logger.info("VideoPainter diagnostics: All components OK")
        except Exception as e:
            logger.error(f"VideoPainter diagnostics failed: {e}")
            _videopainter_diag = {"errors": [str(e)]}
    return _videopainter_diag


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
