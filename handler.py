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
│ DeepSeek-OCR-2     │ 7GB   │ 1 (keep) │
│ PaddleOCR          │ 2GB   │ 1 (fallb)│
├────────────────────┼───────┼──────────┤
│ Peak usage         │ ~45GB │          │
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

# =============================================================================
# Monkey-patch: transformers.utils.FLAX_WEIGHTS_NAME
# Removed in transformers >=4.47, but VideoPainter's custom diffusers imports it.
# Must be applied BEFORE any diffusers/VideoPainter import.
# =============================================================================
try:
    import transformers.utils
    if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
        transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
except ImportError:
    pass

import torch
import numpy as np

# RTL (Right-to-Left) languages for subtitle rendering
RTL_LANGUAGES = {"ar", "he", "fa", "ur", "yi"}

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
    """Video localization pipeline stages.
    Order matters: stages execute in enum order by default.
    TRANSCRIBE moved before DETECT_TEXT — both are independent
    (transcribe uses vocals_path, detect_text uses video_path).
    """
    PREPROCESS = "preprocess"          # Audio separation (Demucs)
    TRANSCRIBE = "transcribe"          # Speech recognition (Whisper) — uses vocals from preprocess
    DETECT_TEXT = "detect_text"        # OCR (DeepSeek/PaddleOCR)
    CREATE_MASK = "create_mask"        # SAM 2.1 mask generation
    INPAINT = "inpaint"                # VideoPainter/ProPainter
    TRANSLATE = "translate"            # Server-side (Gemini 3 Pro)
    RENDER_TEXT = "render_text"        # ASS subtitles (ffmpeg)
    TTS = "tts"                        # ElevenLabs → F5-TTS
    LIPSYNC = "lipsync"                # VideoRetalking / MuseTalk
    ENHANCE = "enhance"                # GFPGAN face enhancement
    UPSCALE = "upscale"                # Real-ESRGAN
    QUALITY_CHECK = "quality_check"    # Auto quality assessment
    ASSEMBLE = "assemble"              # Final mix


# Stages that MUST succeed for the job to produce a valid localized video.
# If any of these fail, the job returns status="error" instead of "success".
CRITICAL_STAGES = {"transcribe", "translate", "tts", "assemble"}

# Stages where failure degrades output quality but doesn't block the job.
QUALITY_CRITICAL_STAGES = {"detect_text", "create_mask", "render_text"}


# =============================================================================
# Language Code Normalization
# =============================================================================

LANGUAGE_CODE_MAP = {
    "en": ["en", "en-US", "en-GB", "en-AU"],
    "pt": ["pt", "pt-PT"],              # European Portuguese
    "pt-BR": ["pt-BR"],                 # Brazilian Portuguese
    "es": ["es", "es-ES", "es-AR"],
    "es-MX": ["es-MX"],                 # Mexican Spanish (distinct)
    "ru": ["ru", "ru-RU"],
    "de": ["de", "de-DE", "de-AT", "de-CH"],
    "fr": ["fr", "fr-FR", "fr-CA"],
    "zh-CN": ["zh-CN", "zh"],           # Simplified Chinese
    "zh-TW": ["zh-TW", "zh-HK"],       # Traditional Chinese
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
    "sr": ["sr", "sr-Cyrl"],            # Serbian Cyrillic
    "sr-Latn": ["sr-Latn"],             # Serbian Latin
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

    # Pre-cloned ElevenLabs voice ID (from server-side caching)
    # If provided, skips per-call voice cloning and uses this voice directly
    elevenlabs_voice_id: Optional[str] = None

    # Adaptive pipeline config from server (VideoProfile)
    video_profile: Optional[Dict] = None

    # Gemini-generated target caption style (highest priority for ASS rendering)
    target_caption_style: Optional[Dict] = None

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
    methods_used: Dict[str, str] = field(default_factory=dict)
    quality_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "stage_times_ms": {k: v * 1000 for k, v in self.stage_times.items()},
            "model_load_times_ms": {k: v * 1000 for k, v in self.model_loads.items()},
            "quality_scores": self.quality_scores,
            "errors": self.errors,
            "methods_used": self.methods_used,
            "quality_warnings": self.quality_warnings,
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
        "deepseek_ocr": (7, 1, True),   # 3B params, BF16 = ~6.8GB
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

    def preload(self):
        """Preload all keep_loaded models at startup to eliminate first-job latency."""
        logger.info("ModelManager: Preloading keep_loaded models...")
        for name, (vram, priority, keep) in self.MODEL_CONFIGS.items():
            if keep:
                try:
                    logger.info(f"  Preloading {name} ({vram}GB)...")
                    self.load(name)
                    logger.info(f"  Preloaded {name} OK")
                except Exception as e:
                    logger.warning(f"  Preload {name} failed: {e}")

        loaded_vram = self._get_used_vram()
        logger.info(f"Preload complete: {len(self.loaded)} models, {loaded_vram}GB VRAM used")

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

        if name == "deepseek_ocr":
            from transformers import AutoModel, AutoTokenizer
            model_id = "deepseek-ai/DeepSeek-OCR-2"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_safetensors=True,
                _attn_implementation="eager",  # flash_attention_2 needs flash-attn pkg (CUDA version mismatch in CI)
            )
            model = model.eval().cuda().to(torch.bfloat16)
            return {"model": model, "tokenizer": tokenizer}

        elif name == "paddleocr":
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
            predictor = SAM2VideoPredictor.from_pretrained(
                "facebook/sam2.1-hiera-large"
            )
            # Force Float32 to avoid BFloat16/Float32 dtype mismatch during propagation
            predictor.float()
            return predictor

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
            from diffusers.models.branch_cogvideox import CogvideoXBranchModel

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

            logger.info("VideoPainter: Loading CogvideoXBranchModel from branch path...")
            branch_model = CogvideoXBranchModel.from_pretrained(
                branch_path, torch_dtype=torch.bfloat16
            )

            logger.info("VideoPainter: Loading CogVideoXI2VInpaintAnyLPipeline...")
            pipe = CogVideoXI2VInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch_model,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            ).to(self.device)

            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("VideoPainter: ✓ Pipeline loaded with xformers")
            except Exception as e:
                logger.warning(f"VideoPainter: xformers not available ({e}), using default attention")
            logger.info("VideoPainter: ✓ Pipeline loaded successfully")
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


def stage_detect_text(video_path: str, mm: ModelManager, pipeline_config: dict = None) -> Dict:
    """Detect text overlays using DeepSeek-OCR-2 + PaddleOCR (merged for best coverage)."""
    config = pipeline_config or {}
    confidence_threshold = config.get("ocr_confidence_threshold", 0.3)
    sample_count = config.get("ocr_sample_count", 8)

    logger.info(f"Stage: DETECT_TEXT (dual-OCR merge strategy, conf_threshold={confidence_threshold}, sample_count={sample_count})")

    deepseek_result = None
    paddle_result = None

    # Run DeepSeek-OCR (high quality, slow)
    try:
        deepseek_result = _detect_text_deepseek(video_path, mm, confidence_threshold=confidence_threshold, sample_count=sample_count)
        logger.info(f"DETECT_TEXT: DeepSeek found {deepseek_result['unique_regions']} unique regions")
    except Exception as e:
        logger.warning(f"DeepSeek-OCR failed: {e}")

    # ALWAYS run PaddleOCR as supplementary scan (fast, catches what DeepSeek misses)
    try:
        paddle_result = _detect_text_paddle(video_path, mm, confidence_threshold=confidence_threshold, sample_count=sample_count)
        logger.info(f"DETECT_TEXT: PaddleOCR found {paddle_result['unique_regions']} unique regions")
    except Exception as e:
        logger.warning(f"PaddleOCR also failed: {e}")

    # Merge results from both OCR engines
    if deepseek_result and paddle_result:
        merged = _merge_ocr_results(deepseek_result, paddle_result)
        mm.metrics.methods_used["ocr"] = "deepseek_ocr+paddleocr"
        logger.info(
            f"DETECT_TEXT: Merged {deepseek_result['unique_regions']} DeepSeek + "
            f"{paddle_result['unique_regions']} PaddleOCR → {merged['unique_regions']} unique regions"
        )
        return merged
    elif deepseek_result:
        mm.metrics.methods_used["ocr"] = "deepseek_ocr"
        return deepseek_result
    elif paddle_result:
        mm.metrics.methods_used["ocr"] = "paddleocr"
        return paddle_result
    else:
        raise RuntimeError("Both DeepSeek-OCR and PaddleOCR failed")


def _merge_ocr_results(primary: Dict, secondary: Dict) -> Dict:
    """Merge detections from two OCR engines, avoiding duplicates."""
    merged_dets = list(primary["detections"])
    added = 0

    for det in secondary["detections"]:
        is_dupe = False
        for existing in merged_dets:
            iou = _calculate_iou(det["bbox_norm"], existing["bbox_norm"])
            if iou > 0.5:  # Lower threshold for cross-engine merge
                is_dupe = True
                break
        if not is_dupe:
            merged_dets.append(det)
            added += 1

    logger.info(f"MERGE_OCR: Added {added} new regions from secondary OCR engine")

    return {
        "detections": merged_dets,
        "total_found": primary["total_found"] + secondary["total_found"],
        "unique_regions": len(merged_dets),
        "fps": primary["fps"],
        "frame_count": primary["frame_count"],
        "resolution": primary["resolution"],
    }


def _is_likely_false_positive(frame_rgb, bbox_norm, frame_w, frame_h):
    """
    Reject OCR detections that are likely false positives on body parts.

    Only reject if BOTH conditions are true (AND logic):
    1. High skin-tone ratio (>65%) AND
    2. Low edge density (<5%)

    Skip filter entirely for bottom 30% of frame (almost always real captions).
    """
    import cv2

    # Skip filter for bottom captions — text in bottom zone is almost always real
    y_center = (bbox_norm[1] + bbox_norm[3]) / 2
    if y_center > 0.70:
        return False  # Bottom zone = trust the OCR

    x1 = int(bbox_norm[0] * frame_w)
    y1 = int(bbox_norm[1] * frame_h)
    x2 = int(bbox_norm[2] * frame_w)
    y2 = int(bbox_norm[3] * frame_h)

    # Clamp to frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    region = frame_rgb[y1:y2, x1:x2]
    if region.size == 0:
        return True  # Empty region = definitely false

    # Check 1: Skin-tone detection (HSV-based)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([25, 180, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Also check for darker skin tones
    lower_skin2 = np.array([0, 20, 40])
    upper_skin2 = np.array([30, 200, 200])
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    skin_ratio = max(
        np.sum(skin_mask > 0) / skin_mask.size,
        np.sum(skin_mask2 > 0) / skin_mask2.size
    )

    # Check 2: Edge density — real text has many sharp edges
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # Only reject if BOTH: high skin AND low edges
    # This prevents rejecting real text that overlaps with skin-toned areas
    if skin_ratio > 0.65 and edge_ratio < 0.05:
        return True  # High skin + smooth = body part, not text

    return False


def _detect_text_deepseek(video_path: str, mm: ModelManager, confidence_threshold: float = 0.3, sample_count: int = 8) -> Dict:
    """Detect text using DeepSeek-OCR-2 (model.infer API with grounding mode).

    DeepSeek-OCR-2 outputs: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>text
    Coordinates are normalized 0-999, independent of image resolution.
    """
    import cv2
    import re
    import tempfile

    deepseek = mm.load("deepseek_ocr")
    model, tokenizer = deepseek["model"], deepseek["tokenizer"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    sample_rate = max(1, int(fps))  # 1 fps sampling (DeepSeek-OCR-2 is ~10-28s/frame)

    # Regex for grounding output: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    det_pattern = re.compile(r'<\|ref\|>([^<]*)<\|/ref\|><\|det\|>([^<]*)<\|/det\|>')

    # Temp dir for frame images (model.infer needs file path)
    tmp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

    try:
        for i in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as temp image (model.infer requires file path)
            frame_path = os.path.join(tmp_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)

            try:
                # Use grounding OCR prompt — returns text + bounding boxes
                res = model.infer(
                    tokenizer,
                    prompt="<image>\n<|grounding|>OCR this image.",
                    image_file=frame_path,
                    output_path=tmp_dir,
                    base_size=1024,
                    image_size=768,
                    crop_mode=False,
                    save_results=False,
                    eval_mode=True,  # Must be True to return text (False streams to stdout, returns None)
                )

                # res is the raw model output string with <|ref|>/<|det|> tags
                if not res or not isinstance(res, str):
                    logger.warning(f"DeepSeek-OCR frame {i}: res is {type(res).__name__}, value={repr(res)[:200]}")
                    continue

                # Log first frame's output for debugging
                if i == 0 or (i > 0 and not detections):
                    logger.info(f"DeepSeek-OCR frame {i}: res type={type(res).__name__}, len={len(res)}, first200={repr(res[:200])}")

                # Parse grounding tags (cap at 50 per frame to avoid hallucinations)
                frame_dets = 0
                MAX_DETS_PER_FRAME = 50
                for match in det_pattern.finditer(res):
                    if frame_dets >= MAX_DETS_PER_FRAME:
                        logger.warning(f"Frame {i}: hit {MAX_DETS_PER_FRAME} det cap, skipping rest")
                        break
                    label = match.group(1).strip()
                    coords_str = match.group(2).strip()

                    try:
                        coords_list = eval(coords_str)  # [[x1,y1,x2,y2], ...]
                        if not isinstance(coords_list, list):
                            continue
                        # Handle both [[x1,y1,x2,y2]] and [x1,y1,x2,y2]
                        if coords_list and not isinstance(coords_list[0], list):
                            coords_list = [coords_list]
                    except Exception:
                        continue

                    for coords in coords_list:
                        if len(coords) < 4:
                            continue
                        # Validate all coords are numbers in 0-999 range
                        if not all(isinstance(c, (int, float)) and 0 <= c <= 999 for c in coords[:4]):
                            continue
                        # Coords are normalized 0-999 → convert to pixel coords
                        nx1, ny1, nx2, ny2 = coords[:4]
                        # Skip tiny boxes but EXEMPT bottom 20% of frame (captions live there)
                        is_bottom_region = ny1 > 800  # Bottom 20% of 0-999 range
                        min_w = 5 if is_bottom_region else 8
                        min_h = 2 if is_bottom_region else 3
                        if (nx2 - nx1) < min_w or (ny2 - ny1) < min_h:
                            logger.debug(f"DeepSeek frame {i}: REJECTED tiny box ({nx2-nx1:.0f}x{ny2-ny1:.0f} norm) text='{label[:30]}' bottom={is_bottom_region}")
                            continue
                        # REJECT oversized boxes — DeepSeek-OCR hallucinates subject-grounding boxes
                        box_w_norm = nx2 - nx1
                        box_h_norm = ny2 - ny1
                        box_area = box_w_norm * box_h_norm
                        # Reject if area > 15% of frame (999*999) or both dims > 50%/25% of frame
                        if box_area > 150000 or (box_w_norm > 500 and box_h_norm > 250):
                            logger.warning(f"DeepSeek frame {i}: REJECTED oversized box ({box_w_norm:.0f}x{box_h_norm:.0f} norm, area={box_area:.0f}) text='{label[:30]}' — likely hallucination")
                            continue
                        px1 = int(nx1 * width / 999)
                        py1 = int(ny1 * height / 999)
                        px2 = int(nx2 * width / 999)
                        py2 = int(ny2 * height / 999)

                        # False positive filter: reject skin-tone / low-edge regions
                        bbox_norm_check = [nx1 / 999, ny1 / 999, nx2 / 999, ny2 / 999]
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            if _is_likely_false_positive(frame_rgb, bbox_norm_check, width, height):
                                logger.info(f"DeepSeek frame {i}: REJECTED: likely false positive (skin/low-edge) text='{label[:30]}' bbox_norm={[f'{v:.3f}' for v in bbox_norm_check]}")
                                continue
                        except Exception as fp_err:
                            logger.debug(f"False positive check failed: {fp_err}")

                        det_text = label
                        if det_text and len(det_text) > 1:
                            detections.append({
                                "frame_idx": i,
                                "timestamp": i / fps,
                                "bbox": [[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                                "bbox_norm": [nx1 / 999, ny1 / 999, nx2 / 999, ny2 / 999],
                                "text": det_text,
                                "confidence": 0.95,
                                "label": label,
                            })
                            frame_dets += 1

            except Exception as frame_err:
                logger.warning(f"DeepSeek-OCR frame {i} error: {frame_err}")

    finally:
        cap.release()
        # Clean up temp frames
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mm.metrics.methods_used["ocr"] = "deepseek_ocr"
    logger.info(f"DeepSeek-OCR raw detections: {len(detections)} across {frame_count} frames")
    unique_detections = _dedupe_detections(detections)
    logger.info(f"DeepSeek-OCR unique detections: {len(unique_detections)}")

    return {
        "detections": unique_detections,
        "total_found": len(detections),
        "unique_regions": len(unique_detections),
        "fps": fps,
        "frame_count": frame_count,
        "resolution": (width, height),
    }


def _detect_text_paddle(video_path: str, mm: ModelManager, confidence_threshold: float = 0.3, sample_count: int = 8) -> Dict:
    """Detect text overlays using PaddleOCR (fallback)."""
    import cv2

    ocr = mm.load("paddleocr")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    sample_rate = max(1, int(fps / 4))  # Sample 4 frames per second (better caption coverage)

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

                if conf > max(0.5, confidence_threshold) and len(text) > 1:
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
    mm.metrics.methods_used["ocr"] = "paddleocr"

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


def _dedupe_detections(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Remove duplicate detections based on spatial IoU + text similarity.

    Uses IoU > 0.5 (spatial overlap) AND Levenshtein > 0.6 (text similarity)
    to identify duplicates. This avoids:
    - Exact-match-only dedup missing OCR variants of same text
    - Over-aggressive dedup removing distinct text at similar positions
    """
    if not detections:
        return []

    unique = []
    for det in detections:
        is_dupe = False
        for existing in unique:
            iou = _calculate_iou(det["bbox_norm"], existing["bbox_norm"])
            if iou > iou_threshold:
                # Spatial overlap — check text similarity
                text_sim = _levenshtein_ratio(det.get("text", ""), existing.get("text", ""))
                if text_sim > 0.6:
                    is_dupe = True
                    # Keep the one with higher confidence
                    if det.get("confidence", 0) > existing.get("confidence", 0):
                        idx = unique.index(existing)
                        unique[idx] = det
                    break
        if not is_dupe:
            unique.append(det)

    logger.info(f"DEDUPE: {len(detections)} raw → {len(unique)} unique (IoU>{iou_threshold}, text_sim>0.6)")
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


def stage_create_mask(video_path: str, detections: List[Dict], mm: ModelManager, pipeline_config: dict = None, video_profile: dict = None) -> str:
    """Create temporally consistent masks using SAM 2.1."""
    config = pipeline_config or {}
    MAX_SAM2_PROMPTS = config.get("sam2_max_prompts", 25)
    zone_quotas = config.get("sam2_zone_quotas", {"top": 8, "mid": 5, "bot": 8})
    dilation_px = config.get("mask_dilation_px", 10)
    face_protection = config.get("face_protection", False)
    face_regions = (video_profile or {}).get("face_regions", [])

    logger.info(f"Stage: CREATE_MASK (SAM 2.1, max_prompts={MAX_SAM2_PROMPTS}, dilation={dilation_px}px, face_protection={face_protection})")

    if not detections:
        logger.info("No text detections, skipping mask creation")
        return None

    sam2 = mm.load("sam2")

    # Initialize video predictor
    inference_state = sam2.init_state(video_path)

    # Zone-balanced selection: ensure top, middle, and bottom all get SAM2 prompts
    # Without this, bottom captions (lower confidence) get crowded out by top text
    filtered = [d for d in detections if d.get("confidence", 0.5) >= 0.3]

    # Split by zone based on bbox_norm y-position
    top_dets = []
    mid_dets = []
    bot_dets = []
    for d in filtered:
        bbox_norm = d.get("bbox_norm", [0, 0, 1, 1])
        y_center = (bbox_norm[1] + bbox_norm[3]) / 2
        if y_center < 0.35:
            top_dets.append(d)
        elif y_center > 0.65:
            bot_dets.append(d)
        else:
            mid_dets.append(d)

    # Sort each zone by confidence
    for zone in [top_dets, mid_dets, bot_dets]:
        zone.sort(key=lambda d: d.get("confidence", 0.5), reverse=True)

    # Allocate prompts using config-driven zone quotas
    sorted_dets = []
    bot_quota = min(len(bot_dets), zone_quotas.get("bot", 8))
    top_quota = min(len(top_dets), zone_quotas.get("top", 8))
    mid_quota = min(len(mid_dets), zone_quotas.get("mid", 5))

    sorted_dets.extend(top_dets[:top_quota])
    sorted_dets.extend(mid_dets[:mid_quota])
    sorted_dets.extend(bot_dets[:bot_quota])

    logger.info(
        f"CREATE_MASK: {len(detections)} total, {len(filtered)} after conf filter. "
        f"Zone split: top={len(top_dets)}(using {top_quota}), mid={len(mid_dets)}(using {mid_quota}), "
        f"bot={len(bot_dets)}(using {bot_quota}). Total prompts: {len(sorted_dets)}"
    )

    # Add prompts for each text region
    for i, det in enumerate(sorted_dets[:MAX_SAM2_PROMPTS]):
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
    total_white_pixels = 0
    for frame_idx, obj_ids, masks in sam2.propagate_in_video(inference_state):
        # masks: torch.Tensor (num_objects, H, W) or (num_objects, 1, H, W)
        # CRITICAL: .float() converts BFloat16→Float32 before numpy (avoids dtype mismatch)
        masks_np = masks.cpu().float().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(1)  # (N, 1, H, W) → (N, H, W)
        combined_mask = np.zeros(masks_np.shape[1:], dtype=np.uint8)
        for mask in masks_np:
            combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8) * 255)
        mask_frames[frame_idx] = combined_mask
        total_white_pixels += int(np.sum(combined_mask > 0))

    # Validate masks are non-empty (all-black masks mean SAM2 failed to segment)
    if total_white_pixels == 0:
        logger.error(
            f"CREATE_MASK: All mask frames are BLACK (0 white pixels across {len(mask_frames)} frames). "
            f"SAM2 failed to segment text regions. Returning None so inpaint knows mask failed."
        )
        return None

    logger.info(f"CREATE_MASK: {len(mask_frames)} frames, {total_white_pixels} total white pixels (pre-dilation)")

    # Face protection: zero out face regions in masks to prevent face corruption
    if face_protection and face_regions:
        face_zeroed = 0
        for fidx in mask_frames:
            h, w = mask_frames[fidx].shape[:2]
            for face in face_regions:
                fx1, fy1, fx2, fy2 = face
                margin = 0.02
                y1 = max(0, int((fy1 - margin) * h))
                y2 = min(h, int((fy2 + margin) * h))
                x1 = max(0, int((fx1 - margin) * w))
                x2 = min(w, int((fx2 + margin) * w))
                pixels_before = int(np.sum(mask_frames[fidx][y1:y2, x1:x2] > 0))
                mask_frames[fidx][y1:y2, x1:x2] = 0
                face_zeroed += pixels_before
        logger.info(f"CREATE_MASK: Face protection zeroed {face_zeroed} mask pixels across {len(face_regions)} face region(s)")

    # Two-stage dilation:
    # 1. Standard elliptical dilation everywhere (covers text edges, shadows, glow)
    # 2. Extra HORIZONTAL dilation in bottom 25% (bridges gaps between separate caption words)
    # NOTE: Morphological closing (merging letters into blocks) caused massive regressions
    # in iter2-4. Zone-specific horizontal dilation is safe — it only affects the caption zone.
    import cv2 as _cv2
    kernel_size = dilation_px * 2 + 1
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Extra horizontal kernel for bottom zone: wide but not tall (bridges word gaps in captions)
    h_kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (61, 11))  # 30px horizontal, 5px vertical (thicker for caption shadows)

    dilated_white = 0
    for fidx in mask_frames:
        h, w = mask_frames[fidx].shape[:2]
        # Step 1: Standard dilation everywhere
        mask_frames[fidx] = _cv2.dilate(mask_frames[fidx], kernel, iterations=1)
        # Step 2: Extra horizontal dilation ONLY in bottom 25% (caption zone)
        bot_start = int(h * 0.75)
        bottom_strip = mask_frames[fidx][bot_start:, :]
        if np.any(bottom_strip > 0):
            bottom_strip = _cv2.dilate(bottom_strip, h_kernel, iterations=1)
            mask_frames[fidx][bot_start:, :] = bottom_strip
        dilated_white += int(np.sum(mask_frames[fidx] > 0))
    logger.info(f"CREATE_MASK: After {dilation_px}px dilation + bottom h-bridge: {dilated_white} total white pixels (+{dilated_white - total_white_pixels})")

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
            # Only use if within reasonable distance (60 frames = ~2 sec)
            if abs(nearest_idx - i) <= 60:
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


def stage_inpaint(video_path: str, mask_path: str, mm: ModelManager, errors: Optional[List[str]] = None, detections: Optional[List] = None, pipeline_config: dict = None, video_profile: dict = None) -> Tuple[str, bool]:
    """
    Remove text using VideoPainter (primary) or ProPainter (fallback).

    Args:
        errors: Optional list to append error messages (for metrics tracking)
        detections: Text detections from OCR stage (to distinguish "no text" vs "mask failed")
        pipeline_config: Adaptive pipeline config (inpaint_strategy per region)
        video_profile: Full VideoProfile with text_regions

    Returns:
        Tuple of (video_path, inpaint_succeeded)
        - inpaint_succeeded=True: Text was removed OR no text detected, eraser plate optional
        - inpaint_succeeded=False: Text detected but NOT removed, eraser plate REQUIRED
    """
    config = pipeline_config or {}
    inpaint_strategies = config.get("inpaint_strategy", {})
    text_regions = (video_profile or {}).get("text_regions", [])

    logger.info(f"Stage: INPAINT (strategies={inpaint_strategies})")

    # If ALL regions are "backplate" or "skip", skip inpainting entirely
    if text_regions and inpaint_strategies:
        non_skip = [s for s in inpaint_strategies.values() if s not in ("backplate", "skip")]
        if not non_skip:
            logger.info("INPAINT: All regions marked as backplate/skip — skipping inpainting entirely")
            mm.metrics.methods_used["inpaint"] = "skipped_backplate_only"
            return video_path, False  # Signal that render_text should use backplates

    if mask_path is None:
        if detections:
            # Text WAS detected by OCR but mask creation failed (e.g. SAM2 all-black)
            # Original text is still visible — eraser plate REQUIRED
            logger.warning(
                f"INPAINT: mask_path is None but {len(detections)} text detections exist. "
                f"SAM2 mask failed — eraser plate REQUIRED to cover original text."
            )
            if errors is not None:
                errors.append("inpaint: mask creation failed (SAM2 all-black), text not removed")
            mm.metrics.methods_used["inpaint"] = "eraser_plate_only"
            return video_path, False
        else:
            # No text detected at all — legitimate skip, no eraser needed
            logger.info("No mask and no text detections, skipping inpainting")
            return video_path, True

    output_path = video_path.replace(".mp4", "_inpainted.mp4")
    all_errors = []

    # ═══════════════════════════════════════════════════════════════════════
    # MASK ZONE PRUNING: Restrict mask to only "videopainter" region Y-ranges
    # This prevents mid-zone false positives from being inpainted (black blobs)
    # ═══════════════════════════════════════════════════════════════════════
    if text_regions and inpaint_strategies and mask_path:
        import cv2 as _cv2_prune
        vp_y_ranges = []
        for region in text_regions:
            rid = region.get("id", "")
            if inpaint_strategies.get(rid) == "videopainter":
                bbox = region.get("bbox_norm", [0, 0, 1, 1])
                vp_y_ranges.append((bbox[1], bbox[3]))

        if vp_y_ranges:
            # Load mask frames, zero out everything outside videopainter zones
            cap = _cv2_prune.VideoCapture(mask_path)
            if cap.isOpened():
                fps = cap.get(_cv2_prune.CAP_PROP_FPS)
                w = int(cap.get(_cv2_prune.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(_cv2_prune.CAP_PROP_FRAME_HEIGHT))
                fourcc = _cv2_prune.VideoWriter_fourcc(*"mp4v")
                pruned_path = mask_path.replace(".mp4", "_pruned.mp4")
                writer = _cv2_prune.VideoWriter(pruned_path, fourcc, fps, (w, h), False)

                pruned_frames = 0
                original_white = 0
                pruned_white = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if len(frame.shape) == 3:
                        frame = _cv2_prune.cvtColor(frame, _cv2_prune.COLOR_BGR2GRAY)
                    original_white += int(np.sum(frame > 0))

                    # Build keep-mask: only rows in videopainter regions
                    keep = np.zeros_like(frame)
                    for y1_n, y2_n in vp_y_ranges:
                        y1 = max(0, int(y1_n * h) - 30)  # 30px margin
                        y2 = min(h, int(y2_n * h) + 30)
                        keep[y1:y2, :] = 255
                    frame = _cv2_prune.bitwise_and(frame, keep)
                    pruned_white += int(np.sum(frame > 0))
                    writer.write(frame)
                    pruned_frames += 1

                cap.release()
                writer.release()
                removed_pct = (1 - pruned_white / max(original_white, 1)) * 100
                logger.info(
                    f"INPAINT MASK PRUNE: {pruned_frames} frames, "
                    f"removed {removed_pct:.0f}% of mask pixels outside videopainter zones "
                    f"(kept Y-ranges: {vp_y_ranges})"
                )
                mask_path = pruned_path  # Use pruned mask for inpainting

    # Try VideoPainter first (best quality, CogVideoX-based)
    try:
        logger.info("INPAINT: Attempting VideoPainter (CogVideoX-based)...")
        with mm.use("videopainter") as videopainter:
            result = _inpaint_videopainter(video_path, mask_path, output_path, videopainter)
            logger.info("INPAINT: VideoPainter succeeded!")
            mm.metrics.methods_used["inpaint"] = "videopainter"
            return result, True
    except Exception as e:
        import traceback
        err_msg = f"VideoPainter failed: {e}"
        logger.warning(err_msg)
        logger.warning(f"VideoPainter traceback:\n{traceback.format_exc()}")
        all_errors.append(err_msg)

    # Fallback to ProPainter
    try:
        logger.info("INPAINT: Attempting ProPainter fallback...")
        result = _inpaint_propainter(video_path, mask_path, output_path)
        logger.info("INPAINT: ProPainter succeeded!")
        mm.metrics.methods_used["inpaint"] = "propainter"
        return result, True
    except Exception as e:
        import traceback
        err_msg = f"ProPainter failed: {e}"
        logger.error(err_msg)
        logger.warning(f"ProPainter traceback:\n{traceback.format_exc()}")
        all_errors.append(err_msg)

    # All methods failed - eraser plate is now REQUIRED
    combined_error = f"inpaint: ALL methods failed - {'; '.join(all_errors)}"
    logger.error(combined_error)
    logger.warning("INPAINT: Falling back to ERASER-PLATE-ONLY mode (original text NOT removed)")
    mm.metrics.methods_used["inpaint"] = "eraser_plate_only"
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
            "--resize_ratio", "1.0",  # Full resolution for better text removal
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

        if result.stdout:
            logger.info(f"ProPainter stdout: {result.stdout.decode()[-300:]}")
        if result.returncode != 0:
            logger.error(f"ProPainter failed (rc={result.returncode}): {result.stderr.decode()[:500]}")
            raise RuntimeError(f"ProPainter subprocess failed: {result.stderr.decode()[:200]}")

        # ProPainter saves to {result_dir}/{video_dir_basename}/inpaint_out.mp4
        # where video_dir_basename = os.path.basename(video_frames_dir)
        video_dir_name = os.path.basename(video_frames_dir)
        expected_output = os.path.join(result_dir, video_dir_name, "inpaint_out.mp4")

        if os.path.exists(expected_output):
            logger.info(f"ProPainter: Found output at {expected_output}")
            shutil.copy(expected_output, output_path)
        else:
            # Fallback: search recursively for any mp4 or png output
            logger.warning(f"ProPainter: Expected output not at {expected_output}")
            logger.warning(f"ProPainter: result_dir contents: {os.listdir(result_dir)}")
            # Check subdirectories
            found = False
            for sub in os.listdir(result_dir):
                sub_path = os.path.join(result_dir, sub)
                if os.path.isdir(sub_path):
                    inpaint_mp4 = os.path.join(sub_path, "inpaint_out.mp4")
                    if os.path.exists(inpaint_mp4):
                        logger.info(f"ProPainter: Found output in subdirectory: {inpaint_mp4}")
                        shutil.copy(inpaint_mp4, output_path)
                        found = True
                        break
            if not found:
                raise RuntimeError(
                    f"ProPainter produced no output. Expected: {expected_output}. "
                    f"result_dir contents: {os.listdir(result_dir)}"
                )

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

    # Process in chunks of 49 frames (CogVideoX constraint) with 10-frame overlap
    # for smooth cross-fade transitions between chunks
    max_frames = 49
    overlap = 10
    step = max_frames - overlap
    all_output_frames = []

    min_chunk_frames = 25  # CogVideoX stride=24, needs at least stride+1 frames

    for chunk_idx, chunk_start in enumerate(range(0, len(video_frames), step)):
        chunk_end = min(chunk_start + max_frames, len(video_frames))

        # If last chunk is too small, extend start backwards to get enough frames
        if chunk_end - chunk_start < min_chunk_frames:
            chunk_start = max(0, chunk_end - min_chunk_frames)
            if chunk_end - chunk_start < min_chunk_frames:
                # Video is too short — pad with last frame
                pad_count = min_chunk_frames - (chunk_end - chunk_start)
                logger.info(f"VideoPainter: Padding last chunk with {pad_count} duplicate frames")

        chunk_masked = masked_frames[chunk_start:chunk_end]
        chunk_masks = mask_frames[chunk_start:chunk_end]

        # Pad if still too short
        while len(chunk_masked) < min_chunk_frames:
            chunk_masked.append(chunk_masked[-1])
            chunk_masks.append(chunk_masks[-1])

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

        if chunk_idx == 0:
            all_output_frames.extend(chunk_frames)
        else:
            # Cross-fade overlapping region with linear blend
            actual_overlap = min(overlap, len(chunk_frames), len(all_output_frames))
            for j in range(actual_overlap):
                alpha = j / overlap
                prev = np.array(all_output_frames[-(actual_overlap - j)])
                curr = np.array(chunk_frames[j])
                if prev.dtype == np.float32 or prev.dtype == np.float64:
                    prev = (prev * 255).clip(0, 255).astype(np.uint8)
                if curr.dtype == np.float32 or curr.dtype == np.float64:
                    curr = (curr * 255).clip(0, 255).astype(np.uint8)
                blended = ((1 - alpha) * prev + alpha * curr).astype(np.uint8)
                all_output_frames[-(actual_overlap - j)] = Image.fromarray(blended)
            # Add non-overlapping frames
            all_output_frames.extend(chunk_frames[actual_overlap:])

        # Break if we've processed all frames
        if chunk_end >= len(video_frames):
            break

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


def stage_blur_plate(video_path: str, regions: List[Dict], video_profile: dict = None) -> str:
    """
    Pipeline B: Gaussian blur + darken over text regions using ffmpeg.

    For each blur_plate region, applies boxblur + brightness reduction
    to the region's bbox area. This creates a clean dark surface for
    text overlay without expensive AI inpainting.

    Zero VRAM — pure ffmpeg CPU operation.
    """
    logger.info(f"Stage: BLUR_PLATE ({len(regions)} regions)")

    if not regions:
        return video_path

    output_path = video_path.replace(".mp4", "_blurred.mp4")

    # Get video dimensions
    import cv2
    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Build ffmpeg filter chain for all blur regions
    # Strategy: split → crop each region → blur+darken → overlay back
    filter_parts = []
    current_input = "[0:v]"

    for i, region in enumerate(regions):
        bbox = region.get("bbox_norm", [0, 0.85, 1, 1])  # Default: bottom strip

        # Convert normalized bbox to pixels with padding
        pad = 5  # pixels
        x = max(0, int(bbox[0] * vid_w) - pad)
        y = max(0, int(bbox[1] * vid_h) - pad)
        w = min(vid_w - x, int((bbox[2] - bbox[0]) * vid_w) + 2 * pad)
        h = min(vid_h - y, int((bbox[3] - bbox[1]) * vid_h) + 2 * pad)

        # Ensure minimum dimensions
        w = max(w, 10)
        h = max(h, 10)

        # Adaptive blur radius — must be <= min(w,h)/2 for boxblur
        blur_radius = max(2, min(25, min(w, h) // 2 - 1))

        # Create blur+darken filter for this region
        # crop → boxblur → darken (eq) → overlay at original position
        filter_parts.append(
            f"{current_input}split[main{i}][blur_src{i}];"
            f"[blur_src{i}]crop={w}:{h}:{x}:{y},"
            f"boxblur=luma_radius={blur_radius}:luma_power=3,"
            f"eq=brightness=-0.3:saturation=0.5"
            f"[blurred{i}];"
            f"[main{i}][blurred{i}]overlay={x}:{y}[out{i}]"
        )
        current_input = f"[out{i}]"

    filter_complex = ";".join(filter_parts)
    final_output = current_input.strip("[]")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", f"[{final_output}]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]

    logger.info(f"BLUR_PLATE: Running ffmpeg with {len(regions)} blur regions")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error(f"BLUR_PLATE ffmpeg failed: {result.stderr[-500:]}")
        return video_path  # Non-fatal, return original

    if not os.path.exists(output_path):
        logger.error("BLUR_PLATE: output file not created")
        return video_path

    logger.info(f"BLUR_PLATE: Applied blur to {len(regions)} regions")
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


# =============================================================================
# Caption Style System — Market-Aware Presets
# =============================================================================
#
# Modern TikTok/Reels/Shorts caption styles:
# - Bold sans-serif fonts (Montserrat, Bebas Neue, Helvetica)
# - Thick outlines (3-4px) for mobile readability
# - Colored accent backgrounds or no-bg with strong stroke
# - Font size 3.5-4.5% of video height (bigger = more engaging)
# - Shadow for depth on bright backgrounds
# - Chunked text (3-7 words per line, not full sentences)
#
# Style Format keys:
#   font: font family name
#   font_size_pct: % of video height (e.g., 0.042 = 4.2%)
#   bold: bool
#   text_color: hex (#RRGGBB)
#   outline_color: hex
#   outline_width: int (0-5)
#   shadow_depth: int (0-4)
#   shadow_color: hex
#   bg_style: "none" | "box" | "strip" | "pill"
#   bg_color: hex (for box/strip/pill backgrounds)
#   bg_alpha: float (0.0=opaque, 1.0=transparent)
#   border_style: 1 (outline+shadow) or 3 (opaque box)
#   spacing: int (letter spacing, 0-3)
#   max_chars_line: int (word wrap threshold)

CAPTION_STYLE_PRESETS = {
    # ── TikTok Native ──────────────────────────────────────────────
    "tiktok_bold": {
        "font": "Montserrat",
        "font_size_pct": 0.042,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 4,
        "shadow_depth": 2,
        "shadow_color": "#000000",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 1,
        "max_chars_line": 25,
    },
    # ── TikTok with dark pill bg (CapCut-style) ───────────────────
    "tiktok_pill": {
        "font": "Montserrat",
        "font_size_pct": 0.038,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#1A1A1A",
        "outline_width": 0,
        "shadow_depth": 0,
        "shadow_color": "#000000",
        "bg_style": "pill",
        "bg_color": "#1A1A1A",
        "bg_alpha": 0.15,
        "border_style": 3,
        "spacing": 1,
        "max_chars_line": 28,
    },
    # ── Reels / YouTube Shorts — clean modern ─────────────────────
    "reels_clean": {
        "font": "Helvetica",
        "font_size_pct": 0.040,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#222222",
        "outline_width": 3,
        "shadow_depth": 3,
        "shadow_color": "#111111",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 0,
        "max_chars_line": 28,
    },
    # ── Blogger / Talking Head — accent color box ─────────────────
    "blogger_accent": {
        "font": "Montserrat",
        "font_size_pct": 0.045,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow_depth": 0,
        "shadow_color": "#000000",
        "bg_style": "box",
        "bg_color": "#FF4757",
        "bg_alpha": 0.1,
        "border_style": 3,
        "spacing": 1,
        "max_chars_line": 22,
    },
    # ── Product / Tutorial — minimal clean ────────────────────────
    "product_minimal": {
        "font": "Helvetica",
        "font_size_pct": 0.035,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#333333",
        "outline_width": 2,
        "shadow_depth": 1,
        "shadow_color": "#222222",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 0,
        "max_chars_line": 30,
    },
    # ── Dark strip — TikTok dubbed content ────────────────────────
    "dubbed_strip": {
        "font": "Montserrat",
        "font_size_pct": 0.038,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow_depth": 0,
        "shadow_color": "#000000",
        "bg_style": "strip",
        "bg_color": "#0A0A0A",
        "bg_alpha": 0.08,  # ASS convention: 0=opaque. 0.08 = 92% opaque (covers source text)
        "border_style": 1,
        "spacing": 1,
        "max_chars_line": 32,
    },
    # ── LATAM market — warm, vibrant ──────────────────────────────
    "latam_vibrant": {
        "font": "Montserrat",
        "font_size_pct": 0.044,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 4,
        "shadow_depth": 2,
        "shadow_color": "#1A1A1A",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 1,
        "max_chars_line": 24,
    },
    # ── Arabic / RTL — Noto Arabic with RTL-friendly sizing ───────
    "arabic_native": {
        "font": "Noto Sans Arabic",
        "font_size_pct": 0.040,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 3,
        "shadow_depth": 2,
        "shadow_color": "#111111",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 0,
        "max_chars_line": 30,
    },
    # ── CJK (Chinese/Japanese/Korean) — wider spacing ─────────────
    "cjk_bold": {
        "font": "Noto Sans CJK SC",
        "font_size_pct": 0.042,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 3,
        "shadow_depth": 2,
        "shadow_color": "#111111",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 2,
        "max_chars_line": 16,
    },
}

# Map video content types to best caption presets
_CONTENT_TYPE_STYLE_MAP = {
    "talking_head": "blogger_accent",
    "interview": "reels_clean",
    "vlog": "tiktok_bold",
    "product": "product_minimal",
    "product_demo": "product_minimal",
    "tutorial": "product_minimal",
    "montage": "tiktok_bold",
    "text_overlay": "tiktok_pill",
    "reaction": "tiktok_bold",
}

# Market-specific overrides (target language → preferred style)
_MARKET_STYLE_MAP = {
    "ar": "arabic_native",
    "he": "arabic_native",
    "fa": "arabic_native",
    "ur": "arabic_native",
    "zh-CN": "cjk_bold",
    "zh-TW": "cjk_bold",
    "ja": "cjk_bold",
    "ko": "cjk_bold",
    "pt-BR": "latam_vibrant",
    "es-MX": "latam_vibrant",
    "es": "latam_vibrant",
}


def _resolve_caption_style(
    target_caption_style: Optional[Dict] = None,
    subtitle_style: Optional[Dict] = None,
    video_profile: Optional[Dict] = None,
    target_language: str = "",
    pipeline_type: str = "",
) -> Dict:
    """
    Resolve the best caption style for ASS rendering.

    Priority:
    1. Gemini-generated target_caption_style (from video analysis) — highest
    2. Market-specific corrections (CJK font, RTL, Indic scripts)
    3. Content-type + pipeline preset fallback
    4. Original video style hints (subtitle_style from Gemini manifest)
    5. Default: tiktok_bold

    Returns a merged style dict ready for ASS generation.
    """
    # ── Priority 1: Use Gemini-generated style if available ──
    if target_caption_style and isinstance(target_caption_style, dict):
        required_keys = {"font", "text_color", "outline_color"}
        if required_keys.issubset(target_caption_style.keys()):
            # Start from Gemini's recommendation
            style = {
                "font": target_caption_style.get("font", "Montserrat"),
                "font_size_pct": target_caption_style.get("font_size_pct", 0.042),
                "bold": target_caption_style.get("bold", True),
                "text_color": target_caption_style.get("text_color", "#FFFFFF"),
                "outline_color": target_caption_style.get("outline_color", "#000000"),
                "outline_width": target_caption_style.get("outline_width", 4),
                "shadow_depth": target_caption_style.get("shadow_depth", 2),
                "shadow_color": target_caption_style.get("shadow_color", "#000000"),
                "bg_style": target_caption_style.get("bg_style", "none"),
                "bg_color": target_caption_style.get("bg_color", "#000000"),
                # Gemini uses CSS convention (0=transparent, 1=opaque)
                # Internal/ASS convention is inverted (0=opaque, 1=transparent)
                "bg_alpha": 1.0 - float(target_caption_style.get("bg_alpha", 1.0)),
                "border_style": target_caption_style.get("border_style", 1),
                "spacing": target_caption_style.get("spacing", 1),
                "max_chars_line": target_caption_style.get("max_chars_line", 25),
            }

            # ── Priority 2: Apply market-specific corrections on top of Gemini style ──
            # CJK needs different font + smaller max_chars
            lang_base = normalize_language_code(target_language) if target_language else ""
            if lang_base in ("zh", "ja", "ko") or target_language in ("zh-CN", "zh-TW"):
                cjk_fonts = {"ja": "Noto Sans CJK JP", "ko": "Noto Sans CJK KR", "zh": "Noto Sans CJK SC"}
                style["font"] = cjk_fonts.get(lang_base, "Noto Sans CJK SC")
                if target_language == "zh-TW":
                    style["font"] = "Noto Sans CJK TC"
                style["max_chars_line"] = min(style["max_chars_line"], 16)
                style["spacing"] = max(style["spacing"], 2)
            elif lang_base in ("ar", "he", "fa", "ur"):
                style["font"] = "Noto Sans Arabic" if lang_base in ("ar", "fa", "ur") else "Noto Sans Hebrew"
                style["max_chars_line"] = min(style["max_chars_line"], 30)
            elif lang_base in ("hi", "bn", "ta", "te"):
                indic_fonts = {"hi": "Noto Sans Devanagari", "bn": "Noto Sans Bengali", "ta": "Noto Sans Tamil", "te": "Noto Sans Telugu"}
                style["font"] = indic_fonts.get(lang_base, "Noto Sans Devanagari")

            style["_preset_name"] = "gemini_generated"
            style["_style_reasoning"] = target_caption_style.get("style_reasoning", "")
            logger.info(f"CAPTION_STYLE: Using Gemini-generated style "
                        f"(font={style['font']}, bg={style['bg_style']}, "
                        f"reason={style.get('_style_reasoning', '')[:60]})")
            return style

    # ── Fallback: existing preset-based resolution ──
    # Start with default
    preset_name = "tiktok_bold"

    # 4. Pipeline-specific defaults
    if pipeline_type == "subtitle_snap":
        preset_name = "dubbed_strip"
    elif pipeline_type == "blur_plate":
        preset_name = "tiktok_pill"

    # 3. Content-type override
    if video_profile:
        video_type = video_profile.get("video_type", "")
        if video_type in _CONTENT_TYPE_STYLE_MAP:
            preset_name = _CONTENT_TYPE_STYLE_MAP[video_type]

    # 2. Market override (highest priority for language fit)
    lang_base = normalize_language_code(target_language) if target_language else ""
    if target_language in _MARKET_STYLE_MAP:
        preset_name = _MARKET_STYLE_MAP[target_language]
    elif lang_base in _MARKET_STYLE_MAP:
        preset_name = _MARKET_STYLE_MAP[lang_base]

    # Get the preset
    style = dict(CAPTION_STYLE_PRESETS.get(preset_name, CAPTION_STYLE_PRESETS["tiktok_bold"]))

    # 1. Apply original video style hints (Gemini manifest overrides)
    if subtitle_style:
        # Use original font if it's a known good font
        orig_font = subtitle_style.get("font_family", "")
        good_fonts = {"Impact", "Montserrat", "Helvetica", "Arial", "Bebas Neue", "Oswald", "Raleway"}
        if orig_font in good_fonts:
            style["font"] = orig_font

        # Use original colors if they have good contrast
        if subtitle_style.get("text_color"):
            style["text_color"] = subtitle_style["text_color"]
        if subtitle_style.get("outline_color"):
            style["outline_color"] = subtitle_style["outline_color"]

    style["_preset_name"] = preset_name
    logger.info(f"CAPTION_STYLE: resolved preset='{preset_name}' "
                f"(lang={target_language}, pipeline={pipeline_type}, "
                f"font={style['font']}, bg={style['bg_style']})")
    return style


def _build_ass_styles(style: Dict, video_height: int, is_rtl: bool = False, rtl_font: str = "") -> str:
    """
    Build ASS [V4+ Styles] section from a resolved caption style preset.

    Returns the full styles block string.
    """
    font = style["font"]
    font_size = int(video_height * style["font_size_pct"])
    bold = -1 if style["bold"] else 0
    text_color = _hex_to_ass_color(style["text_color"])
    outline_color = _hex_to_ass_color(style["outline_color"])
    shadow_color = _hex_to_ass_color(style.get("shadow_color", "#000000"))
    outline_w = style["outline_width"]
    shadow_d = style["shadow_depth"]
    border_style = style["border_style"]
    spacing = style.get("spacing", 0)

    # Background color for BorderStyle=3 (opaque box)
    if border_style == 3:
        bg_color = _hex_to_ass_color(style["bg_color"], alpha=style["bg_alpha"])
    else:
        bg_color = _hex_to_ass_color("#000000", alpha=0.5)  # semi-transparent shadow

    # Main text style
    styles = f"Style: TranslatedText,{font},{font_size},{text_color},{text_color},{outline_color},{bg_color},{bold},0,0,0,100,100,{spacing},0,{border_style},{outline_w},{shadow_d},5,10,10,10,1\n"

    # BackPlate (drawing-only, 1px invisible font)
    styles += f"Style: BackPlate,Arial,1,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"

    # BackPlateBox — for backplate_overlay strategy
    box_bg = _hex_to_ass_color(style["bg_color"], alpha=max(0.05, style["bg_alpha"] - 0.1))
    styles += f"Style: BackPlateBox,{font},{font_size},{text_color},&H000000FF,{_hex_to_ass_color('#0A0A0A', 0.1)},{box_bg},{bold},0,0,0,100,100,{spacing},0,3,10,0,5,10,10,10,1\n"

    # RTL variants
    if is_rtl and rtl_font:
        styles += f"Style: TranslatedTextRTL,{rtl_font},{font_size},{text_color},{text_color},{outline_color},{bg_color},{bold},0,0,0,100,100,{spacing},0,{border_style},{outline_w},{shadow_d},5,10,10,10,1\n"
        styles += f"Style: BackPlateBoxRTL,{rtl_font},{font_size},{text_color},&H000000FF,{_hex_to_ass_color('#0A0A0A', 0.1)},{box_bg},{bold},0,0,0,100,100,{spacing},0,3,10,0,5,10,10,10,1\n"

    return styles, font_size


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def _match_overlay_to_region(overlay: Dict, text_regions: list) -> Optional[Dict]:
    """
    Match a translated overlay to its corresponding VideoProfile text_region.

    Matches by:
    1. Position zone similarity (top/middle/bottom)
    2. Text content overlap (Levenshtein ratio)
    """
    if not text_regions:
        return None

    overlay_pos = overlay.get("position", "top").lower()
    overlay_text = overlay.get("text", "")

    best_match = None
    best_score = 0.0

    for region in text_regions:
        score = 0.0

        # Zone match bonus
        region_zone = region.get("zone", "top")
        if overlay_pos == region_zone:
            score += 0.5
        elif (overlay_pos in ("top", "top_left", "top_right") and region_zone == "top"):
            score += 0.4
        elif (overlay_pos in ("bottom", "bottom_left", "bottom_right") and region_zone == "bottom"):
            score += 0.4

        # Text similarity (source text)
        region_text = region.get("content", "")
        if overlay_text and region_text:
            text_sim = _levenshtein_ratio(overlay_text, region_text)
            score += text_sim * 0.5

        if score > best_score:
            best_score = score
            best_match = region

    if best_score >= 0.3:
        return best_match
    return None


def _render_backplate_overlay(ass_lines: list, overlay: Dict, region: Dict, video_width: int, video_height: int, font_size: int, is_rtl: bool = False, target_language: str = "", caption_style: dict = None):
    """
    Render translated text with TikTok/IG-native dark backplate covering original text.

    Uses two layers:
    Layer 0: Dark opaque rectangle covering original text bbox (hides source text)
    Layer 1: White text with auto-sized dark box (BorderStyle=3) for native look
    """
    bbox = region.get("bbox_norm", [0, 0, 1, 0.15])

    # Convert normalized bbox to pixel coords
    # Use generous padding (8%) to ensure full coverage of original text + shadows/glow
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    pad_x = int(max(bbox_w * video_width * 0.08, 15))  # At least 15px
    pad_y = int(max(bbox_h * video_height * 0.15, 10))  # More vertical for descenders/shadows
    x1 = max(0, int(bbox[0] * video_width) - pad_x)
    y1 = max(0, int(bbox[1] * video_height) - pad_y)
    x2 = min(video_width, int(bbox[2] * video_width) + pad_x)
    y2 = min(video_height, int(bbox[3] * video_height) + pad_y)

    # Center position for text
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    start_time = _format_ass_time(overlay.get("appears_at", 0))
    end_time = _format_ass_time(overlay.get("disappears_at", 0))

    # Layer 0: FULLY OPAQUE dark rectangle covering original text area
    # alpha 0x00 = 100% opaque — any translucency causes ghost text bleed-through
    draw_cmd = f"m {x1} {y1} l {x2} {y1} l {x2} {y2} l {x1} {y2}"
    ass_lines.append(
        f"Dialogue: 0,{start_time},{end_time},BackPlate,,0,0,0,,"
        f"{{\\an7\\pos(0,0)\\1c&H0A0A0A\\1a&H00\\bord0\\shad0\\p1}}{draw_cmd}"
    )

    # Layer 1: Translated text with BackPlateBox style (auto-sized dark box)
    translated_text = overlay.get("translated_text", "")
    wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=28)
    safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

    # Apply RTL reshaping
    style_name = "BackPlateBox"
    if is_rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(safe_text)
            safe_text = get_display(reshaped)
        except ImportError:
            pass
        style_name = "BackPlateBoxRTL"

    ass_lines.append(
        f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
        f"{{\\an5\\pos({cx},{cy})\\fad(200,200)}}{safe_text}"
    )

    logger.info(
        f"BACKPLATE: region={region.get('id', '?')} "
        f"bbox=({x1},{y1},{x2},{y2}) text='{translated_text[:30]}'"
    )


def _render_subtitle_snap(ass_lines: list, overlay: Dict, region: Dict, video_width: int, video_height: int, font_size: int, is_rtl: bool = False, target_language: str = "", caption_style: dict = None):
    """
    Pipeline A: Render translated subtitle with market-native styling.

    Renders a background strip (style-dependent) covering original text area,
    with bold translated text on top. Uses caption_style preset for:
    - Strip appearance (solid dark, semi-transparent, gradient feel)
    - Text color, outline, shadow from market preset
    - Font size and spacing tuned per market
    """
    cs = dict(caption_style or CAPTION_STYLE_PRESETS["dubbed_strip"])
    bbox = region.get("bbox_norm", [0, 0.85, 1, 1])

    # subtitle_snap does NOT remove original text — background MUST be opaque
    # bg_alpha in ASS convention: 0.0=opaque, 1.0=transparent
    # Force max 15% transparency (= min 85% opaque) to cover source text
    cs["bg_alpha"] = min(cs.get("bg_alpha", 0.15), 0.15)

    # Convert to pixels — full width strip for clean subtitle look
    x1 = 0
    y1 = max(0, int(bbox[1] * video_height) - 8)
    x2 = video_width
    y2 = min(video_height, int(bbox[3] * video_height) + 8)

    # Ensure minimum strip height for readability
    min_strip_h = int(video_height * cs["font_size_pct"] * 2.5)
    if y2 - y1 < min_strip_h:
        cy_center = (y1 + y2) // 2
        y1 = max(0, cy_center - min_strip_h // 2)
        y2 = min(video_height, cy_center + min_strip_h // 2)

    cx = video_width // 2
    cy = (y1 + y2) // 2

    start_time = _format_ass_time(overlay.get("appears_at", 0))
    end_time = _format_ass_time(overlay.get("disappears_at", 0))

    # Layer 0: Background strip — style depends on caption preset
    bg_color_ass = _hex_to_ass_color(cs["bg_color"])
    bg_alpha_hex = f"{int(cs['bg_alpha'] * 255):02X}"

    draw_cmd = f"m {x1} {y1} l {x2} {y1} l {x2} {y2} l {x1} {y2}"
    ass_lines.append(
        f"Dialogue: 0,{start_time},{end_time},BackPlate,,0,0,0,,"
        f"{{\\an7\\pos(0,0)\\1c{bg_color_ass}\\1a&H{bg_alpha_hex}\\bord0\\shad0\\p1}}{draw_cmd}"
    )

    # Layer 1: Translated text with market-native styling
    translated_text = overlay.get("translated_text", "")
    max_chars = cs.get("max_chars_line", 32)
    wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=max_chars)
    safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

    style_name = "TranslatedText"
    if is_rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(safe_text)
            safe_text = get_display(reshaped)
        except ImportError:
            pass
        style_name = "TranslatedTextRTL"

    ass_lines.append(
        f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
        f"{{\\an5\\pos({cx},{cy})\\fad(150,150)}}{safe_text}"
    )

    logger.info(
        f"SUBTITLE_SNAP: region={region.get('id', '?')} "
        f"strip=({x1},{y1},{x2},{y2}) style={cs.get('_preset_name', '?')} "
        f"text='{translated_text[:30]}'"
    )


def _render_blur_plate_overlay(ass_lines: list, overlay: Dict, region: Dict, video_width: int, video_height: int, font_size: int, is_rtl: bool = False, target_language: str = "", caption_style: dict = None):
    """
    Pipeline B: Render translated text on already-blurred region.

    The blur_plate stage has already applied gaussian blur + darken to the region.
    Text is rendered with market-native styling on top of the blurred area.
    The blur provides the background — no additional plate needed.
    """
    cs = dict(caption_style or CAPTION_STYLE_PRESETS["tiktok_pill"])
    bbox = region.get("bbox_norm", [0, 0.85, 1, 1])

    # Convert to pixels
    x1 = max(0, int(bbox[0] * video_width))
    y1 = max(0, int(bbox[1] * video_height))
    x2 = min(video_width, int(bbox[2] * video_width))
    y2 = min(video_height, int(bbox[3] * video_height))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    start_time = _format_ass_time(overlay.get("appears_at", 0))
    end_time = _format_ass_time(overlay.get("disappears_at", 0))

    translated_text = overlay.get("translated_text", "")
    max_chars = cs.get("max_chars_line", 28)
    wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=max_chars)
    safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

    style_name = "TranslatedText"
    if is_rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(safe_text)
            safe_text = get_display(reshaped)
        except ImportError:
            pass
        style_name = "TranslatedTextRTL"

    # Layer 0: Semi-opaque backup background in case blur failed or is insufficient
    bg_color_ass = _hex_to_ass_color(cs.get("bg_color", "#0A0A0A"))
    # 60% opaque backup (ASS: 0x66 = ~40% = 60% opaque)
    draw_cmd = f"m {x1} {y1} l {x2} {y1} l {x2} {y2} l {x1} {y2}"
    ass_lines.append(
        f"Dialogue: 0,{start_time},{end_time},BackPlate,,0,0,0,,"
        f"{{\\an7\\pos(0,0)\\1c{bg_color_ass}\\1a&H66\\bord0\\shad0\\p1}}{draw_cmd}"
    )

    # Layer 1: Text with shadow for readability on blurred background
    shadow_color = _hex_to_ass_color(cs.get("shadow_color", "#000000"))
    ass_lines.append(
        f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
        f"{{\\an5\\pos({cx},{cy})\\fad(200,200)\\shad3\\3c{shadow_color}}}{safe_text}"
    )

    logger.info(
        f"BLUR_PLATE_OVERLAY: region={region.get('id', '?')} "
        f"pos=({cx},{cy}) style={cs.get('_preset_name', '?')} "
        f"text='{translated_text[:30]}'"
    )


def _generate_ass_subtitles(
    translated_overlays: List[Dict],
    text_detections: List[Dict],
    video_width: int,
    video_height: int,
    video_duration: float,
    video_fps: float,
    subtitle_style: Optional[Dict] = None,
    inpaint_succeeded: bool = True,
    target_language: str = "",
    video_profile: dict = None,
) -> str:
    """
    Generate ASS subtitle file with SOTA Logic 3.0 + Adaptive Pipeline.

    Strategies per overlay:
    - replace_inplace: Text at OCR bbox position with outline (default)
    - backplate_overlay: Color-matched backplate + translated text on top
    - subtitle_snap: Dark strip + centered text (Pipeline A)
    - blur_plate_overlay: Text on pre-blurred region (Pipeline B)
    - skip: No rendering for this overlay

    Uses market-aware caption style presets based on:
    - Target language/market (LATAM, CJK, Arabic, etc.)
    - Content type (talking_head, product, vlog, etc.)
    - Pipeline type (subtitle_snap, blur_plate, inpaint_backplate)
    - Original video style hints (from Gemini manifest)

    Supports RTL languages (Arabic, Hebrew, Farsi, Urdu, Yiddish).
    """
    # Extract adaptive pipeline config
    vp_config = (video_profile or {}).get("pipeline_config", {})
    text_regions = (video_profile or {}).get("text_regions", [])
    render_strategies = vp_config.get("render_strategy", {})

    # Determine dominant pipeline type for style resolution
    pipelines_used = vp_config.get("pipelines_used", [])
    dominant_pipeline = pipelines_used[0] if pipelines_used else "inpaint_backplate"

    # Extract Gemini-generated target caption style from video_profile
    target_caption_style = (video_profile or {}).get("target_caption_style")

    # Resolve caption style based on Gemini analysis → market → content → pipeline
    caption_style = _resolve_caption_style(
        target_caption_style=target_caption_style,
        subtitle_style=subtitle_style,
        video_profile=video_profile,
        target_language=target_language,
        pipeline_type=dominant_pipeline,
    )

    # Detect RTL language
    is_rtl = any(target_language.startswith(lang) for lang in RTL_LANGUAGES) if target_language else False
    if is_rtl:
        logger.info(f"ASS: RTL mode enabled for language '{target_language}'")

    # RTL font selection
    rtl_font = ""
    if is_rtl:
        if target_language.startswith(("ar", "fa", "ur")):
            rtl_font = "Noto Sans Arabic"
        else:
            rtl_font = "Noto Sans Hebrew"

    # Build ASS styles from resolved caption preset
    styles_block, font_size = _build_ass_styles(
        caption_style, video_height, is_rtl=is_rtl, rtl_font=rtl_font
    )

    # ASS Header
    ass_content = f"""[Script Info]
Title: TrafficPlant SOTA Subtitles v5
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    ass_lines = []  # Collect dialogue lines

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

        # Determine render strategy from adaptive pipeline
        matched_region = _match_overlay_to_region(overlay, text_regions)
        if matched_region:
            strategy = render_strategies.get(matched_region["id"], "replace_inplace")
            logger.info(f"RENDER_TEXT ASS [{i}]: matched region '{matched_region['id']}', strategy='{strategy}'")
        else:
            strategy = "replace_inplace"  # Default fallback

        # Handle skip strategy
        if strategy == "skip":
            logger.info(f"RENDER_TEXT ASS [{i}]: skipping overlay (strategy=skip)")
            continue

        # Handle backplate_overlay strategy
        if strategy == "backplate_overlay" and matched_region:
            # Pass adjusted times (disappears_at may have been extended to video_duration)
            overlay_with_times = {**overlay, "appears_at": appears_at, "disappears_at": disappears_at}
            _render_backplate_overlay(
                ass_lines, overlay_with_times, matched_region,
                video_width, video_height, font_size,
                is_rtl=is_rtl, target_language=target_language,
                caption_style=caption_style,
            )
            continue

        # Handle subtitle_snap strategy (Pipeline A)
        if strategy == "subtitle_snap" and matched_region:
            _render_subtitle_snap(
                ass_lines, {**overlay, "appears_at": appears_at, "disappears_at": disappears_at},
                matched_region, video_width, video_height, font_size,
                is_rtl=is_rtl, target_language=target_language,
                caption_style=caption_style,
            )
            continue

        # Handle blur_plate_overlay strategy (Pipeline B)
        if strategy == "blur_plate_overlay" and matched_region:
            _render_blur_plate_overlay(
                ass_lines, {**overlay, "appears_at": appears_at, "disappears_at": disappears_at},
                matched_region, video_width, video_height, font_size,
                is_rtl=is_rtl, target_language=target_language,
                caption_style=caption_style,
            )
            continue

        # Default: replace_inplace (or subtitle_bottom) — original behavior
        # Get bounding box from OCR (zone + time filtered)
        x, y, box_w, box_h = _find_overlay_bbox(
            overlay, text_detections, video_width, video_height,
            appears_at=appears_at, disappears_at=disappears_at, video_fps=video_fps
        )

        # Format times
        start_time = _format_ass_time(appears_at)
        end_time = _format_ass_time(disappears_at)

        # Skip entire overlay if no valid bbox (middle/center zone without OCR)
        # Minimal bbox (0,0,1,1) means "skip this overlay entirely"
        if x == 0 and y == 0 and box_w == 1 and box_h == 1:
            logger.info(
                f"RENDER_TEXT ASS [{i}]: skipping overlay for middle/center zone "
                f"(no OCR detections, would obscure content)"
            )
            continue

        # Translated text renders with its own outline+shadow for readability
        # Word-wrap using market-aware char limit from caption style
        max_chars = caption_style.get("max_chars_line", 28) if caption_style else 28
        wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=max_chars)

        # Escape special ASS characters
        safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

        # Apply RTL reshaping for Arabic/Persian/Urdu
        if is_rtl:
            try:
                import arabic_reshaper
                from bidi.algorithm import get_display
                reshaped = arabic_reshaper.reshape(safe_text)
                safe_text = get_display(reshaped)
            except ImportError:
                logger.warning("arabic_reshaper/python-bidi not installed, RTL rendering may be incorrect")
            style_name = "TranslatedTextRTL"
        else:
            style_name = "TranslatedText"

        # Position: center of OCR bbox
        cx = x + box_w // 2
        cy = y + box_h // 2

        # {\an5} = center alignment, {\pos(x,y)} = absolute position
        # {\fad(200,200)} = 200ms fade in/out
        text_event = (
            f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
            f"{{\\an5\\pos({cx},{cy})\\fad(200,200)}}{safe_text}"
        )
        ass_lines.append(text_event)

        logger.info(
            f"RENDER_TEXT ASS [{i}]: strategy='{strategy}' position='{position}' "
            f"bbox=({x},{y},{box_w}x{box_h}) "
            f"text_center=({cx},{cy}) "
            f"time={start_time}-{end_time}"
        )

    # Combine all lines
    ass_content += "\n".join(ass_lines) + "\n"

    return ass_content


def stage_render_text(
    video_path: str,
    translated_overlays: List[Dict],
    text_detections: List[Dict],
    resolution: Tuple[int, int],
    fps: float,
    subtitle_style: Optional[Dict] = None,
    inpaint_succeeded: bool = True,
    target_language: str = "",
    video_profile: dict = None,
) -> str:
    """
    Render translated text overlays onto video using ASS subtitles.

    ASS (Advanced SubStation Alpha) provides:
    - Precise positioning with {\\pos(x,y)}
    - Opaque background boxes with BorderStyle=3
    - Fade-in/fade-out animations with {\\fad(200,200)}
    - Smart word-wrapping (~25 chars per line for mobile)
    - Safe zone awareness (avoid TikTok UI elements)
    - Backplate rendering (adaptive pipeline: color-matched plates)

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
        target_language=target_language,
        video_profile=video_profile,
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


def _tts_elevenlabs(text: str, reference_audio: str, target_language: str = "en", voice_id: Optional[str] = None) -> Optional[str]:
    """Generate speech using ElevenLabs API (best quality, runs from GPU worker IP to avoid geo-blocks).

    Args:
        voice_id: Pre-cloned voice ID from server. If provided, skips cloning and
                  does NOT delete the voice (server handles cleanup).
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.info("ElevenLabs: no API key, skipping")
        return None

    import httpx

    output_path = tempfile.mktemp(suffix=".mp3")
    locally_cloned = False  # Track whether we cloned locally (for cleanup)

    try:
        # Step 1: Use pre-cloned voice or create a new clone
        if voice_id:
            logger.info(f"ElevenLabs: using pre-cloned voice {voice_id}")
        else:
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

            locally_cloned = True
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
            # Pass full locale (e.g. "zh-TW") — ElevenLabs v3 handles locale-specific voices
            if target_language:
                tts_payload["language_code"] = target_language
                logger.info(f"ElevenLabs: using language_code={target_language}")

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
            # Step 3: Delete voice clone ONLY if we created it locally
            if locally_cloned and voice_id:
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


def stage_tts(text: str, reference_audio: str, mm: ModelManager, target_language: str = "en", voice_id: Optional[str] = None) -> Tuple[str, str]:
    """Generate speech: ElevenLabs API (primary) → F5-TTS local (fallback).
    Returns: (audio_path, method_used)
    """
    logger.info(f"Stage: TTS (target_language={target_language}, voice_id={'yes' if voice_id else 'no'})")

    # Primary: ElevenLabs (runs from US GPU IP — no geo-block)
    result = _tts_elevenlabs(text, reference_audio, target_language=target_language, voice_id=voice_id)
    if result:
        return result, "elevenlabs"

    # Fallback: F5-TTS on local GPU
    return _tts_f5(text, reference_audio, mm), "f5tts"


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

    mm.metrics.methods_used["lipsync"] = model_name

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
    """Assess output quality using no-reference metric (MUSIQ or NIQE fallback)."""
    logger.info("Stage: QUALITY_CHECK")

    import pyiqa
    import cv2

    metric_name = "musiq"
    try:
        metric = pyiqa.create_metric('musiq', device='cuda')
    except Exception:
        metric_name = "niqe"
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
            scores.append(score)

    cap.release()

    if not scores:
        return {
            "score": 0, "raw_score": 0, "threshold": threshold,
            "passed": False, "frame_scores": [], "metric": metric_name,
        }

    avg_score = sum(scores) / len(scores)

    # Normalize based on metric type
    if metric_name == "musiq":
        # MUSIQ: 0-100 scale, higher = better
        normalized = avg_score / 100.0 if avg_score > 1.0 else avg_score
        passed = normalized >= threshold
        effective_threshold = threshold
    else:
        # NIQE: lower = better, typical range 2-8
        # Don't normalize to 0-1, use NIQE-specific threshold
        normalized = avg_score
        passed = avg_score <= 5.0
        effective_threshold = 5.0

    logger.info(
        f"Quality ({metric_name}): raw={avg_score:.3f} normalized={normalized:.3f} "
        f"threshold={effective_threshold} {'PASS' if passed else 'FAIL'}"
    )

    return {
        "score": normalized,
        "raw_score": avg_score,
        "threshold": effective_threshold,
        "passed": passed,
        "frame_scores": scores,
        "metric": metric_name,
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
        # Speechless video — just copy video with original audio (text-only localization)
        logger.info("No TTS audio — assembling with original audio only")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-c", "copy",
            output_path
        ], capture_output=True, check=False)
        if not os.path.exists(output_path):
            raise RuntimeError(f"ffmpeg assemble (no-TTS) failed — output not created: {output_path}")
        return output_path

    if background_audio and os.path.exists(background_audio):
        # Mix TTS voice with background audio using sidechain ducking:
        # Background plays at ~70% volume, ducks to ~25% when TTS voice is active.
        # This gives clean dubbing: voice is clear, background music/SFX preserved.
        mixed_audio = output_path.replace(".mp4", "_mixed.wav")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tts_audio,
            "-i", background_audio,
            "-filter_complex",
            # Sidechain compressor: TTS voice triggers ducking on background
            # [1:a] background gets compressed when [0:a] TTS is loud
            # threshold=-25dB: start ducking when TTS > -25dB
            # ratio=3: moderate ducking (not too aggressive)
            # attack=50ms: fast attack when voice starts
            # release=300ms: smooth release when voice stops
            # Then mix: TTS at full volume + ducked background at 70%
            "[1:a]volume=0.7[bg];"
            "[bg][0:a]sidechaincompress=threshold=0.02:ratio=3:attack=50:release=300:level_sc=1[ducked];"
            "[0:a][ducked]amix=inputs=2:duration=longest:weights=1 1[a]",
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
            elevenlabs_voice_id=job_input.get("elevenlabs_voice_id"),  # Pre-cloned voice ID
            video_profile=job_input.get("video_profile"),  # Adaptive pipeline config
            target_caption_style=job_input.get("target_caption_style"),  # Gemini-generated caption style
        )

        mm = get_model_manager()
        mm.metrics = PipelineMetrics()  # Reset metrics for each request
        metrics = mm.metrics

        # Parse adaptive pipeline config
        video_profile = config.video_profile
        if video_profile:
            logger.info(f"ADAPTIVE: VideoProfile loaded — type={video_profile.get('video_type')}, "
                        f"{len(video_profile.get('text_regions', []))} text regions")
            pipeline_config = video_profile.get("pipeline_config", {})
        else:
            logger.info("ADAPTIVE: No VideoProfile, using defaults")
            pipeline_config = {}

        # Merge Gemini target_caption_style into video_profile for ASS generator
        target_caption_style = config.target_caption_style
        if target_caption_style:
            if video_profile:
                video_profile["target_caption_style"] = target_caption_style
            else:
                video_profile = {"target_caption_style": target_caption_style}
            logger.info(f"CAPTION: Gemini target_caption_style injected into video_profile "
                        f"(font={target_caption_style.get('font')}, bg={target_caption_style.get('bg_style')})")

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
            "transcript": None,
            "video_profile": video_profile,
            "pipeline_config": pipeline_config,
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
                    # Skip if all regions use pipelines that don't need OCR
                    skip_ocr = False
                    if state.get("video_profile"):
                        text_regions = state["video_profile"].get("text_regions", [])
                        pipeline_types = set(r.get("pipeline_type", "inpaint_backplate") for r in text_regions)
                        skip_ocr = pipeline_types and pipeline_types.issubset({"subtitle_snap", "blur_plate"})

                    if skip_ocr:
                        logger.info("DETECT_TEXT: Skipped — all regions use subtitle_snap/blur_plate")
                        state["text_detections"] = []
                        state["text_resolution"] = (1920, 1080)  # Default
                        state["text_fps"] = 30.0
                    else:
                        result = stage_detect_text(state["video_path"], mm, pipeline_config=state.get("pipeline_config"))
                        state["text_detections"] = result["detections"]
                        state["text_resolution"] = result["resolution"]  # (width, height)
                        state["text_fps"] = result["fps"]

                elif stage_name == "create_mask":
                    # Skip if all regions use pipelines that don't need masks
                    skip_mask = False
                    if state.get("video_profile"):
                        text_regions = state["video_profile"].get("text_regions", [])
                        pipeline_types = set(r.get("pipeline_type", "inpaint_backplate") for r in text_regions)
                        skip_mask = pipeline_types and pipeline_types.issubset({"subtitle_snap", "blur_plate"})

                    if skip_mask:
                        logger.info("CREATE_MASK: Skipped — all regions use subtitle_snap/blur_plate")
                        state["mask_path"] = None
                    else:
                        state["mask_path"] = stage_create_mask(
                            state["video_path"],
                            state.get("text_detections", []),
                            mm,
                            pipeline_config=state.get("pipeline_config"),
                            video_profile=state.get("video_profile"),
                        )

                elif stage_name == "inpaint":
                    # Determine which pipelines are needed
                    needs_inpaint = False
                    blur_regions = []

                    if state.get("video_profile"):
                        text_regions = state["video_profile"].get("text_regions", [])
                        for r in text_regions:
                            pt = r.get("pipeline_type", "inpaint_backplate")
                            if pt == "inpaint_backplate":
                                needs_inpaint = True
                            elif pt == "blur_plate":
                                blur_regions.append(r)
                    else:
                        needs_inpaint = True  # No profile = default behavior

                    # Run AI inpainting only for inpaint_backplate regions
                    if needs_inpaint:
                        video_path, inpaint_ok = stage_inpaint(
                            state["video_path"],
                            state["mask_path"],
                            mm,
                            errors=metrics.errors,
                            detections=state.get("text_detections", []),
                            pipeline_config=state.get("pipeline_config"),
                            video_profile=state.get("video_profile"),
                        )
                        state["video_path"] = video_path
                        state["inpaint_succeeded"] = inpaint_ok
                    else:
                        logger.info("INPAINT: Skipped — no regions need AI inpainting")
                        state["inpaint_succeeded"] = True

                    # Run blur_plate for Pipeline B regions
                    if blur_regions:
                        try:
                            state["video_path"] = stage_blur_plate(
                                state["video_path"], blur_regions,
                                video_profile=state.get("video_profile"),
                            )
                        except Exception as e:
                            logger.warning(f"BLUR_PLATE failed (non-fatal): {e}")
                            metrics.errors.append(f"blur_plate: {str(e)}")

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
                            target_language=config.target_language,
                            video_profile=state.get("video_profile"),
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
                        tts_result, tts_method = stage_tts(
                            state["translated_text"],
                            state.get("vocals_path") or video_path,
                            mm,
                            target_language=config.target_language,
                            voice_id=config.elevenlabs_voice_id,
                        )
                        state["tts_audio"] = tts_result
                        metrics.methods_used["tts"] = tts_method
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
                import traceback
                logger.error(f"Stage {stage_name} failed: {e}")
                logger.error(f"Stage {stage_name} traceback:\n{traceback.format_exc()}")
                metrics.errors.append(f"{stage_name}: {str(e)}")

                if stage_name in CRITICAL_STAGES:
                    critical_failure = f"Critical stage '{stage_name}' failed: {e}"
                    logger.error(f"CRITICAL FAILURE: {critical_failure}")
                elif stage_name in QUALITY_CRITICAL_STAGES:
                    metrics.quality_warnings.append(f"{stage_name}: {str(e)}")
                    logger.warning(f"QUALITY WARNING: Stage '{stage_name}' failed — output may be degraded")
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
