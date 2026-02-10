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

# =============================================================================
# Geo-Aware Font Map
# =============================================================================
# Maps language codes → (primary_font, fallback_font)
# Primary fonts are Google Fonts downloaded at runtime; fallback fonts are
# system fonts from Dockerfile.base packages (fonts-noto-core, fonts-noto-cjk,
# fonts-dejavu-core, fonts-freefont-ttf).
GEO_FONT_MAP = {
    # Latin script — Google Fonts primary
    "en": ("Montserrat", "DejaVu Sans"),
    "es": ("Poppins", "DejaVu Sans"),
    "pt": ("Poppins", "DejaVu Sans"),
    "fr": ("Inter", "DejaVu Sans"),
    "de": ("Inter", "DejaVu Sans"),
    "it": ("Inter", "DejaVu Sans"),
    "nl": ("Inter", "DejaVu Sans"),
    "pl": ("Inter", "DejaVu Sans"),
    "sv": ("Inter", "DejaVu Sans"),
    "tr": ("Montserrat", "DejaVu Sans"),
    "vi": ("Montserrat", "DejaVu Sans"),
    "id": ("Montserrat", "DejaVu Sans"),
    "sr-Latn": ("Montserrat", "DejaVu Sans"),
    # Cyrillic script
    "ru": ("Montserrat", "DejaVu Sans"),
    "uk": ("Montserrat", "DejaVu Sans"),
    "sr": ("Montserrat", "DejaVu Sans"),
    # Arabic/RTL script — bundled via fonts-noto-core
    "ar": ("Noto Sans Arabic", "Noto Sans Arabic"),
    "fa": ("Noto Sans Arabic", "Noto Sans Arabic"),
    "ur": ("Noto Sans Arabic", "Noto Sans Arabic"),
    "he": ("Noto Sans Hebrew", "Noto Sans Hebrew"),
    # CJK script — bundled via fonts-noto-cjk
    "ja": ("Noto Sans JP", "Noto Sans CJK JP"),
    "ko": ("Noto Sans KR", "Noto Sans CJK KR"),
    "zh": ("Noto Sans SC", "Noto Sans CJK SC"),
    "zh-CN": ("Noto Sans SC", "Noto Sans CJK SC"),
    "zh-TW": ("Noto Sans TC", "Noto Sans CJK TC"),
    # Indic scripts — bundled via fonts-noto-core
    "hi": ("Noto Sans Devanagari", "Noto Sans Devanagari"),
    "bn": ("Noto Sans Bengali", "Noto Sans Bengali"),
    "ta": ("Noto Sans Tamil", "Noto Sans Tamil"),
    "te": ("Noto Sans Telugu", "Noto Sans Telugu"),
    # Thai
    "th": ("Noto Sans Thai", "Noto Sans Thai"),
}

# Google Fonts that need downloading (not in system packages)
_GOOGLE_FONTS_TO_DOWNLOAD = {
    "Montserrat": "Montserrat:wght@400;700;800;900",
    "Poppins": "Poppins:wght@400;600;700;800",
    "Inter": "Inter:wght@400;500;600;700",
}

_FONT_DIR = Path("/workspace/fonts")
_fonts_initialized = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def _ensure_geo_fonts():
    """
    Download Google Fonts for geo-aware rendering and register with fontconfig.

    Fonts are cached in /workspace/fonts/ which persists across Vast.ai runs.
    Only downloads missing fonts — subsequent calls are no-ops.
    """
    global _fonts_initialized
    if _fonts_initialized:
        return

    _FONT_DIR.mkdir(parents=True, exist_ok=True)
    import re as _re_fonts

    for family, url_spec in _GOOGLE_FONTS_TO_DOWNLOAD.items():
        family_dir = _FONT_DIR / family.replace(" ", "_")
        if family_dir.exists() and any(family_dir.glob("*.ttf")):
            continue  # Already downloaded

        family_dir.mkdir(parents=True, exist_ok=True)
        try:
            api_url = f"https://fonts.googleapis.com/css2?family={url_spec}&display=swap"
            result = subprocess.run(
                ["curl", "-sL", "-H", "User-Agent: Mozilla/5.0", api_url],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning(f"FONTS: Failed to fetch CSS for {family}: {result.stderr[:100]}")
                continue

            ttf_urls = _re_fonts.findall(r'url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)', result.stdout)
            if not ttf_urls:
                logger.warning(f"FONTS: No TTF URLs found for {family} (Google may serve woff2 only)")
                continue

            for i, url in enumerate(ttf_urls):
                out_file = family_dir / f"{family.replace(' ', '_')}_{i}.ttf"
                subprocess.run(
                    ["curl", "-sL", "-o", str(out_file), url],
                    capture_output=True, timeout=30,
                )

            ttf_count = len(list(family_dir.glob("*.ttf")))
            logger.info(f"FONTS: Downloaded {ttf_count} TTF files for '{family}'")
        except Exception as e:
            logger.warning(f"FONTS: Error downloading {family}: {e}")

    # Register /workspace/fonts/ with fontconfig so ffmpeg/libass can find them
    fontconfig_file = Path("/etc/fonts/conf.d/99-trafficplant-fonts.conf")
    if not fontconfig_file.exists():
        try:
            fontconfig_file.parent.mkdir(parents=True, exist_ok=True)
            fontconfig_file.write_text(
                '<?xml version="1.0"?>\n'
                '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
                '<fontconfig>\n'
                f'  <dir>{_FONT_DIR}</dir>\n'
                '</fontconfig>\n'
            )
            subprocess.run(["fc-cache", "-fv", str(_FONT_DIR)], capture_output=True, timeout=30)
            logger.info(f"FONTS: Registered {_FONT_DIR} with fontconfig")
        except Exception as e:
            logger.warning(f"FONTS: Failed to register with fontconfig: {e}")

    _fonts_initialized = True
    logger.info("FONTS: Geo font initialization complete")


def get_geo_font(target_language: str) -> str:
    """Get the best available font family name for a target language."""
    lang_base = normalize_language_code(target_language) if target_language else "en"
    primary, fallback = GEO_FONT_MAP.get(lang_base, GEO_FONT_MAP.get("en", ("Montserrat", "DejaVu Sans")))

    # Check if primary font was downloaded
    family_dir = _FONT_DIR / primary.replace(" ", "_")
    if family_dir.exists() and any(family_dir.glob("*.ttf")):
        return primary

    # Check system fontconfig
    try:
        result = subprocess.run(
            ["fc-list", f":family={primary}", "family"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            return primary
    except Exception:
        pass

    return fallback


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

    # Original transcript (source language) — used for TTS speed calculation
    original_transcript: Optional[str] = None

    # Persisted ElevenLabs voice ID (from campaign)
    # If provided, reuses this voice instead of cloning each time
    elevenlabs_voice_id: Optional[str] = None

    # Campaign ID for voice naming (traceability)
    campaign_id: Optional[int] = None

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
    to identify duplicates. Also tracks frame_count for temporal persistence filtering.
    """
    if not detections:
        return []

    unique = []
    frame_counts = []  # Track how many source frames each unique detection appears in
    for det in detections:
        is_dupe = False
        for idx, existing in enumerate(unique):
            iou = _calculate_iou(det["bbox_norm"], existing["bbox_norm"])
            if iou > iou_threshold:
                # Spatial overlap — check text similarity
                text_sim = _levenshtein_ratio(det.get("text", ""), existing.get("text", ""))
                if text_sim > 0.6:
                    is_dupe = True
                    frame_counts[idx] += 1
                    # Keep the one with higher confidence
                    if det.get("confidence", 0) > existing.get("confidence", 0):
                        unique[idx] = det
                    break
        if not is_dupe:
            unique.append(det)
            frame_counts.append(1)

    # Temporal persistence filter: require detection in ≥2 frames
    # Single-frame detections are likely false positives (noise, body parts, reflections)
    filtered = []
    rejected_count = 0
    for det, fc in zip(unique, frame_counts):
        if fc >= 2:
            det["frame_count"] = fc
            filtered.append(det)
        else:
            rejected_count += 1
            logger.info(
                f"DEDUPE: REJECTED single-frame detection text='{det.get('text', '')[:30]}' "
                f"bbox_norm={[f'{v:.3f}' for v in det.get('bbox_norm', [])]} — likely false positive"
            )

    logger.info(f"DEDUPE: {len(detections)} raw → {len(unique)} unique → {len(filtered)} persistent (IoU>{iou_threshold}, text_sim>0.6, min_frames=2)")
    if rejected_count:
        logger.info(f"DEDUPE: Rejected {rejected_count} single-frame detections")
    return filtered


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
        - inpaint_succeeded=True: Text was removed OR no text detected
        - inpaint_succeeded=False: Text detected but NOT removed; watermarks get eraser
          plates, non-watermark regions fall through to styled overlays
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
            # Only watermark/logo regions need eraser plates (black rectangles).
            # Non-watermark regions (captions, subtitles, usernames) fall through
            # to styled overlay rendering — black rectangles look worse than original text.
            has_watermark_regions = any(
                r.get("type", "").lower() in ("watermark", "logo")
                for r in text_regions
            )
            if has_watermark_regions:
                logger.warning(
                    f"INPAINT: mask_path is None but {len(detections)} text detections exist. "
                    f"SAM2 mask failed — eraser plate REQUIRED for watermark regions only."
                )
                mm.metrics.methods_used["inpaint"] = "eraser_plate_watermarks_only"
            else:
                logger.warning(
                    f"INPAINT: mask_path is None but {len(detections)} text detections exist. "
                    f"SAM2 mask failed — no watermarks, styled overlays will cover text."
                )
                mm.metrics.methods_used["inpaint"] = "skipped_overlay_fallback"
            if errors is not None:
                errors.append("inpaint: mask creation failed (SAM2 all-black), text not removed")
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
                bbox = _normalize_bbox(region.get("bbox_norm", [0, 0, 1, 1]))
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

    # Determine video orientation — ProPainter preserves resolution, VideoPainter downsamples to ~480p
    import cv2 as _cv2_orient
    _cap = _cv2_orient.VideoCapture(video_path)
    _vw = int(_cap.get(_cv2_orient.CAP_PROP_FRAME_WIDTH))
    _vh = int(_cap.get(_cv2_orient.CAP_PROP_FRAME_HEIGHT))
    _cap.release()
    is_vertical = _vh > _vw

    # For vertical video: ProPainter first (preserves resolution), VideoPainter as fallback
    # For landscape: VideoPainter first (better inpainting quality when resolution isn't destroyed)
    if is_vertical:
        logger.info(f"INPAINT: Vertical video ({_vw}x{_vh}) — using ProPainter first (preserves resolution)")
        inpaint_order = [
            ("propainter", _inpaint_propainter, [video_path, mask_path, output_path]),
            ("videopainter", None, None),  # Fallback, needs mm context
        ]
    else:
        logger.info(f"INPAINT: Landscape video ({_vw}x{_vh}) — using VideoPainter first")
        inpaint_order = [
            ("videopainter", None, None),
            ("propainter", _inpaint_propainter, [video_path, mask_path, output_path]),
        ]

    for method_name, method_fn, method_args in inpaint_order:
        try:
            if method_name == "videopainter":
                logger.info("INPAINT: Attempting VideoPainter (CogVideoX-based)...")
                with mm.use("videopainter") as videopainter:
                    result = _inpaint_videopainter(video_path, mask_path, output_path, videopainter)
                    logger.info("INPAINT: VideoPainter succeeded!")
                    mm.metrics.methods_used["inpaint"] = "videopainter"
                    return result, True
            else:
                logger.info(f"INPAINT: Attempting ProPainter...")
                result = _inpaint_propainter(video_path, mask_path, output_path)
                logger.info("INPAINT: ProPainter succeeded!")
                mm.metrics.methods_used["inpaint"] = "propainter"
                return result, True
        except Exception as e:
            import traceback
            err_msg = f"{method_name} failed: {e}"
            logger.warning(err_msg)
            logger.warning(f"{method_name} traceback:\n{traceback.format_exc()}")
            all_errors.append(err_msg)

    # All methods failed — check if any regions are watermarks
    # Only watermarks get eraser plates; non-watermark content uses styled overlays
    combined_error = f"inpaint: ALL methods failed - {'; '.join(all_errors)}"
    logger.error(combined_error)
    has_watermark_regions = any(
        r.get("type", "").lower() in ("watermark", "logo")
        for r in text_regions
    )
    if has_watermark_regions:
        logger.warning("INPAINT: Eraser plates enabled for WATERMARK regions only (original text NOT removed)")
        mm.metrics.methods_used["inpaint"] = "eraser_plate_watermarks_only"
    else:
        logger.warning("INPAINT: No watermarks — styled overlays will cover text (no eraser plates)")
        mm.metrics.methods_used["inpaint"] = "failed_overlay_fallback"
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

    # Write output video using ffmpeg pipe (h.264) instead of cv2.VideoWriter (mp4v)
    # mp4v codec produces macroblocking/datamoshing artifacts; h.264 is much higher quality
    logger.info(f"VideoPainter: Writing {len(all_output_frames)} frames via ffmpeg pipe (h.264, {width}x{height} @ {fps:.1f}fps)")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    ffproc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for frame in all_output_frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
        frame_resized = cv2.resize(frame_bgr, (width, height))
        ffproc.stdin.write(frame_resized.tobytes())

    ffproc.stdin.close()
    ffproc.wait(timeout=120)
    if ffproc.returncode != 0:
        stderr = ffproc.stderr.read().decode()[-500:]
        logger.error(f"VideoPainter: ffmpeg pipe failed (rc={ffproc.returncode}): {stderr}")
        raise RuntimeError(f"VideoPainter ffmpeg encoding failed: {stderr}")

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
        bbox = _normalize_bbox(region.get("bbox_norm", [0, 0.85, 1, 1]), vid_w, vid_h)

        # Convert normalized bbox to pixels with generous padding
        # V5: Gemini bboxes are approximate — extra padding prevents text bleed-through
        pad_x = max(25, int((bbox[2] - bbox[0]) * vid_w * 0.15))  # 15% of bbox width, min 25px
        pad_y = max(20, int((bbox[3] - bbox[1]) * vid_h * 0.20))  # 20% of bbox height, min 20px
        x = max(0, int(bbox[0] * vid_w) - pad_x)
        y = max(0, int(bbox[1] * vid_h) - pad_y)
        w = min(vid_w - x, int((bbox[2] - bbox[0]) * vid_w) + 2 * pad_x)
        h = min(vid_h - y, int((bbox[3] - bbox[1]) * vid_h) + 2 * pad_y)

        # Ensure minimum dimensions
        w = max(w, 10)
        h = max(h, 10)

        # Use gblur (gaussian blur) — strong enough to fully obscure text
        blur_sigma = max(15, min(40, min(w, h) // 3))

        # Create blur+darken filter for this region
        # crop → gblur → darken (eq) → overlay at original position
        filter_parts.append(
            f"{current_input}split[main{i}][blur_src{i}];"
            f"[blur_src{i}]crop={w}:{h}:{x}:{y},"
            f"gblur=sigma={blur_sigma},"
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
            pos_lower = position.lower()
            if pos_lower.startswith("top") and y_bottom < 0.45:
                in_correct_zone = True
            elif pos_lower.startswith("bottom") and y_top > 0.55:
                in_correct_zone = True
            elif pos_lower in ("middle", "center") and y_top < 0.65 and y_bottom > 0.35:
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
        pos_lower = position.lower()
        if pos_lower.startswith("top") and y_bottom < 0.40:  # Entire bbox in top 40%
            filtered.append(d)
        elif pos_lower.startswith("bottom") and y_top > 0.60:  # Entire bbox in bottom 40%
            filtered.append(d)
        elif pos_lower in ("middle", "center"):
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
    pos_lower = position.lower()
    if pos_lower in ("middle", "center"):
        logger.warning(
            f"RENDER_TEXT: No OCR in zone '{position}' — skipping (would obscure content)"
        )
        return (0, 0, 1, 1)  # Signal to skip this overlay

    logger.warning(f"RENDER_TEXT: No detections, using position fallback for '{original_text[:30]}...'")

    # Narrower fallback box (70% width instead of 94%)
    margin = int(video_width * 0.15)
    box_w = int(video_width * 0.70)
    box_h = int(video_height * 0.08)

    if pos_lower.startswith("top"):
        y = int(video_height * 0.03)
    else:  # bottom, bottom_right, bottom_left, etc.
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


def _normalize_bbox(bbox: list, video_width: int = 1920, video_height: int = 1080) -> list:
    """Normalize bbox values to 0.0-1.0 range.

    Gemini sometimes returns mixed coordinates: x as 0-1 normalized, y as pixel values.
    Detect and fix any value > 1.0 by dividing by the appropriate dimension.
    """
    if not bbox or len(bbox) < 4:
        return [0, 0.85, 1, 1]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    # If any value > 1.5, treat it as pixel coordinate and normalize
    if x1 > 1.5:
        x1 /= video_width
    if x2 > 1.5:
        x2 /= video_width
    if y1 > 1.5:
        y1 /= video_height
    if y2 > 1.5:
        y2 /= video_height
    # Clamp to 0-1
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return [x1, y1, x2, y2]


def _strip_emoji(text: str) -> str:
    """Strip emoji characters from text — ASS/libass can't render color emoji glyphs."""
    import re
    # Remove all Unicode emoji ranges (emoticons, symbols, dingbats, etc.)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"   # Symbols & pictographs
        "\U0001F680-\U0001F6FF"   # Transport & map
        "\U0001F1E0-\U0001F1FF"   # Flags
        "\U00002702-\U000027B0"   # Dingbats
        "\U000024C2-\U0001F251"   # Enclosed chars
        "\U0001F900-\U0001F9FF"   # Supplemental
        "\U0001FA00-\U0001FA6F"   # Chess symbols
        "\U0001FA70-\U0001FAFF"   # Extended-A
        "\U00002600-\U000026FF"   # Misc symbols
        "\U0000FE00-\U0000FE0F"   # Variation selectors
        "\U0000200D"              # Zero-width joiner
        "\U00000023\U0000FE0F\U000020E3"  # Keycap #
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub("", text).strip()


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


def _is_light_hex_color(hex_color: str) -> bool:
    """Return True if a hex color is visually light (and bad for subtitle backplates)."""
    if not hex_color:
        return False
    hc = hex_color.strip().lstrip("#")
    if len(hc) == 3:
        hc = "".join(ch * 2 for ch in hc)
    if len(hc) != 6:
        return False
    try:
        r = int(hc[0:2], 16)
        g = int(hc[2:4], 16)
        b = int(hc[4:6], 16)
    except ValueError:
        return False
    # Relative luminance approximation (0..255)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 145


def _safe_int(value: Any, default: int) -> int:
    """Best-effort integer parsing with fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """Best-effort float parsing with fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _density_from_max_chars(max_chars: int) -> str:
    """Map max chars per line to coarse text density."""
    if max_chars <= 20:
        return "compact"
    if max_chars >= 30:
        return "airy"
    return "balanced"


def _coerce_container_pref(value: Any, fallback: str = "pill") -> str:
    """Normalize free-form container preference values."""
    pref = str(value or "").strip().lower()
    if pref in ("pill", "rounded_pill", "round_pill"):
        return "pill"
    if pref in ("box", "boxed", "rect", "rectangle"):
        return "box"
    if pref in ("strip", "bar", "band"):
        return "strip"
    if pref in ("none", "transparent", "text_only"):
        return "none"
    return fallback


def _extract_style_intent(
    target_caption_style: Optional[Dict],
    subtitle_style: Optional[Dict],
    video_profile: Optional[Dict],
    pipeline_type: str,
) -> Dict[str, Any]:
    """
    Build a stable style intent from multiple sources.

    Gemini style is treated as intent (what look to aim for), while rendering
    policy later enforces hard guardrails.
    """
    account_profile = (video_profile or {}).get("account_style_profile", {}) or {}
    intent = {
        "container_preference": "pill" if pipeline_type in ("subtitle_snap", "blur_plate") else "none",
        "density": "balanced",
        "emphasis": "medium",
        "motion": "none",
        "source": "default",
    }

    # Account profile gives cross-video consistency baseline.
    if isinstance(account_profile, dict):
        if account_profile.get("container_preference"):
            intent["container_preference"] = _coerce_container_pref(
                account_profile.get("container_preference"), intent["container_preference"]
            )
            intent["source"] = "account_profile"
        if account_profile.get("density") in ("compact", "balanced", "airy"):
            intent["density"] = account_profile["density"]
        if account_profile.get("emphasis") in ("low", "medium", "high"):
            intent["emphasis"] = account_profile["emphasis"]
        if account_profile.get("motion") in ("none", "fade", "pop"):
            intent["motion"] = account_profile["motion"]

    # Per-video Gemini suggestion can refine baseline.
    if target_caption_style and isinstance(target_caption_style, dict):
        intent["source"] = "gemini_target"
        if target_caption_style.get("bg_style"):
            intent["container_preference"] = _coerce_container_pref(
                target_caption_style.get("bg_style"), intent["container_preference"]
            )

        max_chars = _safe_int(target_caption_style.get("max_chars_line"), 25)
        if max_chars > 0:
            intent["density"] = _density_from_max_chars(max_chars)

        emphasis_score = 0
        if bool(target_caption_style.get("bold", True)):
            emphasis_score += 1
        if _safe_int(target_caption_style.get("outline_width"), 0) >= 3:
            emphasis_score += 1
        if _safe_int(target_caption_style.get("shadow_depth"), 0) >= 2:
            emphasis_score += 1
        if emphasis_score >= 2:
            intent["emphasis"] = "high"
        elif emphasis_score <= 0:
            intent["emphasis"] = "low"
        else:
            intent["emphasis"] = "medium"

    # Original subtitle style is weaker signal, only fills gaps.
    if subtitle_style and isinstance(subtitle_style, dict):
        animation = str(subtitle_style.get("animation", "")).lower()
        if intent["motion"] == "none" and animation in ("pop", "fade"):
            intent["motion"] = animation
        if not target_caption_style:
            fw = str(subtitle_style.get("font_weight", "")).lower()
            if fw in ("black", "extrabold", "bold"):
                intent["emphasis"] = "high"

    # Pipeline guardrails at intent level.
    if pipeline_type == "subtitle_snap":
        if intent["container_preference"] == "none":
            intent["container_preference"] = "pill"
    elif pipeline_type == "blur_plate":
        # Blur already adds texture reduction, so avoid "strip" intent here.
        if intent["container_preference"] == "strip":
            intent["container_preference"] = "pill"

    return intent


def _build_layout_policy(
    pipeline_type: str,
    density: str = "balanced",
    container_preference: str = "pill",
) -> Dict[str, Any]:
    """Deterministic layout caps used by renderers to prevent giant blocks."""
    # Base for subtitle-like overlays
    policy = {
        "zone_caps": {
            "top": {"width": 0.70, "height": 0.16},
            "middle": {"width": 0.60, "height": 0.22},
            "bottom": {"width": 0.70, "height": 0.16},
        },
        "cover_pad_x_ratio": 0.08,
        "cover_pad_y_ratio": 0.12,
        "cover_pad_x_min": 8,
        "cover_pad_y_min": 8,
        "text_bias_w_ratio": 1.6,
        "text_bias_h_ratio": 1.2,
    }

    if pipeline_type == "blur_plate":
        # Slightly wider/shorter allowed on blur plate.
        policy["zone_caps"]["top"]["width"] = 0.78
        policy["zone_caps"]["middle"]["width"] = 0.70
        policy["zone_caps"]["bottom"]["width"] = 0.78
        policy["zone_caps"]["top"]["height"] = 0.19
        policy["zone_caps"]["bottom"]["height"] = 0.19

    if density == "compact":
        for z in policy["zone_caps"].values():
            z["width"] = max(0.58, z["width"] - 0.04)
        policy["cover_pad_x_ratio"] = 0.07
    elif density == "airy":
        for z in policy["zone_caps"].values():
            z["width"] = min(0.82, z["width"] + 0.03)
        policy["cover_pad_x_ratio"] = 0.10

    if container_preference == "box":
        # Box styles can be a bit wider; still capped.
        for z in policy["zone_caps"].values():
            z["width"] = min(0.84, z["width"] + 0.03)
    elif container_preference == "none":
        # Minimal container use: keep tight.
        for z in policy["zone_caps"].values():
            z["height"] = max(0.16, z["height"] - 0.03)

    return policy


def _apply_caption_style_policy(
    style: Dict[str, Any],
    style_intent: Dict[str, Any],
    pipeline_type: str,
    target_language: str,
    video_profile: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convert style intent + loose style into a safe render policy.
    This is the hard-guardrail layer.
    """
    out = dict(style or {})
    account_profile = (video_profile or {}).get("account_style_profile", {}) or {}

    out["font_size_pct"] = max(0.032, min(_safe_float(out.get("font_size_pct"), 0.04), 0.055))
    out["outline_width"] = max(0, min(_safe_int(out.get("outline_width"), 2), 6))
    out["shadow_depth"] = max(0, min(_safe_int(out.get("shadow_depth"), 2), 4))
    out["max_chars_line"] = max(12, min(_safe_int(out.get("max_chars_line"), 25), 36))
    out["spacing"] = max(0, min(_safe_int(out.get("spacing"), 1), 3))

    density = style_intent.get("density", "balanced")
    emphasis = style_intent.get("emphasis", "medium")
    container_pref = style_intent.get("container_preference", "pill")

    if pipeline_type == "subtitle_snap":
        out["bg_style"] = "pill"
        out["border_style"] = 3
        out["container_preference"] = container_pref

        if density == "compact":
            chars = 20
            bg_alpha_target = 0.12
        elif density == "airy":
            chars = 28
            bg_alpha_target = 0.19
        else:
            chars = 24
            bg_alpha_target = 0.16

        if emphasis == "high":
            bg_alpha_target = max(0.08, bg_alpha_target - 0.03)
        elif emphasis == "low":
            bg_alpha_target = min(0.24, bg_alpha_target + 0.03)

        # account profile hint is CSS alpha (0 transparent, 1 opaque) -> convert to ASS convention.
        hint_css = _safe_float(account_profile.get("bg_alpha_hint_css"), -1.0)
        if 0.0 <= hint_css <= 1.0:
            hint_ass = 1.0 - hint_css
            bg_alpha_target = (bg_alpha_target * 0.7) + (hint_ass * 0.3)

        out["max_chars_line"] = max(16, min(chars, out["max_chars_line"]))
        out["bg_alpha"] = max(0.08, min(bg_alpha_target, 0.24))

        if _is_light_hex_color(out.get("bg_color", "#000000")):
            out["bg_color"] = "#1A1A1A"

    elif pipeline_type == "blur_plate":
        # Blur is already applied in pixels; keep overlay readable but compact.
        out["bg_style"] = "pill"
        out["border_style"] = 3
        out["max_chars_line"] = max(16, min(out["max_chars_line"], 28))
        out["bg_alpha"] = max(0.10, min(_safe_float(out.get("bg_alpha"), 0.16), 0.26))
        if _is_light_hex_color(out.get("bg_color", "#000000")):
            out["bg_color"] = "#1A1A1A"

    # Language/script-specific policy clamps
    lang_base = normalize_language_code(target_language) if target_language else ""
    if lang_base in ("zh", "ja", "ko") or target_language in ("zh-CN", "zh-TW"):
        out["max_chars_line"] = min(out["max_chars_line"], 16)
        out["spacing"] = max(out["spacing"], 1)
    elif lang_base in ("ar", "he", "fa", "ur"):
        out["max_chars_line"] = min(out["max_chars_line"], 30)

    out["_style_intent"] = {
        "container_preference": container_pref,
        "density": density,
        "emphasis": emphasis,
        "motion": style_intent.get("motion", "none"),
        "source": style_intent.get("source", "default"),
    }
    out["_layout_policy"] = _build_layout_policy(
        pipeline_type=pipeline_type,
        density=density,
        container_preference=container_pref,
    )
    return out


def _solve_container_dimensions(
    orig_w: int,
    orig_h: int,
    text_container_w: int,
    text_container_h: int,
    font_size: int,
    zone: str,
    video_width: int,
    video_height: int,
    layout_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """Compute robust container dimensions with policy guardrails."""
    lp = layout_policy or _build_layout_policy("subtitle_snap")
    caps = (lp.get("zone_caps", {}) or {}).get(zone, {"width": 0.64, "height": 0.24})
    width_cap_ratio = _safe_float(caps.get("width"), 0.64)
    height_cap_ratio = _safe_float(caps.get("height"), 0.24)

    cover_pad_x_ratio = _safe_float(lp.get("cover_pad_x_ratio"), 0.08)
    cover_pad_y_ratio = _safe_float(lp.get("cover_pad_y_ratio"), 0.12)
    cover_pad_x_min = _safe_int(lp.get("cover_pad_x_min"), 8)
    cover_pad_y_min = _safe_int(lp.get("cover_pad_y_min"), 8)
    text_bias_w_ratio = _safe_float(lp.get("text_bias_w_ratio"), 1.6)
    text_bias_h_ratio = _safe_float(lp.get("text_bias_h_ratio"), 1.2)

    cover_pad_x = int(max(orig_w * cover_pad_x_ratio, cover_pad_x_min))
    cover_pad_y = int(max(orig_h * cover_pad_y_ratio, cover_pad_y_min))
    cover_w = orig_w + cover_pad_x * 2
    cover_h = orig_h + cover_pad_y * 2

    # Detect suspiciously wide source bboxes (likely defaults, not real text)
    if orig_w > int(video_width * 0.85):
        cover_w = min(cover_w, max(text_container_w + int(font_size * 2), int(video_width * 0.45)))

    # Over-wide/over-tall source bboxes are often detector artifacts.
    if orig_w > int(video_width * 0.62) and text_container_w < int(video_width * 0.55):
        cover_w = max(text_container_w + int(font_size * text_bias_w_ratio), int(video_width * 0.42))
    if orig_h > int(video_height * 0.20) and text_container_h < int(video_height * 0.14):
        cover_h = max(text_container_h + int(font_size * text_bias_h_ratio), int(video_height * 0.10))

    container_w = max(text_container_w, cover_w)
    container_h = max(text_container_h, cover_h)
    container_w = min(container_w, int(video_width * width_cap_ratio))
    container_h = min(container_h, int(video_height * height_cap_ratio))
    container_w = max(container_w, text_container_w)
    container_h = max(container_h, text_container_h)
    return container_w, container_h


def _enforce_container_guardrails(
    container_w: int,
    container_h: int,
    *,
    orig_w: int,
    orig_h: int,
    text_container_w: int,
    text_container_h: int,
    font_size: int,
    zone: str,
    video_width: int,
    video_height: int,
    layout_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Enforce deterministic sizing guardrails and auto-fallback to compact mode.

    The solver above computes the first-pass size. This layer handles two failure
    modes seen in production:
    1) low source coverage (source text can leak through);
    2) oversized slabs caused by noisy region bboxes.
    """
    lp = layout_policy or _build_layout_policy("subtitle_snap")
    caps = (lp.get("zone_caps", {}) or {}).get(zone, {"width": 0.64, "height": 0.24})
    width_cap_ratio = _safe_float(caps.get("width"), 0.64)
    height_cap_ratio = _safe_float(caps.get("height"), 0.24)

    zone = (zone or "middle").lower()
    zone_area_caps = {"top": 0.14, "middle": 0.16, "bottom": 0.14}
    max_area_ratio = zone_area_caps.get(zone, 0.16)

    # Treat very wide/tall source boxes as suspect when text itself is compact.
    effective_orig_w = max(1, orig_w)
    effective_orig_h = max(1, orig_h)
    suspect_source_bbox = False
    if orig_w > int(video_width * 0.58) and text_container_w < int(video_width * 0.50):
        effective_orig_w = max(text_container_w + int(font_size * 1.6), int(video_width * 0.34))
        suspect_source_bbox = True
    if orig_h > int(video_height * 0.22) and text_container_h < int(video_height * 0.16):
        effective_orig_h = max(text_container_h + int(font_size * 1.2), int(video_height * 0.08))
        suspect_source_bbox = True

    # Coverage floor.
    min_cover_w = max(text_container_w, int(effective_orig_w * 1.03))
    min_cover_h = max(text_container_h, int(effective_orig_h * 1.03))
    container_w = max(container_w, min_cover_w)
    container_h = max(container_h, min_cover_h)

    fallback_applied = False
    fallback_reasons: List[str] = []

    area_ratio = (container_w * container_h) / max(1, video_width * video_height)
    oversized_before = area_ratio > max_area_ratio
    if oversized_before:
        # Compact fallback bound by zone caps and text envelope.
        max_w = int(video_width * min(0.80, max(0.52, width_cap_ratio - 0.02)))
        max_h = int(video_height * min(0.24, max(0.14, height_cap_ratio - 0.01)))
        if zone in ("top", "bottom"):
            max_h = min(max_h, int(video_height * 0.20))

        # Prevent giant slabs: never exceed a multiple of text envelope.
        max_w = min(max_w, max(text_container_w, int(text_container_w * 2.20)))
        max_h = min(max_h, max(text_container_h, int(text_container_h * 2.40)))

        # If source bbox looks noisy, bias harder toward compact text envelope.
        if suspect_source_bbox:
            max_w = min(max_w, max(text_container_w, int(text_container_w * 1.70)))
            max_h = min(max_h, max(text_container_h, int(text_container_h * 1.85)))
            fallback_reasons.append("suspect_source_bbox")

        new_w = min(container_w, max_w)
        new_h = min(container_h, max_h)

        # Keep minimum source coverage against effective bbox.
        new_w = max(new_w, text_container_w, int(effective_orig_w * 1.01))
        new_h = max(new_h, text_container_h, int(effective_orig_h * 1.01))

        # Respect hard zone caps unless they're smaller than text itself.
        hard_w_cap = int(video_width * width_cap_ratio)
        hard_h_cap = int(video_height * height_cap_ratio)
        if new_w > hard_w_cap and text_container_w <= hard_w_cap:
            new_w = hard_w_cap
        if new_h > hard_h_cap and text_container_h <= hard_h_cap:
            new_h = hard_h_cap

        if new_w < container_w or new_h < container_h:
            container_w = max(text_container_w, new_w)
            container_h = max(text_container_h, new_h)
            fallback_applied = True
            fallback_reasons.append("compact_fallback")

    # Recheck coverage after fallback and lift if needed.
    coverage_x_eff = container_w / max(1, effective_orig_w)
    coverage_y_eff = container_h / max(1, effective_orig_h)
    if coverage_x_eff < 1.02 or coverage_y_eff < 1.02:
        target_w = max(container_w, int(effective_orig_w * 1.03))
        target_h = max(container_h, int(effective_orig_h * 1.03))
        hard_w_cap = int(video_width * width_cap_ratio)
        hard_h_cap = int(video_height * height_cap_ratio)
        if text_container_w <= hard_w_cap:
            target_w = min(target_w, hard_w_cap)
        if text_container_h <= hard_h_cap:
            target_h = min(target_h, hard_h_cap)
        if target_w > container_w or target_h > container_h:
            container_w = max(text_container_w, target_w)
            container_h = max(text_container_h, target_h)
            fallback_applied = True
            fallback_reasons.append("coverage_boost")

    coverage_x_eff = container_w / max(1, effective_orig_w)
    coverage_y_eff = container_h / max(1, effective_orig_h)
    coverage_x_raw = container_w / max(1, orig_w)
    coverage_y_raw = container_h / max(1, orig_h)
    area_ratio = (container_w * container_h) / max(1, video_width * video_height)

    metrics = {
        "coverage_x_eff": coverage_x_eff,
        "coverage_y_eff": coverage_y_eff,
        "coverage_x_raw": coverage_x_raw,
        "coverage_y_raw": coverage_y_raw,
        "area_ratio": area_ratio,
        "max_area_ratio": max_area_ratio,
        "coverage_low": coverage_x_eff < 1.02 or coverage_y_eff < 1.02,
        "oversized": area_ratio > max_area_ratio,
        "oversized_before": oversized_before,
        "fallback_applied": fallback_applied,
        "fallback_reasons": fallback_reasons,
        "suspect_source_bbox": suspect_source_bbox,
        "effective_orig_w": effective_orig_w,
        "effective_orig_h": effective_orig_h,
    }
    return container_w, container_h, metrics


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
# Style Format keys (core — used by ASS renderer):
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
#
# Extended metadata keys (geo-native styling):
#   font_family: canonical font family name (e.g., "Montserrat")
#   font_weight: weight string ("Bold"/"ExtraBold"/"Black"/"Regular")
#   border_radius: px (0=square, 999=full pill)
#   padding_x_ratio: float (horizontal padding relative to font_size)
#   padding_y_ratio: float (vertical padding relative to font_size)
#   text_transform: "uppercase" | "none"
#   animation: "fade" | "pop" | "none"

CAPTION_STYLE_PRESETS = {
    # ══════════════════════════════════════════════════════════════════
    # PLATFORM PRESETS (content-type based)
    # ══════════════════════════════════════════════════════════════════

    # ── TikTok Native — bold outline, no bg ──────────────────────────
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
        "font_family": "Montserrat",
        "font_weight": "Bold",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "none",
        "animation": "none",
    },
    # ── EN (US/UK) — TikTok pill, CapCut-style ──────────────────────
    # Montserrat Bold, white on dark semi-transparent pill (rgba 0,0,0,0.85),
    # full pill border-radius, text-shadow for depth
    "tiktok_pill": {
        "font": "Montserrat",
        "font_size_pct": 0.038,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#1A1A1A",
        "outline_width": 0,
        "shadow_depth": 2,
        "shadow_color": "#000000",
        "bg_style": "pill",
        "bg_color": "#000000",
        "bg_alpha": 0.15,   # ASS: 0=opaque → 0.15 = 85% opaque (rgba 0,0,0,0.85)
        "border_style": 3,
        "spacing": 1,
        "max_chars_line": 28,
        "font_family": "Montserrat",
        "font_weight": "Bold",
        "border_radius": 999,     # full pill
        "padding_x_ratio": 0.6,
        "padding_y_ratio": 0.25,
        "text_transform": "none",
        "animation": "fade",
    },
    # ── Reels / YouTube Shorts — clean modern ────────────────────────
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
        "font_family": "Helvetica",
        "font_weight": "Bold",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "none",
        "animation": "none",
    },
    # ── Blogger / Talking Head — accent color box ────────────────────
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
        "font_family": "Montserrat",
        "font_weight": "Bold",
        "border_radius": 6,
        "padding_x_ratio": 0.5,
        "padding_y_ratio": 0.2,
        "text_transform": "none",
        "animation": "pop",
    },
    # ── Product / Tutorial — minimal clean ───────────────────────────
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
        "font_family": "Helvetica",
        "font_weight": "Bold",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "none",
        "animation": "none",
    },
    # ── Dark strip — TikTok dubbed content ───────────────────────────
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
        "bg_color": "#1A1A1A",
        "bg_alpha": 0.18,   # ASS convention: 0=opaque. 0.18 = 82% opaque (semi-transparent)
        "border_style": 1,
        "spacing": 1,
        "max_chars_line": 32,
        "font_family": "Montserrat",
        "font_weight": "Bold",
        "border_radius": 0,
        "padding_x_ratio": 0.3,
        "padding_y_ratio": 0.15,
        "text_transform": "none",
        "animation": "fade",
    },

    # ══════════════════════════════════════════════════════════════════
    # GEO-NATIVE PRESETS (market/language based)
    # ══════════════════════════════════════════════════════════════════

    # ── ES (LATAM) — vibrant, no bg, heavy stroke + hard shadow ──────
    # Poppins ExtraBold, white, 3px black stroke, uppercase.
    # Matches LATAM TikTok/Reels trend: loud, colorful, high contrast.
    "latam_vibrant": {
        "font": "Poppins",
        "font_size_pct": 0.044,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 3,
        "shadow_depth": 3,
        "shadow_color": "#1A1A1A",
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 1,
        "max_chars_line": 24,
        "font_family": "Poppins",
        "font_weight": "ExtraBold",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "uppercase",
        "animation": "pop",
    },
    # ── PT-BR — Brazil energy, similar to LATAM with color accent ────
    # Poppins ExtraBold, white text, heavy black stroke, dark green
    # shadow accent (Brazil flag energy), uppercase.
    "brazil_energy": {
        "font": "Poppins",
        "font_size_pct": 0.044,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 3,
        "shadow_depth": 3,
        "shadow_color": "#1B5E20",   # dark green accent (Brazil)
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 1,
        "max_chars_line": 24,
        "font_family": "Poppins",
        "font_weight": "ExtraBold",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "uppercase",
        "animation": "pop",
    },
    # ── AR (MENA) — clean dark box, RTL-optimized ────────────────────
    # Cairo Bold (or Noto Sans Arabic fallback), white text on dark
    # opaque box (0.85 alpha), 8px radius, no outline. RTL-friendly.
    "arabic_clean": {
        "font": "Cairo",
        "font_size_pct": 0.040,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow_depth": 1,
        "shadow_color": "#000000",
        "bg_style": "box",
        "bg_color": "#000000",
        "bg_alpha": 0.15,   # 85% opaque dark box
        "border_style": 3,
        "spacing": 0,
        "max_chars_line": 30,
        "font_family": "Cairo",
        "font_weight": "Bold",
        "border_radius": 8,
        "padding_x_ratio": 0.5,
        "padding_y_ratio": 0.25,
        "text_transform": "none",
        "animation": "fade",
    },
    # ── DE — European clean, corporate, high legibility ──────────────
    # Inter 500-700 weight, dark bg with 0.9 alpha, 4px radius,
    # subtle shadow. German audiences prefer understated clarity.
    "european_clean": {
        "font": "Inter",
        "font_size_pct": 0.038,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#1A1A1A",
        "outline_width": 0,
        "shadow_depth": 1,
        "shadow_color": "#111111",
        "bg_style": "box",
        "bg_color": "#0A0A0A",
        "bg_alpha": 0.10,   # 90% opaque dark bg
        "border_style": 3,
        "spacing": 0,
        "max_chars_line": 30,
        "font_family": "Inter",
        "font_weight": "Bold",
        "border_radius": 4,
        "padding_x_ratio": 0.5,
        "padding_y_ratio": 0.2,
        "text_transform": "none",
        "animation": "fade",
    },
    # ── FR — French elegant, refined European clean ──────────────────
    # Same base as DE but slightly larger text, more padding, slightly
    # more shadow for cinematic depth. French content leans refined.
    "french_elegant": {
        "font": "Inter",
        "font_size_pct": 0.039,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#1A1A1A",
        "outline_width": 0,
        "shadow_depth": 2,
        "shadow_color": "#0D0D0D",
        "bg_style": "box",
        "bg_color": "#0A0A0A",
        "bg_alpha": 0.10,   # 90% opaque dark bg
        "border_style": 3,
        "spacing": 0,
        "max_chars_line": 32,
        "font_family": "Inter",
        "font_weight": "Bold",
        "border_radius": 4,
        "padding_x_ratio": 0.55,
        "padding_y_ratio": 0.22,
        "text_transform": "none",
        "animation": "fade",
    },
    # ── JA — Japanese telop, anime/variety show style ────────────────
    # Noto Sans CJK JP Black, white text, NO background, heavy 4px
    # colored outline (#FF6699 pink), glow effect. Matches telop tradition.
    "jp_telop": {
        "font": "Noto Sans CJK JP",
        "font_size_pct": 0.042,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#FF6699",
        "outline_width": 4,
        "shadow_depth": 3,
        "shadow_color": "#CC3366",   # darker pink glow
        "bg_style": "none",
        "bg_color": "#000000",
        "bg_alpha": 0.0,
        "border_style": 1,
        "spacing": 2,
        "max_chars_line": 14,
        "font_family": "Noto Sans JP",
        "font_weight": "Black",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "none",
        "animation": "pop",
    },
    # ── KO — Korean cafe aesthetic, light & soft ─────────────────────
    # Noto Sans CJK KR light (300-400 weight), dark text on white/pastel
    # bg (85% opaque). Clean, airy feel matching Korean design trends.
    "korean_cafe": {
        "font": "Noto Sans CJK KR",
        "font_size_pct": 0.040,
        "bold": False,
        "text_color": "#1A1A1A",
        "outline_color": "#FFFFFF",
        "outline_width": 0,
        "shadow_depth": 1,
        "shadow_color": "#CCCCCC",
        "bg_style": "box",
        "bg_color": "#FFFFFF",
        "bg_alpha": 0.15,   # 85% opaque white/pastel bg
        "border_style": 3,
        "spacing": 1,
        "max_chars_line": 16,
        "font_family": "Noto Sans KR",
        "font_weight": "Regular",
        "border_radius": 8,
        "padding_x_ratio": 0.5,
        "padding_y_ratio": 0.25,
        "text_transform": "none",
        "animation": "fade",
    },

    # ══════════════════════════════════════════════════════════════════
    # LEGACY / COMPAT ALIASES
    # ══════════════════════════════════════════════════════════════════

    # arabic_native — alias for arabic_clean (backwards compat)
    "arabic_native": {
        "font": "Cairo",
        "font_size_pct": 0.040,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow_depth": 1,
        "shadow_color": "#000000",
        "bg_style": "box",
        "bg_color": "#000000",
        "bg_alpha": 0.15,
        "border_style": 3,
        "spacing": 0,
        "max_chars_line": 30,
        "font_family": "Cairo",
        "font_weight": "Bold",
        "border_radius": 8,
        "padding_x_ratio": 0.5,
        "padding_y_ratio": 0.25,
        "text_transform": "none",
        "animation": "fade",
    },
    # cjk_bold — Chinese generic (wider spacing)
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
        "font_family": "Noto Sans CJK SC",
        "font_weight": "Bold",
        "border_radius": 0,
        "padding_x_ratio": 0.0,
        "padding_y_ratio": 0.0,
        "text_transform": "none",
        "animation": "none",
    },
    # en_production — alias for tiktok_pill with higher opacity
    "en_production": {
        "font": "Montserrat",
        "font_size_pct": 0.040,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#1A1A1A",
        "outline_width": 0,
        "shadow_depth": 2,
        "shadow_color": "#000000",
        "bg_style": "pill",
        "bg_color": "#000000",
        "bg_alpha": 0.08,   # 92% opaque for overlay coverage
        "border_style": 3,
        "spacing": 1,
        "max_chars_line": 28,
        "font_family": "Montserrat",
        "font_weight": "Bold",
        "border_radius": 999,
        "padding_x_ratio": 0.6,
        "padding_y_ratio": 0.25,
        "text_transform": "none",
        "animation": "fade",
    },
    # arabic_production — alias for arabic_clean with higher opacity
    "arabic_production": {
        "font": "Cairo",
        "font_size_pct": 0.040,
        "bold": True,
        "text_color": "#FFFFFF",
        "outline_color": "#0A0A0A",
        "outline_width": 0,
        "shadow_depth": 1,
        "shadow_color": "#000000",
        "bg_style": "pill",
        "bg_color": "#0A0A0A",
        "bg_alpha": 0.08,   # 92% opaque for overlay coverage
        "border_style": 3,
        "spacing": 0,
        "max_chars_line": 28,
        "font_family": "Cairo",
        "font_weight": "Bold",
        "border_radius": 8,
        "padding_x_ratio": 0.5,
        "padding_y_ratio": 0.25,
        "text_transform": "none",
        "animation": "fade",
    },
}

# ── GEO_STYLE_MAP — canonical lang code → preset name ────────────────
# Primary mapping for geo-native caption styles.
# Default fallback: "tiktok_pill"
GEO_STYLE_MAP = {
    "en": "tiktok_pill",
    "es": "latam_vibrant",
    "pt": "brazil_energy",
    "ar": "arabic_clean",
    "de": "european_clean",
    "fr": "french_elegant",
    "ja": "jp_telop",
    "ko": "korean_cafe",
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
# Production: each geo gets its own native-looking style.
# Uses GEO_STYLE_MAP presets + variant codes for regional specificity.
_MARKET_STYLE_MAP = {
    # English markets — TikTok pill
    "en": "tiktok_pill",
    "en-US": "tiktok_pill",
    "en-GB": "tiktok_pill",
    # Arabic/RTL — clean dark box
    "ar": "arabic_clean",
    "he": "arabic_clean",
    "fa": "arabic_clean",
    "ur": "arabic_clean",
    # CJK — language-specific styles
    "zh-CN": "cjk_bold",
    "zh-TW": "cjk_bold",
    "ja": "jp_telop",
    "ko": "korean_cafe",
    # LATAM — vibrant, heavy stroke
    "es": "latam_vibrant",
    "es-MX": "latam_vibrant",
    "es-AR": "latam_vibrant",
    # Brazil — separate energy style
    "pt": "brazil_energy",
    "pt-BR": "brazil_energy",
    # European — geo-specific styles
    "de": "european_clean",
    "fr": "french_elegant",
    "it": "european_clean",
    "nl": "european_clean",
    "pl": "european_clean",
    "sv": "european_clean",
    # Turkish — bold sans, no background
    "tr": "tiktok_bold",
    # Hindi/Indic — default to tiktok_pill (covers well)
    "hi": "tiktok_pill",
    "bn": "tiktok_pill",
    "ta": "tiktok_pill",
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
    style_intent = _extract_style_intent(
        target_caption_style=target_caption_style,
        subtitle_style=subtitle_style,
        video_profile=video_profile,
        pipeline_type=pipeline_type,
    )

    style = None
    preset_name = "tiktok_bold"
    style_source = "preset"
    style_reasoning = ""

    # ── Priority 1: Use Gemini-generated style as intent source if available ──
    if target_caption_style and isinstance(target_caption_style, dict):
        required_keys = {"font", "text_color", "outline_color"}
        if required_keys.issubset(target_caption_style.keys()):
            css_alpha = max(0.0, min(_safe_float(target_caption_style.get("bg_alpha"), 1.0), 1.0))
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
                # Gemini uses CSS alpha (0 transparent, 1 opaque).
                # Internal ASS convention is inverted (0 opaque, 1 transparent).
                "bg_alpha": 1.0 - css_alpha,
                "border_style": target_caption_style.get("border_style", 1),
                "spacing": target_caption_style.get("spacing", 1),
                "max_chars_line": target_caption_style.get("max_chars_line", 25),
            }
            style_source = "gemini_generated"
            style_reasoning = target_caption_style.get("style_reasoning", "")

    # ── Fallback: preset-based style ──
    if style is None:
        if pipeline_type == "subtitle_snap":
            preset_name = "tiktok_pill"
        elif pipeline_type == "blur_plate":
            preset_name = "tiktok_pill"

        if video_profile:
            video_type = video_profile.get("video_type", "")
            if video_type in _CONTENT_TYPE_STYLE_MAP:
                preset_name = _CONTENT_TYPE_STYLE_MAP[video_type]

        lang_base = normalize_language_code(target_language) if target_language else ""
        if target_language in _MARKET_STYLE_MAP:
            preset_name = _MARKET_STYLE_MAP[target_language]
        elif lang_base in _MARKET_STYLE_MAP:
            preset_name = _MARKET_STYLE_MAP[lang_base]

        style = dict(CAPTION_STYLE_PRESETS.get(preset_name, CAPTION_STYLE_PRESETS["tiktok_bold"]))

        # Apply original source style hints only for preset fallback path.
        if subtitle_style:
            orig_font = subtitle_style.get("font_family", "")
            good_fonts = {"Impact", "Montserrat", "Helvetica", "Arial", "Bebas Neue", "Oswald", "Raleway"}
            if orig_font in good_fonts:
                style["font"] = orig_font
            if subtitle_style.get("text_color"):
                style["text_color"] = subtitle_style["text_color"]
            if subtitle_style.get("outline_color"):
                style["outline_color"] = subtitle_style["outline_color"]

    # Market/script corrections are applied in both gemini + preset paths.
    lang_base = normalize_language_code(target_language) if target_language else ""
    if lang_base in ("zh", "ja", "ko") or target_language in ("zh-CN", "zh-TW"):
        cjk_fonts = {"ja": "Noto Sans CJK JP", "ko": "Noto Sans CJK KR", "zh": "Noto Sans CJK SC"}
        style["font"] = cjk_fonts.get(lang_base, "Noto Sans CJK SC")
        if target_language == "zh-TW":
            style["font"] = "Noto Sans CJK TC"
        style["max_chars_line"] = min(_safe_int(style.get("max_chars_line"), 20), 16)
        style["spacing"] = max(_safe_int(style.get("spacing"), 1), 1)
    elif lang_base in ("ar", "he", "fa", "ur"):
        style["font"] = "Noto Sans Arabic" if lang_base in ("ar", "fa", "ur") else "Noto Sans Hebrew"
        style["max_chars_line"] = min(_safe_int(style.get("max_chars_line"), 28), 30)
    elif lang_base in ("hi", "bn", "ta", "te"):
        indic_fonts = {"hi": "Noto Sans Devanagari", "bn": "Noto Sans Bengali", "ta": "Noto Sans Tamil", "te": "Noto Sans Telugu"}
        style["font"] = indic_fonts.get(lang_base, "Noto Sans Devanagari")
    else:
        geo_font = get_geo_font(target_language) if target_language else style.get("font", "Montserrat")
        if geo_font != style.get("font"):
            logger.info(f"CAPTION_STYLE: Geo font override: '{style.get('font')}' -> '{geo_font}' for lang={target_language}")
            style["font"] = geo_font

    # Final deterministic policy pass.
    style = _apply_caption_style_policy(
        style=style,
        style_intent=style_intent,
        pipeline_type=pipeline_type,
        target_language=target_language,
        video_profile=video_profile,
    )
    style["_preset_name"] = "gemini_generated" if style_source == "gemini_generated" else preset_name
    style["_style_reasoning"] = style_reasoning

    logger.info(
        "CAPTION_STYLE: resolved source=%s preset=%s lang=%s pipeline=%s "
        "font=%s bg=%s intent=%s",
        style_source,
        style.get("_preset_name"),
        target_language,
        pipeline_type,
        style.get("font"),
        style.get("bg_style"),
        style.get("_style_intent"),
    )
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

    # No-box text style for custom overlay renderers (subtitle_snap / blur_plate).
    # Avoids nested black boxes when main style uses BorderStyle=3.
    no_box_outline = max(2, outline_w) if border_style == 3 else outline_w
    no_box_shadow = max(2, shadow_d)
    styles += f"Style: TranslatedTextNoBox,{font},{font_size},{text_color},{text_color},{outline_color},&H00000000,{bold},0,0,0,100,100,{spacing},0,1,{no_box_outline},{no_box_shadow},5,10,10,10,1\n"

    # BackPlate (drawing-only, 1px invisible font)
    styles += f"Style: BackPlate,Arial,1,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"

    # BackPlateBox — for backplate_overlay strategy (BorderStyle=3: opaque box behind text)
    box_bg = _hex_to_ass_color(style["bg_color"], alpha=max(0.05, style["bg_alpha"]))
    box_outline = _hex_to_ass_color(style["bg_color"], alpha=max(0.05, style["bg_alpha"]))
    styles += f"Style: BackPlateBox,{font},{font_size},{text_color},&H000000FF,{box_outline},{box_bg},{bold},0,0,0,100,100,{spacing},0,3,10,2,5,10,10,10,1\n"

    # RTL variants
    if is_rtl and rtl_font:
        styles += f"Style: TranslatedTextRTL,{rtl_font},{font_size},{text_color},{text_color},{outline_color},{bg_color},{bold},0,0,0,100,100,{spacing},0,{border_style},{outline_w},{shadow_d},5,10,10,10,1\n"
        styles += f"Style: TranslatedTextNoBoxRTL,{rtl_font},{font_size},{text_color},{text_color},{outline_color},&H00000000,{bold},0,0,0,100,100,{spacing},0,1,{no_box_outline},{no_box_shadow},5,10,10,10,1\n"
        styles += f"Style: BackPlateBoxRTL,{rtl_font},{font_size},{text_color},&H000000FF,{box_outline},{box_bg},{bold},0,0,0,100,100,{spacing},0,3,10,2,5,10,10,10,1\n"

    return styles, font_size


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format (H:MM:SS.CC)."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def _position_to_zone(position: str) -> str:
    """Map position strings to canonical zones: top/middle/bottom."""
    pos = (position or "").lower()
    if pos in ("top", "top_left", "top_right", "upper"):
        return "top"
    if pos in ("bottom", "bottom_left", "bottom_right", "lower"):
        return "bottom"
    return "middle"


def _bbox_area_norm(bbox: list) -> float:
    """Area of normalized bbox."""
    if not bbox or len(bbox) < 4:
        return 0.0
    w = max(0.0, float(bbox[2]) - float(bbox[0]))
    h = max(0.0, float(bbox[3]) - float(bbox[1]))
    return w * h


def _bbox_center_norm(bbox: list) -> Tuple[float, float]:
    """Center of normalized bbox."""
    if not bbox or len(bbox) < 4:
        return 0.5, 0.5
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _bbox_iou_norm(a: list, b: list) -> float:
    """IoU for two normalized bboxes."""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _blend_bbox_norm(base_bbox: list, next_bbox: list, next_weight: float) -> list:
    """Linear blend of two normalized bboxes."""
    wb = max(0.0, min(1.0, 1.0 - float(next_weight)))
    wn = max(0.0, min(1.0, float(next_weight)))
    out = []
    for i in range(4):
        out.append((float(base_bbox[i]) * wb) + (float(next_bbox[i]) * wn))
    # Ensure sorted + clamped bounds
    x1, y1, x2, y2 = out
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def _clamp_bbox_by_zone(bbox: list, zone: str) -> Tuple[list, bool]:
    """Clamp bbox dimensions to zone-specific caps to avoid giant slabs."""
    zone = (zone or "middle").lower()
    zone_caps = {
        "top": {"w": 0.78, "h": 0.22},
        "middle": {"w": 0.72, "h": 0.28},
        "bottom": {"w": 0.78, "h": 0.22},
    }
    caps = zone_caps.get(zone, zone_caps["middle"])
    max_w = caps["w"]
    max_h = caps["h"]

    x1, y1, x2, y2 = _normalize_bbox(bbox)
    cx, cy = _bbox_center_norm([x1, y1, x2, y2])
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    changed = False

    if w > max_w:
        w = max_w
        changed = True
    if h > max_h:
        h = max_h
        changed = True

    # Keep zone semantics if detector drifts heavily.
    if zone == "top" and cy > 0.55:
        cy = 0.28
        changed = True
    elif zone == "bottom" and cy < 0.45:
        cy = 0.72
        changed = True

    nx1 = max(0.0, cx - w / 2.0)
    ny1 = max(0.0, cy - h / 2.0)
    nx2 = min(1.0, cx + w / 2.0)
    ny2 = min(1.0, cy + h / 2.0)

    # Preserve dimensions after edge clipping.
    if (nx2 - nx1) < w:
        if nx1 <= 0.0:
            nx2 = min(1.0, nx1 + w)
        elif nx2 >= 1.0:
            nx1 = max(0.0, nx2 - w)
        changed = True
    if (ny2 - ny1) < h:
        if ny1 <= 0.0:
            ny2 = min(1.0, ny1 + h)
        elif ny2 >= 1.0:
            ny1 = max(0.0, ny2 - h)
        changed = True

    return [nx1, ny1, nx2, ny2], changed


def _stabilize_matched_region_bbox(
    overlay: Dict[str, Any],
    region: Dict[str, Any],
    overlay_zone: str,
    appears_at: float,
    temporal_state: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Stabilize matched region bbox across time and noisy detector outliers.

    Uses:
    - overlay bbox as corrective hint when region bbox is implausibly large;
    - temporal smoothing per region/zone to suppress one-frame jumps;
    - zone caps to avoid giant full-screen slabs.
    """
    if not region or not isinstance(region, dict):
        return region
    if not isinstance(region.get("bbox_norm"), (list, tuple)) or len(region.get("bbox_norm", [])) < 4:
        return region

    zone = _position_to_zone(region.get("zone") or overlay_zone or overlay.get("position", "middle"))
    original_bbox = _normalize_bbox(region.get("bbox_norm", [0, 0.85, 1, 1]))
    candidate_bbox = list(original_bbox)
    reasons: List[str] = []

    overlay_bbox = None
    if isinstance(overlay.get("bbox_norm"), (list, tuple)) and len(overlay.get("bbox_norm", [])) >= 4:
        overlay_bbox = _normalize_bbox(overlay.get("bbox_norm", [0, 0.85, 1, 1]))

    # Correct giant region boxes with overlay-local hint.
    if overlay_bbox:
        region_area = _bbox_area_norm(original_bbox)
        overlay_area = _bbox_area_norm(overlay_bbox)
        iou = _bbox_iou_norm(original_bbox, overlay_bbox)
        if overlay_area > 0:
            if region_area > (overlay_area * 2.30) and iou < 0.25:
                candidate_bbox = _blend_bbox_norm(original_bbox, overlay_bbox, 0.48)
                reasons.append("overlay_blend_large_region")
            elif region_area > 0.15 and iou < 0.08:
                candidate_bbox = _blend_bbox_norm(original_bbox, overlay_bbox, 0.60)
                reasons.append("overlay_blend_low_iou")

    candidate_bbox, zone_clamped = _clamp_bbox_by_zone(candidate_bbox, zone)
    if zone_clamped:
        reasons.append("zone_cap")

    region_id = str(region.get("id", "") or "")
    temporal_key = f"region:{region_id}" if region_id else f"zone:{zone}"
    prev_entry = temporal_state.get(temporal_key) or temporal_state.get(f"zone:{zone}")
    if prev_entry and isinstance(prev_entry.get("bbox"), list):
        prev_bbox = _normalize_bbox(prev_entry["bbox"])
        prev_time = _safe_float(prev_entry.get("time"), appears_at)
        delta_t = appears_at - prev_time

        if 0.0 <= delta_t <= 8.0:
            iou_prev = _bbox_iou_norm(prev_bbox, candidate_bbox)
            prev_area = max(1e-9, _bbox_area_norm(prev_bbox))
            curr_area = max(1e-9, _bbox_area_norm(candidate_bbox))
            area_scale = curr_area / prev_area
            px, py = _bbox_center_norm(prev_bbox)
            cx, cy = _bbox_center_norm(candidate_bbox)
            center_drift = max(abs(cx - px), abs(cy - py))

            if (iou_prev < 0.10 and area_scale > 1.55) or center_drift > 0.22:
                candidate_bbox = _blend_bbox_norm(prev_bbox, candidate_bbox, 0.25)
                reasons.append("temporal_jump_guard")
            elif iou_prev < 0.28 or area_scale > 1.30 or area_scale < 0.72:
                candidate_bbox = _blend_bbox_norm(prev_bbox, candidate_bbox, 0.45)
                reasons.append("temporal_smooth")

            candidate_bbox, zone_clamped_2 = _clamp_bbox_by_zone(candidate_bbox, zone)
            if zone_clamped_2 and "zone_cap" not in reasons:
                reasons.append("zone_cap")

    state_value = {"bbox": candidate_bbox, "time": appears_at}
    temporal_state[temporal_key] = state_value
    temporal_state[f"zone:{zone}"] = state_value

    delta = max(abs(candidate_bbox[i] - original_bbox[i]) for i in range(4))
    if delta < 1e-4:
        return region

    stabilized = dict(region)
    stabilized["bbox_norm"] = candidate_bbox
    logger.info(
        "BBOX_STABILIZE: region=%s zone=%s reasons=%s "
        "bbox=[%.3f,%.3f,%.3f,%.3f] -> [%.3f,%.3f,%.3f,%.3f]",
        region.get("id", "?"),
        zone,
        ",".join(reasons) if reasons else "none",
        original_bbox[0], original_bbox[1], original_bbox[2], original_bbox[3],
        candidate_bbox[0], candidate_bbox[1], candidate_bbox[2], candidate_bbox[3],
    )
    return stabilized


def _next_future_start(starts: list, current_t: float, min_gap: float = 0.35) -> Optional[float]:
    """Get first future start timestamp with minimum gap from current time."""
    if not starts:
        return None
    threshold = current_t + min_gap
    for t in starts:
        tv = _safe_float(t, -1.0)
        if tv >= threshold:
            return tv
    return None


def _apply_style_profile(region: Optional[Dict], default_bg_hex: str = "#1A1A1A",
                         default_text_hex: str = "#FFFFFF", default_alpha: float = 0.08
                         ) -> Dict[str, Any]:
    """
    Extract ASS-compatible style parameters from a text_region's style_profile.

    Returns a dict with:
        bg_hex, text_hex, bg_alpha, font_size_rel_factor, bold, outline, shadow, bg_color_ass, text_color_ass
    If no style_profile exists, returns defaults (dark box with white text).
    """
    sp = (region or {}).get("style_profile", {}) if region else {}
    if not sp:
        return {
            "bg_hex": default_bg_hex,
            "text_hex": default_text_hex,
            "bg_alpha": default_alpha,
            "font_size_rel_factor": 1.0,
            "bold": True,
            "outline_width": 2,
            "shadow_depth": 2,
            "bg_color_ass": _hex_to_ass_color(default_bg_hex),
            "text_color_ass": _hex_to_ass_color(default_text_hex),
            "has_style_profile": False,
        }

    bg_hex = sp.get("bg_color", default_bg_hex)
    text_hex = sp.get("text_color", default_text_hex)
    opacity = _safe_float(sp.get("opacity"), 1.0)
    # Convert opacity (1.0=opaque) to ASS alpha (0x00=opaque, 0xFF=transparent)
    bg_alpha = max(0.0, 1.0 - opacity)

    # Font size relative factor
    size_map = {"small": 0.75, "medium": 1.0, "large": 1.25}
    font_size_rel = size_map.get(sp.get("font_size_rel", "medium"), 1.0)

    bold = sp.get("font_weight", "normal") == "bold"
    # V5: Thinner outlines for modern TikTok look (min 1, not 0)
    outline_width = 2 if sp.get("has_outline", False) else 1
    shadow_depth = 1 if sp.get("has_shadow", False) else 1

    return {
        "bg_hex": bg_hex,
        "text_hex": text_hex,
        "bg_alpha": bg_alpha,
        "font_size_rel_factor": font_size_rel,
        "bold": bold,
        "outline_width": outline_width,
        "shadow_depth": shadow_depth,
        "bg_color_ass": _hex_to_ass_color(bg_hex),
        "text_color_ass": _hex_to_ass_color(text_hex),
        "has_style_profile": True,
    }


def _estimate_overlay_max_hold_seconds(
    translated_text: str,
    region: Optional[Dict[str, Any]],
    strategy: str,
) -> float:
    """
    Estimate max safe on-screen duration for one overlay to prevent slab buildup.
    """
    text = (translated_text or "").strip()
    words = [w for w in text.split() if w.strip()]
    word_count = len(words)
    char_count = len(text)

    if word_count <= 2 and char_count <= 14:
        hold = 3.2
    elif word_count <= 6:
        hold = 4.8
    elif word_count <= 12:
        hold = 6.6
    else:
        hold = 8.4

    # Overlay pipelines should not linger too long by default.
    if strategy in ("subtitle_snap", "blur_plate_overlay"):
        hold = min(hold, 8.5)

    if region and isinstance(region, dict):
        temporal = str(region.get("temporal", "")).lower()
        rtype = str(region.get("type", "")).lower()

        if temporal == "static":
            # Static text is visible throughout the video — allow full duration.
            # Return a very large number; the caller will clamp to video_duration.
            return 999999.0
        if rtype in ("watermark", "logo", "username"):
            hold = max(hold, 12.0)
        if rtype in ("caption", "subtitle"):
            hold = min(hold, 8.0)
        if rtype in ("title", "headline"):
            hold = max(hold, 10.0)

    return max(2.0, min(14.0, hold))


def _match_overlay_to_region(overlay: Dict, text_regions: list) -> Optional[Dict]:
    """
    Match a translated overlay to its corresponding VideoProfile text_region.

    Matches by:
    1. Position zone similarity (top/middle/bottom)
    2. Text content overlap (Levenshtein ratio)
    3. BBox overlap (IoU) when both sides provide bbox_norm
    """
    if not text_regions:
        return None

    # Deterministic direct match if server provided stable region_id link.
    overlay_region_id = str(
        overlay.get("region_id")
        or overlay.get("text_region_id")
        or ""
    ).strip()
    if overlay_region_id:
        for region in text_regions:
            if str(region.get("id", "")).strip() == overlay_region_id:
                return region

    overlay_pos = overlay.get("position", "top").lower()
    overlay_zone = _position_to_zone(overlay_pos)
    overlay_text = overlay.get("text", "")
    overlay_bbox = None
    if isinstance(overlay.get("bbox_norm"), (list, tuple)) and len(overlay.get("bbox_norm", [])) >= 4:
        overlay_bbox = _normalize_bbox(overlay.get("bbox_norm", [0, 0, 1, 1]))

    best_match = None
    best_score = 0.0

    for region in text_regions:
        score = 0.0

        # Zone match bonus
        region_zone = region.get("zone", "top")
        if overlay_zone == region_zone:
            score += 0.35

        # Text similarity (source text)
        region_text = region.get("content", "")
        if overlay_text and region_text:
            text_sim = _levenshtein_ratio(overlay_text, region_text)
            score += text_sim * 0.55

        # BBox overlap (helps avoid matching top overlay to bottom region and vice versa)
        if overlay_bbox and isinstance(region.get("bbox_norm"), (list, tuple)) and len(region.get("bbox_norm", [])) >= 4:
            region_bbox = _normalize_bbox(region.get("bbox_norm", [0, 0, 1, 1]))
            iou = _bbox_iou_norm(overlay_bbox, region_bbox)
            if iou > 0:
                score += min(0.8, iou * 1.3)
            elif overlay_zone == region_zone:
                # Same-zone but zero overlap is a strong anti-signal (often wrong region).
                score -= 0.15

        if score > best_score:
            best_score = score
            best_match = region

    if best_score >= 0.45:
        return best_match
    return None


def _render_backplate_overlay(ass_lines: list, overlay: Dict, region: Dict, video_width: int, video_height: int, font_size: int, is_rtl: bool = False, target_language: str = "", caption_style: dict = None):
    """
    Render translated text with TikTok/IG-native dark backplate covering original text.

    Uses two layers:
    Layer 0: Dark opaque rectangle covering original text bbox (hides source text)
    Layer 1: White text with auto-sized dark box (BorderStyle=3) for native look
    """
    bbox = _normalize_bbox(region.get("bbox_norm", [0, 0, 1, 0.15]), video_width, video_height)

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

    # Apply style_profile from Gemini analysis (if available)
    sp = _apply_style_profile(region, default_bg_hex="#0A0A0A", default_text_hex="#FFFFFF", default_alpha=0.0)

    # Layer 0: Rectangle covering original text area using style-matched colors
    bg_ass = sp["bg_color_ass"]
    bg_alpha_hex = f"{int(sp['bg_alpha'] * 255):02X}"
    # Use 95% opacity minimum to ensure source text is hidden
    if int(bg_alpha_hex, 16) > 0x0D:
        bg_alpha_hex = "0D"  # Cap at 95% opaque for backplates
    draw_cmd = f"m {x1} {y1} l {x2} {y1} l {x2} {y2} l {x1} {y2}"
    ass_lines.append(
        f"Dialogue: 0,{start_time},{end_time},BackPlate,,0,0,0,,"
        f"{{\\an7\\pos(0,0)\\1c{bg_ass}\\1a&H{bg_alpha_hex}\\bord0\\shad0\\blur1\\p1}}{draw_cmd}"
    )

    # Layer 1: Translated text with style-matched colors
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

    # Build text overrides from style_profile
    text_ass = sp["text_color_ass"]
    bold_tag = "\\b1" if sp["bold"] else ""
    outline_tag = f"\\bord{sp['outline_width']}"
    shadow_tag = f"\\shad{sp['shadow_depth']}"

    ass_lines.append(
        f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
        f"{{\\an5\\pos({cx},{cy})\\fad(200,200)\\1c{text_ass}{bold_tag}{outline_tag}{shadow_tag}}}{safe_text}"
    )

    logger.info(
        f"BACKPLATE: region={region.get('id', '?')} "
        f"bbox=({x1},{y1},{x2},{y2}) style_profile={sp['has_style_profile']} "
        f"text='{translated_text[:30]}'"
    )


def _render_subtitle_snap(ass_lines: list, overlay: Dict, region: Dict, video_width: int, video_height: int, font_size: int, is_rtl: bool = False, target_language: str = "", caption_style: dict = None):
    """
    Pipeline A: Render translated subtitle with style-cloned container.

    Instead of full-width opaque strips, renders a fitted rounded-rect
    background sized to the text content. Supports per-overlay font_style:
    - ios_default_bg: dark semi-transparent rounded rect, white text
    - impact_shadow: no background, white text with black shadow
    - minimal_outline: no background, white text with thin outline
    - sticker: colored background, contrasting text
    Falls back to caption_style preset for unrecognized styles.
    """
    cs = dict(caption_style or CAPTION_STYLE_PRESETS["tiktok_pill"])
    bbox = _normalize_bbox(region.get("bbox_norm", [0, 0.85, 1, 1]), video_width, video_height)

    # Resolve per-overlay font_style (from Gemini manifest)
    font_style = overlay.get("font_style", "ios_default_bg")

    # Prepare translated text first (needed for fitted container sizing)
    translated_text = _strip_emoji(overlay.get("translated_text", ""))
    max_chars = max(12, min(_safe_int(cs.get("max_chars_line"), 28), 22))
    wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=max_chars)
    safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

    style_name = "TranslatedTextNoBox"
    if is_rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(safe_text)
            safe_text = get_display(reshaped)
        except ImportError:
            pass
        style_name = "TranslatedTextNoBoxRTL"

    # Estimate text dimensions for fitted container
    # Count lines and max line width in characters
    text_lines = wrapped_text.split("\\N")
    num_lines = len(text_lines)
    max_line_chars = max(len(line) for line in text_lines) if text_lines else 1

    # Approximate pixel width: ~0.55 * font_size per char (bold sans-serif average)
    char_width = font_size * 0.55
    text_pixel_w = int(max_line_chars * char_width)
    text_pixel_h = int(num_lines * font_size * 1.3)  # 1.3x line height

    # Padding around text inside the container.
    pad_x = int(font_size * 0.42)
    pad_y = int(font_size * 0.28)

    # Container dimensions — sized to fit translated text only
    # V5: Inpaint removes source text perfectly, so container is decorative only
    text_container_w = text_pixel_w + pad_x * 2
    text_container_h = text_pixel_h + pad_y * 2

    # Original text bbox in pixels (used for positioning center, not sizing)
    orig_w = int((bbox[2] - bbox[0]) * video_width)
    orig_h = int((bbox[3] - bbox[1]) * video_height)

    # Deterministic layout solver (policy layer).
    zone = (region.get("zone") or _position_to_zone(overlay.get("position", "middle")) or "middle").lower()
    style_intent = cs.get("_style_intent", {}) if isinstance(cs, dict) else {}
    layout_policy = _build_layout_policy(
        "subtitle_snap",
        density=style_intent.get("density", "balanced"),
        container_preference=style_intent.get("container_preference", "pill"),
    )
    # V5: Text-only sizing — no need to cover original bbox (inpaint handles removal)
    container_w = text_container_w
    container_h = text_container_h
    container_w, container_h, guard_metrics = _enforce_container_guardrails(
        container_w=container_w,
        container_h=container_h,
        orig_w=orig_w,
        orig_h=orig_h,
        text_container_w=text_container_w,
        text_container_h=text_container_h,
        font_size=font_size,
        zone=zone,
        video_width=video_width,
        video_height=video_height,
        layout_policy=layout_policy,
    )
    if guard_metrics.get("fallback_applied"):
        logger.info(
            "STYLE_GUARD: auto-compact applied region=%s zone=%s reasons=%s "
            "raw_coverage=(%.2f,%.2f) eff_coverage=(%.2f,%.2f) area=%.3f",
            region.get("id", "?"),
            zone,
            ",".join(guard_metrics.get("fallback_reasons", [])) or "none",
            guard_metrics.get("coverage_x_raw", 0.0),
            guard_metrics.get("coverage_y_raw", 0.0),
            guard_metrics.get("coverage_x_eff", 0.0),
            guard_metrics.get("coverage_y_eff", 0.0),
            guard_metrics.get("area_ratio", 0.0),
        )

    # Position: center container over the ORIGINAL text bbox (not video center)
    # This ensures the opaque container fully covers the source text
    bbox_cx = int(((bbox[0] + bbox[2]) / 2) * video_width)
    bbox_cy = int(((bbox[1] + bbox[3]) / 2) * video_height)
    cx = bbox_cx
    cy = bbox_cy

    # Container top-left for ASS drawing
    rect_x1 = cx - container_w // 2
    rect_y1 = cy - container_h // 2
    rect_x2 = cx + container_w // 2
    rect_y2 = cy + container_h // 2

    # Clamp to video bounds (with padding)
    margin = 4
    if rect_x1 < margin:
        shift = margin - rect_x1
        rect_x1 += shift
        rect_x2 += shift
        cx += shift
    if rect_x2 > video_width - margin:
        shift = rect_x2 - (video_width - margin)
        rect_x1 -= shift
        rect_x2 -= shift
        cx -= shift
    if rect_y1 < margin:
        shift = margin - rect_y1
        rect_y1 += shift
        rect_y2 += shift
        cy += shift
    if rect_y2 > video_height - margin:
        shift = rect_y2 - (video_height - margin)
        rect_y1 -= shift
        rect_y2 -= shift
        cy -= shift

    start_time = _format_ass_time(overlay.get("appears_at", 0))
    end_time = _format_ass_time(overlay.get("disappears_at", 0))

    # Corner radius for rounded rect (ASS bezier curves)
    # V5: Larger radius for modern pill shape
    R = max(10, min(int(font_size * 0.6), 18, container_w // 3, container_h // 3))

    # Shorthand positions for the rounded rect
    rx1, ry1, rx2, ry2 = rect_x1, rect_y1, rect_x2, rect_y2

    # --- V5: Unified modern frosted pill for ALL subtitle_snap overlays ---
    # Gemini font_style values don't match handler vocabulary (dead code eliminated).
    # Inpaint/blur removes source text; overlay is purely decorative.
    # Force ios_default_bg frosted pill for consistent modern TikTok look.

    # Style: modern semi-transparent frosted pill
    bg_hex = "#000000"
    bg_alpha = 0.50  # 50% opacity — semi-transparent, modern look
    text_color_ass = _hex_to_ass_color("#FFFFFF")
    bold_tag = "\\b1"  # Bold for readability at small sizes
    outline_tag = "\\bord1"  # Thin outline — modern TikTok style
    shadow_tag = "\\shad1"
    plate_blur = "\\blur6"  # Strong blur for frosted glass effect
    plate_shad = ""  # No shadow on frosted glass plate

    bg_color_ass = _hex_to_ass_color(bg_hex)
    bg_alpha_hex = f"{int(bg_alpha * 255):02X}"

    # Rounded-rect pill shape with smooth bezier quarter-circle corners
    C = int(R * 0.55)
    draw_cmd = (
        f"m {rx1+R} {ry1} "
        f"l {rx2-R} {ry1} "
        f"b {rx2-R+C} {ry1} {rx2} {ry1+R-C} {rx2} {ry1+R} "
        f"l {rx2} {ry2-R} "
        f"b {rx2} {ry2-R+C} {rx2-R+C} {ry2} {rx2-R} {ry2} "
        f"l {rx1+R} {ry2} "
        f"b {rx1+R-C} {ry2} {rx1} {ry2-R+C} {rx1} {ry2-R} "
        f"l {rx1} {ry1+R} "
        f"b {rx1} {ry1+R-C} {rx1+R-C} {ry1} {rx1+R} {ry1}"
    )

    # BackPlate: frosted pill with fade-out to prevent ghost rectangle
    ass_lines.append(
        f"Dialogue: 0,{start_time},{end_time},BackPlate,,0,0,0,,"
        f"{{\\an7\\pos(0,0)\\fad(0,200)\\1c{bg_color_ass}\\1a&H{bg_alpha_hex}\\bord0{plate_shad}{plate_blur}\\p1}}{draw_cmd}"
    )

    # Text: white bold with thin outline, centered inside the pill
    text_overrides = f"\\an5\\pos({cx},{cy})\\fad(150,200)\\1c{text_color_ass}{bold_tag}{outline_tag}\\3c&H000000{shadow_tag}\\4c&H70000000"
    ass_lines.append(
        f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
        f"{{{text_overrides}}}{safe_text}"
    )

    # Hard QA metrics (deterministic checks before model-based QA).
    coverage_x_raw = guard_metrics.get("coverage_x_raw", container_w / max(1, orig_w))
    coverage_y_raw = guard_metrics.get("coverage_y_raw", container_h / max(1, orig_h))
    coverage_x_eff = guard_metrics.get("coverage_x_eff", coverage_x_raw)
    coverage_y_eff = guard_metrics.get("coverage_y_eff", coverage_y_raw)
    area_ratio = guard_metrics.get("area_ratio", (container_w * container_h) / max(1, video_width * video_height))
    if guard_metrics.get("coverage_low", False):
        logger.warning(
            f"STYLE_GUARD: low source coverage region={region.get('id', '?')} "
            f"zone={zone} raw=({coverage_x_raw:.2f},{coverage_y_raw:.2f}) "
            f"eff=({coverage_x_eff:.2f},{coverage_y_eff:.2f})"
        )
    if guard_metrics.get("oversized", False):
        logger.warning(
            f"STYLE_GUARD: oversized container region={region.get('id', '?')} "
            f"zone={zone} area_ratio={area_ratio:.3f} "
            f"max={guard_metrics.get('max_area_ratio', 0.16):.3f}"
        )

    logger.info(
        f"SUBTITLE_SNAP: region={region.get('id', '?')} "
        f"container=({rect_x1},{rect_y1},{rect_x2},{rect_y2}) R={R} "
        f"zone={zone} coverage_raw=({coverage_x_raw:.2f},{coverage_y_raw:.2f}) "
        f"coverage_eff=({coverage_x_eff:.2f},{coverage_y_eff:.2f}) "
        f"font_style='{font_style}' render_style='ios_default_bg_v5' "
        f"bg_alpha={bg_alpha} plate_blur='{plate_blur}' "
        f"time={start_time}-{end_time} "
        f"text='{translated_text[:40]}'"
    )


def _render_blur_plate_overlay(ass_lines: list, overlay: Dict, region: Dict, video_width: int, video_height: int, font_size: int, is_rtl: bool = False, target_language: str = "", caption_style: dict = None):
    """
    Pipeline B: Render translated text on already-blurred region.

    The blur_plate stage has already applied gaussian blur + darken to the region.
    Uses a fitted rounded-rect backup container (not full-width) that covers
    the original text area. Text is rendered with market-native styling on top.
    """
    cs = dict(caption_style or CAPTION_STYLE_PRESETS["tiktok_pill"])
    bbox = _normalize_bbox(region.get("bbox_norm", [0, 0.85, 1, 1]), video_width, video_height)

    # Convert to pixels — use the actual region bbox, not full width
    x1 = max(0, int(bbox[0] * video_width))
    y1 = max(0, int(bbox[1] * video_height))
    x2 = min(video_width, int(bbox[2] * video_width))
    y2 = min(video_height, int(bbox[3] * video_height))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    start_time = _format_ass_time(overlay.get("appears_at", 0))
    end_time = _format_ass_time(overlay.get("disappears_at", 0))

    translated_text = _strip_emoji(overlay.get("translated_text", ""))
    max_chars = max(14, min(_safe_int(cs.get("max_chars_line"), 28), 28))
    wrapped_text = _wrap_text_for_ass(translated_text, max_chars_per_line=max_chars)
    safe_text = wrapped_text.replace("{", "\\{").replace("}", "\\}")

    style_name = "TranslatedTextNoBox"
    if is_rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(safe_text)
            safe_text = get_display(reshaped)
        except ImportError:
            pass
        style_name = "TranslatedTextNoBoxRTL"

    # Estimate text dimensions for fitted container
    text_lines = wrapped_text.split("\\N")
    num_lines = len(text_lines)
    max_line_chars = max(len(line) for line in text_lines) if text_lines else 1
    char_width = font_size * 0.55
    text_pixel_w = int(max_line_chars * char_width)
    text_pixel_h = int(num_lines * font_size * 1.3)

    # Container must cover original text area OR fit translated text.
    pad_x = int(font_size * 0.55)
    pad_y = int(font_size * 0.35)
    text_container_w = text_pixel_w + pad_x * 2
    text_container_h = text_pixel_h + pad_y * 2
    region_w = x2 - x1
    region_h = y2 - y1

    zone = (region.get("zone") or _position_to_zone(overlay.get("position", "middle")) or "middle").lower()
    style_intent = cs.get("_style_intent", {}) if isinstance(cs, dict) else {}
    layout_policy = _build_layout_policy(
        "blur_plate",
        density=style_intent.get("density", "balanced"),
        container_preference=style_intent.get("container_preference", "pill"),
    )
    container_w, container_h = _solve_container_dimensions(
        orig_w=region_w,
        orig_h=region_h,
        text_container_w=text_container_w,
        text_container_h=text_container_h,
        font_size=font_size,
        zone=zone,
        video_width=video_width,
        video_height=video_height,
        layout_policy=layout_policy,
    )
    container_w, container_h, guard_metrics = _enforce_container_guardrails(
        container_w=container_w,
        container_h=container_h,
        orig_w=region_w,
        orig_h=region_h,
        text_container_w=text_container_w,
        text_container_h=text_container_h,
        font_size=font_size,
        zone=zone,
        video_width=video_width,
        video_height=video_height,
        layout_policy=layout_policy,
    )
    if guard_metrics.get("fallback_applied"):
        logger.info(
            "BLUR_STYLE_GUARD: auto-compact region=%s zone=%s reasons=%s "
            "raw_coverage=(%.2f,%.2f) eff_coverage=(%.2f,%.2f) area=%.3f",
            region.get("id", "?"),
            zone,
            ",".join(guard_metrics.get("fallback_reasons", [])) or "none",
            guard_metrics.get("coverage_x_raw", 0.0),
            guard_metrics.get("coverage_y_raw", 0.0),
            guard_metrics.get("coverage_x_eff", 0.0),
            guard_metrics.get("coverage_y_eff", 0.0),
            guard_metrics.get("area_ratio", 0.0),
        )

    # Fitted container position (centered on original region center)
    rx1 = cx - container_w // 2
    ry1 = cy - container_h // 2
    rx2 = cx + container_w // 2
    ry2 = cy + container_h // 2

    # Clamp to video bounds
    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(video_width, rx2)
    ry2 = min(video_height, ry2)

    # Corner radius for rounded rect
    R = min(int(font_size * 0.4), 12, container_w // 4, container_h // 4)

    # Apply style_profile from Gemini analysis (if available)
    sp = _apply_style_profile(region, default_bg_hex="#0A0A0A", default_text_hex="#FFFFFF", default_alpha=0.1)

    # Layer 0: Fitted rounded-rect backup background with style-matched color
    bg_color_ass = sp["bg_color_ass"]
    bg_alpha_hex = f"{int(sp['bg_alpha'] * 255):02X}"
    # Blur plate already obscures text; cap transparency at 90% opaque
    if int(bg_alpha_hex, 16) > 0x1A:
        bg_alpha_hex = "1A"
    draw_cmd = (
        f"m {rx1+R} {ry1} "
        f"l {rx2-R} {ry1} "
        f"b {rx2} {ry1} {rx2} {ry1+R} {rx2} {ry1+R} "
        f"l {rx2} {ry2-R} "
        f"b {rx2} {ry2} {rx2-R} {ry2} {rx2-R} {ry2} "
        f"l {rx1+R} {ry2} "
        f"b {rx1} {ry2} {rx1} {ry2-R} {rx1} {ry2-R} "
        f"l {rx1} {ry1+R} "
        f"b {rx1} {ry1} {rx1+R} {ry1} {rx1+R} {ry1}"
    )
    ass_lines.append(
        f"Dialogue: 0,{start_time},{end_time},BackPlate,,0,0,0,,"
        f"{{\\an7\\pos(0,0)\\1c{bg_color_ass}\\1a&H{bg_alpha_hex}\\bord0\\shad0\\p1}}{draw_cmd}"
    )

    # Layer 1: Text on blurred background
    # CRITICAL: Blur plate creates a DARK background. Style profile text colors are designed
    # for the ORIGINAL background, not the blur. Dark text on dark blur = invisible.
    # Always use white text with strong outline for readability on blur.
    text_ass = _hex_to_ass_color("#FFFFFF")
    bold_tag = "\\b1"
    outline_tag = "\\bord3"
    shadow_tag = "\\shad2"
    shadow_color = _hex_to_ass_color("#000000")
    ass_lines.append(
        f"Dialogue: 1,{start_time},{end_time},{style_name},,0,0,0,,"
        f"{{\\an5\\pos({cx},{cy})\\fad(200,200)\\1c{text_ass}{bold_tag}{outline_tag}{shadow_tag}\\3c{shadow_color}}}{safe_text}"
    )

    coverage_x_raw = guard_metrics.get("coverage_x_raw", container_w / max(1, region_w))
    coverage_y_raw = guard_metrics.get("coverage_y_raw", container_h / max(1, region_h))
    coverage_x_eff = guard_metrics.get("coverage_x_eff", coverage_x_raw)
    coverage_y_eff = guard_metrics.get("coverage_y_eff", coverage_y_raw)
    area_ratio = guard_metrics.get("area_ratio", (container_w * container_h) / max(1, video_width * video_height))
    if guard_metrics.get("coverage_low", False):
        logger.warning(
            f"BLUR_STYLE_GUARD: low source coverage region={region.get('id', '?')} "
            f"zone={zone} raw=({coverage_x_raw:.2f},{coverage_y_raw:.2f}) "
            f"eff=({coverage_x_eff:.2f},{coverage_y_eff:.2f})"
        )
    if guard_metrics.get("oversized", False):
        logger.warning(
            f"BLUR_STYLE_GUARD: oversized container region={region.get('id', '?')} "
            f"zone={zone} area_ratio={area_ratio:.3f} "
            f"max={guard_metrics.get('max_area_ratio', 0.16):.3f}"
        )
    logger.info(
        f"BLUR_PLATE_OVERLAY: region={region.get('id', '?')} "
        f"container=({rx1},{ry1},{rx2},{ry2}) R={R} "
        f"zone={zone} coverage_raw=({coverage_x_raw:.2f},{coverage_y_raw:.2f}) "
        f"coverage_eff=({coverage_x_eff:.2f},{coverage_y_eff:.2f}) "
        f"time={start_time}-{end_time} "
        f"text='{translated_text[:40]}'"
    )


def _refine_regions_from_ocr(text_regions: list, text_detections: list) -> list:
    """Refine Gemini text_region bboxes using precise OCR detections.

    For each Gemini region, find overlapping OCR detections and expand
    the region bbox to cover ALL matched detections. This ensures
    subtitle_snap containers fully cover original text.
    """
    if not text_detections or not text_regions:
        return text_regions

    for region in text_regions:
        bbox = region.get("bbox_norm", [0, 0.85, 1, 1])
        region_zone = region.get("zone", "middle")
        region_w = max(1e-4, bbox[2] - bbox[0])
        region_h = max(1e-4, bbox[3] - bbox[1])

        # Expand search area moderately; aggressive expansion causes giant merged slabs.
        margin_x = max(0.02, min(0.08, region_w * 0.35))
        margin_y = max(0.015, min(0.07, region_h * 0.45))
        search_x1 = max(0, bbox[0] - margin_x)
        search_y1 = max(0, bbox[1] - margin_y)
        search_x2 = min(1, bbox[2] + margin_x)
        search_y2 = min(1, bbox[3] + margin_y)
        region_cx = (bbox[0] + bbox[2]) / 2
        region_cy = (bbox[1] + bbox[3]) / 2

        matched_dets = []
        for det in text_detections:
            det_bbox = det.get("bbox_norm", [0, 0, 0, 0])
            det_x1, det_y1, det_x2, det_y2 = det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3]

            # Overlap-based matching: calculate bbox overlap between OCR det and expanded region
            overlap_x = max(0.0, min(det_x2, search_x2) - max(det_x1, search_x1))
            overlap_y = max(0.0, min(det_y2, search_y2) - max(det_y1, search_y1))
            overlap_area = overlap_x * overlap_y
            det_area = max(1e-6, (det_x2 - det_x1) * (det_y2 - det_y1))
            overlap_ratio = overlap_area / det_area

            if overlap_ratio < 0.05:
                continue  # Skip only if truly no overlap with expanded search area

            # Zone penalty: penalize score instead of hard reject
            zone_penalty = 0.0
            if region_zone == "bottom" and det_y1 < 0.35:
                zone_penalty = 0.3
            elif region_zone == "top" and det_y2 > 0.65:
                zone_penalty = 0.3
            elif region_zone == "middle" and not (det_y1 < 0.75 and det_y2 > 0.25):
                zone_penalty = 0.2

            # Center-distance check (soft)
            det_cx = (det_x1 + det_x2) / 2
            det_cy = (det_y1 + det_y2) / 2
            max_dx = region_w * 0.75 + 0.06
            max_dy = region_h * 0.90 + 0.08
            close_enough = abs(det_cx - region_cx) <= max_dx and abs(det_cy - region_cy) <= max_dy

            # Combined score: overlap strength minus zone penalty
            match_score = overlap_ratio - zone_penalty

            if close_enough and match_score > 0.0:
                matched_dets.append(det_bbox)

        if matched_dets:
            # OCR union
            ocr_x1 = min(b[0] for b in matched_dets)
            ocr_y1 = min(b[1] for b in matched_dets)
            ocr_x2 = max(b[2] for b in matched_dets)
            ocr_y2 = max(b[3] for b in matched_dets)

            # Compare Gemini bbox area vs OCR area.
            # If Gemini bbox is heavily over-wide, trust OCR more to avoid giant subtitle blocks.
            orig_area = max(1e-9, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            ocr_area = max(1e-9, (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1))
            if ocr_area > (orig_area * 1.6):
                logger.info(
                    f"OCR_REFINE: region={region.get('id', '?')} skipped (ocr area too large: "
                    f"orig={orig_area:.4f}, ocr={ocr_area:.4f})"
                )
                continue
            use_ocr_only = orig_area > (ocr_area * 1.8)

            if use_ocr_only:
                margin_x = 0.01
                margin_y = 0.015
                union_x1 = max(0.0, ocr_x1 - margin_x)
                union_y1 = max(0.0, ocr_y1 - margin_y)
                union_x2 = min(1.0, ocr_x2 + margin_x)
                union_y2 = min(1.0, ocr_y2 + margin_y)
            else:
                # Standard union: keep Gemini guidance + OCR precision
                all_bboxes = [bbox] + matched_dets
                union_x1 = min(b[0] for b in all_bboxes)
                union_y1 = min(b[1] for b in all_bboxes)
                union_x2 = max(b[2] for b in all_bboxes)
                union_y2 = max(b[3] for b in all_bboxes)

            # Clamp refinement growth to reduce accidental over-expansion.
            max_expand_x = max(0.05, region_w * 0.35)
            max_expand_y = max(0.05, region_h * 0.45)
            union_x1 = max(0.0, max(union_x1, bbox[0] - max_expand_x))
            union_y1 = max(0.0, max(union_y1, bbox[1] - max_expand_y))
            union_x2 = min(1.0, min(union_x2, bbox[2] + max_expand_x))
            union_y2 = min(1.0, min(union_y2, bbox[3] + max_expand_y))

            old_bbox = region["bbox_norm"]
            region["bbox_norm"] = [union_x1, union_y1, union_x2, union_y2]
            logger.info(
                f"OCR_REFINE: region={region.get('id', '?')} matched {len(matched_dets)} OCR dets, "
                f"mode={'ocr_only' if use_ocr_only else 'union'} "
                f"bbox [{old_bbox[0]:.3f},{old_bbox[1]:.3f},{old_bbox[2]:.3f},{old_bbox[3]:.3f}] → "
                f"[{union_x1:.3f},{union_y1:.3f},{union_x2:.3f},{union_y2:.3f}]"
            )

    return text_regions


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

    # Determine dominant pipeline type for style resolution (deterministic).
    # Do not trust pipelines_used order from upstream set/list conversions.
    strategy_to_pipeline = {
        "subtitle_snap": "subtitle_snap",
        "blur_plate_overlay": "blur_plate",
        "backplate_overlay": "inpaint_backplate",
        "replace_inplace": "inpaint_backplate",
    }
    strategy_counts: Dict[str, int] = {}
    for _rid, _strategy in (render_strategies or {}).items():
        strategy_counts[_strategy] = strategy_counts.get(_strategy, 0) + 1

    dominant_pipeline = "inpaint_backplate"
    if strategy_counts:
        priority = {"subtitle_snap": 4, "blur_plate_overlay": 3, "backplate_overlay": 2, "replace_inplace": 1}
        dominant_strategy = sorted(
            strategy_counts.keys(),
            key=lambda s: (strategy_counts.get(s, 0), priority.get(s, 0)),
            reverse=True,
        )[0]
        dominant_pipeline = strategy_to_pipeline.get(dominant_strategy, "inpaint_backplate")
    else:
        pipelines_used = vp_config.get("pipelines_used", [])
        if pipelines_used:
            pref = {"subtitle_snap": 4, "blur_plate": 3, "inpaint_backplate": 2}
            dominant_pipeline = sorted(
                pipelines_used,
                key=lambda p: pref.get(p, 0),
                reverse=True,
            )[0]

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

    # Pre-compute future overlay starts by position and by canonical zone.
    # Zone index avoids overlap when Gemini alternates labels like top/top_left.
    _next_start_by_pos = {}
    _next_start_by_zone = {"top": [], "middle": [], "bottom": []}
    for i, ov in enumerate(translated_overlays):
        pos = ov.get("position", "top")
        t = ov.get("appears_at", 0.0)
        tv = _safe_float(t, 0.0)
        _next_start_by_pos.setdefault(pos, []).append(tv)
        _next_start_by_zone.setdefault(_position_to_zone(pos), []).append(tv)
    for pos in _next_start_by_pos:
        _next_start_by_pos[pos] = sorted(_next_start_by_pos[pos])
    for zone in _next_start_by_zone:
        _next_start_by_zone[zone] = sorted(_next_start_by_zone[zone])

    # Temporal state for bbox stabilization (per region + per zone fallback).
    _bbox_temporal_state: Dict[str, Dict[str, Any]] = {}
    # Duplicate suppression per zone to avoid stacked identical overlays.
    _zone_dedupe_state: Dict[str, Dict[str, Any]] = {}

    for i, overlay in enumerate(translated_overlays):
        translated_text = _strip_emoji(overlay.get("translated_text", ""))
        if not translated_text:
            continue

        # Garbage text validation: reject high-entropy alphanumeric noise
        # e.g. "5ef4re3ft2r.. 1" — corrupted OCR output that should not be rendered
        import re as _re_validate
        alpha_count = sum(1 for c in translated_text if c.isalpha())
        digit_count = sum(1 for c in translated_text if c.isdigit())
        total_alnum = alpha_count + digit_count
        _is_garbage = False
        if total_alnum > 3 and digit_count > 0:
            digit_ratio = digit_count / total_alnum
            # High digit-to-alpha mixing = likely garbage (normal text has <10% digits)
            if digit_ratio > 0.25 and len(translated_text) < 30:
                # Additional check: does it look like a random hash?
                has_mixed_runs = bool(_re_validate.search(r'[a-zA-Z]\d|\d[a-zA-Z]', translated_text))
                if has_mixed_runs:
                    _is_garbage = True

        # Detect OCR-concatenated text: "THatitookme#romi64kg", "Snd:lookingdlikethis"
        # Patterns: long runs without spaces, embedded hashtags/colons, camelCase mid-word
        if not _is_garbage and len(translated_text) > 12:
            space_count = translated_text.count(" ")
            word_ratio = space_count / max(1, len(translated_text))
            # Normal text has ~1 space per 5-6 chars; OCR concat has almost none
            if word_ratio < 0.04 and alpha_count > 10:
                # Check for embedded hashtags, colons, or erratic casing
                has_concat_markers = bool(_re_validate.search(r'[a-z]#|#[a-z]|[a-z]:[a-z]', translated_text))
                has_erratic_case = len(_re_validate.findall(r'[a-z][A-Z]', translated_text)) >= 2
                if has_concat_markers or has_erratic_case:
                    _is_garbage = True

        if _is_garbage:
            logger.warning(
                f"RENDER_TEXT ASS [{i}]: GARBAGE TEXT REJECTED: '{translated_text}' "
                f"(likely OCR concatenation, not proper translation)"
            )
            continue

        appears_at = overlay.get("appears_at", 0.0)
        disappears_at = overlay.get("disappears_at", 0.0)
        position = overlay.get("position", "top")
        overlay_zone = _position_to_zone(position)

        # FIX: ASS timing bug — overlays starting at 0.0 don't render on frame 0.
        # ASS "0:00:00.00" start is sometimes skipped by renderers on the very first
        # frame. Shift to -0.05s so the subtitle is already active at t=0.
        if appears_at < 0.05:
            appears_at = -0.05

        # Check if the matched text_region is "static" (visible throughout video).
        # Static regions should always span the full video duration regardless of
        # model-provided timestamps, which are often wrong (e.g., disappears_at=5.0
        # for text that's visible for 44 seconds).
        matched_region_for_timing = _match_overlay_to_region(overlay, text_regions) if text_regions else None
        is_static_region = (
            matched_region_for_timing
            and str(matched_region_for_timing.get("temporal", "")).lower() == "static"
            and not overlay.get("type") == "countdown"  # Countdowns are NOT static
        )

        # Check model-provided end time.
        model_end_raw = _safe_float(overlay.get("disappears_at"), 0.0)
        model_end_valid = (
            model_end_raw > (appears_at + 0.3)
            and model_end_raw <= (video_duration + 0.5)
        )

        # For static regions, model timing is often wrong (e.g. 12s for text
        # visible throughout a 44s video). Only trust model timing for static
        # regions if it covers >50% of video duration; otherwise extend to full.
        if is_static_region and model_end_valid and video_duration > 0:
            model_coverage = (model_end_raw - appears_at) / video_duration
            if model_coverage < 0.50:
                logger.info(
                    f"RENDER_TEXT ASS [{i}]: STATIC region but model_end={model_end_raw:.1f}s "
                    f"covers only {model_coverage:.0%} of {video_duration:.1f}s — extending to full"
                )
                model_end_valid = False  # Fall through to static handler

        if is_static_region and not model_end_valid:
            # Static text with no valid model end: force full video duration
            appears_at = -0.05 if appears_at < 0.05 else appears_at
            disappears_at = video_duration
            logger.info(
                f"RENDER_TEXT ASS [{i}]: STATIC region '{matched_region_for_timing.get('id', '?')}' "
                f"— forcing full duration 0..{video_duration:.1f}s"
            )
        elif model_end_valid:
            # Explicit timing from model (non-static region, or static with good coverage)
            disappears_at = min(video_duration, model_end_raw)
            logger.info(
                f"RENDER_TEXT ASS [{i}]: using model timing "
                f"{appears_at:.1f}..{disappears_at:.1f}s (explicit)"
            )
        else:
            # Hybrid timing resolution:
            # - bridge to next overlay start (by position/zone);
            # - avoid indefinite slabs by max-hold clamp later.
            pos_starts = _next_start_by_pos.get(position, [])
            zone_starts = _next_start_by_zone.get(overlay_zone, [])
            next_same_pos = _next_future_start(pos_starts, appears_at, min_gap=0.35)
            next_same_zone = _next_future_start(zone_starts, appears_at, min_gap=0.35)
            next_candidates = [t for t in (next_same_pos, next_same_zone) if t is not None]
            next_start = min(next_candidates) if next_candidates else None

            if next_start is not None:
                disappears_at = min(video_duration, next_start)
            else:
                disappears_at = video_duration

            if next_start is not None:
                disappears_at = min(disappears_at, next_start)

            # Safety: minimum display time.
            if disappears_at <= appears_at + 0.2:
                disappears_at = appears_at + 2.0

            # FIX: If disappears_at is very close to appears_at (model gave bad data)
            # and there's no next overlay, extend to video end
            if disappears_at <= appears_at + 0.5 and next_start is None:
                disappears_at = video_duration

        # Determine render strategy from adaptive pipeline
        matched_region = _match_overlay_to_region(overlay, text_regions)
        if not matched_region and text_regions:
            # Fallback: if exactly one region exists in this zone, use it.
            # This avoids defaulting to replace_inplace with a bad OCR fallback bbox.
            zone_candidates = [r for r in text_regions if r.get("zone", "middle") == overlay_zone]
            if len(zone_candidates) == 1:
                matched_region = zone_candidates[0]
                logger.info(
                    f"RENDER_TEXT ASS [{i}]: zone fallback matched region "
                    f"'{matched_region.get('id', '?')}' for zone='{overlay_zone}'"
                )
        render_region = matched_region
        if matched_region and matched_region.get("bbox_norm"):
            render_region = _stabilize_matched_region_bbox(
                overlay=overlay,
                region=matched_region,
                overlay_zone=overlay_zone,
                appears_at=appears_at,
                temporal_state=_bbox_temporal_state,
            )
        if matched_region:
            region_id = matched_region.get("id")
            strategy = render_strategies.get(region_id, "replace_inplace")
            logger.info(
                f"RENDER_TEXT ASS [{i}]: matched region '{matched_region.get('id', '?')}', "
                f"strategy='{strategy}'"
            )
        else:
            strategy = "replace_inplace"  # Default fallback

        # Final timing guard based on text/region/strategy.
        max_hold = _estimate_overlay_max_hold_seconds(
            translated_text=translated_text,
            region=render_region,
            strategy=strategy,
        )
        if (disappears_at - appears_at) > max_hold:
            old_end = disappears_at
            disappears_at = min(disappears_at, appears_at + max_hold)
            logger.info(
                "TIMING_GUARD: clamped overlay[%s] zone=%s strategy=%s duration %.2fs -> %.2fs",
                i,
                overlay_zone,
                strategy,
                old_end - appears_at,
                disappears_at - appears_at,
            )

        # Duplicate suppression in same zone (model sometimes emits near-identical clones).
        prev_zone = _zone_dedupe_state.get(overlay_zone)
        if prev_zone:
            prev_end = _safe_float(prev_zone.get("end"), -1.0)
            prev_text = str(prev_zone.get("text", ""))
            sim = _levenshtein_ratio(translated_text.lower(), prev_text.lower()) if prev_text else 0.0
            if appears_at <= (prev_end + 0.08) and sim >= 0.86:
                logger.info(
                    "TIMING_GUARD: skipped duplicate overlay[%s] zone=%s sim=%.2f",
                    i,
                    overlay_zone,
                    sim,
                )
                continue

        # FIX: When inpainting failed, non-watermark regions MUST use subtitle_snap.
        # backplate_overlay draws a fully opaque black rectangle (eraser plate) which
        # looks terrible for captions/subtitles/usernames. Only watermarks should get
        # eraser plates — everything else gets a styled overlay on top of original text.
        # This applies to ALL strategies that could produce eraser plates:
        # backplate_overlay (opaque rect) and replace_inplace (may show through).
        if not inpaint_succeeded and render_region:
            region_type = render_region.get("type", "").lower()
            if region_type not in ("watermark", "logo"):
                if strategy in ("backplate_overlay", "replace_inplace"):
                    logger.info(
                        f"RENDER_TEXT ASS [{i}]: ERASER PLATE DISABLED for non-watermark region "
                        f"(type='{region_type}', was='{strategy}'). Switching to subtitle_snap."
                    )
                    strategy = "subtitle_snap"

        # Handle skip strategy
        if strategy == "skip":
            logger.info(f"RENDER_TEXT ASS [{i}]: skipping overlay (strategy=skip)")
            continue

        # Handle backplate_overlay strategy
        if strategy == "backplate_overlay" and render_region:
            # Pass adjusted times (disappears_at may have been extended to video_duration)
            overlay_with_times = {**overlay, "appears_at": appears_at, "disappears_at": disappears_at}
            _render_backplate_overlay(
                ass_lines, overlay_with_times, render_region,
                video_width, video_height, font_size,
                is_rtl=is_rtl, target_language=target_language,
                caption_style=caption_style,
            )
            _zone_dedupe_state[overlay_zone] = {"text": translated_text, "end": disappears_at}
            continue

        # Handle subtitle_snap strategy (Pipeline A)
        if strategy == "subtitle_snap" and render_region:
            _render_subtitle_snap(
                ass_lines, {**overlay, "appears_at": appears_at, "disappears_at": disappears_at},
                render_region, video_width, video_height, font_size,
                is_rtl=is_rtl, target_language=target_language,
                caption_style=caption_style,
            )
            _zone_dedupe_state[overlay_zone] = {"text": translated_text, "end": disappears_at}
            continue

        # Handle blur_plate_overlay strategy (Pipeline B)
        if strategy == "blur_plate_overlay" and render_region:
            _render_blur_plate_overlay(
                ass_lines, {**overlay, "appears_at": appears_at, "disappears_at": disappears_at},
                render_region, video_width, video_height, font_size,
                is_rtl=is_rtl, target_language=target_language,
                caption_style=caption_style,
            )
            _zone_dedupe_state[overlay_zone] = {"text": translated_text, "end": disappears_at}
            continue

        # Default: replace_inplace (or subtitle_bottom) — original behavior
        # If we have a matched region with bbox, use it directly (more reliable than OCR zone search)
        if render_region and render_region.get("bbox_norm"):
            bbox = _normalize_bbox(render_region.get("bbox_norm", [0, 0.85, 1, 1]), video_width, video_height)
            pad_x = (bbox[2] - bbox[0]) * 0.10
            pad_y = (bbox[3] - bbox[1]) * 0.15
            x = max(0, int((bbox[0] - pad_x) * video_width))
            y = max(0, int((bbox[1] - pad_y) * video_height))
            box_w = min(video_width, int((bbox[2] - bbox[0] + 2 * pad_x) * video_width))
            box_h = max(int(video_height * 0.04), int((bbox[3] - bbox[1] + 2 * pad_y) * video_height))
            logger.info(f"RENDER_TEXT ASS [{i}]: using matched region bbox: ({x},{y},{box_w}x{box_h})")
        else:
            # Fallback: search OCR detections by zone + time
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
            style_name = "TranslatedTextNoBoxRTL"
        else:
            style_name = "TranslatedTextNoBox"

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
        _zone_dedupe_state[overlay_zone] = {"text": translated_text, "end": disappears_at}

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

    When inpaint_succeeded=False:
    - Watermark/logo regions: eraser plates (opaque black) to cover branding
    - All other regions (caption, subtitle, username): forced to subtitle_snap
      for styled overlay rendering — no black rectangles on content

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


def _tts_elevenlabs(text: str, reference_audio: str, target_language: str = "en",
                    voice_id: Optional[str] = None, campaign_id: Optional[int] = None,
                    speed: float = 1.0) -> Optional[Tuple[str, str]]:
    """Generate speech using ElevenLabs API (best quality, runs from GPU worker IP to avoid geo-blocks).

    Args:
        voice_id: Pre-cloned voice ID. If provided, skips cloning and reuses the voice.
        campaign_id: Campaign ID for naming the cloned voice (for traceability).

    Returns:
        Tuple of (audio_path, voice_id) or None if failed.
        voice_id is returned so server can persist it for future reuse.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.info("ElevenLabs: no API key, skipping")
        return None

    import httpx

    output_path = tempfile.mktemp(suffix=".mp3")

    try:
        # Step 1: Use persisted voice or create a new clone
        if voice_id:
            logger.info(f"ElevenLabs: reusing persisted voice {voice_id}")
        else:
            voice_name = f"tp_campaign_{campaign_id}" if campaign_id else "tp_clone_temp"
            logger.info(f"ElevenLabs: cloning voice as '{voice_name}' from reference audio...")
            with open(reference_audio, "rb") as f:
                clone_resp = httpx.post(
                    "https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": api_key},
                    data={"name": voice_name, "description": f"TrafficPlant voice clone (campaign={campaign_id})"},
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

            # Don't delete — server will persist this voice_id for reuse
            logger.info(f"ElevenLabs: voice cloned as {voice_id} (persisted, not deleting)")

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
                    "speed": speed,
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
            return output_path, voice_id

        except Exception as e:
            logger.warning(f"ElevenLabs TTS generation failed: {e}")
            return None

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


def stage_tts(text: str, reference_audio: str, mm: ModelManager, target_language: str = "en",
              voice_id: Optional[str] = None, campaign_id: Optional[int] = None,
              original_transcript: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    """Generate speech using ElevenLabs API (only engine — F5-TTS removed).
    Returns: (audio_path, method_used, voice_id_for_persistence)
    Raises RuntimeError if ElevenLabs fails (TTS is a critical stage).
    """
    # Calculate speech speed based on text length ratio.
    # When translation is significantly longer (e.g. RU→EN), slow down to avoid rushing.
    speed = 1.0
    source_chars = len(original_transcript or "")
    target_chars = len(text or "")
    if source_chars > 0 and target_chars > source_chars * 1.2:
        ratio = source_chars / target_chars
        speed = max(0.8, min(1.0, ratio * 1.05))  # Clamp 0.8-1.0
    logger.info(f"Stage: TTS (target_language={target_language}, voice_id={'reuse' if voice_id else 'clone'}, "
                f"campaign={campaign_id}, speed={speed:.2f}, src={source_chars} chars, tgt={target_chars} chars)")

    # ElevenLabs only (runs from US GPU IP — no geo-block)
    result = _tts_elevenlabs(text, reference_audio, target_language=target_language,
                             voice_id=voice_id, campaign_id=campaign_id, speed=speed)
    if result:
        audio_path, used_voice_id = result
        return audio_path, "elevenlabs", used_voice_id

    # No fallback — ElevenLabs is the only TTS engine.
    # F5-TTS removed: sounds robotic and degrades quality.
    raise RuntimeError(
        "ElevenLabs TTS failed and no fallback available. "
        "Check ELEVENLABS_API_KEY env var and API quota."
    )


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


def stage_dubbing(video_url: str, source_lang: str, target_lang: str,
                   video_path: str) -> str:
    """ElevenLabs Dubbing API — replaces preprocess+transcribe+translate+tts+assemble.

    Sends the video URL to ElevenLabs, which handles:
    - Transcription (auto speaker detection)
    - Translation to target language
    - Voice cloning of original speaker(s)
    - Audio mixing with background music/sounds

    Returns path to the dubbed audio file (MP3/MP4).
    The caller must then replace the video's audio track with this.
    """
    import httpx

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set — dubbing requires ElevenLabs API")

    logger.info(f"Stage: DUBBING (ElevenLabs) {source_lang}→{target_lang}")

    # Step 1: Create dubbing job — upload file directly (not source_url)
    # R2 URLs return application/octet-stream which ElevenLabs rejects.
    # We must upload the local video file with explicit video/mp4 MIME type.
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")

    with httpx.Client(timeout=120) as client:
        with open(video_path, "rb") as vf:
            resp = client.post(
                "https://api.elevenlabs.io/v1/dubbing",
                headers={"xi-api-key": api_key},
                data={
                    "source_lang": source_lang if source_lang != "auto" else "auto",
                    "target_lang": target_lang,
                    "num_speakers": "0",
                    "watermark": "false",
                },
                files={"file": ("video.mp4", vf, "video/mp4")},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs dubbing create failed: {resp.status_code} {resp.text[:300]}")

        dub_data = resp.json()
        dubbing_id = dub_data.get("dubbing_id")
        expected = dub_data.get("expected_duration_sec", 0)
        if not dubbing_id:
            raise RuntimeError(f"ElevenLabs dubbing: no dubbing_id returned: {dub_data}")

    logger.info(f"ElevenLabs dubbing created: {dubbing_id} (expected ~{expected:.0f}s)")

    # Step 2: Poll until done (max 10 minutes)
    max_wait = 600
    poll_interval = 5
    elapsed = 0

    with httpx.Client(timeout=30) as client:
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            resp = client.get(
                f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}",
                headers={"xi-api-key": api_key},
            )
            if resp.status_code != 200:
                logger.warning(f"ElevenLabs dubbing poll failed: {resp.status_code}")
                continue

            status_data = resp.json()
            status = status_data.get("status", "unknown")
            logger.info(f"ElevenLabs dubbing {dubbing_id}: status={status} ({elapsed}s)")

            if status == "dubbed":
                break
            elif status in ("failed", "error"):
                error_msg = status_data.get("error", "unknown error")
                raise RuntimeError(f"ElevenLabs dubbing failed: {error_msg}")

        else:
            raise RuntimeError(f"ElevenLabs dubbing timed out after {max_wait}s")

    # Step 3: Download dubbed audio
    logger.info(f"ElevenLabs dubbing complete, downloading audio for {target_lang}...")

    output_path = video_path.replace(".mp4", f"_dubbed_{target_lang}.mp4")

    with httpx.Client(timeout=120) as client:
        resp = client.get(
            f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}/audio/{target_lang}",
            headers={"xi-api-key": api_key},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs dubbing download failed: {resp.status_code} {resp.text[:200]}")

        with open(output_path, "wb") as f:
            f.write(resp.content)

    logger.info(f"ElevenLabs dubbing: downloaded {len(resp.content)} bytes → {output_path}")
    return output_path


def _verify_text_removal(output_path: str, source_lang: str, mm: "ModelManager",
                         num_frames: int = 6) -> Dict[str, Any]:
    """
    Post-pipeline quality gate: extract frames from output video, run OCR,
    check for remaining source-language text.

    Returns a quality report with detection count and text_removal_score (0-100).
    """
    import cv2
    import subprocess

    logger.info(f"QUALITY_GATE: verifying text removal on {output_path} (source_lang={source_lang})")

    if not os.path.exists(output_path):
        return {"source_text_found": False, "detection_count": 0, "detections": [],
                "text_removal_score": 100, "error": "output file not found"}

    # Get video duration
    dur_result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", output_path
    ], capture_output=True, text=True)
    duration = float(dur_result.stdout.strip()) if dur_result.returncode == 0 else 0
    if duration <= 0:
        return {"source_text_found": False, "detection_count": 0, "detections": [],
                "text_removal_score": 100, "error": "could not determine duration"}

    # Extract evenly-spaced frames using cv2
    cap = cv2.VideoCapture(output_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // (num_frames + 1))

    # Load PaddleOCR (should already be loaded from detect_text stage)
    try:
        ocr = mm.load("paddleocr")
    except Exception as e:
        cap.release()
        logger.warning(f"QUALITY_GATE: PaddleOCR load failed: {e}")
        return {"source_text_found": False, "detection_count": 0, "detections": [],
                "text_removal_score": -1, "error": str(e)}

    detections = []
    for i in range(num_frames):
        frame_idx = interval * (i + 1)
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = frame_idx / fps

        try:
            result = ocr.ocr(frame, cls=False)
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.6 and len(text) > 1:
                        if _is_source_language_text(text, source_lang):
                            detections.append({
                                "frame": i,
                                "frame_idx": frame_idx,
                                "timestamp": round(timestamp, 2),
                                "text": text,
                                "confidence": round(conf, 3),
                            })
        except Exception as e:
            logger.warning(f"QUALITY_GATE: OCR failed on frame {frame_idx}: {e}")

    cap.release()

    # Score: 100 = perfect removal, each detection costs 15 points
    score = max(0, 100 - len(detections) * 15)

    report = {
        "source_text_found": len(detections) > 0,
        "detection_count": len(detections),
        "detections": detections[:10],  # Limit to first 10
        "text_removal_score": score,
    }

    if detections:
        texts = [d["text"] for d in detections[:5]]
        logger.warning(
            f"QUALITY_GATE: {len(detections)} source-language text fragments still visible! "
            f"score={score}/100 samples={texts}"
        )
    else:
        logger.info(f"QUALITY_GATE: PASS — no source-language text detected (score={score}/100)")

    return report


def _is_source_language_text(text: str, lang: str) -> bool:
    """Check if text contains characters from the source language."""
    lang = (lang or "").lower().split("-")[0]
    if lang in ("ru", "russian"):
        # Cyrillic characters
        return any('\u0400' <= c <= '\u04FF' for c in text)
    elif lang in ("zh", "chinese"):
        # CJK Unified Ideographs
        return any('\u4E00' <= c <= '\u9FFF' for c in text)
    elif lang in ("ja", "japanese"):
        # Hiragana + Katakana + CJK
        return any(('\u3040' <= c <= '\u309F') or ('\u30A0' <= c <= '\u30FF') or ('\u4E00' <= c <= '\u9FFF') for c in text)
    elif lang in ("ko", "korean"):
        # Hangul
        return any('\uAC00' <= c <= '\uD7AF' for c in text)
    elif lang in ("ar", "arabic"):
        # Arabic script
        return any('\u0600' <= c <= '\u06FF' for c in text)
    elif lang in ("he", "hebrew"):
        return any('\u0590' <= c <= '\u05FF' for c in text)
    elif lang in ("th", "thai"):
        return any('\u0E00' <= c <= '\u0E7F' for c in text)
    elif lang in ("hi", "hindi"):
        # Devanagari
        return any('\u0900' <= c <= '\u097F' for c in text)
    # For Latin-script source languages, we cannot easily distinguish from target
    # (e.g., Portuguese source vs English target both use Latin chars)
    return False


def stage_assemble(
    video_path: str,
    tts_audio: Optional[str],
    background_audio: Optional[str],
    output_path: str
) -> str:
    """Assemble final video with mixed audio.

    Rules for localized output:
    - TTS + background: Mix TTS (full volume) + background (30%), fade out last 500ms
    - TTS only: Use TTS audio with fade out
    - No TTS + background: Use background only (vocals already stripped by Demucs)
    - No TTS + no background: Strip all audio (silent video — NEVER keep original vocals)
    """
    logger.info("Stage: ASSEMBLE")

    # Get video duration for fade-out calculation
    duration_result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ], capture_output=True, text=True)
    video_duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 0
    fade_start = max(0, video_duration - 0.5) if video_duration > 0.5 else 0

    if not tts_audio:
        # No TTS — use background-only audio (vocals stripped) or go silent.
        # NEVER copy original audio — it contains source-language vocals.
        if background_audio and os.path.exists(background_audio):
            logger.info("No TTS audio — using Demucs background only (vocals stripped)")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", background_audio,
                "-c:v", "copy",
                "-af", f"afade=t=out:st={fade_start}:d=0.5",
                "-c:a", "aac",
                "-ar", "44100",
                "-ac", "2",
                "-map", "0:v",
                "-map", "1:a",
                "-shortest",
                output_path
            ], capture_output=True, check=False)
        else:
            # No Demucs ran, no TTS — this is a text-only video (music/ambient only).
            # Preserve original audio since there's no source-language speech to hide.
            logger.info("No TTS audio, no background audio — preserving original audio (text-only pipeline)")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", "44100",
                "-ac", "2",
                output_path
            ], capture_output=True, check=False)
        if not os.path.exists(output_path):
            raise RuntimeError(f"ffmpeg assemble (no-TTS) failed — output not created: {output_path}")
        return output_path

    if background_audio and os.path.exists(background_audio):
        # Mix TTS voice with background audio.
        # TTS at full volume, background at 30% (ducked).
        # 500ms fade-out at end to prevent audio tail leak.
        #
        # CRITICAL: Resample BOTH inputs to 44100Hz stereo FIRST.
        # ElevenLabs outputs ~44.1kHz, Demucs 44.1kHz stereo.
        mixed_audio = output_path.replace(".mp4", "_mixed.wav")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", tts_audio,
            "-i", background_audio,
            "-filter_complex",
            # Resample both to same format, mix, then fade out last 500ms
            "[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[tts];"
            "[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.3[bg];"
            f"[tts][bg]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,afade=t=in:st=0:d=0.2,afade=t=out:st={fade_start}:d=0.5[a]",
            "-map", "[a]",
            "-ar", "44100", "-ac", "2",
            mixed_audio
        ], capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode()[-500:] if result.stderr else "unknown"
            logger.warning(f"Audio mixing failed (rc={result.returncode}): {stderr}")
        audio_to_use = mixed_audio if os.path.exists(mixed_audio) else tts_audio
    else:
        # No background audio — use TTS audio directly (still apply fade-out)
        faded_tts = output_path.replace(".mp4", "_faded_tts.wav")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", tts_audio,
            "-af", f"aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,afade=t=out:st={fade_start}:d=0.5",
            "-ar", "44100", "-ac", "2",
            faded_tts
        ], capture_output=True, check=False)
        audio_to_use = faded_tts if (result.returncode == 0 and os.path.exists(faded_tts)) else tts_audio

    # Combine video with audio
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_to_use,
        "-c:v", "copy",
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
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

    # Ensure geo-aware fonts are downloaded and registered (lazy init, cached)
    try:
        _ensure_geo_fonts()
    except Exception as e:
        logger.warning(f"Font initialization failed (non-fatal): {e}")

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
            original_transcript=job_input.get("original_transcript"),  # Source text for TTS speed calc
            translated_overlays=job_input.get("translated_overlays"),  # Pre-translated text overlays
            subtitle_style=job_input.get("subtitle_style"),  # Original subtitle style from manifest
            elevenlabs_voice_id=job_input.get("elevenlabs_voice_id"),  # Persisted voice ID
            campaign_id=job_input.get("campaign_id"),  # For voice naming
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
            asp = video_profile.get("account_style_profile", {}) or {}
            if asp:
                logger.info(
                    "ADAPTIVE: AccountStyle profile loaded — container=%s density=%s emphasis=%s overlays=%s",
                    asp.get("container_preference"),
                    asp.get("density"),
                    asp.get("emphasis"),
                    asp.get("overlay_count"),
                )
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
            "translated_text": config.translated_text,  # Pre-loaded from server (Gemini 3 Pro)
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
                        logger.info("DETECT_TEXT: All regions use subtitle_snap/blur_plate — running OCR for precise bbox positioning")
                    # Always run OCR when detect_text is in stages — gives precise bboxes
                    if True:
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
                    snap_blur_regions = []  # subtitle_snap regions that need pre-blur

                    # Build set of region IDs that have non-empty translated text.
                    # Only blur/inpaint regions that will actually get text rendered;
                    # otherwise we leave visible blur patches with no overlay text.
                    _renderable_region_ids = set()
                    if config.translated_overlays:
                        for _ov in config.translated_overlays:
                            _tt = (_ov.get("translated_text") or "").strip()
                            _rid = _ov.get("region_id", "")
                            if _tt and _rid:
                                _renderable_region_ids.add(_rid)

                    if state.get("video_profile"):
                        text_regions = state["video_profile"].get("text_regions", [])
                        for r in text_regions:
                            pt = r.get("pipeline_type", "inpaint_backplate")
                            rid = r.get("id", "")
                            # Skip regions whose translated overlay is empty/garbage
                            if _renderable_region_ids and rid and rid not in _renderable_region_ids:
                                logger.info(f"INPAINT: Skipping region '{rid}' — no translated text to render")
                                continue
                            if pt == "inpaint_backplate":
                                needs_inpaint = True
                            elif pt == "blur_plate":
                                blur_regions.append(r)
                            elif pt == "subtitle_snap":
                                # Pre-blur to remove original text before overlay
                                snap_blur_regions.append(r)
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

                    # Pre-blur subtitle_snap regions to erase original text
                    # before ASS overlay is rendered on top
                    if snap_blur_regions:
                        try:
                            logger.info(f"INPAINT: Pre-blurring {len(snap_blur_regions)} subtitle_snap regions to remove original text")
                            state["video_path"] = stage_blur_plate(
                                state["video_path"], snap_blur_regions,
                                video_profile=state.get("video_profile"),
                            )
                        except Exception as e:
                            logger.warning(f"SNAP_PRE_BLUR failed (non-fatal): {e}")
                            metrics.errors.append(f"blur_plate: {str(e)}")

                elif stage_name == "render_text":
                    # Refine Gemini bboxes with precise OCR detections (if available)
                    if state.get("text_detections") and state.get("video_profile"):
                        vp = state["video_profile"]
                        vp["text_regions"] = _refine_regions_from_ocr(
                            vp.get("text_regions", []),
                            state["text_detections"],
                        )
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
                        tts_result, tts_method, used_voice_id = stage_tts(
                            state["translated_text"],
                            state.get("vocals_path") or video_path,
                            mm,
                            target_language=config.target_language,
                            voice_id=config.elevenlabs_voice_id,
                            campaign_id=config.campaign_id,
                            original_transcript=config.original_transcript,
                        )
                        state["tts_audio"] = tts_result
                        state["cloned_voice_id"] = used_voice_id
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

                    # Post-assembly quality gate: verify source text was removed
                    try:
                        qg_report = _verify_text_removal(
                            state["video_path"],
                            source_lang=config.source_language,
                            mm=mm,
                        )
                        if qg_report.get("source_text_found"):
                            metrics.quality_warnings.append(
                                f"source_text_visible: {qg_report['detection_count']} detections, "
                                f"score={qg_report['text_removal_score']}/100"
                            )
                        metrics.quality_scores["text_removal"] = qg_report
                    except Exception as e:
                        logger.warning(f"QUALITY_GATE: verification failed (non-fatal): {e}")
                        metrics.errors.append(f"quality_gate: {str(e)}")

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

        result = {
            "status": "success",
            "output_url": output_url,
            "metrics": metrics.to_dict(),
            "transcript": state.get("transcript"),
        }
        # Return cloned voice_id so server can persist it for future reuse
        if state.get("cloned_voice_id"):
            result["cloned_voice_id"] = state["cloned_voice_id"]
        return result

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
