"""
Hardware Profile System — Probe Once, Cache Forever.

On first launch, detects CPU core count, GPU properties (VRAM, CUDA cores,
NVDEC engine count, compute capability), and calculates optimal pipeline
parameters. Saves to hw_profile.json so subsequent launches skip all probing.

Design: always leave headroom (2 CPU threads, ~10% VRAM) for user tasks.
"""
import json
import os
import logging
import multiprocessing
from typing import Optional

logger = logging.getLogger(__name__)

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_profile.json")

# ── Known NVDEC engine counts per GPU architecture ────────────────────────────
# Source: NVIDIA Video Codec SDK documentation
# These are the number of *hardware* NVDEC decoder instances per chip.
_NVDEC_BY_ARCH = {
    # Turing (RTX 20xx)
    "TU102": 2, "TU104": 2, "TU106": 2, "TU116": 1, "TU117": 1,
    # Ampere (RTX 30xx)
    "GA102": 3, "GA103": 3, "GA104": 2, "GA106": 2, "GA107": 1,
    # Ada Lovelace (RTX 40xx)
    "AD102": 3, "AD103": 3, "AD104": 2, "AD106": 2, "AD107": 1,
    # Blackwell (RTX 50xx)
    "GB202": 3, "GB203": 3, "GB205": 2,
}

# Mapping from common GPU names → rough NVDEC count
_NVDEC_BY_NAME = {
    # RTX 30 series
    "RTX 3090": 2, "RTX 3080": 2, "RTX 3070": 2, "RTX 3060": 2, "RTX 3050": 1,
    # RTX 40 series
    "RTX 4090": 3, "RTX 4080": 3, "RTX 4070": 2, "RTX 4060": 2,
    # RTX 50 series
    "RTX 5090": 3, "RTX 5080": 3, "RTX 5070": 2, "RTX 5060": 2,
    # GTX 16 series
    "GTX 1660": 1, "GTX 1650": 1,
}


def _estimate_nvdec_count(gpu_name: str) -> int:
    """Estimate NVDEC engine count from the GPU product name."""
    name_upper = gpu_name.upper()
    for key, count in _NVDEC_BY_NAME.items():
        if key.upper() in name_upper:
            return count
    # Conservative fallback
    return 1


def _probe_hardware() -> dict:
    """
    Full hardware probe. Called only once; result is cached to disk.
    Returns a dict with all detected parameters + computed optimal settings.
    """
    profile = {
        "version": 2,
        "cpu_cores_physical": 1,
        "cpu_cores_logical": 1,
        "gpu_available": False,
        "gpu_name": "",
        "gpu_vram_total_mb": 0,
        "gpu_compute_capability": "",
        "gpu_sm_count": 0,
        "nvdec_engines": 1,
        # ── Computed optimal settings (with headroom) ──
        "batch_v_size": 8,         # videos per pipeline stage
        "extractor_threads": 4,    # concurrent FFmpeg workers
        "gpu_batch_size": 64,      # tensor batch size for model inference
        "vram_usage_pct": 0.85,    # how much VRAM to target (leaves 15% headroom)
        "mb_per_frame_fp16": 6,    # memory per frame in FP16 mode
    }

    # ── CPU ──────────────────────────────────────────────────────────────
    try:
        physical = multiprocessing.cpu_count()
    except Exception:
        physical = os.cpu_count() or 4
    profile["cpu_cores_physical"] = physical
    profile["cpu_cores_logical"] = os.cpu_count() or physical

    # ── GPU (via PyTorch + pynvml) ───────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            profile["gpu_available"] = True
            props = torch.cuda.get_device_properties(0)
            profile["gpu_name"] = props.name
            profile["gpu_vram_total_mb"] = props.total_memory // (1024 ** 2)
            profile["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            profile["gpu_sm_count"] = props.multi_processor_count
            profile["nvdec_engines"] = _estimate_nvdec_count(props.name)
    except Exception as e:
        logger.warning(f"GPU probe failed: {e}")

    # ── Compute optimal settings (with headroom) ────────────────────────
    logical_cores = profile["cpu_cores_logical"]
    nvdec = profile["nvdec_engines"]
    vram_mb = profile["gpu_vram_total_mb"]

    # FFmpeg extractor threads:
    # - Each NVDEC engine can handle ~4 concurrent decode streams efficiently
    # - But also limited by CPU cores (FFmpeg demuxing is CPU-bound)
    # - Reserve 2 logical cores for user headroom + Flask server
    max_by_nvdec = nvdec * 4
    max_by_cpu = max(2, logical_cores - 2)  # leave 2 cores headroom
    profile["extractor_threads"] = min(max_by_nvdec, max_by_cpu)

    # Video batch size (how many videos to process per pipeline stage):
    # Tied to extractor threads — no point having more videos than threads
    profile["batch_v_size"] = max(4, profile["extractor_threads"])

    # GPU tensor batch size:
    # FP16 EfficientNet-B0 uses ~6 MB per frame
    # Leave 15% VRAM headroom for OS/compositor/other apps
    if vram_mb > 0:
        usable_vram = vram_mb * profile["vram_usage_pct"]
        profile["gpu_batch_size"] = max(32, int(usable_vram / profile["mb_per_frame_fp16"]))
    else:
        profile["gpu_batch_size"] = 64

    logger.info(
        f"Hardware probe complete: "
        f"CPU={logical_cores} cores, "
        f"GPU={profile['gpu_name']} ({vram_mb} MB VRAM, {nvdec} NVDEC), "
        f"→ extractor_threads={profile['extractor_threads']}, "
        f"batch_v_size={profile['batch_v_size']}, "
        f"gpu_batch_size={profile['gpu_batch_size']}"
    )

    return profile


def load_profile(force_reprobe: bool = False) -> dict:
    """
    Load the cached hardware profile, or probe + save if missing.
    Set force_reprobe=True to regenerate (e.g. after GPU swap).
    """
    if not force_reprobe and os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r") as f:
                profile = json.load(f)
            # Validate version to catch stale profiles
            if profile.get("version", 0) >= 2:
                logger.info(
                    f"Loaded cached hardware profile: "
                    f"extractor_threads={profile['extractor_threads']}, "
                    f"batch_v_size={profile['batch_v_size']}, "
                    f"gpu_batch_size={profile['gpu_batch_size']}"
                )
                return profile
            else:
                logger.info("Stale profile version detected, re-probing.")
        except Exception as e:
            logger.warning(f"Failed to load hw_profile.json: {e}")

    # Probe and save
    profile = _probe_hardware()
    try:
        with open(PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Hardware profile saved to {PROFILE_PATH}")
    except Exception as e:
        logger.warning(f"Could not save hardware profile: {e}")

    return profile


def get_profile_summary(profile: dict) -> dict:
    """Return a UI-friendly summary of the hardware profile."""
    return {
        "cpu_cores": profile.get("cpu_cores_logical", "?"),
        "gpu": profile.get("gpu_name", "None"),
        "vram_mb": profile.get("gpu_vram_total_mb", 0),
        "nvdec_engines": profile.get("nvdec_engines", 0),
        "extractor_threads": profile.get("extractor_threads", 4),
        "batch_v_size": profile.get("batch_v_size", 8),
        "gpu_batch_size": profile.get("gpu_batch_size", 64),
    }
