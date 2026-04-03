"""
Frame extractor using FFmpeg - extracts keyframes from videos for comparison.
Auto-discovers ffmpeg/ffprobe from common install locations if not on PATH.
Hardware acceleration: auto-probes CUDA → Vulkan → OpenCL → D3D11VA → DXVA2 → software.
"""
import subprocess
import json
import io
import os
import shutil
import logging
from typing import List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# ── FFmpeg auto-discovery ─────────────────────────────────────────────────────

_FFMPEG_SEARCH_PATHS = [
    # Common Windows locations
    r"C:\msys64\mingw64\bin",
    r"C:\msys64\usr\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"D:\ffmpeg\bin",
    r"D:\Program Files\ffmpeg\bin",
    # Chocolatey / Scoop
    r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin",
    r"C:\Users\Administrator\scoop\apps\ffmpeg\current\bin",
    # Common bundled locations
    r"C:\Program Files\Kdenlive\bin",
    r"C:\Program Files\HandBrake",
    r"C:\Program Files\obs-studio\bin\64bit",
]


def _find_exe(name: str) -> str:
    """Try to locate an executable from PATH or known search paths."""
    # Try shutil first (respects PATH)
    found = shutil.which(name)
    if found:
        return found

    # Search known locations
    for folder in _FFMPEG_SEARCH_PATHS:
        candidate = os.path.join(folder, name + ".exe")
        if os.path.isfile(candidate):
            logger.info(f"Found {name} at: {candidate}")
            return candidate

    return name  # Fall back to bare name and let it fail with a clear error


# Resolve once at import time
FFMPEG_BIN = _find_exe("ffmpeg")
FFPROBE_BIN = _find_exe("ffprobe")

logger.info(f"ffmpeg  -> {FFMPEG_BIN}")
logger.info(f"ffprobe -> {FFPROBE_BIN}")


# ── Hardware acceleration detection ───────────────────────────────────────────

# Priority order: NVIDIA CUDA > cross-platform Vulkan > cross-platform OpenCL
# > Windows-native D3D11VA > older Windows DXVA2 > pure software
_HWACCEL_PRIORITY = ["cuda", "vulkan", "opencl", "d3d11va", "dxva2"]


def _detect_hwaccel() -> str:
    """Probe FFmpeg for the best available hardware decoder on this machine.

    Returns the hwaccel name string (e.g. 'cuda', 'vulkan', 'opencl') or
    an empty string when no GPU acceleration is available.
    """
    try:
        r = subprocess.run(
            [FFMPEG_BIN, "-hwaccels"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        available = {line.strip().lower() for line in r.stdout.splitlines()} | \
                    {line.strip().lower() for line in r.stderr.splitlines()}
    except Exception as exc:
        logger.warning(f"Could not probe FFmpeg hwaccels: {exc}")
        return ""

    for accel in _HWACCEL_PRIORITY:
        if accel in available:
            logger.info(f"FFmpeg hwaccel selected: {accel}")
            return accel

    logger.info("No GPU hwaccel found – FFmpeg will use software decoding.")
    return ""


FFMPEG_HWACCEL: str = _detect_hwaccel()


def check_ffmpeg() -> dict:
    """Check if ffmpeg/ffprobe are available. Returns status dict."""
    result = {"ffmpeg": False, "ffprobe": False, "ffmpeg_path": FFMPEG_BIN, "ffprobe_path": FFPROBE_BIN}
    for key, exe in [("ffmpeg", FFMPEG_BIN), ("ffprobe", FFPROBE_BIN)]:
        try:
            r = subprocess.run(
                [exe, "-version"], capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            result[key] = r.returncode == 0
        except Exception:
            result[key] = False
    return result


# ── Video info / frame extraction ─────────────────────────────────────────────

def extract_metadata(filepath: str) -> dict:
    """
    CPU Fast-Filter: Wraps ffprobe to instantly return duration, size, and
    dimensions WITHOUT decoding any video frames.  O(1) per file.
    Returns a dict with keys: duration, size, width, height.
    """
    default = {"duration": 0.0, "size": 0, "width": 0, "height": 0}
    if not os.path.exists(filepath):
        return default
    try:
        cmd = [
            FFPROBE_BIN, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            filepath
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        if r.returncode != 0:
            return default
        data = json.loads(r.stdout)
    except Exception as exc:
        logger.warning(f"extract_metadata failed for {filepath}: {exc}")
        return default

    def _f(val) -> float:
        try:
            return float(val) if val and str(val).lower() != "n/a" else 0.0
        except (ValueError, TypeError):
            return 0.0

    fmt = data.get("format", {})
    duration = _f(fmt.get("duration", 0))
    size = int(fmt.get("size", 0) or 0)
    width = height = 0
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            width = int(s.get("width", 0) or 0)
            height = int(s.get("height", 0) or 0)
            if duration <= 0:
                duration = _f(s.get("duration", 0))
            break
    if size == 0:
        try:
            size = os.path.getsize(filepath)
        except OSError:
            pass
    return {"duration": duration, "size": size, "width": width, "height": height}


def get_vram_free_mb() -> float:
    """
    Return free VRAM in MB on the primary CUDA device.
    Returns 0.0 when CUDA is unavailable or pynvml is not installed.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.free / 1024 ** 2
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free / 1024 ** 2
    except Exception:
        pass
    return 0.0


def get_video_info(video_path: str) -> Optional[dict]:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            FFPROBE_BIN, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)

        duration = 0.0
        width = 0
        height = 0

        # Safely parse float string that might be "N/A"
        def _parse_float(val) -> float:
            try:
                if val and str(val).lower() != "n/a":
                    return float(val)
            except (ValueError, TypeError):
                pass
            return 0.0

        if "format" in data and "duration" in data["format"]:
            duration = _parse_float(data["format"]["duration"])

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = int(stream.get("width", 0) or 0)
                height = int(stream.get("height", 0) or 0)
                if duration <= 0 and "duration" in stream:
                    duration = _parse_float(stream["duration"])
                break

        return {"duration": float(duration), "width": int(width), "height": int(height)}
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
        return None


def _run_ffmpeg_extract(cmd: List[str], num_frames: int) -> List[Image.Image]:
    """Run an FFmpeg command that pipes PNG frames to stdout; return parsed frames."""
    result = subprocess.run(
        cmd, capture_output=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if result.returncode == 0 and len(result.stdout) > 0:
        frames: List[Image.Image] = []
        raw_frames = result.stdout.split(b"\x89PNG\r\n\x1a\n")
        for chunk in raw_frames[1:]:
            if chunk:
                frames.append(Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + chunk)).convert("RGB"))
        return frames[:num_frames]
    return []


def extract_frames(video_path: str, num_frames: int = 24,
                   target_size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """
    Extract frames at fixed intervals using a single FFmpeg process.
    Hardware acceleration priority: CUDA → Vulkan → OpenCL → D3D11VA → DXVA2 → software.
    Falls back automatically if the accelerated decode fails.
    """
    info = get_video_info(video_path)
    if not info:
        return []

    duration = info.get("duration", 0.0)
    interval = 1.5
    timestamps = [float(i * interval) for i in range(num_frames)]

    if duration > 0:
        timestamps = [ts for ts in timestamps if ts < duration]
        if len(timestamps) < num_frames:
            new_interval = duration / (num_frames + 1)
            timestamps = [new_interval * (i + 1) for i in range(num_frames)]

    vf = (
        f"select=eq(pict_type\\,I),"
        f"scale={target_size[0]}:{target_size[1]}:force_original_aspect_ratio=increase,"
        f"crop={target_size[0]}:{target_size[1]}"
    )
    base_tail = [
        "-i", video_path,
        "-vf", vf,
        "-vsync", "vfr",
        "-vframes", str(num_frames),
        "-f", "image2pipe",
        "-vcodec", "png",
        "-an",
        "pipe:1",
    ]

    # Build candidate command list: accelerated first, then plain software
    candidates: List[List[str]] = []
    if FFMPEG_HWACCEL:
        candidates.append([FFMPEG_BIN, "-y", "-hwaccel", FFMPEG_HWACCEL] + base_tail)
    candidates.append([FFMPEG_BIN, "-y"] + base_tail)  # pure software fallback

    for cmd in candidates:
        accel_label = cmd[3] if "-hwaccel" in cmd else "software"
        try:
            frames = _run_ffmpeg_extract(cmd, num_frames)
            if frames:
                logger.debug(f"extract_frames: {len(frames)} frames via {accel_label}")
                return frames
            logger.debug(f"extract_frames: {accel_label} returned 0 frames, trying next...")
        except Exception as exc:
            logger.debug(f"extract_frames [{accel_label}] error: {exc}, trying next...")

    return []


def extract_thumbnail(video_path: str, size: Tuple[int, int] = (320, 180)) -> Optional[bytes]:
    """Extract a single thumbnail from a video at ~10% mark."""
    info = get_video_info(video_path)
    if not info:
        return None

    duration = info.get("duration", 0.0)
    ts = max(duration * 0.1, 0.5) if duration > 0 else 0.0

    try:
        cmd = [
            FFMPEG_BIN,
            "-ss", str(ts),
            "-i", video_path,
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-s", f"{size[0]}x{size[1]}",
            "-q:v", "5",
            "-an", "-y",
            "pipe:1"
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        if result.returncode == 0 and len(result.stdout) > 0:
            return result.stdout
    except Exception as e:
        logger.debug(f"Thumbnail extract error: {e}")
    return None
