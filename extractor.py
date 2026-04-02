"""
Frame extractor using FFmpeg - extracts keyframes from videos for comparison.
Auto-discovers ffmpeg/ffprobe from common install locations if not on PATH.
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


def extract_frames(video_path: str, num_frames: int = 24,
                   target_size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """
    Extract frames at fixed intervals using concurrent FFmpeg processes.
    If possible, uses CUDA hardware acceleration for decoding.
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

    hw_args = ["-hwaccel", "cuda"] if "cuda" in str(FFMPEG_BIN).lower() or True else []

    try:
        # Instead of launching 24 ffmpeg calls per video (which exhausts NVDEC limits),
        # launch exactly ONE ffmpeg process that extracts num_frames sequentially.
        cmd = [
            FFMPEG_BIN, '-y'
        ] + hw_args + [
            '-i', video_path,
            # Fastest way: just pull I-frames (keyframes). Avoids full decoding.
            '-vf', f'select=eq(pict_type\\,I),scale={target_size[0]}:{target_size[1]}:force_original_aspect_ratio=increase,crop={target_size[0]}:{target_size[1]}',
            '-vsync', 'vfr',
            '-vframes', str(num_frames),
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-an',
            'pipe:1'
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        
        if result.returncode == 0 and len(result.stdout) > 0:
            frames = []
            # Split the pipe data into individual PNG sequences
            raw_frames = result.stdout.split(b'\x89PNG\r\n\x1a\n')
            for chunk in raw_frames[1:]:
                if chunk:
                    frames.append(Image.open(io.BytesIO(b'\x89PNG\r\n\x1a\n' + chunk)).convert("RGB"))
            return frames[:num_frames]
        
    except Exception as e:
        logger.debug(f"Frame extraction error: {e}")

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
