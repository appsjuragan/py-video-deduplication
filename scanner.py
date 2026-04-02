"""
Video file scanner - walks directories recursively to find all video files.
"""
import os
from pathlib import Path
from typing import List, Set

VIDEO_EXTENSIONS: Set[str] = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.ts', '.mts',
    '.m2ts', '.vob', '.ogv', '.rm', '.rmvb', '.asf', '.divx',
    '.f4v', '.swf', '.amv', '.svi',
}


def scan_folders(folders: List[str]) -> List[dict]:
    """
    Walk through all given folders recursively and collect video files.
    Returns list of dicts with file metadata.
    """
    videos = []
    seen_paths: Set[str] = set()

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for root, _dirs, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    full_path = os.path.join(root, fname)
                    real_path = os.path.realpath(full_path)

                    if real_path in seen_paths:
                        continue
                    seen_paths.add(real_path)

                    try:
                        stat = os.stat(full_path)
                        videos.append({
                            'path': full_path,
                            'name': fname,
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'folder': root,
                        })
                    except OSError:
                        continue

    return videos
