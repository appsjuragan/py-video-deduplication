"""
Video file scanner - walks directories recursively to find all video files.
"""
import os
from pathlib import Path
from typing import List, Set, Optional

VIDEO_EXTENSIONS: Set[str] = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.ts', '.mts',
    '.m2ts', '.vob', '.ogv', '.rm', '.rmvb', '.asf', '.divx',
    '.f4v', '.swf', '.amv', '.svi',
}


def scan_folders(
    folders: List[str],
    size_min: Optional[int] = None,
    size_max: Optional[int] = None,
    file_types: Optional[List[str]] = None,
) -> List[dict]:
    """
    Walk through all given folders recursively and collect video files.

    Args:
        folders: List of directory paths to scan.
        size_min: Minimum file size in bytes (inclusive). None = no limit.
        size_max: Maximum file size in bytes (inclusive). None = no limit.
        file_types: List of extensions to include (e.g. ['.mp4', '.mkv']).
                     None = use all VIDEO_EXTENSIONS.

    Returns list of dicts with file metadata.
    """
    allowed_exts = set(file_types) if file_types else VIDEO_EXTENSIONS
    videos = []
    seen_paths: Set[str] = set()

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for root, _dirs, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in allowed_exts:
                    continue

                full_path = os.path.join(root, fname)
                real_path = os.path.realpath(full_path)

                if real_path in seen_paths:
                    continue
                seen_paths.add(real_path)

                try:
                    stat = os.stat(full_path)
                    fsize = stat.st_size

                    # Apply size filters
                    if size_min is not None and fsize < size_min:
                        continue
                    if size_max is not None and fsize > size_max:
                        continue

                    videos.append({
                        'path': full_path,
                        'name': fname,
                        'size': fsize,
                        'modified': stat.st_mtime,
                        'folder': root,
                    })
                except OSError:
                    continue

    return videos
