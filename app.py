"""
Video Deduplicator - Main Flask Application
GPU-accelerated video duplicate finder with comprehensive web UI.
"""
import os
import sys

# Ensure local dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import base64
import logging
import threading
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file

from scanner import scan_folders
from extractor import extract_frames, extract_thumbnail, get_video_info
from hasher import VideoHasher
from comparator import find_duplicate_groups

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comparer.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Global State ─────────────────────────────────────────────────────────────
scan_state = {
    'status': 'idle',          # idle, scanning, extracting, hashing, processing, comparing, done, error, paused, aborted
    'progress': 0,
    'total': 0,
    'current_file': '',
    'message': '',
    'start_time': None,
    'elapsed': 0,
    'folders': [],
    'videos': [],
    'groups': [],
    'stats': {},
    'threshold': 0.88,
    'error': None,
}
scan_lock = threading.RLock()
hasher_instance: Optional[VideoHasher] = None

scan_thread = None
abort_flag = False
pause_flag = False
SESSION_FILE = 'session_state.pkl'

# Persistent data across resumes
current_videos = []
current_fingerprints = []
current_params = {}

def save_session():
    global current_videos, current_fingerprints, scan_state, current_params
    try:
        import pickle
        with scan_lock:
            state_copy = dict(scan_state)
            state_copy['start_time'] = 0  # not useful to resume
        with open(SESSION_FILE, 'wb') as f:
            pickle.dump({
                'videos': current_videos,
                'fingerprints': current_fingerprints,
                'scan_state': state_copy,
                'params': current_params
            }, f)
    except Exception as e:
        logger.error(f"Failed to save session: {e}")

def load_session():
    global current_videos, current_fingerprints, scan_state, current_params
    import pickle
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'rb') as f:
                data = pickle.load(f)
            current_videos = data.get('videos', [])
            current_fingerprints = data.get('fingerprints', [])
            current_params = data.get('params', {})
            loaded_state = data.get('scan_state', {})
            with scan_lock:
                scan_state.update(loaded_state)
                # If it was interrupted mid-process, it should load as paused
                if scan_state['status'] in ('processing', 'scanning', 'extracting', 'hashing', 'comparing'):
                    scan_state['status'] = 'paused'
                    scan_state['message'] = 'Scan paused. Ready to resume.'
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            try:
                os.remove(SESSION_FILE)
            except:
                pass

# Attempt to load session at startup
load_session()


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def run_scan(folders: List[str], threshold: float, num_frames: int, batch_size: int, is_resume: bool = False):
    """Background task to discover videos, hash them, and group them."""
    global abort_flag, pause_flag, hasher_instance, current_videos, current_fingerprints, current_params, scan_state

    try:
        from scanner import scan_folders
        from extractor import extract_frames, get_video_info
        from hasher import VideoHasher
        from comparator import find_duplicate_groups

        if not is_resume:
            # ── Phase 1: Scan Directory ───────────────────────────────────
            with scan_lock:
                scan_state['status'] = 'scanning'
                scan_state['start_time'] = time.time()
                scan_state['folders'] = folders
                scan_state['threshold'] = threshold
                scan_state['error'] = None
            
            logger.info(f"Scanning folders: {folders}")
            videos = scan_folders(folders)
            current_videos = videos
            current_fingerprints = []
            
            # Use current_params to store the run settings
            current_params['num_frames'] = num_frames
            
            if not videos:
                with scan_lock:
                    scan_state['status'] = 'done'
                    scan_state['message'] = 'No video files found in selected folders.'
                    scan_state['videos'] = []
                    scan_state['groups'] = []
                return

            with scan_lock:
                scan_state['total'] = len(videos)
                scan_state['message'] = f"Found {len(videos)} videos. Gathering metadata..."
            
            # Gather metadata quickly
            for i in range(len(videos)):
                if abort_flag: break
                while pause_flag and not abort_flag: time.sleep(0.5)

                vid = videos[i]
                # Only fetch if not already present (safeguard)
                if 'duration' not in vid:
                    info = get_video_info(vid['path'])
                    if info:
                        vid.update({
                            'duration': info['duration'],
                            'width': info['width'],
                            'height': info['height'],
                            'resolution': f"{info['width']}x{info['height']}",
                            'duration_str': format_duration(info['duration'])
                        })
                    else:
                        vid.update({'duration': 0, 'width': 0, 'height': 0, 'resolution': 'Unknown', 'duration_str': '0:00'})
                    vid['size_str'] = format_size(vid['size'])
                
                with scan_lock:
                    scan_state['progress'] = i + 1
                    scan_state['current_file'] = vid['name']
                    # Save progress during metadata phase too
                    if i % 20 == 0: save_session()
        
        else:
            # We are resuming from where we left off
            with scan_lock:
                scan_state['status'] = 'processing'
                if scan_state['start_time'] == 0:
                    scan_state['start_time'] = time.time() - scan_state.get('elapsed', 0)

        videos = current_videos
        
        # ── Phase 2: Massive Parallel Extract and Hash ──────────────────
        if hasher_instance is None:
            hasher_instance = VideoHasher(batch_size=128)

        # Batch videos for processing (concurrency)
        batch_v_size = 8
        import concurrent.futures
        
        # Ensure num_frames is set from params if resuming
        if is_resume and 'num_frames' in current_params:
            num_frames = current_params['num_frames']

        start_index = len(current_fingerprints)
        
        # We start from start_index and process in batches
        for i in range(start_index, len(videos), batch_v_size):
            if abort_flag: break
            while pause_flag and not abort_flag: time.sleep(0.5)

            # Ensure we don't go out of bounds
            batch_chunk = videos[i : min(i + batch_v_size, len(videos))]
            
            with scan_lock:
                scan_state['status'] = 'processing'
                scan_state['total'] = len(videos)  # Ensure total is always correct
                scan_state['progress'] = min(i + batch_v_size, len(videos))
                scan_state['current_file'] = f"Processing batch {i//batch_v_size + 1}..."
                elapsed = time.time() - scan_state['start_time']
                scan_state['elapsed'] = elapsed
                if i > 0:
                    eta = (elapsed / i) * (len(videos) - i)
                    scan_state['message'] = f'Matching videos: {scan_state["progress"]}/{len(videos)} (ETA: {format_duration(eta)})'
                else:
                    scan_state['message'] = f'Start matching ({len(videos)} videos)...'

            # 1. Concurrent Extraction (Threaded FFmpeg)
            all_video_frames = [[] for _ in range(len(batch_chunk))]
            
            def extract_batch_v(idx, path):
                all_video_frames[idx] = extract_frames(path, num_frames=num_frames)

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_v_size) as executor:
                futures = [executor.submit(extract_batch_v, j, vid['path']) for j, vid in enumerate(batch_chunk)]
                concurrent.futures.wait(futures)

            # 2. Massive GPU Fingerprinting (Batch across videos)
            fps = hasher_instance.compute_batch_fingerprints(all_video_frames)
            current_fingerprints.extend(fps)
            
            # Periodically save state
            save_session()

        if abort_flag:
            with scan_lock:
                scan_state.update({
                    'status': 'idle',
                    'progress': 0,
                    'total': 0,
                    'current_file': '',
                    'message': 'Scan aborted. System ready.',
                    'start_time': None,
                    'elapsed': 0,
                    'videos': [],
                    'groups': [],
                    'stats': {},
                    'error': None,
                })
            # Clean up persistence
            current_videos.clear()
            current_fingerprints.clear()
            try:
                if os.path.exists(SESSION_FILE):
                    os.remove(SESSION_FILE)
            except: pass
            return

        # ── Phase 4: Compare & Group ───────────────────────────────────
        with scan_lock:
            scan_state['status'] = 'comparing'
            scan_state['message'] = 'Computing similarity matrix and finding duplicates...'

        groups = find_duplicate_groups(videos, current_fingerprints, threshold=threshold)

        # ── Done ───────────────────────────────────────────────────────
        total_dupes = sum(g['count'] for g in groups)
        total_size = sum(
            v['size'] for g in groups for v in g['videos'][1:]  # skip the "original"
        )

        with scan_lock:
            scan_state['status'] = 'done'
            scan_state['videos'] = videos
            scan_state['groups'] = groups
            scan_state['elapsed'] = time.time() - scan_state['start_time']
            scan_state['stats'] = {
                'total_videos': len(videos),
                'duplicate_groups': len(groups),
                'total_duplicates': total_dupes,
                'potential_savings': format_size(total_size),
                'potential_savings_bytes': total_size,
                'elapsed': format_duration(scan_state['elapsed']),
                'device': str(hasher_instance.device) if hasher_instance else 'N/A',
            }
            scan_state['message'] = (
                f'Done! Found {len(groups)} duplicate groups '
                f'({total_dupes} files, {format_size(total_size)} recoverable)'
            )

        logger.info(f"Scan complete: {len(groups)} groups, {total_dupes} duplicates")
        
        # Scan finished completely, we can delete the resume session file
        try:
            if os.path.exists(SESSION_FILE):
                os.remove(SESSION_FILE)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Scan failed")
        with scan_lock:
            scan_state['status'] = 'error'
            scan_state['error'] = str(e)
            scan_state['message'] = f'Error: {str(e)}'


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start a new scan or resume a paused one."""
    global scan_thread, abort_flag, pause_flag, current_params
    
    data = request.get_json() or {}
    is_resume = data.get('resume', False)

    if not is_resume:
        folders = data.get('folders', [])
        threshold = float(data.get('threshold', 0.88))
        num_frames = int(data.get('num_frames', 32))
        batch_size = int(data.get('batch_size', 128))

        if not folders:
            return jsonify({'error': 'No folders selected'}), 400

        # Validate folders exist
        valid_folders = [f for f in folders if os.path.isdir(f)]
        if not valid_folders:
            return jsonify({'error': 'No valid folders found'}), 400
            
        current_params = {
            'folders': valid_folders,
            'threshold': threshold,
            'num_frames': num_frames,
            'batch_size': batch_size
        }
    else:
        # Load params from current_params if they exist
        if not current_params:
            return jsonify({'error': 'No paused session found to resume'}), 400

    # Ensure thread isn't running
    if scan_thread and scan_thread.is_alive():
        # If it's just paused, we unpause it
        if pause_flag and is_resume:
            pause_flag = False
            return jsonify({'status': 'resumed'})
        return jsonify({'error': 'Scan already in progress'}), 400

    with scan_lock:
        if scan_state['status'] not in ('idle', 'done', 'error'):
            return jsonify({'error': 'Scan already in progress'}), 409

        scan_state['status'] = 'starting'
        scan_state['groups'] = []
        scan_state['videos'] = []
        scan_state['stats'] = {}

    # Safety reset of flags
    abort_flag = False
    pause_flag = False
    
    # Folders are not needed in call if resuming (handled by global current_videos)
    folders_to_scan = valid_folders if not is_resume else []

    scan_thread = threading.Thread(
        target=run_scan,
        args=(folders_to_scan, threshold, num_frames, batch_size, is_resume),
        daemon=True
    )
    scan_thread.start()

    return jsonify({'status': 'started', 'folders': valid_folders if not is_resume else current_params.get('folders', [])})


@app.route('/api/status')
def get_status():
    """Get current scan status."""
    gpu_util = 0.0
    try:
        # Check GPU usage quickly
        res = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
            encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        gpu_util = float(res.strip().split('\n')[0])
    except Exception:
        pass

    with scan_lock:
        return jsonify({
            'status': scan_state['status'],
            'progress': scan_state['progress'],
            'total': scan_state['total'],
            'current_file': scan_state['current_file'],
            'message': scan_state['message'],
            'elapsed': scan_state.get('elapsed', 0),
            'gpu_util': gpu_util
        })


@app.route('/api/results')
def get_results():
    """Get scan results."""
    with scan_lock:
        # Sanitize paths and prepare for JSON
        groups = []
        for g in scan_state['groups']:
            group = dict(g)
            group['videos'] = []
            for v in g['videos']:
                vid = {k: v[k] for k in [
                    'path', 'name', 'size', 'size_str', 'folder',
                    'duration', 'duration_str', 'resolution',
                    'width', 'height', 'index'
                ] if k in v}
                if 'match_score' in v:
                    vid['similarity'] = v['match_score']
                elif 'similarity' in v:
                    vid['similarity'] = v['similarity']
                group['videos'].append(vid)
            groups.append(group)

        all_videos = []
        for v in scan_state['videos']:
            vid = {k: v[k] for k in [
                'path', 'name', 'size', 'size_str', 'folder',
                'duration', 'duration_str', 'resolution',
                'width', 'height', 'index'
            ] if k in v}
            all_videos.append(vid)

        return jsonify({
            'groups': groups,
            'videos': all_videos,
            'stats': scan_state['stats'],
            'total_videos': len(scan_state['videos']),
        })


@app.route('/api/thumbnail/<int:video_index>')
def get_thumbnail(video_index):
    """Get a JPEG thumbnail for a video."""
    with scan_lock:
        videos = scan_state['videos']
        if video_index < 0 or video_index >= len(videos):
            return '', 404
        video_path = videos[video_index]['path']

    thumb_data = extract_thumbnail(video_path)
    if thumb_data:
        return thumb_data, 200, {'Content-Type': 'image/jpeg'}
    return '', 404


@app.route('/api/open-folder', methods=['POST'])
def open_folder():
    """Open file location in Windows Explorer."""
    data = request.get_json()
    file_path = data.get('path', '')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        # Open Windows Explorer with the file selected
        subprocess.Popen(
            f'explorer /select,"{file_path}"',
            shell=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/open-file', methods=['POST'])
def open_file():
    """Open a video file with the default player."""
    data = request.get_json()
    file_path = data.get('path', '')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        os.startfile(file_path)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete', methods=['POST'])
def delete_file():
    """Move a file to recycle bin (or delete if not possible)."""
    data = request.get_json()
    file_path = data.get('path', '')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        # Try to use send2trash for recycle bin support
        try:
            from send2trash import send2trash
            send2trash(file_path)
        except ImportError:
            os.remove(file_path)

        return jsonify({'status': 'deleted', 'path': file_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/abort', methods=['POST'])
def abort_scan_route():
    global abort_flag, pause_flag
    abort_flag = True
    pause_flag = False
    return jsonify({'status': 'aborting'})

@app.route('/api/hard-reset', methods=['POST'])
def hard_reset_route():
    """Wipe everything, including folder list and parameters."""
    global current_videos, current_fingerprints, current_params, abort_flag, pause_flag
    abort_flag = True
    pause_flag = False
    with scan_lock:
        scan_state.update({
            'status': 'idle',
            'progress': 0,
            'total': 0,
            'current_file': '',
            'message': 'System factory reset complete.',
            'start_time': None,
            'elapsed': 0,
            'folders': [],
            'videos': [],
            'groups': [],
            'stats': {},
            'threshold': 0.88,
            'error': None,
        })
    current_videos.clear()
    current_fingerprints.clear()
    current_params.clear()
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
    except: pass
    return jsonify({'status': 'hard-reset'})

@app.route('/api/reset', methods=['POST'])
def reset_scan():
    """Reset scan state except folders."""
    global current_videos, current_fingerprints
    with scan_lock:
        scan_state.update({
            'status': 'idle',
            'progress': 0,
            'total': 0,
            'current_file': '',
            'message': '',
            'start_time': None,
            'elapsed': 0,
            # Folders and threshold are preserved
            'videos': [],
            'groups': [],
            'stats': {},
            'error': None,
        })
    current_videos.clear()
    current_fingerprints.clear()
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
    except: pass
    return jsonify({'status': 'reset'})


@app.route('/api/ffmpeg-status')
def ffmpeg_status():
    """Check if FFmpeg/FFprobe are available."""
    from extractor import check_ffmpeg, FFMPEG_BIN, FFPROBE_BIN
    status = check_ffmpeg()
    return jsonify(status)


@app.route('/api/system-info')
def system_info():
    """Get system GPU/CUDA info."""
    import torch
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': [],
        'torch_version': torch.__version__,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'name': props.name,
                'total_memory': format_size(props.total_memory),
                'compute_capability': f"{props.major}.{props.minor}",
            })
        info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info['cuda_version'] = torch.version.cuda

    return jsonify(info)


# ── Folder Picker API for pywebview ─────────────────────────────────────────

class FolderPickerApi:
    """Exposed to JS via pywebview so we can open a native folder dialog."""

    def pick_folder(self):
        """Open native folder browser and return the selected path, or None."""
        try:
            import webview
            windows = webview.windows
            if not windows:
                return None
            # Support both old and new pywebview API
            try:
                dialog_type = webview.FileDialog.FOLDER
            except AttributeError:
                dialog_type = webview.FOLDER_DIALOG  # legacy < 4.x
            result = windows[0].create_file_dialog(dialog_type)
            if result and len(result) > 0:
                return result[0]
        except Exception as e:
            logger.error(f"pick_folder error: {e}")
        return None


# ── Fallback route when running in plain browser (not pywebview) ─────────────

@app.route('/api/pick-folder', methods=['POST'])
def pick_folder_route():
    """Fallback: open a Tkinter folder dialog when not using pywebview."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title='Select a folder to scan')
        root.destroy()
        if folder:
            return jsonify({'path': folder})
        return jsonify({'path': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = 5599
    logger.info(f"Starting Video Deduplicator on http://localhost:{port}")

    try:
        import webview

        # Expose native folder-picker to JavaScript
        api = FolderPickerApi()

        # Start Flask in a daemon thread
        flask_thread = threading.Thread(
            target=lambda: app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False),
            daemon=True
        )
        flask_thread.start()
        # Give Flask a moment to bind
        time.sleep(1.0)

        logger.info("Initializing Desktop UI (pywebview)...")
        window = webview.create_window(
            'Video Deduplicator',
            f'http://localhost:{port}',
            width=1440, height=900,
            min_size=(1024, 700),
            js_api=api,
        )
        webview.start(debug=False)

    except ImportError:
        logger.info("pywebview not found, falling back to webbrowser...")
        import webbrowser

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f'http://localhost:{port}')

        threading.Thread(target=open_browser, daemon=True).start()
        app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
