# GPU-Accelerated Video Deduplicator

A high-performance pipeline architecture desktop application for finding and removing duplicate video files. It leverages PyTorch, TorchVision, and CUDA to quickly analyze video content through a pre-trained EfficientNet-B0 model. 

It is highly robust for detecting videos that have been resized, compressed, or slightly visually altered, backed by an optimized hardware-aware pipeline designed to saturate RTX NVIDIA hardware using FP16 mixed precision and asynchronous double buffering.

![Video Deduplicator UI Preview](static/img/validating_files.gif)

## Features

- **Blazing Fast GPU Acceleration:** Uses CUDA Tensor Cores (FP16 mixed precision) for neural network-based video frame fingerprinting.
- **Asynchronous Pipelining:** Features a double-buffered architecture, aggressively extracting the next video batch via FFmpeg threading while the GPU simultaneously crunches the current batch.
- **Hardware-Aware Auto Tuning:** Probes runtime VRAM availability to dynamically scale tensor batch limits while enabling `cuDNN` benchmarking auto-tuning. Graceful OOM recoveries prevent crashes.
- **O(1) CPU Fast Filtering:** Prunes impossible matches strictly through zero-decode `ffprobe` metadata queries before ANY frames touch the deep learning layer.
- **Content-Based Similarity:** Employs Deep Feature embeddings (EfficientNet-B0) rather than basic file hashing, effectively detecting compressed, slightly altered, or differing format videos as exact matches.
- **Interactive UI:** Built using a Python backend (Flask) with an elegant, responsive frontend inside a native desktop window (using `pywebview`).
- **One-Click Bulk Delete:** Auto-selects the "Best" video based on resolution and size, letting you batch delete duplicates securely to the Recycle Bin.
- **Smart Pause & Resume:** Robust SQLite session caching allows for pause and resume mid-scan without losing structural progress.

## Architecture & Infrastructure

Check out the deep-dive on data flow, pipeline architecture, and structural optimizations inside [INFRASTRUCTURE.md](INFRASTRUCTURE.md).

## Requirements

1. **Python 3.9+** (Tested working seamlessly in embedded portable environments).
2. **FFmpeg & ffprobe:** Must be installed and available on your system `PATH`.
3. **NVIDIA GPU (CUDA):** Designed for systems with Compute Capability. Maximum performance requires the `torch` module built with CUDA support. (Will gracefully fallback to CPU if not found).

### System Path Check
The application validates the `ffmpeg` presence at runtime. If not found, you will get a clear error inside the application header allowing you to correct the configuration.

## Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:appsjuragan/py-video-deduplication.git
   cd py-video-deduplication
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Creating a Standalone Executable
You can bundle the entire application (including dependencies and deep learning models) into a standalone `.exe` using PyInstaller.

```bash
pyinstaller app.spec --clean
```
The resulting executable will be generated inside the `dist/` directory.

## Usage Guide
1. **Launch** the app and add paths to the Folders to Scan list.
2. **Tweak Settings:** Adjust the similarity threshold to catch partial clips, or increase frames. Batch sizes automatically scale to fit VRAM dynamically.
3. **Scan:** Click **Start Scan** and watch the fast-filter drop disparate files, while the remainder hit the GPU batch queue.
4. **Compare:** Evaluate duplicate groups. The app automatically flags the lower-quality or smaller videos for deletion.
5. **Clean:** Click "Delete Checked" to send duplicate files safely to the trash.

## License
MIT License.
