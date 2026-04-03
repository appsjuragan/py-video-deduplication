# GPU-Accelerated Video Deduplicator

A high-performance desktop application for finding and removing duplicate video files. It leverages PyTorch, TorchVision, and CUDA to quickly analyze video content through a pre-trained ResNet-18 model, making it highly effective at detecting videos that have been resized, compressed, or slightly altered.

![Video Deduplicator UI Preview](static/img/validating_files.gif)

## Features

- **Blazing Fast GPU Acceleration:** Uses CUDA for neural network-based video frame fingerprinting.
- **Content-Based Similarity:** Uses image hashing (perceptual hashing over ResNet embeddings/features) rather than basic file hashing (MD5/SHA), allowing it to detect visually identical videos regardless of format or compression.
- **Advanced Filtering:** Skip small files (Min/Max size filters) and select specific video file types (MP4, MKV, AVI, etc.) to optimize the scan process.
- **Interactive UI:** Built using a local Python backend (Flask) with an elegant, responsive frontend inside a native desktop window (using `pywebview`).
- **One-Click Bulk Delete:** Auto-selects the "Best" video based on resolution and size, letting you batch delete duplicates securely to the Recycle Bin.
- **Smart Pause & Resume:** Need to shut down? Pause long scans and resume later without losing progress.

## Architecture

- **Backend:** Python + Flask, handling background threads for scanning and extracting.
- **Deep Learning Layer:** `onnxruntime` or PyTorch (via torchvision models) applied to frames extracted via `ffmpeg`.
- **Frontend:** Vanilla JS, CSS (Grid+Flexbox design), and HTML in an electron-like desktop wrapper using `pywebview`.

## Requirements

1. **Python 3.9+** (Tested to work perfectly in portable embedded environments).
2. **FFmpeg:** Must be installed and available on your system `PATH`.
3. **NVIDIA GPU (CUDA):** Designed for systems with CUDA. Performance relies on the `torch` module built with CUDA support.

### System Path Check
The application validates the `ffmpeg` presence at runtime. If not found, you will get a clear error right inside the application header allowing you to correct the configuration.

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
2. **Tweak Settings:** Drop the similarity threshold to catch partial clips, or increase frames/batch size for more rigorous speed parameters.
3. **Scan:** Click **Start Scan** and watch the process in the Progress Panel. You can view the extracted video thumbnails mid-scan.
4. **Compare:** Evaluate duplicate groups. The app automatically flags the lower-quality videos for deletion.
5. **Clean:** Click "Delete Checked" to send duplicate files safely to the trash.

## License
MIT License.
