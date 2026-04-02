# Video Deduplicator

A GPU-accelerated video duplicate finder that uses deep learning (EfficientNet-B0 + PyTorch) to find near-identical videos, even if they have been resized, cropped, re-encoded, or trimmed.

## Prerequisites
1. Python 3.9+
2. CUDA toolkit (install via PyTorch)
3. FFmpeg installed on your system and added to your PATH environment variable.

## Setup
1. Create a virtual environment or use global python
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

A browser window will open automatically at `http://localhost:5599`.

## Features
- **Deep Hash Matching**: Instead of file-hash (MD5), it computes cognitive hashes using a neural network on evenly spaced frames, allowing 98%+ accuracy on visual similarity.
- **VRAM Batching**: Batches frames into GPU VRAM for the highest inference throughput possible.
- **Interactive UI**: Web-based preview UI allowing you to see the duplicate match score, preview duration/resolution/size, launch the file physically in your system player, open it in explorer, or delete it (moves to Windows Recycle Bin if possible).
