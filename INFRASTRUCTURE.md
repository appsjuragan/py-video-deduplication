# Infrastructure & Pipeline Architecture

This document details the hardware-aware design and data flow of the Video Deduplicator. The entire system is built to minimize CPU-GPU bottlenecks by keeping the GPU fed continuously using overlapped asynchronous execution and discarding obvious non-duplicates before decoding even starts.

## Architecture Diagram

```mermaid
graph TD
    classDef cpu fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef gpu fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    classDef storage fill:#6b7280,stroke:#374151,stroke-width:2px,color:#fff
    classDef fail fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff

    UI[🖥️ pywebview / Vanilla JS UI]:::cpu --> API[🐍 Flask Backend / app.py]:::cpu
    API --> DB[(SQLAlchemy Cache)]:::storage

    subgraph Phase 1: O(1) CPU Fast-Filter
        API --> Prob[ffprobe Metadata Extractor]:::cpu
        Prob -- Retrieves Duration & Resolution --> Filter{Resolution Match?}
        Filter -- Unique / No Match --> EarlyStop[Discard early. Save VRAM]:::fail
    end

    subgraph Phase 2: Asynchronous Pipelining (Double-Buffering)
        Filter -- Match possible --> ThreadPool[Concurrent CPU Extraction Pool]:::cpu
        ThreadPool -- Extract N Frames --> FFmpeg[24x FFmpeg Workers]:::cpu
        FFmpeg --> Buffer[RAM Frame Buffer Layer]:::storage
        
        Buffer -- Prefetched Batch N+1 --> GPUStream[CUDA Non-Blocking Stream]:::gpu
        GPUStream -- Feeds -- FP16 Tensors --> TensorCores(NVIDIA Tensor Cores):::gpu
    end

    subgraph GPU-Bound Semantic Analysis
        TensorCores -- Batch size dynamically tuned via VRAM probe --> EfficientNet[EfficientNet-B0 Feature Extractor]:::gpu
        EfficientNet --> Norm[L2 Normalization]:::gpu
        Norm --> FeatureSpace[1280-Dim Vectors]:::storage
    end

    subgraph Phase 3: Vectorized Similarity
        FeatureSpace --> Numpy[Vectorized Chamfer Metric]:::cpu
        Numpy --> Threshold{Similarity > 95%?}
        Threshold -- Yes --> Groups[Duplicate Video Groups]:::storage
        Threshold -- No --> Safe[Unique Video]:::fail
    end

    Groups --> API
```

## Core Pipeline Flow

### 1. The Orchestrator (`app.py`)
Responsible for scanning directories, orchestrating threads, and handling the double-buffered GPU feedback loop. It ensures that background worker contention (OpenMP/MKL thread collisions) is avoided by strict environment overrides.

### 2. The Initial Screen (CPU Fast-Filter) (`extractor.py`)
Using lightweight `ffprobe` bindings, the system groups all source files by intrinsic metadata (resolution, ratio, duration).
**Advantage:** If a file has an entirely unique resolution and aspect ratio against the rest of the pool, it is statistically impossible to be an exact duplicate. It is immediately pruned from the queue, saving seconds of heavy ffmpeg CPU extraction and Tensor Core computation.

### 3. Pipeline Double Buffering (`app.py` / `hasher.py`)
To prevent the GPU from idling while the CPU runs `ffmpeg` extractions:
*   **Batch $N$** is processed by the Torch neural network.
*   Concurrently, a thread pool of 24 ffmpeg workers uses NVDEC/CPU decoding to fetch raw RGB frames into RAM for **Batch $N+1$**.
*   Using PyTorch CUDA streams (`torch.cuda.Stream`), tensors are streamed onto the GPU asynchronously.

### 4. GPU Subsystem Optimization
*   **Dynamic VRAM Sizing:** During `init()`, the `VideoHasher` uses `pynvml` to probe free VRAM capacity. It dynamically sets the batch size to consume up to 90% of available free overhead, making it immediately compatible seamlessly across RTX 3x/4x/5x architectures.
*   **Mixed Precision:** Using `model.half()`, the neural net operates on FP16 types. This slashes memory footprints by 50% and doubles computational throughput by utilizing specialized NVIDIA Tensor Cores.
*   **OOM Resiliency:** Catching `torch.cuda.OutOfMemoryError`, the algorithm can clear the cache, halve the batch size, and immediately retry the segment, ensuring the server never crashes even if VRAM gets unpredictably saturated by OS spikes.

### 5. Vectorized Distance Mapping (`comparator.py`)
Instead of slow $O(n^2)$ nested loops performing `scipy.spatial.cosine` checks, the comparator engine pivots to vectorizing operations via fast NumPy matrix multiplication. It computes an adjusted chamfer distance matrix globally across the entire space output, resulting in near-instant clustering of visually identical groups.
