"""
GPU-accelerated video fingerprinting using PyTorch.
Extracts deep feature vectors from video frames using a pretrained CNN,
processes everything in batches on the best available compute device.
Device priority: CUDA → Vulkan (experimental) → CPU.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoHasher:
    """GPU-accelerated video fingerprinter using EfficientNet features."""

    def __init__(self, device: Optional[str] = None, batch_size: int = 128):
        self.batch_size = batch_size
        self.backend_name: str = "cpu"  # human-readable label set below

        if device is None:
            self.device, self.backend_name = self._pick_best_device(batch_size)
        else:
            self.device = torch.device(device)
            self.backend_name = device

        # Use EfficientNet-B0 for robust semantic features
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Identity()
        self.model = self.model.to(self.device)
        
        # Performance Tuning: Enable cuDNN benchmarking for auto-tuning kernels
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Use Mixed Precision (FP16) for ~2x throughput on Tensor Cores
            self.model = self.model.half() 
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.feature_dim = 1280
        logger.info(f"VideoHasher initialized (EfficientNet-B0) on {self.device} [{self.backend_name}]")

    # ------------------------------------------------------------------
    # Device selection helper
    # ------------------------------------------------------------------

    def _pick_best_device(self, batch_size: int) -> Tuple[torch.device, str]:
        """Return the best (device, label) pair available on this machine.

        Priority: CUDA → Vulkan (experimental torch backend) → CPU.
        """
        # 1. CUDA / NVIDIA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)

            # Dynamic batch sizing: use free VRAM for optimal throughput
            # FP16 reduced MB_PER_FRAME significantly (~6 MB vs ~12 MB)
            MB_PER_FRAME = 6  # Headroom for FP16 EfficientNet-B0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_vram_mb = mem_info.free / 1024 ** 2
            except Exception:
                try:
                    free, _ = torch.cuda.mem_get_info(0)
                    free_vram_mb = free / 1024 ** 2
                except Exception:
                    free_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2 * 0.8

            # Be bolder with VRAM: Use 90% of free VRAM
            computed = max(32, int(free_vram_mb * 0.90 / MB_PER_FRAME))
            self.batch_size = max(batch_size, computed)
            logger.info(
                f"CUDA GPU: {gpu_name} ({free_vram_mb:.0f} MB free) "
                f"→ dynamic batch_size={self.batch_size}"
            )
            return device, f"cuda:{gpu_name}"

        # 2. Vulkan (experimental torch-vulkan backend, available in some nightly builds)
        try:
            if hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
                # Vulkan tensors live on CPU but ops run on GPU via Vulkan compute shaders.
                # We keep model on CPU and only move tensors to vulkan for the forward pass.
                logger.info("Vulkan compute available – using torch vulkan backend.")
                return torch.device("vulkan"), "vulkan"
        except Exception:
            pass

        # 3. CPU fallback
        logger.warning("No GPU acceleration available for PyTorch – using CPU.")
        return torch.device("cpu"), "cpu"

    def get_device_info(self) -> dict:
        """Return a dict describing the active compute backend."""
        info = {"backend": self.backend_name, "batch_size": self.batch_size}
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_mb"] = props.total_memory // 1024 ** 2
        return info

    def frames_to_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Convert list of PIL Images to a batched tensor."""
        tensors = [self.transform(f) for f in frames]
        return torch.stack(tensors)

    @torch.no_grad()
    def extract_features_batch(self, frames_batch: torch.Tensor) -> np.ndarray:
        """
        Extract feature vectors from a batch of frames on the active device.
        Returns L2-normalized vectors as a NumPy array.
        Includes OOM recovery: if VRAM runs out, batch_size is halved and retried.
        """
        features_list = []
        n = frames_batch.shape[0]
        i = 0
        working_bs = self.batch_size

        while i < n:
            try:
                # ── Asynchronous Stream ───────────────────────────────────
                with torch.cuda.stream(self.stream):
                    # Slice and move to device as FP16
                    batch = frames_batch[i:i + working_bs]
                    if self.device.type == "cuda":
                        batch = batch.to(self.device, non_blocking=True).half()
                    else:
                        batch = batch.to(self.device)

                    features = self.model(batch)          # (B, feature_dim)

                    # L2-normalize every frame feature
                    norms = torch.norm(features, p=2, dim=1, keepdim=True)
                    features = features / (norms + 1e-6)

                    # Accumulate results (transfer back to CPU)
                    features_list.append(features.cpu().float().numpy())
                    i += working_bs  # advance only on success

            except torch.cuda.OutOfMemoryError:
                # ── OOM Recovery ──────────────────────────────────────────
                del batch
                torch.cuda.empty_cache()
                new_bs = max(1, working_bs // 2)
                logger.warning(
                    f"CUDA OOM! Halving batch size: {working_bs} → {new_bs}"
                )
                working_bs = new_bs
                self.batch_size = new_bs  # persist for future calls
                continue   # retry the same slice with smaller batch

            finally:
                del batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return np.concatenate(features_list, axis=0) if features_list else np.empty((0, self.feature_dim))

    def compute_video_fingerprint(self, frames: List[Image.Image]) -> Optional[np.ndarray]:
        """
        Compute a single fingerprint vector for a video.
        Returns a (NumFrames, 1280) matrix.
        """
        if not frames:
            return None

        try:
            tensor = self.frames_to_tensor(frames)
            features = self.extract_features_batch(tensor)  # (K, 1280)
            return features
        except Exception as e:
            logger.error(f"Error computing fingerprint: {e}")
            return None

    def compute_batch_fingerprints(
        self, all_frames: List[List[Image.Image]]
    ) -> List[Optional[np.ndarray]]:
        """
        Compute fingerprints for multiple videos at once.
        Returns list of (FrameCount, Dim) matrices.
        """
        flat_frames = []
        video_indices = []
        for vid_idx, frames in enumerate(all_frames):
            if frames:
                for f in frames:
                    flat_frames.append(f)
                    video_indices.append(vid_idx)

        if not flat_frames:
            return [None] * len(all_frames)

        try:
            tensor = self.frames_to_tensor(flat_frames)
            all_features = self.extract_features_batch(tensor)
        except Exception as e:
            logger.exception(f"Batch feature extraction failed: {e}")
            return [None] * len(all_frames)

        # Re-group by video
        results = [[] for _ in range(len(all_frames))]
        for i, vid_idx in enumerate(video_indices):
            results[vid_idx].append(all_features[i])

        final_results = []
        for frames_feats in results:
            if frames_feats:
                final_results.append(np.array(frames_feats))
            else:
                final_results.append(None)

        return final_results
