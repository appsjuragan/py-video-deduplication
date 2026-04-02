"""
GPU-accelerated video fingerprinting using PyTorch + CUDA.
Extracts deep feature vectors from video frames using a pretrained CNN,
processes everything in batches on VRAM for maximum throughput.
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
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                
                # Maximizing VRAM usage with nvidia-ml-py
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free_vram_mb = mem_info.free / 1024**2
                    
                    # EfficientNet-B0 no_grad uses ~10MB per image
                    optimal_batch = int((free_vram_mb * 0.9) / 10)
                    self.batch_size = max(batch_size, optimal_batch)
                    logger.info(f"Using CUDA GPU: {gpu_name} ({free_vram_mb:.0f} MB Free VRAM) -> Auto batch_size={self.batch_size}")
                except Exception as e:
                    logger.warning(f"Could not use pynvml: {e}")
                    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    logger.info(f"Using CUDA GPU: {gpu_name} ({vram_mb:.0f} MB Total VRAM) -> batch_size={self.batch_size}")
            else:
                self.device = torch.device('cpu')
                logger.warning("CUDA not available, falling back to CPU")
        else:
            self.device = torch.device(device)

        # Use EfficientNet-B0 for robust semantic features
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Identity()
        self.model = self.model.to(self.device)
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
        logger.info(f"VideoHasher re-initialized (EfficientNet) on {self.device}")

    def frames_to_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Convert list of PIL Images to a batched tensor."""
        tensors = [self.transform(f) for f in frames]
        return torch.stack(tensors)

    @torch.no_grad()
    def extract_features_batch(self, frames_batch: torch.Tensor) -> np.ndarray:
        """
        Extract feature vectors from a batch of frames on GPU.
        Returns normalized vectors.
        """
        features_list = []
        n = frames_batch.shape[0]

        for i in range(0, n, self.batch_size):
            batch = frames_batch[i:i + self.batch_size].to(self.device)
            features = self.model(batch) # (B, 1280)
            
            # L2 Normalize every frame feature
            norms = torch.norm(features, p=2, dim=1, keepdim=True)
            features = features / (norms + 1e-6)
            
            features_list.append(features.cpu().numpy())

            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(features_list, axis=0)

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
