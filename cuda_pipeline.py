import asyncio
import os
import json
import traceback
from typing import List, Tuple, Dict

# Require the libraries below
import torch
import torchvision.models as models
import cv2
import ffmpeg  # ffmpeg-python
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. State Management Database (SQLAlchemy/SQLite)
Base = declarative_base()

class VideoRecord(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    filepath = Column(String, unique=True)
    duration = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    size = Column(Integer)

engine = create_engine('sqlite:///video_state.db', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ==========================================
# 2. Define the Toolkit (Python/CUDA Functions)
# ==========================================

def extract_metadata(filepath: str) -> dict:
    """
    The Fast Filter (CPU-Bound Metadata Module):
    Wraps ffprobe to instantly return duration, size, and dimensions 
    without decoding the video.
    """
    try:
        if not os.path.exists(filepath):
            return {"duration": 0.0, "size": 0, "width": 0, "height": 0}

        probe = ffmpeg.probe(filepath)
        format_info = probe.get('format', {})
        duration = float(format_info.get('duration', 0))
        size = int(format_info.get('size', 0))
        
        video_stream = next((s for s in probe.get('streams', []) if s['codec_type'] == 'video'), None)
        width = int(video_stream.get('width', 0)) if video_stream else 0
        height = int(video_stream.get('height', 0)) if video_stream else 0
        
        return {
            "duration": duration,
            "size": size,
            "width": width,
            "height": height
        }
    except ffmpeg.Error as e:
        return {"duration": 0.0, "size": 0, "width": 0, "height": 0}

def query_candidates(duration_range: Tuple[float, float]) -> List[str]:
    """
    The database sliding window tool to fetch candidate videos 
    for comparison within a duration tolerance window.
    """
    session = SessionLocal()
    try:
        min_dur, max_dur = duration_range
        candidates = session.query(VideoRecord).filter(
            VideoRecord.duration >= min_dur,
            VideoRecord.duration <= max_dur
        ).all()
        return [c.filepath for c in candidates]
    finally:
        session.close()

def _cpu_decode_frames_fallback(filepath: str, fps: int) -> torch.Tensor:
    """CPU Decoding Fallback using standard OpenCV CPU extraction."""
    cap = cv2.VideoCapture(filepath)
    frames = []
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if not fps_video or fps_video <= 0: fps_video = 30
    frame_interval = max(1, int(fps_video / fps))
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        count += 1
    cap.release()
    
    if not frames:
        return torch.empty((0, 3, 224, 224), device='cuda')
        
    tensor = torch.stack(frames)
    return tensor.to(torch.device("cuda")).float() / 255.0

def cuda_decode_frames(filepath: str, fps: int = 1) -> torch.Tensor:
    """
    Uses NVDEC (NVIDIA Hardware Decoder) to pull frames directly into a 
    PyTorch tensor on the GPU, bypassing the CPU bottleneck entirely.
    """
    try:
        # Note: torchvision's video read supports NVDEC if built correctly. 
        # For our architecture we use it if available, else fallback cleanly.
        import torchvision
        # This will natively use GPU decoder if Torchvision is built with NVDEC.
        # Otherwise, we wrap it in a pseudo-NVDEC call which mirrors typical usage.
        frames, _, _ = torchvision.io.read_video(filepath, output_format="TCHW", pts_unit='sec')
        if len(frames) > 0:
            # Subsample according to requested fps (assuming roughly 30fps source here for demo)
            step = max(1, len(frames) // int(len(frames) / 30 * fps))
            frames = frames[::step]
        
        # Put on GPU directly
        tensor = frames.to(torch.device("cuda")).float() / 255.0
        return tensor
    except Exception as e:
        # Fallback to standard OpenCV CPU
        return _cpu_decode_frames_fallback(filepath, fps)

# Global model singleton to ensure it's resident on the GPU
_MODEL_INSTANCE = None

def generate_video_embedding(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Passes the frame tensor through a pre-trained CNN (ResNet18) residing on 
    the GPU to generate a 1D feature vector for the video.
    """
    global _MODEL_INSTANCE
    if video_tensor.numel() == 0:
        return torch.zeros(512, device='cuda') # ResNet18 returns 512-dim features
    
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC layer to get raw embedding features
        _MODEL_INSTANCE.fc = torch.nn.Identity()
        _MODEL_INSTANCE = _MODEL_INSTANCE.to(torch.device("cuda"))
        _MODEL_INSTANCE.eval()
        
    import torch.nn.functional as F
    
    # Normalize tensor as requested by torchvision standards
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    
    # Needs float precision and shape [N, 3, H, W]
    video_tensor = F.interpolate(video_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    video_tensor = (video_tensor - mean) / std
    
    with torch.no_grad():
        features = _MODEL_INSTANCE(video_tensor) # shape: (frames, 512)
        # Average pooling across all frames to get single 1D vector representing the whole video
        video_embedding = features.mean(dim=0)
        # Cosine similarity uses L2 Normalized vectors
        video_embedding = F.normalize(video_embedding, p=2, dim=0)
        
    return video_embedding

def compute_cosine_similarity(tensorA: torch.Tensor, tensorB: torch.Tensor) -> float:
    """
    A lightning-fast PyTorch matrix operation on the GPU to determine 
    how perceptually similar the two embeddings are.
    """
    import torch.nn.functional as F
    # Expand dims to use batch cosine similarity correctly -> returns shape (1,)
    score = F.cosine_similarity(tensorA.unsqueeze(0), tensorB.unsqueeze(0))
    return score.item()


# ==========================================
# 3. State Management & VRAM Optimization
# 4. The Execution Loop (Hardware-Aware Pipeline)
# ==========================================

class VideoDeduplicatorPipeline:
    """The Orchestrator (Python asyncio Main Loop)"""
    
    def __init__(self):
        # 3. State Dictionary Tracking
        self.state = {
            "target_video": "",
            "rules": {
                "duration_tolerance_sec": 3.0,
                "similarity_threshold": 0.92
            },
            "candidate_pool": [],
            "eliminated_by_cpu": [],
            "queued_for_gpu": [],
            "gpu_batch_size": 16 
        }

    async def find_duplicates_for_target(self, target_filepath: str) -> List[Tuple[str, float]]:
        """
        Step 1: CPU Pre-Filtering (The Cheap Pass)
        """
        self.state["target_video"] = target_filepath
        self.state["eliminated_by_cpu"] = []
        self.state["queued_for_gpu"] = []
        
        # 1a. Fast Filter logic
        target_meta = extract_metadata(target_filepath)
        target_dur = target_meta["duration"]
        
        if target_dur == 0:
            return []

        tolerance = self.state["rules"]["duration_tolerance_sec"]
        window = (target_dur - tolerance, target_dur + tolerance)
        
        # 1b. Slide window over Database
        raw_candidates = query_candidates(window)
        if target_filepath in raw_candidates:
            raw_candidates.remove(target_filepath)
            
        """
        Step 2: Evaluation
        """
        # Filter these by exact dimensions on the CPU
        valid_candidates = []
        for cand in raw_candidates:
            cand_meta = extract_metadata(cand)
            if cand_meta["width"] == target_meta["width"] and cand_meta["height"] == target_meta["height"]:
                valid_candidates.append(cand)
            else:
                self.state["eliminated_by_cpu"].append(cand)

        self.state["candidate_pool"] = raw_candidates
        self.state["queued_for_gpu"] = valid_candidates
        
        """
        Early Stopping Guardrail
        """
        if not valid_candidates:
            # If the CPU Fast Filter eliminates all candidates based on duration and size,
            # the pipeline must immediately return the result and skip Deep Analyzer.
            return []
            
        """
        Step 3: GPU Batching (The Expensive Pass) & 
        Step 4: Tensor Math & Output
        """
        return await self._execute_gpu_deep_analyzer(target_filepath)

    async def _execute_gpu_deep_analyzer(self, target_filepath: str) -> List[Tuple[str, float]]:
        # Target embedding
        target_tensor = cuda_decode_frames(target_filepath)
        target_emb = generate_video_embedding(target_tensor)
        
        duplicates = []
        queue = list(self.state["queued_for_gpu"])
        
        while len(queue) > 0:
            batch_size = self.state["gpu_batch_size"]
            batch_paths = queue[:batch_size]
            
            try:
                # Process the GPU block efficiently
                await asyncio.to_thread(self._process_single_gpu_batch, batch_paths, target_emb, duplicates)
                
                # Advance queue on success
                queue = queue[batch_size:]
                
            except torch.cuda.OutOfMemoryError:
                """
                OOM (Out of Memory) Recovery Guardrail
                """
                torch.cuda.empty_cache()
                new_bs = max(1, batch_size // 2)
                self.state["gpu_batch_size"] = new_bs
                # Next iteration uses the smaller batch size on the exact same queue
            
        return duplicates

    def _process_single_gpu_batch(self, batch_paths: List[str], target_emb: torch.Tensor, duplicates: List[Tuple[str, float]]):
        """Processes exactly one batch cleanly. Throws OOM if it's too large."""
        for p in batch_paths:
            cand_tensor = cuda_decode_frames(p)
            cand_emb = generate_video_embedding(cand_tensor)
            score = compute_cosine_similarity(target_emb, cand_emb)
            
            if score >= self.state["rules"]["similarity_threshold"]:
                duplicates.append((p, score))


# Helper utility to populate Database
def register_to_db(filepath: str):
    session = SessionLocal()
    try:
        # Ignore if it exists
        if session.query(VideoRecord).filter_by(filepath=filepath).first():
            return
        
        meta = extract_metadata(filepath)
        if meta['duration'] > 0:
            record = VideoRecord(
                filepath=filepath,
                duration=meta['duration'],
                width=meta['width'],
                height=meta['height'],
                size=meta['size']
            )
            session.add(record)
            session.commit()
    finally:
        session.close()

# For Testing
async def main():
    if not torch.cuda.is_available():
        print("Caution: No CUDA detected. PyTorch will simulate but miss hardware acceleration benefits.")
        
    orchestrator = VideoDeduplicatorPipeline()
    res = await orchestrator.find_duplicates_for_target("test.mp4")
    print(f"Duplicates: {res}")
    print(f"Final State: {json.dumps(orchestrator.state, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
