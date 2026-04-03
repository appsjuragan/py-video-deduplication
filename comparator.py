"""
Vectorized similarity computation and clustering for video fingerprints.
Uses Chamfer-style frame-to-frame matching for robust clip detection.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable

def compute_video_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Computes bidirectional Chamfer similarity between two frame feature sets.
    fp1/fp2: Matrices of shape (Frames, 1280)
    Returns: scalar float [0, 1]
    """
    if fp1 is None or fp2 is None:
        return 0.0
    
    # Cosine similarity matrix between all frames
    # Since features are normalized, dot product is cosine similarity
    # Add a tiny epsilon and clip to avoid NaN from floating point noise
    sim_mtx = np.dot(fp1, fp2.T)
    if sim_mtx.size == 0:
        return 0.0
    
    # Bidirectional matching:
    # How well does every frame in v1 find a match in v2?
    v1_to_v2 = np.mean(np.max(sim_mtx, axis=1))
    
    # How well does every frame in v2 find a match in v1?
    v2_to_v1 = np.mean(np.max(sim_mtx, axis=0))
    
    # Average them and clip to valid range
    score = (v1_to_v2 + v2_to_v1) / 2.0
    if np.isnan(score):
        return 0.0
        
    return float(max(0.0, min(1.0, score)))

def find_duplicate_groups(
    videos: List[Dict], 
    fingerprints: List[Optional[np.ndarray]], 
    threshold: float = 0.90,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict]:
    """
    Clusters videos into duplicate groups based on frame-wise Chamfer similarity.
    Highly optimized for large batches using vectorized NumPy operations.
    threshold: 0.90 is usually safe for semantic matching.
    """
    num_videos = len(videos)
    if num_videos < 2:
        return []

    # Filter out videos without fingerprints
    valid_indices = [i for i, fp in enumerate(fingerprints) if fp is not None]
    N = len(valid_indices)
    if N < 2:
        return []

    # 1. Fast Vectorization Precomputations
    # Get frame counts and flat offsets mapping
    lengths = np.array([len(fingerprints[i]) for i in valid_indices], dtype=int)
    starts = np.zeros(N, dtype=int)
    ends = np.zeros(N, dtype=int)
    
    curr = 0
    for i in range(N):
        starts[i] = curr
        ends[i] = curr + lengths[i]
        curr += lengths[i]
        
    # Concatenate all valid features into a single universe matrix shape (TotalFrames, Dim)
    X = np.concatenate([fingerprints[i] for i in valid_indices], axis=0)

    # 2. Compute Adjacency Matrix
    adjacency = {i: [] for i in valid_indices}
    
    # Compute pairwise similarities for all valid videos
    for i_idx, i in enumerate(valid_indices):
        if progress_callback and i_idx % 10 == 0:
            progress_callback(i_idx, N)
            
        fp_i = fingerprints[i] # (F_i, Dim)
        sim_i_all = np.dot(fp_i, X.T) # (F_i, TotalFrames)
        
        # Score v2_to_v1: How well frames of j match frames of i
        max_sim_to_i = np.max(sim_i_all, axis=0) # (TotalFrames,)
        v2_to_v1_sum = np.add.reduceat(max_sim_to_i, starts) # (N,)
        v2_to_v1_all = v2_to_v1_sum / lengths
        
        # Score v1_to_v2: How well frames of i match frames of j
        max_j = np.maximum.reduceat(sim_i_all, starts, axis=1) # (F_i, N)
        v1_to_v2_all = np.mean(max_j, axis=0) # (N,)
        
        # Combined Score
        sims = (v1_to_v2_all + v2_to_v1_all) * 0.5
        
        # Process valid matches
        for j_idx in range(i_idx + 1, N):
            sim = float(sims[j_idx])
            if sim >= threshold:
                j = valid_indices[j_idx]
                adjacency[i].append((j, sim))
                adjacency[j].append((i, sim))

    if progress_callback:
        progress_callback(N, N)

    # Find connected components (groups)
    visited = set()
    groups = []

    for i in valid_indices:
        if i not in visited:
            # New group
            component = []
            queue = [(i, 1.0)] # (index, similarity to original) - similarity to "leader"
            visited.add(i)
            
            while queue:
                curr_idx, curr_sim = queue.pop(0)
                # Attach video info
                v_info = dict(videos[curr_idx])
                v_info['index'] = curr_idx
                v_info['match_score'] = curr_sim
                component.append(v_info)
                
                for neighbor, sim_val in adjacency[curr_idx]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, sim_val))

            if len(component) > 1:
                # Format group object
                # Sort by size to keep the largest as "original"
                component.sort(key=lambda x: x['size'], reverse=True)
                
                groups.append({
                    'id': component[0]['index'],
                    'name': component[0]['name'],
                    'count': len(component) - 1,
                    'videos': component
                })

    return groups
