"""
Vectorized similarity computation and clustering for video fingerprints.
Uses Chamfer-style frame-to-frame matching for robust clip detection.
"""
import numpy as np
from typing import List, Dict, Any, Optional

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

def find_duplicate_groups(videos: List[Dict], fingerprints: List[Optional[np.ndarray]], threshold: float = 0.90) -> List[Dict]:
    """
    Clusters videos into duplicate groups based on frame-wise Chamfer similarity.
    threshold: 0.90 is usually safe for semantic matching.
    """
    num_videos = len(videos)
    if num_videos < 2:
        return []

    # Filter out videos without fingerprints
    valid_indices = [i for i, fp in enumerate(fingerprints) if fp is not None]
    if len(valid_indices) < 2:
        return []

    # Map for union-find or adjacency
    adjacency = {i: [] for i in valid_indices}
    
    # Compute pairwise similarities for all valid videos
    for i_idx, i in enumerate(valid_indices):
        for j in valid_indices[i_idx + 1:]:
            sim = compute_video_similarity(fingerprints[i], fingerprints[j])
            if sim >= threshold:
                adjacency[i].append((j, sim))
                adjacency[j].append((i, sim))

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
