#!/usr/bin/env python3
import h5py
import numpy as np
import sys
import os
from sklearn.metrics import average_precision_score
sys.path.append('.')

def cosine_similarity(a, b):
    """Cosine similarity between two embedding matrices"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def test_single_subset():
    """Test evaluation on a single subset to understand the 100% results"""
    query_file = 'centroids_reid/Market1501/256_resnet50/multi_camera_long_term_query_embeddings.h5'
    gallery_file = 'centroids_reid/Market1501/256_resnet50/multi_camera_long_term_gallery_embeddings.h5'
    
    # Load data
    with h5py.File(query_file, 'r') as f:
        query_emb = f['embeddings'][:]
        query_ids = f['ids'][:]
        query_paths = f['paths'][:].astype(str)
    
    with h5py.File(gallery_file, 'r') as f:
        gallery_emb = f['embeddings'][:]
        gallery_ids = f['ids'][:]
        gallery_paths = f['paths'][:].astype(str)
    
    # Focus on test_1 vs train_1 (from our debug output)
    test_1_indices = []
    train_1_indices = []
    
    for i, path in enumerate(query_paths):
        if '/test/test_1/' in path:
            test_1_indices.append(i)
    
    for i, path in enumerate(gallery_paths):
        if '/train/train_1/' in path:
            train_1_indices.append(i)
    
    print(f"Found {len(test_1_indices)} test_1 queries")
    print(f"Found {len(train_1_indices)} train_1 gallery items")
    
    # Extract subset data
    q_emb = query_emb[test_1_indices]
    q_ids = query_ids[test_1_indices]
    q_paths = query_paths[test_1_indices]
    
    g_emb = gallery_emb[train_1_indices]
    g_ids = gallery_ids[train_1_indices]
    g_paths = gallery_paths[train_1_indices]
    
    print(f"Query IDs: {sorted(np.unique(q_ids))}")
    print(f"Gallery IDs: {sorted(np.unique(g_ids))}")
    
    # Check overlaps
    q_ids_set = set(q_ids)
    g_ids_set = set(g_ids)
    overlap_ids = q_ids_set.intersection(g_ids_set)
    print(f"Overlapping IDs: {sorted(list(overlap_ids))}")
    
    # Only evaluate queries that have matching gallery IDs (closed-set)
    known_mask = np.array([qid in g_ids_set for qid in q_ids])
    known_q_emb = q_emb[known_mask]
    known_q_ids = q_ids[known_mask]
    
    print(f"Known queries: {known_mask.sum()} out of {len(q_ids)}")
    print(f"Known query IDs: {sorted(known_q_ids)}")
    
    if len(known_q_emb) == 0:
        print("No known queries found!")
        return
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(known_q_emb, g_emb)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    
    # Manual CMC calculation
    cmc_scores = []
    mAP_scores = []
    
    print(f"\\n=== DETAILED MATCHING ANALYSIS ===")
    for i, (q_emb_single, q_id) in enumerate(zip(known_q_emb, known_q_ids)):
        # Get similarities for this query
        sims = sim_matrix[i]
        
        # Find positive matches (same person ID)
        positive_mask = (g_ids == q_id)
        positive_count = positive_mask.sum()
        
        print(f"\\nQuery {i} (ID {q_id}):")
        print(f"  Positive gallery items: {positive_count}")
        if positive_count == 0:
            print(f"  WARNING: No positive matches in gallery!")
            continue
        
        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(-sims)
        sorted_g_ids = g_ids[sorted_indices]
        sorted_sims = sims[sorted_indices]
        
        print(f"  Top 5 matches:")
        for j in range(min(5, len(sorted_g_ids))):
            idx = sorted_indices[j]
            is_positive = g_ids[idx] == q_id
            print(f"    Rank {j+1}: ID {g_ids[idx]} (sim: {sorted_sims[j]:.4f}) {'✓ CORRECT' if is_positive else '✗ wrong'}")
        
        # Check if first match is correct
        first_match_correct = (sorted_g_ids[0] == q_id)
        print(f"  Rank-1 correct: {first_match_correct}")
        
        # Compute AP for this query
        y_true = (sorted_g_ids == q_id).astype(int)
        if y_true.sum() > 0:  # Only compute if there are positives
            ap = average_precision_score(y_true, sorted_sims)
            print(f"  AP: {ap:.4f}")
            mAP_scores.append(ap)
        
        if not first_match_correct:
            print(f"  ERROR: First match should be correct but it's not!")
            print(f"  Query embedding norm: {np.linalg.norm(q_emb_single):.4f}")
            print(f"  Best match embedding norm: {np.linalg.norm(g_emb[sorted_indices[0]]):.4f}")
    
    overall_mAP = np.mean(mAP_scores) if mAP_scores else 0.0
    print(f"\\nOverall mAP for test_1: {overall_mAP:.4f} ({overall_mAP*100:.2f}%)")

if __name__ == "__main__":
    test_single_subset()
