#!/usr/bin/env python3
import h5py
import numpy as np
import sys
import os
sys.path.append('.')

def extract_subset_from_path(path):
    """Extract subset from path - same logic as evaluate_reid.py"""
    if '/test/' in path or '/train/' in path:
        parts = path.split('/')
        for i, part in enumerate(parts):
            if part in ['test', 'train'] and i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part.startswith(('test_', 'train_')):
                    return next_part
        return "unknown"
    else:
        filename = os.path.basename(path)
        parts = filename.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "unknown"

def group_by_subset(embeddings, ids, paths):
    """Group data by subset - same logic as evaluate_reid.py"""
    subsets = {}
    for i, path in enumerate(paths):
        subset = extract_subset_from_path(path)
        if subset not in subsets:
            subsets[subset] = {
                'embeddings': [],
                'ids': [],
                'paths': []
            }
        subsets[subset]['embeddings'].append(embeddings[i])
        subsets[subset]['ids'].append(ids[i])
        subsets[subset]['paths'].append(path)
    
    # Convert lists to numpy arrays
    for subset in subsets:
        subsets[subset]['embeddings'] = np.array(subsets[subset]['embeddings'])
        subsets[subset]['ids'] = np.array(subsets[subset]['ids'])
        subsets[subset]['paths'] = np.array(subsets[subset]['paths'])
    
    return subsets

def debug_evaluation():
    query_file = 'centroids_reid/Market1501/256_resnet50/multi_camera_long_term_query_embeddings.h5'
    gallery_file = 'centroids_reid/Market1501/256_resnet50/multi_camera_long_term_gallery_embeddings.h5'
    
    print("=== LOADING DATA ===")
    
    # Load query data
    with h5py.File(query_file, 'r') as f:
        query_emb = f['embeddings'][:]
        query_ids = f['ids'][:]
        query_paths = f['paths'][:].astype(str)
    
    # Load gallery data  
    with h5py.File(gallery_file, 'r') as f:
        gallery_emb = f['embeddings'][:]
        gallery_ids = f['ids'][:]
        gallery_paths = f['paths'][:].astype(str)
    
    print(f"Query: {query_emb.shape[0]} samples, {len(np.unique(query_ids))} unique IDs")
    print(f"Gallery: {gallery_emb.shape[0]} samples, {len(np.unique(gallery_ids))} unique IDs")
    
    # Group by subsets
    query_subsets = group_by_subset(query_emb, query_ids, query_paths)
    gallery_subsets = group_by_subset(gallery_emb, gallery_ids, gallery_paths)
    
    print(f"\nQuery subsets: {sorted(query_subsets.keys())}")
    print(f"Gallery subsets: {sorted(gallery_subsets.keys())}")
    
    # Detailed analysis
    print("\n=== DETAILED SUBSET ANALYSIS ===")
    for query_subset in sorted(query_subsets.keys()):
        q_data = query_subsets[query_subset]
        print(f"\n--- Query subset: {query_subset} ---")
        print(f"  Samples: {len(q_data['ids'])}")
        print(f"  Unique IDs: {sorted(np.unique(q_data['ids']))}")
        
        # Find corresponding gallery subset
        corresponding_gallery_subset = query_subset.replace('test_', 'train_')
        
        if corresponding_gallery_subset in gallery_subsets:
            g_data = gallery_subsets[corresponding_gallery_subset]
            print(f"  -> Found gallery subset: {corresponding_gallery_subset}")
            print(f"     Gallery samples: {len(g_data['ids'])}")
            print(f"     Gallery unique IDs: {sorted(np.unique(g_data['ids']))}")
            
            # Check ID overlap
            q_ids_set = set(q_data['ids'])
            g_ids_set = set(g_data['ids'])
            overlap = q_ids_set.intersection(g_ids_set)
            print(f"     ID overlap: {sorted(list(overlap))} ({len(overlap)} IDs)")
            print(f"     Query-only IDs: {sorted(list(q_ids_set - g_ids_set))}")
            print(f"     Gallery-only IDs: {sorted(list(g_ids_set - q_ids_set))}")
        else:
            print(f"  -> No corresponding gallery subset found (looking for {corresponding_gallery_subset})")
            print(f"     Will use entire gallery with {len(gallery_ids)} samples")
            
            # Check overlap with entire gallery
            q_ids_set = set(q_data['ids'])
            g_ids_set = set(gallery_ids)
            overlap = q_ids_set.intersection(g_ids_set)
            print(f"     ID overlap with full gallery: {sorted(list(overlap))} ({len(overlap)} IDs)")

if __name__ == "__main__":
    debug_evaluation()
