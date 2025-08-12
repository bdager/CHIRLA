#!/usr/bin/env python3
import h5py
import numpy as np
import sys
import os

def inspect_h5_file(h5_path):
    """Inspect the contents of an H5 file"""
    if not os.path.exists(h5_path):
        print(f"File not found: {h5_path}")
        return
    
    print(f"\n=== Inspecting: {h5_path} ===")
    
    try:
        with h5py.File(h5_path, "r") as f:
            print(f"Keys in file: {list(f.keys())}")
            
            if "embeddings" in f:
                embeddings = f["embeddings"][:]
                print(f"Embeddings shape: {embeddings.shape}")
                print(f"Embeddings dtype: {embeddings.dtype}")
                print(f"Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            
            if "ids" in f:
                ids = f["ids"][:]
                print(f"IDs shape: {ids.shape}")
                print(f"IDs dtype: {ids.dtype}")
                print(f"Unique IDs: {len(np.unique(ids))}")
                print(f"ID range: [{ids.min()}, {ids.max()}]")
                print(f"First 10 IDs: {ids[:10]}")
            
            if "paths" in f:
                paths = f["paths"][:].astype(str)
                print(f"Paths shape: {paths.shape}")
                print(f"First 3 paths:")
                for i, path in enumerate(paths[:3]):
                    print(f"  {i}: {path}")
                
                # Extract subsets from paths using improved detection
                subsets = set()
                for path in paths:
                    # Check if it's a hierarchical FastReID path format
                    if '/test/' in path or '/train/' in path:
                        # Extract from hierarchical path structure
                        parts = path.split('/')
                        for i, part in enumerate(parts):
                            if part in ['test', 'train'] and i + 1 < len(parts):
                                next_part = parts[i + 1]
                                if next_part.startswith(('test_', 'train_')):
                                    subsets.add(next_part)
                                    break
                    else:
                        # Flattened Centroids-ReID format - extract from filename
                        filename = os.path.basename(path)
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            subset = f"{parts[0]}_{parts[1]}"
                            subsets.add(subset)
                
                print(f"Detected subsets: {sorted(subsets)}")
    
    except Exception as e:
        print(f"Error reading H5 file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <h5_file1> [h5_file2] ...")
        sys.exit(1)
    
    for h5_file in sys.argv[1:]:
        inspect_h5_file(h5_file)
