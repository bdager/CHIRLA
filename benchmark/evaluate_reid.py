#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------
# 1. Cosine similarity
# -----------------------
def cosine_similarity(a, b):
    # Normalize embeddings
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)  # shape [num_queries, num_gallery]


# -----------------------
# 2. CMC and mAP computation
# -----------------------
def evaluate_cmc_map(query_emb, query_ids, gallery_emb, gallery_ids, topk=[1,5,10]):
    sim_matrix = cosine_similarity(query_emb, gallery_emb)
    
    num_queries = query_emb.shape[0]
    cmc_scores = np.zeros(len(topk))
    ap_list = []

    for i in tqdm(range(num_queries), desc="Evaluating"):
        sim_scores = sim_matrix[i]
        sort_idx = np.argsort(-sim_scores)  # sort by descending similarity
        sorted_gallery_ids = gallery_ids[sort_idx]

        gt_id = query_ids[i]
        matches = (sorted_gallery_ids == gt_id).astype(int)

        # Compute CMC
        for j, k in enumerate(topk):
            cmc_scores[j] += (matches[:k].sum() > 0)

        # Compute AP (average precision)
        if matches.sum() > 0:
            ap = average_precision_score(matches, sim_scores[sort_idx])
            ap_list.append(ap)
        else:
            ap_list.append(0.0)

    cmc_scores /= num_queries
    mAP = np.mean(ap_list)

    return cmc_scores, mAP


# -----------------------
# 3. Visualization of top-k matches
# -----------------------
def visualize_topk(query_path, gallery_paths, gallery_ids, query_id, matches, topk=5):
    """
    Display the query image and top-k gallery matches.
    Matches = binary array indicating correct IDs.
    """
    fig, axes = plt.subplots(1, topk+1, figsize=(15, 5))
    # Show query
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title(f"Query ID: {query_id}", color="blue")
    axes[0].axis("off")

    # Show top-k matches
    for k in range(topk):
        axes[k+1].imshow(Image.open(gallery_paths[k]))
        axes[k+1].set_title(f"ID: {gallery_ids[k]}\n{'✔' if matches[k] else '✘'}",
                            color="green" if matches[k] else "red")
        axes[k+1].axis("off")

    plt.show()


# -----------------------
# 4. Load HDF5 embeddings
# -----------------------
def load_h5_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        ids = f["ids"][:]
        paths = f["paths"][:].astype(str)
    return embeddings, ids, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CHIRLA ReID embeddings with CMC and mAP.")
    parser.add_argument("--gallery", required=True, help="Path to gallery HDF5 (train embeddings)")
    parser.add_argument("--query", required=True, help="Path to query HDF5 (test embeddings)")
    parser.add_argument("--topk", nargs="+", type=int, default=[1,5,10], help="Top-k values for CMC")
    parser.add_argument("--visualize", action="store_true", help="Show query and top-k gallery matches")

    args = parser.parse_args()

    # Load embeddings
    gallery_emb, gallery_ids, gallery_paths = load_h5_embeddings(args.gallery)
    query_emb, query_ids, query_paths = load_h5_embeddings(args.query)

    # Evaluate CMC and mAP
    cmc_scores, mAP = evaluate_cmc_map(query_emb, query_ids, gallery_emb, gallery_ids, topk=args.topk)

    print("\n=== Evaluation Results ===")
    for k, score in zip(args.topk, cmc_scores):
        print(f"CMC Rank-{k}: {score*100:.2f}%")
    print(f"mAP: {mAP*100:.2f}%")

    # Optional visualization
    if args.visualize:
        for i in range(min(5, len(query_paths))):
            sim_scores = cosine_similarity(query_emb[i:i+1], gallery_emb)[0]
            sort_idx = np.argsort(-sim_scores)
            top_gallery_paths = gallery_paths[sort_idx][:args.topk[0]]
            top_gallery_ids = gallery_ids[sort_idx][:args.topk[0]]
            matches = (top_gallery_ids == query_ids[i])
            visualize_topk(query_paths[i], top_gallery_paths, top_gallery_ids, query_ids[i], matches, topk=args.topk[0])
