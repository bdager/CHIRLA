#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Initialize logger
logger = logging.getLogger("reid_eval")
if not logger.handlers:
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


# -----------------------
# 1. Cosine similarity
# -----------------------
def cosine_similarity(a, b):
    # Normalize embeddings
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)  # shape [num_queries, num_gallery]


# -----------------------
# 2. Subset extraction
# -----------------------
def extract_subset_from_path(path):
    """
    Extract subset information from image path.
    Handles both FastReID and Centroids-ReID path formats:
    - FastReID: '/path/to/reid/multi_camera/test/test_2/seq_002/imgs/camera_4_2023-06-02-12:09:07/18/frame_2657.png'
    - Centroids: '/tmp/tmpv88ca6d4/reid_multi_camera_query/test_2_seq_002_camera_4_2023-06-02-12:09:07_18_frame_2657.png'
    Returns: 'test_2', 'train_1', etc.
    """
    # Check if it's a hierarchical FastReID path format
    if '/test/' in path or '/train/' in path:
        # Extract from hierarchical path structure
        # Look for patterns like '/test/test_2/' or '/train/train_1/'
        parts = path.split('/')
        for i, part in enumerate(parts):
            if part in ['test', 'train'] and i + 1 < len(parts):
                next_part = parts[i + 1]
                # Check if next part follows the pattern test_N or train_N
                if next_part.startswith(('test_', 'train_')):
                    return next_part
        return "unknown"
    else:
        # Flattened Centroids-ReID format - extract from filename
        filename = os.path.basename(path)
        parts = filename.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # e.g., 'test_2', 'train_1', etc.
        return "unknown"


def group_by_subset(embeddings, ids, paths):
    """
    Group embeddings, IDs, and paths by subset.
    Returns a dictionary where keys are subset names and values are (embeddings, ids, paths) tuples.
    """
    subset_data = {}
    
    for i, path in enumerate(paths):
        subset = extract_subset_from_path(path)
        
        if subset not in subset_data:
            subset_data[subset] = {'embeddings': [], 'ids': [], 'paths': [], 'indices': []}
        
        subset_data[subset]['embeddings'].append(embeddings[i])
        subset_data[subset]['ids'].append(ids[i])
        subset_data[subset]['paths'].append(path)
        subset_data[subset]['indices'].append(i)
    
    # Convert lists to numpy arrays
    for subset in subset_data:
        subset_data[subset]['embeddings'] = np.array(subset_data[subset]['embeddings'])
        subset_data[subset]['ids'] = np.array(subset_data[subset]['ids'])
        subset_data[subset]['paths'] = np.array(subset_data[subset]['paths'])
        subset_data[subset]['indices'] = np.array(subset_data[subset]['indices'])
    
    return subset_data


# -----------------------
# 3. CMC and mAP computation
# -----------------------
def evaluate_cmc_map_with_unknowns(query_emb, query_ids, gallery_emb, gallery_ids, topk=[1,5,10], open_set_map_threshold=None):
    """
    Enhanced ReID evaluation that handles unknown identities properly.
    Returns both closed-set (known only) and open-set (with unknowns) results.
    Optionally applies a similarity threshold when computing mAP for the open-set.
    """
    # Identify known vs unknown identities
    known_mask = query_ids >= 0  # Assuming negative IDs are unknown
    known_gallery_ids = set(gallery_ids)
    
    # Known IDs must actually exist in gallery
    truly_known_mask = known_mask & np.isin(query_ids, list(known_gallery_ids))
    
    logger.info(f"Query ID analysis:")
    logger.info(f"  Total queries: {len(query_ids)}")
    logger.info(f"  Known IDs (positive): {known_mask.sum()}")
    logger.info(f"  Unknown IDs (negative): {(~known_mask).sum()}")
    logger.info(f"  Known IDs in gallery: {truly_known_mask.sum()}")
    error_unexpected = (known_mask & ~truly_known_mask).sum()
    if error_unexpected > 0:
        logger.warning(f"  ❌ Known IDs not in gallery: {error_unexpected}," \
              f"{query_ids[~truly_known_mask]}")
    
    results = {
        'known_queries': truly_known_mask.sum(),
        'unknown_queries': (~known_mask).sum(),
        'total_queries': len(query_ids),
        'positive_pairs': 0,  # Will be calculated if needed
        'negative_pairs': 0   # Will be calculated if needed
    }
    
    # 1. Closed-set evaluation (only known identities)
    if truly_known_mask.sum() > 0:
        closed_query_emb = query_emb[truly_known_mask]
        closed_query_ids = query_ids[truly_known_mask]
        
        cmc_closed, mAP_closed = evaluate_cmc_map(
            closed_query_emb, closed_query_ids, gallery_emb, gallery_ids, topk
        )
        results['closed_set'] = {'cmc': cmc_closed, 'mAP': mAP_closed, 'num_queries': len(closed_query_ids)}
        
        logger.info(f"\n--- Closed-set Results (Known IDs only: {len(closed_query_ids)} queries) ---")
        for k, score in zip(topk, cmc_closed):
            logger.info(f"  CMC Rank-{k}: {score*100:.2f}%")
        logger.info(f"  mAP: {mAP_closed*100:.2f}%")
    
    # 2. Open-set evaluation (all identities including unknowns)
    cmc_open, mAP_open = evaluate_cmc_map(query_emb, query_ids, gallery_emb, gallery_ids, topk, min_sim_threshold=open_set_map_threshold)
    results['open_set'] = {'cmc': cmc_open, 'mAP': mAP_open, 'num_queries': len(query_ids), 'map_threshold': open_set_map_threshold}
    
    logger.info(f"\n--- Open-set Results (All IDs including unknowns: {len(query_ids)} queries) ---")
    if open_set_map_threshold is not None:
        logger.info(f"  mAP threshold: {open_set_map_threshold}")
    for k, score in zip(topk, cmc_open):
        logger.info(f"  CMC Rank-{k}: {score*100:.2f}%")
    logger.info(f"  mAP: {mAP_open*100:.2f}%")
    
    # Return closed-set results as primary (standard ReID practice)
    if 'closed_set' in results:
        return results['closed_set']['cmc'], results['closed_set']['mAP'], results
    else:
        return cmc_open, mAP_open, results


def evaluate_cmc_map(query_emb, query_ids, gallery_emb, gallery_ids, topk=[1,5,10], min_sim_threshold=None):
    sim_matrix = cosine_similarity(query_emb, gallery_emb)
    
    num_queries = query_emb.shape[0]
    cmc_scores = np.zeros(len(topk))
    ap_list = []

    for i in tqdm(range(num_queries), desc="Evaluating"):
        sim_scores = sim_matrix[i]
        sort_idx = np.argsort(-sim_scores)  # sort by descending similarity
        sorted_gallery_ids = gallery_ids[sort_idx]
        scores_sorted = sim_scores[sort_idx]

        gt_id = query_ids[i]
        matches = (sorted_gallery_ids == gt_id).astype(int)

        # Compute CMC (threshold does not affect CMC ranking)
        for j, k in enumerate(topk):
            cmc_scores[j] += (matches[:k].sum() > 0)

        # Compute AP (average precision) with optional similarity threshold
        if min_sim_threshold is not None:
            mask = scores_sorted >= min_sim_threshold
            if mask.any():
                matches_masked = matches[mask]
                scores_masked = scores_sorted[mask]
                if matches_masked.sum() > 0:
                    ap = average_precision_score(matches_masked, scores_masked)
                else:
                    ap = 0.0
            else:
                ap = 0.0
        else:
            if matches.sum() > 0:
                ap = average_precision_score(matches, scores_sorted)
            else:
                ap = 0.0
        ap_list.append(ap)

    cmc_scores /= num_queries
    mAP = np.mean(ap_list)

    return cmc_scores, mAP


def evaluate_per_subset(query_emb, query_ids, query_paths, gallery_emb, gallery_ids, gallery_paths, topk=[1,5,10], open_set_map_threshold=None):
    """
    Evaluate metrics per subset and return both per-subset and averaged results.
    """
    logger.info("\n=== Per-Subset Evaluation ===")

    print(query_paths)
    
    # Group query data by subset
    query_subsets = group_by_subset(query_emb, query_ids, query_paths)
    gallery_subsets = group_by_subset(gallery_emb, gallery_ids, gallery_paths)
    
    logger.info(f"Found query subsets: {sorted(list(query_subsets.keys()))}")
    logger.info(f"Found gallery subsets: {sorted(list(gallery_subsets.keys()))}")
    
    valid_results = []
    
    # Sort query subsets for consistent ordering
    for query_subset in sorted(query_subsets.keys()):
        logger.info(f"\n--- Evaluating subset: {query_subset} ---")
        
        query_subset_data = query_subsets[query_subset]
        q_emb = query_subset_data['embeddings']
        q_ids = query_subset_data['ids']
        
        logger.info(f"Query subset {query_subset}: {q_emb.shape[0]} samples, {len(np.unique(q_ids))} unique IDs")
        
        # Find corresponding gallery subset (e.g., test_1 -> train_1)
        corresponding_gallery_subset = query_subset.replace('test_', 'train_')
        
        if corresponding_gallery_subset in gallery_subsets:
            gallery_subset_data = gallery_subsets[corresponding_gallery_subset]
            g_emb = gallery_subset_data['embeddings']
            g_ids = gallery_subset_data['ids']
            
            logger.info(f"Using corresponding gallery subset {corresponding_gallery_subset}: {g_emb.shape[0]} samples, {len(np.unique(g_ids))} unique IDs")
            
            # Use enhanced evaluation that handles unknown identities
            cmc_scores, mAP, detailed_results = evaluate_cmc_map_with_unknowns(q_emb, q_ids, g_emb, g_ids, topk=topk, open_set_map_threshold=open_set_map_threshold)
        else:
            logger.warning(f"No corresponding gallery subset found for {query_subset} (looking for {corresponding_gallery_subset})")
            logger.info("Falling back to using entire gallery")
            cmc_scores, mAP, detailed_results = evaluate_cmc_map_with_unknowns(q_emb, q_ids, gallery_emb, gallery_ids, topk=topk, open_set_map_threshold=open_set_map_threshold)
        
        logger.info(f"Results for {query_subset}:")
        for k, score in zip(topk, cmc_scores):
            logger.info(f"  CMC Rank-{k}: {score*100:.2f}%")
        logger.info(f"  mAP: {mAP*100:.2f}%")
        
        # Only include subsets with valid results (mAP > 0 or CMC > 0)
        if mAP > 0 or any(cmc_scores > 0):
            valid_results.append({
                'cmc_scores': cmc_scores,
                'mAP': mAP,
                'subset': query_subset,
                'num_queries': q_emb.shape[0]
            })
    
    # Calculate weighted average across subsets
    if valid_results:        
        logger.info(f"\n--- Summary of {len(valid_results)} valid subsets ---")
        for result in valid_results:
            logger.info(f"  {result['subset']}: {result['num_queries']} queries, mAP: {result['mAP']*100:.2f}%")
        
        simple_avg_cmc = np.mean([r['cmc_scores'] for r in valid_results], axis=0)
        simple_avg_mAP = np.mean([r['mAP'] for r in valid_results])
        
        logger.info(f"\n--- Average Across All {len(valid_results)} Subsets ---")
        logger.info("Simple Average (equal weight per subset):")
        for k, score in zip(topk, simple_avg_cmc):
            logger.info(f"  CMC Rank-{k}: {score*100:.2f}%")
        logger.info(f"  mAP: {simple_avg_mAP*100:.2f}%")
        
        return simple_avg_cmc, simple_avg_mAP
    else:
        # No valid results
        return np.zeros(len(topk)), 0.0


# -----------------------
# 4. Visualization of top-k matches
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
# 5. Load HDF5 embeddings
# -----------------------
def load_h5_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        ids = f["ids"][:]
        paths = f["paths"][:].astype(str)
    return embeddings, ids, paths


def run_evaluation(gallery_h5, query_h5, topk=[1, 5, 10], per_subset=True, 
                   open_map_threshold=None, visualize=False, show_logs=True):
    # Adjust logger level based on show_logs flag
    prev_level = logger.level
    if show_logs:
        logger.setLevel(logging.INFO)
    else:
        # Suppress INFO (keep warnings/errors)
        logger.setLevel(logging.WARNING)

    # Initialize placeholders to avoid NameError when returning
    cmc_scores = mAP = simple_avg_cmc = simple_avg_mAP = None

    try:
        # Load embeddings
        gallery_emb, gallery_ids, gallery_paths = load_h5_embeddings(gallery_h5)
        query_emb, query_ids, query_paths = load_h5_embeddings(query_h5)

        logger.info(f"Loaded gallery: {gallery_emb.shape[0]} embeddings, {len(np.unique(gallery_ids))} unique IDs")
        logger.info(f"Loaded query: {query_emb.shape[0]} embeddings, {len(np.unique(query_ids))} unique IDs")

        if per_subset:
            # Per-subset evaluation with averaging
            simple_avg_cmc, simple_avg_mAP = evaluate_per_subset(
                query_emb, query_ids, query_paths, gallery_emb, gallery_ids,
                gallery_paths, topk=topk, open_set_map_threshold=open_map_threshold
            )
            logger.info("\n" + "="*60)
            logger.info("FINAL AVERAGED RESULTS")
            logger.info("="*60)
            logger.info("\n--- SUBSET AVERAGE ---")
            logger.info("Simple Average (equal weight per subset):")
            for k, score in zip(topk, simple_avg_cmc):
                logger.info(f"CMC Rank-{k}: {score*100:.2f}%")
            logger.info(f"mAP: {simple_avg_mAP*100:.2f}%")       
        else:
            # Standard evaluation with unknown identity handling
            cmc_scores, mAP, detailed_results = evaluate_cmc_map_with_unknowns(
                query_emb, query_ids, gallery_emb, gallery_ids, topk=topk, 
                open_set_map_threshold=open_map_threshold
            )
            logger.info("\n=== Evaluation Results ===")
            for k, score in zip(topk, cmc_scores):
                logger.info(f"CMC Rank-{k}: {score*100:.2f}%")
            logger.info(f"mAP: {mAP*100:.2f}%")
            logger.info("\nDetailed unknown identity statistics:")
            logger.info(f"Known queries: {detailed_results['known_queries']}")
            logger.info(f"Unknown queries: {detailed_results['unknown_queries']}")
            logger.info(f"Total queries processed: {detailed_results['total_queries']}")
            logger.info(f"Positive pairs (same identity): {detailed_results['positive_pairs']}")
            logger.info(f"Negative pairs (different identity): {detailed_results['negative_pairs']}")

        # Optional visualization (always performed irrespective of log level)
        if visualize:
            for i in range(min(5, len(query_paths))):
                sim_scores = cosine_similarity(query_emb[i:i+1], gallery_emb)[0]
                sort_idx = np.argsort(-sim_scores)
                top_gallery_paths = gallery_paths[sort_idx][:topk[0]]
                top_gallery_ids = gallery_ids[sort_idx][:topk[0]]
                matches = (top_gallery_ids == query_ids[i])
                visualize_topk(query_paths[i], top_gallery_paths, top_gallery_ids, query_ids[i], matches, topk=topk[0])
    finally:
        # Restore previous logger level
        logger.setLevel(prev_level)

    return {
        'cmc_scores': simple_avg_cmc if per_subset else cmc_scores,
        'mAP': simple_avg_mAP if per_subset else mAP
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CHIRLA ReID embeddings with CMC and mAP.")
    parser.add_argument("--gallery", required=True, help="Path to gallery HDF5 (train embeddings)")
    parser.add_argument("--query", required=True, help="Path to query HDF5 (test embeddings)")
    parser.add_argument("--topk", nargs="+", type=int, default=[1,5,10], help="Top-k values for CMC")
    parser.add_argument("--visualize", action="store_true", help="Show query and top-k gallery matches")
    parser.add_argument("--per-subset", action="store_true", help="Evaluate per subset and average results")
    parser.add_argument("--open-map-threshold", dest="open_map_threshold", 
        type=float, default=None, help="Similarity threshold applied when computing mAP for open-set evaluation (no effect on CMC or closed-set mAP)")
    parser.add_argument("--no-show-logs", action="store_false", help="Show detailed logs during evaluation")

    args = parser.parse_args()

    # Decide logging based on flag (flag name implies disabling when provided)
    show_logs = args.no_show_logs  # True if user did NOT pass --no-show-logs

    run_evaluation(args.gallery, args.query, topk=args.topk, 
                   per_subset=args.per_subset,
                   open_map_threshold=args.open_map_threshold,
                   visualize=args.visualize,
                   show_logs=show_logs)


