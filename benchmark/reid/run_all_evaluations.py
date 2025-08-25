#!/usr/bin/env python3
import os
import argparse
import csv


def find_h5_pairs(base_dir):
    """
    Find all gallery/query H5 file pairs in the given base directory.
    Returns a list of (gallery_h5, query_h5, model_name, scenario) tuples.
    """
    pairs = []
    
    # Walk through all subdirectories to find H5 files
    for root, dirs, files in os.walk(base_dir):
        h5_files = [f for f in files if f.endswith('.h5')]
        
        if not h5_files:
            continue
            
        # Group H5 files by scenario (gallery vs query)
        gallery_files = [f for f in h5_files if 'gallery' in f]
        query_files = [f for f in h5_files if 'query' in f and 'val' not in f]
        
        # Extract model name from directory structure
        rel_path = os.path.relpath(root, base_dir)
        model_name = rel_path.replace('/', '_') if rel_path != '.' else 'default'
        
        # Match gallery and query files for the same scenario
        for gallery_file in gallery_files:
            gallery_scenario = extract_scenario_from_filename(gallery_file)
            gallery_path = os.path.join(root, gallery_file)
            
            for query_file in query_files:
                query_scenario = extract_scenario_from_filename(query_file)
                
                # Only pair files from the same scenario
                if gallery_scenario == query_scenario:
                    query_path = os.path.join(root, query_file)
                    pairs.append((gallery_path, query_path, model_name, gallery_scenario))
    
    return pairs


def extract_scenario_from_filename(filename):
    """
    Extract scenario name from H5 filename.
    Examples:
      'long_term_gallery_embeddings.h5' -> 'long_term'
      'reid_multi_camera_query_embeddings.h5' -> 'reid_multi_camera'
    """
    name = filename
    # Remove suffixes
    if name.endswith('_embeddings.h5'):
        name = name[:-len('_embeddings.h5')]
    elif name.endswith('.h5'):
        name = name[:-len('.h5')]

    # Remove trailing gallery/query token if present
    parts = name.split('_')
    if parts and parts[-1] in ('gallery', 'query'):
        parts = parts[:-1]
    scenario = '_'.join(parts)
    return scenario


def get_evaluation(gallery_h5, query_h5, topk=[1, 5, 10], per_subset=True, open_map_threshold=None):
    """Direct evaluation using evaluate_reid functions (no subprocess).
    Returns metrics dict similar to previous parser-based version.
    """
    try:
        from evaluate_reid import (
            run_evaluation
        )
    except ImportError as e:
        print(f"❌ Failed to import evaluation utilities: {e}")
        return None
    return run_evaluation(
        gallery_h5=gallery_h5,
        query_h5=query_h5,
        topk=topk,
        per_subset=per_subset,
        open_map_threshold=open_map_threshold,
        visualize=False,
        show_logs=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on all gallery/query H5 pairs in a folder")
    parser.add_argument("--base-dir", default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/fastreid",
                        help="Base directory containing H5 files (e.g., fastreid or CION subfolders)")
    parser.add_argument("--output", default="evaluation_results.csv",
                        help="Output CSV file for results")
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10],
                        help="Top-k values for CMC evaluation")
    parser.add_argument("--no-per-subset", action="store_true",
                        help="Disable per-subset evaluation (use overall evaluation only)")
    parser.add_argument("--open-map-threshold", dest="open_map_threshold", type=float, default=None,
                        help="Similarity threshold applied to open-set mAP in evaluate_reid (forwarded to script)")
    
    args = parser.parse_args()
    
    # Find all H5 file pairs
    print(f"Searching for H5 files in: {args.base_dir}")
    pairs = find_h5_pairs(args.base_dir)
    
    if not pairs:
        print("No gallery/query H5 file pairs found!")
        return
    
    print(f"Found {len(pairs)} gallery/query pairs to evaluate:")
    for gallery, query, model, scenario in pairs:
        print(f"  - {model}/{scenario}: {os.path.basename(gallery)} vs {os.path.basename(query)}")
    
    # Run evaluations
    results = []
    use_per_subset = not args.no_per_subset
    
    for gallery_h5, query_h5, model_name, scenario in pairs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} - {scenario}")
        print(f"Gallery: {gallery_h5}")
        print(f"Query: {query_h5}")
        print(f"{'='*60}")
        metrics = get_evaluation(
            gallery_h5, query_h5, args.topk, per_subset=use_per_subset, open_map_threshold=args.open_map_threshold
        )
        filter_metrics = {}
        if metrics:
            # metrics['cmc_scores'] is an array aligned with args.topk
            for k, score in zip(args.topk, metrics['cmc_scores']):
                filter_metrics[f"CMC@{k}"] = round(float(score) * 100, 2)
            filter_metrics["mAP"] = round(float(metrics['mAP']) * 100, 2)
            result_row = {
                'model': model_name,
                'scenario': scenario,
                'gallery_file': os.path.basename(gallery_h5),
                'query_file': os.path.basename(query_h5),
                # 'per_subset': use_per_subset,
                # 'mode': 'per_subset' if use_per_subset else 'overall',
                # 'open_map_threshold': args.open_map_threshold if args.open_map_threshold is not None else '',
                **filter_metrics
            }
            results.append(result_row)
    
    # Save results to CSV
    if results:
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ Saved evaluation results to: {args.output}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        fieldnames = list(results[0].keys())
        header = " | ".join([f"{name:15}" for name in fieldnames])
        print(header)
        print("-" * len(header))
        
        for result in results:
            row = " | ".join([f"{str(result[field]):15}" for field in fieldnames])
            print(row)
        
        # Print best performing models per scenario using the primary (Simple/Weighted/Overall) mAP
        print(f"\n{'='*80}")
        print("BEST PERFORMING MODELS PER SCENARIO")
        print(f"{'='*80}")
        scenarios = {}
        for result in results:
            map_value = result.get('mAP') or 0
            scenario_key = result['scenario']
            if scenario_key not in scenarios or map_value > (scenarios[scenario_key].get('mAP') or 0):
                scenarios[scenario_key] = result
        for scenario, best_result in scenarios.items():
            mAP = best_result.get('mAP') or 0
            model = best_result.get('model', 'Unknown')
            print(f"{scenario}: {model} (mAP: {mAP:.2f}%)")
    else:
        print("No successful evaluations completed.")


if __name__ == "__main__":
    main()
