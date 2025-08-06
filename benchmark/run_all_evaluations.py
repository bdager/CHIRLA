#!/usr/bin/env python3
import os
import subprocess
import argparse
import glob
import csv
from pathlib import Path

def find_h5_pairs(base_dir):
    """
    Find all train/test H5 file pairs in the benchmark/fastreid directory.
    Returns a list of (train_h5, test_h5, model_name, scenario) tuples.
    """
    pairs = []
    
    # Walk through all subdirectories to find H5 files
    for root, dirs, files in os.walk(base_dir):
        h5_files = [f for f in files if f.endswith('.h5')]
        
        if not h5_files:
            continue
            
        # Group H5 files by scenario (train vs test)
        train_files = [f for f in h5_files if 'train' in f]
        test_files = [f for f in h5_files if 'test' in f and 'val' not in f]
        
        # Extract model name from directory structure
        rel_path = os.path.relpath(root, base_dir)
        model_name = rel_path.replace('/', '_') if rel_path != '.' else 'default'
        
        # Match train and test files for the same scenario
        for train_file in train_files:
            train_scenario = extract_scenario_from_filename(train_file)
            train_path = os.path.join(root, train_file)
            
            for test_file in test_files:
                test_scenario = extract_scenario_from_filename(test_file)
                
                # Only pair files from the same scenario
                if train_scenario == test_scenario:
                    test_path = os.path.join(root, test_file)
                    pairs.append((train_path, test_path, model_name, train_scenario))
    
    return pairs

def extract_scenario_from_filename(filename):
    """
    Extract scenario name from H5 filename.
    e.g., 'long_term_train_embeddings.h5' -> 'long_term'
    """
    # Remove common suffixes
    name = filename.replace('_embeddings.h5', '').replace('.h5', '')
    
    # Remove train/test suffix
    if name.endswith('_train'):
        name = name[:-6]
    elif name.endswith('_test'):
        name = name[:-5]
    
    return name

def run_evaluation(gallery_h5, query_h5, output_file, topk=[1, 5, 10]):
    """
    Run the evaluate_reid.py script for a specific gallery/query pair.
    Returns the evaluation results.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'evaluate_reid.py')
    
    cmd = [
        'python', script_path,
        '--gallery', gallery_h5,
        '--query', query_h5,
        '--topk'] + [str(k) for k in topk]
    
    try:
        # Run the evaluation script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to extract metrics
        lines = result.stdout.strip().split('\n')
        metrics = {}
        
        for line in lines:
            if 'CMC Rank-' in line:
                parts = line.split(':')
                rank = parts[0].split('Rank-')[1]
                score = float(parts[1].strip().replace('%', ''))
                metrics[f'CMC_Rank_{rank}'] = score
            elif 'mAP:' in line:
                score = float(line.split(':')[1].strip().replace('%', ''))
                metrics['mAP'] = score
        
        print(f"✅ Evaluated: {os.path.basename(gallery_h5)} vs {os.path.basename(query_h5)}")
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error evaluating {gallery_h5} vs {query_h5}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on all train/test H5 pairs in benchmark/fastreid folder")
    parser.add_argument("--base-dir", default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/fastreid",
                        help="Base directory containing H5 files")
    parser.add_argument("--output", default="evaluation_results.csv",
                        help="Output CSV file for results")
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10],
                        help="Top-k values for CMC evaluation")
    
    args = parser.parse_args()
    
    # Find all H5 file pairs
    print(f"Searching for H5 files in: {args.base_dir}")
    pairs = find_h5_pairs(args.base_dir)
    
    if not pairs:
        print("No train/test H5 file pairs found!")
        return
    
    print(f"Found {len(pairs)} train/test pairs to evaluate:")
    for gallery, query, model, scenario in pairs:
        print(f"  - {model}/{scenario}: {os.path.basename(gallery)} vs {os.path.basename(query)}")
    
    # Run evaluations
    results = []
    
    for gallery_h5, query_h5, model_name, scenario in pairs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} - {scenario}")
        print(f"Gallery: {gallery_h5}")
        print(f"Query: {query_h5}")
        print(f"{'='*60}")
        
        metrics = run_evaluation(gallery_h5, query_h5, None, args.topk)
        
        if metrics:
            result_row = {
                'model': model_name,
                'scenario': scenario,
                'gallery_file': os.path.basename(gallery_h5),
                'query_file': os.path.basename(query_h5),
                **metrics
            }
            results.append(result_row)
    
    # Save results to CSV
    if results:
        # Write CSV file manually
        with open(args.output, 'w', newline='') as csvfile:
            if results:
                fieldnames = list(results[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        
        print(f"\n✅ Saved evaluation results to: {args.output}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        # Print header
        if results:
            fieldnames = list(results[0].keys())
            header = " | ".join([f"{name:15}" for name in fieldnames])
            print(header)
            print("-" * len(header))
            
            # Print each result row
            for result in results:
                row = " | ".join([f"{str(result[field]):15}" for field in fieldnames])
                print(row)
        
        # Print best performing models per scenario
        print(f"\n{'='*80}")
        print("BEST PERFORMING MODELS PER SCENARIO")
        print(f"{'='*80}")
        
        # Group results by scenario and find best mAP
        scenarios = {}
        for result in results:
            scenario = result['scenario']
            if scenario not in scenarios or result.get('mAP', 0) > scenarios[scenario].get('mAP', 0):
                scenarios[scenario] = result
        
        for scenario, best_result in scenarios.items():
            mAP = best_result.get('mAP', 0)
            model = best_result.get('model', 'Unknown')
            print(f"{scenario}: {model} (mAP: {mAP:.2f}%)")
    else:
        print("No successful evaluations completed.")

if __name__ == "__main__":
    main()
