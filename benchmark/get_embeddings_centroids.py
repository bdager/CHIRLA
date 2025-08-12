#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
import shutil
import tempfile
from pathlib import Path
import logging
import numpy as np
import h5py
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_name(config_file):
    """Extract dataset name from config file path"""
    if "market1501" in config_file:
        return "Market1501"
    elif "dukemtmc" in config_file:
        return "DukeMTMC"
    else:
        return "Unknown"
    

def get_model_name(config_file, checkpoint_file):
    """Generate model name from config and checkpoint"""
    dataset = get_dataset_name(checkpoint_file)  
    
    if "256_resnet50" in config_file:
        backbone = "256_resnet50"
    elif "320_resnet50_ibn_a" in config_file:
        backbone = "320_resnet50_ibn_a"
    else:
        backbone = "unknown"
    
    return f"{dataset}/{backbone}"


def find_csv_files(metadata_dir):
    """
    Find all CSV metadata files in the benchmark/metadata directory.
    Returns a list of (csv_name, csv_path) tuples.
    """
    csv_files = []
    
    if not os.path.exists(metadata_dir):
        logger.error(f"Metadata directory not found: {metadata_dir}")
        return csv_files
    
    for file in os.listdir(metadata_dir):
        if file.endswith('.csv'):
            if "reid" not in file:
                continue
            csv_path = os.path.join(metadata_dir, file)
            csv_files.append((file, csv_path))
    
    logger.info(f"Found {len(csv_files)} CSV files: {[f[0] for f in csv_files]}")
    return csv_files

def create_flat_structure(csv_path, data_root, temp_dir):
    """
    Create a flat directory structure from a CHIRLA CSV file.
    Returns (flat_dir, image_count, mapping) where mapping maps flattened filename -> original absolute path.
    """
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    flat_dir = os.path.join(temp_dir, csv_name)
    os.makedirs(flat_dir, exist_ok=True)
    mapping = {}
    try:
        image_count = 0
        missing_count = 0
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            row_count = 0
            for row in reader:
                row_count += 1
                image_path = row.get('image_path', '')
                person_id = row.get('id', '')
                camera = row.get('camera', '')
                sequence = row.get('sequence', '')
                subset = row.get('subset', '')
                if not image_path:
                    logger.warning(f"Row {row_count}: No image_path found")
                    continue
                full_image_path = os.path.join(data_root, image_path)
                if os.path.exists(full_image_path):
                    original_filename = os.path.basename(image_path)
                    new_name = f"{subset}_{sequence}_{camera}_{person_id}_{original_filename}"
                    dst_path = os.path.join(flat_dir, new_name)
                    shutil.copy2(full_image_path, dst_path)
                    mapping[new_name] = full_image_path  # store original absolute path
                    image_count += 1
                else:
                    missing_count += 1
                    if missing_count <= 5:
                        logger.warning(f"Image not found: {full_image_path}")
            logger.info(f"Processed {row_count} CSV entries")
        if missing_count > 5:
            logger.warning(f"... and {missing_count - 5} more missing images")
        logger.info(f"Created flat structure with {image_count} images in {flat_dir}")
        logger.info(f"Missing images: {missing_count}")
        return flat_dir, image_count, mapping
    except Exception as e:
        logger.error(f"Error processing CSV {csv_path}: {e}")
        return flat_dir, 0, mapping

def extract_ids_from_paths(paths):
    """
    Extract person IDs from paths supporting two formats:
      1) Flattened: subset_sequence_camera_personid_originalname (person id is 4th token)
      2) Hierarchical: .../test/test_4/seq_006/.../<camera_dir>/<person_id>/frame_xxx.png
         The person ID is taken as the numeric directory immediately above the frame file.
    Returns list of ints (unknown -> -1).
    """
    ids = []
    warn_limit = 10
    warns = 0
    for path in paths:
        pid = None  # Use None to indicate parsing failure
        filename = os.path.basename(path)
        parts_fname = filename.split('_')
        # Try flattened pattern
        if len(parts_fname) >= 4:
            try:
                pid = int(parts_fname[3])
            except ValueError:
                pid = None
        # If not flattened (pid still None), try hierarchical pattern
        if pid is None:
            path_parts = os.path.normpath(path).split(os.sep)
            if len(path_parts) >= 2:
                candidate = path_parts[-2]  # directory containing frame file
                # Support negative IDs too (e.g., '-4', '-1')
                if candidate.lstrip('-').isdigit():
                    pid = int(candidate)
                else:
                    # Fallback: search backwards for first (possibly negative) digit folder
                    for seg in reversed(path_parts[:-2]):
                        if seg.lstrip('-').isdigit():
                            pid = int(seg)
                            break
        if pid is None and warns < warn_limit:
            logger.warning(f"Could not parse ID from path: {path}")
            warns += 1
        ids.append(pid if pid is not None else -1)  # Use -1 as final fallback for storage
    if warns == warn_limit:
        logger.warning("Further ID parse warnings suppressed...")
    return ids

def check_existing_hdf5(output_dir, csv_name):
    """
    Check if HDF5 file already exists for this CSV.
    """
    h5_filename = f"{csv_name}_embeddings.h5"
    h5_filename = h5_filename.replace("reid_", "").replace(".csv", "")
    h5_file = os.path.join(output_dir, h5_filename)
    return os.path.exists(h5_file), h5_file

def convert_numpy_to_hdf5(output_dir, csv_name, path_mapping=None):
    """
    Convert NumPy embeddings format to HDF5 format matching FastReID.
    If path_mapping is provided (flattened filename -> original path), use original paths in HDF5.
    """
    embeddings_file = os.path.join(output_dir, "embeddings.npy")
    paths_file = os.path.join(output_dir, "paths.npy")
    if not os.path.exists(embeddings_file) or not os.path.exists(paths_file):
        logger.error(f"NumPy files not found in {output_dir}")
        return False
    try:
        embeddings = np.load(embeddings_file)
        paths = np.load(paths_file)
        # Map flattened paths back to originals if mapping provided
        if path_mapping:
            remapped_paths = []
            for p in paths:
                base = os.path.basename(p)
                remapped_paths.append(path_mapping.get(base, p))
            paths = np.array(remapped_paths)
        ids = extract_ids_from_paths(paths)
        h5_filename = f"{csv_name}_embeddings.h5".replace("reid_", "").replace(".csv", "")
        h5_file = os.path.join(output_dir, h5_filename)
        with h5py.File(h5_file, "w") as h5f:
            h5f.create_dataset("embeddings", data=embeddings, dtype=np.float32)
            h5f.create_dataset("ids", data=np.array(ids, dtype=np.int32))
            paths_str = [str(p) for p in paths]
            h5f.create_dataset("paths", data=paths_str, dtype=h5py.string_dtype())
        logger.info(f"✅ Converted to HDF5 format: {h5_file}")
        os.remove(embeddings_file)
        os.remove(paths_file)
        logger.info(f"📁 Removed original NumPy files")
        return True
    except Exception as e:
        logger.error(f"Error converting to HDF5: {e}")
        return False

def run_embedding_extraction(input_dir, output_dir, config_file, 
                            model_path, gpu_id=0, batch_size=8, repo_root='.'):
    """Run the embedding extraction for a directory of images from centroids-reid repo root."""
    script_path = os.path.join(repo_root, 'inference', 'create_embeddings.py')
    if not os.path.isfile(script_path):
        logger.error(f"create_embeddings.py not found at {script_path}")
        return False
    cmd = [
        sys.executable, 'inference/create_embeddings.py',
        '--config_file', config_file,
        'GPU_IDS', f'[{gpu_id}]',
        'DATASETS.ROOT_DIR', input_dir,
        'TEST.IMS_PER_BATCH', str(batch_size),
        'OUTPUT_DIR', output_dir,
        'MODEL.PRETRAIN_PATH', model_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=repo_root)
        logger.info("✅ Successfully extracted embeddings")
        if result.stdout:
            logger.debug(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error("❌ Failed to extract embeddings")
        logger.error(f"Command: {' '.join(cmd)} (cwd={repo_root})")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout.strip()[:1000]}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr.strip()[:1000]}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for CHIRLA benchmark using centroids-reid and CSV metadata")
    parser.add_argument("--metadata-dir", 
                       default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/metadata",
                       help="Directory containing CSV metadata files")
    parser.add_argument("--data-root", 
                       default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/data/CHIRLA/benchmark",
                       help="Root directory of CHIRLA benchmark data")
    parser.add_argument("--output-root", 
                       default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/centroids_reid",
                       help="Root directory for output embeddings")
    parser.add_argument("--config-file", 
                       default="configs/256_resnet50.yml",
                       help="Path to config file")
    parser.add_argument("--model", 
                       default="models/market1501_resnet50_256_128_epoch_120.ckpt",
                       help="Path to model checkpoint")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--csv-files", nargs="+", default=None,
                       help="Specific CSV files to process (if None, process all)")
    parser.add_argument("--centroids_dir",
        default="/home/bdager/Dropbox/work/phd/rebuttal_2/centroids-reid",
        help="centroids-reid repository path")
    parser.add_argument("--exclude-val", action="store_true",
                       help="Exclude validation files (*_val.csv)")
    parser.add_argument("--exclude-train", action="store_true",
                       help="Exclude training files (*_train.csv)")
    parser.add_argument("--skip-existing", action="store_true",
        help="Skip files that have already been processed (HDF5 file exists)")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory    
    centroids_reid_repo = os.path.abspath(args.centroids_dir)
    if centroids_reid_repo not in sys.path:
        sys.path.insert(0, centroids_reid_repo)
    # Also add current benchmark script directory (for create_embeddings.py)
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)

    # try:
    #     from create_embeddings import extract_embeddings
    # except ModuleNotFoundError as e:
    #     print(f"❌ Could not import FastReID modules: {e}")
    #     print("   Make sure the fast-reid repository path is correct and contains the 'fastreid' package.")
    #     print("   Suggested fixes:\n     - git clone https://github.com/JDAI-CV/fast-reid.git\n     - pip install -e fast-reid (or) ensure this script's --fastreid-dir points to repo root")
    #     return
    
    centroids_reid_repo = os.path.abspath(args.centroids_dir)
    if centroids_reid_repo not in sys.path:
        sys.path.insert(0, centroids_reid_repo)
    # Also add current benchmark script directory (for create_embeddings.py)
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    
    # Find all CSV files
    csv_files = find_csv_files(args.metadata_dir)
    if not csv_files:
        logger.error("No CSV files found!")
        return
    
    # Filter CSV files if specified
    if args.csv_files:
        csv_files = [(name, path) for name, path in csv_files if name in args.csv_files]
        logger.info(f"Processing specified CSV files: {[f[0] for f in csv_files]}")
    
    # Exclude validation files if requested
    if args.exclude_val:
        csv_files = [(name, path) for name, path in csv_files if '_val.csv' not in name]
        logger.info(f"Excluding validation files. Processing: {[f[0] for f in csv_files]}")
    
    # Exclude training files if requested
    if args.exclude_train:
        csv_files = [(name, path) for name, path in csv_files if '_train.csv' not in name]
        logger.info(f"Excluding training files. Processing: {[f[0] for f in csv_files]}")
    
    total_processed = 0
    total_failed = 0
    total_skipped = 0
    
    # Create temporary directory for flat structures
    with tempfile.TemporaryDirectory() as temp_root:
        logger.info(f"Using temporary directory: {temp_root}")
        
        # Process each CSV file
        for csv_name, csv_path in csv_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing CSV: {csv_name}")
            logger.info(f"{'='*60}")
            
            # Create output directory
            model_name = get_model_name(args.config_file, args.model)
            output_dir = os.path.join(args.output_root, model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if file already exists and skip if requested
            if args.skip_existing:
                exists, h5_path = check_existing_hdf5(output_dir, csv_name)
                if exists:
                    logger.info(f"⏭️  Skipping {csv_name} - HDF5 file already exists: {h5_path}")
                    total_skipped += 1
                    continue
            
            # Create flat structure from CSV
            flat_dir, image_count, path_mapping = create_flat_structure(
                csv_path, args.data_root, temp_root
            )
            
            if image_count == 0:
                logger.warning(f"No images found for {csv_name}")
                total_failed += 1
                continue
            
            # Run embedding extraction
            logger.info(f"Extracting embeddings for {image_count} images...")
            success = run_embedding_extraction(
                flat_dir, output_dir, args.config_file, args.model,
                args.gpu_id, args.batch_size, repo_root=centroids_reid_repo
            )
            
            if success:
                logger.info("Converting to HDF5 format (restoring original image paths)...")
                h5_success = convert_numpy_to_hdf5(output_dir, csv_name, path_mapping)
                
                if h5_success:
                    total_processed += 1
                    logger.info(f"✅ Completed {csv_name}")
                else:
                    total_failed += 1
                    logger.error(f"❌ Failed HDF5 conversion for {csv_name}")
            else:
                total_failed += 1
                logger.error(f"❌ Failed {csv_name}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total CSV files processed: {total_processed}")
    logger.info(f"Total failed: {total_failed}")
    logger.info(f"Total skipped: {total_skipped}")
    total_attempted = total_processed + total_failed
    logger.info(f"Success rate: {total_processed/total_attempted*100:.1f}%" if total_attempted > 0 else "No files attempted")
    logger.info(f"Output directory: {args.output_root}")

if __name__ == "__main__":
    main()
