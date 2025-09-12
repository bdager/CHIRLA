import os
import sys
import argparse


def get_dataset_name(config_file):
    """Extract dataset name from config file path"""
    if "Market1501" in config_file:
        return "Market1501"
    elif "DukeMTMC" in config_file:
        return "DukeMTMC"
    elif "MSMT17" in config_file:
        return "MSMT17"
    elif "VeRi" in config_file:
        return "VeRi"
    elif "VehicleID" in config_file:
        return "VehicleID"
    elif "VERIWild" in config_file:
        return "VERIWild"
    else:
        return "Unknown"

def get_model_name(config_file, checkpoint_file):
    """Generate model name from config and checkpoint"""
    dataset = get_dataset_name(config_file)
    
    if "bagtricks" in config_file or "bot" in checkpoint_file:
        method = "bagtricks"
    elif "AGW" in config_file or "agw" in checkpoint_file:
        method = "AGW"
    elif "sbs" in config_file:
        method = "sbs"
    elif "mgn" in config_file:
        method = "mgn"
    else:
        method = "unknown"
    
    if "R101" in config_file:
        backbone = "R101-ibn"
    elif "R50" in config_file:
        backbone = "R50-ibn"
    else:
        backbone = "unknown"
    
    return f"{dataset}/{method}_{backbone}"


def filter_csv_files(csv_files, exclude_val=False, exclude_train=False):
    """
    Filter CSV files based on exclusion criteria.
    
    Args:
        csv_files (list): List of CSV file paths
        exclude_val (bool): If True, exclude validation files (*_val.csv)
        exclude_train (bool): If True, exclude training files (*_train.csv)
    
    Returns:
        list: Filtered list of CSV file paths
    """
    filtered_files = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # Check if file should be excluded
        if exclude_val and filename.endswith('_val.csv'):
            print(f"⏭️  Excluding validation file: {filename}")
            continue
        
        if exclude_train and filename.endswith('_train.csv'):
            print(f"⏭️  Excluding training file: {filename}")
            continue
        
        filtered_files.append(csv_file)
    
    return filtered_files


def main():
    parser = argparse.ArgumentParser(description="Generate FastReID embeddings for CHIRLA benchmark metadata")
    parser.add_argument("--input-dir", 
                        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/data/CHIRLA/benchmark",
                        help="Input directory containing the CHIRLA benchmark data")
    parser.add_argument("--output-dir", 
                        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/fastreid",
                        help="Output directory for embeddings")
    parser.add_argument("--metadata-dir", 
                        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/metadata",
                        help="Directory containing metadata CSV files")
    parser.add_argument("--fastreid-dir", 
                        default="/home/bdager/Dropbox/work/phd/rebuttal_2/fast-reid",
                        help="FastReID repository directory")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding extraction")
    parser.add_argument("--device", default="cuda",
                        help="Device for computation (cuda/cpu)")
    parser.add_argument("--exclude-val", action="store_true",
                        help="Exclude validation CSV files (*_val.csv)")
    parser.add_argument("--exclude-train", action="store_true",
                        help="Exclude training CSV files (*_train.csv)")
    parser.add_argument("--models", nargs="+", 
                        choices=["market", "duke", "msmt", "veri", "vehicleid", "veriwild", "all"],
                        default=["all"],
                        help="Select specific models to run (default: all)")
    parser.add_argument("--skip", action="store_true", help="Skip CSV/model pairs whose embedding file already exists")
    
    args = parser.parse_args()

    # Ensure fast-reid repo is on sys.path BEFORE importing create_embeddings / fastreid
    fastreid_repo = os.path.abspath(args.fastreid_dir)
    if fastreid_repo not in sys.path:
        sys.path.insert(0, fastreid_repo)
    # Also add current benchmark script directory (for create_embeddings.py)
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)

    try:
        from create_embeddings import extract_embeddings
    except ModuleNotFoundError as e:
        print(f"❌ Could not import FastReID modules: {e}")
        print("   Make sure the fast-reid repository path is correct and contains the 'fastreid' package.")
        print("   Suggested fixes:\n     - git clone https://github.com/JDAI-CV/fast-reid.git\n     - pip install -e fast-reid (or) ensure this script's --fastreid-dir points to repo root")
        return

    # Define all model configurations (restored)
    all_configs = {
        "market": [
            ("configs/Market1501/bagtricks_R101-ibn.yml", "checkpoints/market_bot_R101-ibn.pth"),
            ("configs/Market1501/AGW_R101-ibn.yml", "checkpoints/market_agw_R101-ibn.pth"),
            ("configs/Market1501/sbs_R101-ibn.yml", "checkpoints/market_sbs_R101-ibn.pth"),
            ("configs/Market1501/mgn_R50-ibn.yml", "checkpoints/market_mgn_R50-ibn.pth"),
        ],
        "duke": [
            ("configs/DukeMTMC/bagtricks_R101-ibn.yml", "checkpoints/duke_bot_R101-ibn.pth"),
            ("configs/DukeMTMC/AGW_R101-ibn.yml", "checkpoints/duke_agw_R101-ibn.pth"),
            ("configs/DukeMTMC/sbs_R101-ibn.yml", "checkpoints/duke_sbs_R101-ibn.pth"),
            ("configs/DukeMTMC/mgn_R50-ibn.yml", "checkpoints/duke_mgn_R50-ibn.pth"),
        ],
        "msmt": [
            ("configs/MSMT17/bagtricks_R101-ibn.yml", "checkpoints/msmt_bot_R101-ibn.pth"),
            ("configs/MSMT17/AGW_R101-ibn.yml", "checkpoints/msmt_agw_R101-ibn.pth"),
            ("configs/MSMT17/sbs_R101-ibn.yml", "checkpoints/msmt_sbs_R101-ibn.pth"),
        ],
        "veri": [
            ("configs/VeRi/sbs_R50-ibn.yml", "checkpoints/veri_sbs_R50-ibn.pth"),
        ],
        "vehicleid": [
            ("configs/VehicleID/bagtricks_R50-ibn.yml", "checkpoints/vehicleid_bot_R50-ibn.pth"),
        ],
        "veriwild": [
            ("configs/VERIWild/bagtricks_R50-ibn.yml", "checkpoints/veriwild_bot_R50-ibn.pth"),
        ]
    }

    # Select models to run
    selected_models = []
    if "all" in args.models:
        for model_configs in all_configs.values():
            selected_models.extend(model_configs)
    else:
        for model_name in args.models:
            if model_name in all_configs:
                selected_models.extend(all_configs[model_name])

    print(f"Selected {len(selected_models)} model configurations to run")

    # Change to fast-reid directory for proper relative paths
    original_dir = os.getcwd()
    os.chdir(args.fastreid_dir)
    
    try:
        # Collect all ReID metadata CSV files
        csv_files = []
        if not os.path.exists(args.metadata_dir):
            raise FileNotFoundError(f"Metadata directory not found: {args.metadata_dir}")
        
        for csv_file in os.listdir(args.metadata_dir):
            if "reid" in csv_file and csv_file.endswith(".csv"):
                csv_files.append(os.path.join(args.metadata_dir, csv_file))
        
        if not csv_files:
            print("❌ No ReID CSV files found in metadata directory")
            return
        
        print(f"Found {len(csv_files)} ReID CSV files")
        
        # Filter CSV files based on exclusion criteria
        csv_files = filter_csv_files(csv_files, args.exclude_val, args.exclude_train)
        
        if not csv_files:
            print("❌ No CSV files remaining after filtering")
            return
        
        print(f"Processing {len(csv_files)} CSV files after filtering:")
        for csv_file in csv_files:
            print(f"  - {os.path.basename(csv_file)}")
        
        # Process each model configuration
        total_processed = 0
        total_failed = 0
        
        for config_file, checkpoint_file in selected_models:
            model_name = get_model_name(config_file, checkpoint_file)
            print(f"\n{'='*60}")
            print(f"Processing model: {model_name}")
            print(f"Config: {config_file}")
            print(f"Checkpoint: {checkpoint_file}")
            print(f"{'='*60}")
            
            # Create output directory for this model
            model_output_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Process each CSV file
            for csv_file in csv_files:
                csv_name = os.path.basename(csv_file).replace('.csv', '')
                # Determine expected output file path
                output_name = os.path.basename(csv_file) + "_embeddings.h5"
                output_name = output_name.replace("reid_", "").replace(".csv", "")
                output_file = os.path.join(model_output_dir, output_name)
                if args.skip and os.path.exists(output_file):
                    print(f"⏭️  Skipping {csv_name} (exists): {output_file}")
                    continue
                print(f"\nProcessing CSV: {csv_name}")
                
                try:
                    # Extract embeddings
                    extract_embeddings(
                        csv_path=csv_file,
                        input_dir=args.input_dir,
                        output_dir=model_output_dir,
                        model_name="fastreid",
                        cfg_file=config_file,
                        cktp_file=checkpoint_file,
                        batch_size=args.batch_size,
                        device=args.device
                    )
                    print(f"✅ Successfully processed {csv_name} with {model_name}")
                    total_processed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {csv_name} with {model_name}: {e}")
                    total_failed += 1
                    continue
        
        print(f"\n{'='*60}")
        print("EMBEDDING EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total successful: {total_processed}")
        print(f"Total failed: {total_failed}")
        print(f"Success rate: {total_processed/(total_processed+total_failed)*100:.1f}%" if (total_processed+total_failed) > 0 else "N/A")
        print(f"{'='*60}")
    
    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()