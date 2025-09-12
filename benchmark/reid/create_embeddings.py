#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
import h5py


# -----------------------
# 1. Load your model here
# -----------------------
def load_model(model_name="resnet50", config_file=None, checkpoint_file=None, device="cuda"):
    """Load a backbone model for feature extraction."""
    if model_name == "fastreid":
        from inference_fastreid import FastReIDModel
        model = FastReIDModel(config_file=config_file, cktp=checkpoint_file) 
        return model, 2048  # embedding dimension
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


# -----------------------sudo logidsudo logidsudo logidsudo logid
# 3. Embedding Extraction to HDF5sudo logid
# -----------------------
def extract_embeddings(csv_path, input_dir, output_dir, sudo logid
                    model_name="resnet50",
                    cfg_file="configs/Market1501/bagtricks_R101-ibn.yml",
                    cktp_file="checkpoints/market_bot_R101-ibn.pth",
                    batch_size=32, device="cuda"):
    df = pd.read_csv(csv_path)
    image_paths = df["image_path"].tolist()
    image_paths = [os.path.join(input_dir, path) for path in image_paths]
    ids = df["id"].astype(int).tolist()

    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # HDF5 output file
    output_name = csv_path.split("/")[-1] + "_embeddings.h5"
    output_name = output_name.replace("reid_","").replace(".csv", "")
    output_file = os.path.join(output_dir, output_name)
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping extraction.")
        return

    model, emb_dim = load_model(model_name, cfg_file, cktp_file, device)

    num_images = len(image_paths)
    print(f"Processing {num_images} images from {csv_path}")

    # Create HDF5 file
    with h5py.File(output_file, "w") as h5f:
        emb_dataset = h5f.create_dataset("embeddings", shape=(num_images, emb_dim), dtype=np.float32)
        ids_dataset = h5f.create_dataset("ids", shape=(num_images,), dtype=np.int32)
        paths_dataset = h5f.create_dataset("paths", shape=(num_images,), dtype=h5py.string_dtype())

        for start in tqdm(range(0, num_images, batch_size), 
                          desc="Extracting embeddings"):
            end = min(start + batch_size, num_images)
            batch_paths = image_paths[start:end]
            batch_ids = ids[start:end]

            # Load and transform batch
            batch_images = []
            valid_indices = []
            for i, path in enumerate(batch_paths):
                try:
                    # Paths in CSV are relative to benchmark root
                    img = Image.open(path).convert("RGB")
                    img = model.transform(img)
                    batch_images.append(img)
                    valid_indices.append(i)
                except (OSError, IOError, Exception) as e:
                    print(f"Warning: Skipping corrupted image {path}: {e}")
                    continue

            if not batch_images:
                print(f"Warning: No valid images in batch starting at {start}")
                continue

            # Update batch data to only include valid images
            valid_batch_ids = [batch_ids[i] for i in valid_indices]
            valid_batch_paths = [batch_paths[i] for i in valid_indices]

            batch_tensor = torch.stack(batch_images).to(device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = model(batch_tensor)  # [B, emb_dim]
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)  # L2 normalize

            # Save to HDF5 (only for valid images)
            for j, (emb, img_id, img_path) in enumerate(zip(embeddings.cpu().numpy(), valid_batch_ids, valid_batch_paths)):
                actual_idx = start + valid_indices[j]
                emb_dataset[actual_idx] = emb
                ids_dataset[actual_idx] = img_id
                paths_dataset[actual_idx] = img_path

    print(f"✅ Saved embeddings to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for CHIRLA ReID benchmark and save to HDF5.")
    parser.add_argument("--csv", required=True, 
            help="Path to metadata CSV (e.g., reid_long_term_train.csv)")
    parser.add_argument("--input", required=True, 
            help="Directory containing input images")
    parser.add_argument("--output", required=True, 
            help="Directory to save embeddings.h5")
    parser.add_argument("--model", default="fastreid", 
            help="Model name (resnet50, arcface, adaface, etc.)")
    parser.add_argument("--batch-size", type=int, default=8,
            help="Batch size for embedding extraction")
    parser.add_argument("--config",
            default="configs/Market1501/bagtricks_R101-ibn.yml",
            help="Path to model configuration file")
    parser.add_argument("--cktp",
            default="checkpoints/market_bot_R101-ibn.pth",
            help="Path to model checkpoint file")
    parser.add_argument("--device", default="cuda", help="Device for model inference (cuda or cpu)")

    args = parser.parse_args()

    extract_embeddings(
        csv_path=args.csv,
        input_dir=args.input,
        output_dir=args.output,
        model_name=args.model,
        cfg_file=args.config,
        cktp_file=args.cktp,
        batch_size=args.batch_size,
        device=args.device
    )
