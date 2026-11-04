#!/usr/bin/env python3
"""
Generate embeddings for the CHIRLA Re-ID benchmark and save them to HDF5.

Supported models
----------------
fastreid   FastReID (https://github.com/JDAI-CV/fast-reid), see inference_fastreid.py
differ     DIFFER (https://github.com/xliangp/DIFFER)
sapiensid  SapiensID (https://github.com/mk-minchul/sapiensid)

DIFFER and SapiensID are used from a local clone of their repositories, by default models/DIFFER and models/sapiensid next to this script (override with --repo-dir).
See README.md for the setup.

HDF5 output layout:
    embeddings  float32  [N, D]   L2-normalized
    ids         int32    [N]
    paths       str      [N]      image paths as in the CSV, relative to --input

Usage
-----
    python create_embeddings.py --csv metadata/reid_long_term_gallery.csv \\
        --input /path/to/CHIRLA/benchmark --output embeddings/differ \\
        --model differ \\
        --config checkpoints/DIFFER/Celeb_light/config.yml \\
        --cktp   checkpoints/DIFFER/Celeb_light/eva02_l_bio_best.pth

    python create_embeddings.py --csv metadata/reid_long_term_gallery.csv \\
        --input /path/to/CHIRLA/benchmark --output embeddings/sapiensid \\
        --model sapiensid --cktp checkpoints/SapiensID/sapiensid_wb12m
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_DIRS = {
    "differ": os.path.join(SCRIPT_DIR, "models", "DIFFER"),
    "sapiensid": os.path.join(SCRIPT_DIR, "models", "sapiensid"),
}


def _prepend_to_sys_path(*dirs):
    # Both repos import their own top-level modules (`config`, `model`, `src`, ...),
    # so they have to be importable from their root instead of as a package.
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Repository not found at {d}. See README.md (Create Embeddings).")
        if d not in sys.path:
            sys.path.insert(0, d)


# -----------------------
# 1. Model loading
# -----------------------
def load_model(model_name="fastreid", config_file=None, checkpoint_file=None, device="cuda",
               repo_dir=None):
    """Load a backbone model for feature extraction.

    Returns (model, emb_dim), where model has .transform(PIL image) -> Tensor [C, H, W]
    and .__call__(Tensor [B, C, H, W]) -> Tensor [B, D].
    """
    if model_name == "fastreid":
        from inference_fastreid import FastReIDModel
        model = FastReIDModel(config_file=config_file, cktp=checkpoint_file)
        return model, 2048  # embedding dimension
    elif model_name == "differ":
        model = DIFFERModel(config_file, checkpoint_file, device=device,
                            repo_dir=repo_dir or DEFAULT_REPO_DIRS["differ"])
        return model, model.emb_dim
    elif model_name == "sapiensid":
        model = SapiensIDModel(checkpoint_file, device=device,
                               repo_dir=repo_dir or DEFAULT_REPO_DIRS["sapiensid"])
        return model, model.emb_dim
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


# -----------------------
# 2. DIFFER
# -----------------------
class DIFFERModel:
    """DIFFER (CVPR 2025), an EVA02-CLIP based clothes-changing Re-ID model.

    DIFFER's test.py needs a dataset to know `num_classes` and `camera_num`. Both are read here from the checkpoint's `head`/`camera_embed` shapes instead, so the model can be built without a dataloader. In eval mode it returns the biometric feature.
    """

    def __init__(self, config_file, checkpoint_file, device="cuda", repo_dir=DEFAULT_REPO_DIRS["differ"]):
        from torchvision import transforms as T

        _prepend_to_sys_path(repo_dir)
        from config import cfg as differ_cfg
        from model import build_model

        cfg = differ_cfg.clone()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["TEST.WEIGHT", checkpoint_file])

        state_dict = torch.load(checkpoint_file, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        num_classes = state_dict["head.weight"].shape[0]
        camera_num = state_dict["camera_embed"].shape[0]

        self.model = build_model(cfg, num_classes, camera_num)
        # strict=False: some checkpoints carry a `head_fc.bias` for the training-only
        # gradient-reversal classifiers, a layer that is built with bias=False.
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device).eval()

        self.device = device
        self.emb_dim = self.model.embed_dim
        self._transform = T.Compose([
            T.Resize((cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATA.PIXEL_MEAN, std=cfg.DATA.PIXEL_STD),
        ])

    def transform(self, img):
        return self._transform(img)

    @torch.no_grad()
    def __call__(self, batch_tensor):
        cam_label = torch.zeros(batch_tensor.shape[0], dtype=torch.long, device=self.device)
        return self.model(batch_tensor, cam_label=cam_label).float()


# -----------------------
# 3. SapiensID
# -----------------------
class SapiensIDModel:
    """SapiensID, a ViT Re-ID model conditioned on body and face keypoints.

    The keypoints come from SapiensID's YOLO + DFA aligner (17 body + 5 face points),
    which runs on every batch before the model.
    """

    def __init__(self, checkpoint_dir, device="cuda", repo_dir=DEFAULT_REPO_DIRS["sapiensid"]):
        src_dir = os.path.join(repo_dir, "tasks", "sapiensID")
        _prepend_to_sys_path(repo_dir, src_dir)

        # SapiensID's dependencies still use NumPy aliases removed in NumPy 2.
        if not hasattr(np, "float_"):
            np.float_ = np.float64
        if not hasattr(np, "float"):
            np.float = np.float64
        if not hasattr(np, "object"):
            np.object = np.object_

        from omegaconf import OmegaConf
        from src.models.sapiens_id import load_model as load_sapiensid
        from src.aligners import get_aligner

        config = OmegaConf.load(os.path.join(checkpoint_dir, "model.yaml"))
        self.model = load_sapiensid(config)
        self.model.load_state_dict_from_path(os.path.join(checkpoint_dir, "model.pth"))
        self.model.to(device).eval()

        aligner_config = OmegaConf.load(os.path.join(src_dir, "src", "aligners", "configs", "yolo_dfa.yaml"))
        aligner_config.rgb_mean = config.rgb_mean
        aligner_config.rgb_std = config.rgb_std
        self.aligner = get_aligner(aligner_config)
        self.aligner.to(device).eval()

        self.device = device
        self.emb_dim = config.output_dim
        self._transform = self.model.make_test_transform()

    def transform(self, img):
        return self._transform(img)

    @torch.no_grad()
    def __call__(self, batch_tensor):
        keypoints, foreground_masks = self.aligner(batch_tensor)
        return self.model(batch_tensor, foreground_masks=foreground_masks, ldmks=keypoints).float()


# -----------------------
# 4. Embedding Extraction to HDF5
# -----------------------
def extract_embeddings(csv_path, input_dir, output_dir,
                    model_name="fastreid",
                    cfg_file="configs/Market1501/bagtricks_R101-ibn.yml",
                    cktp_file="checkpoints/market_bot_R101-ibn.pth",
                    batch_size=32, device="cuda", repo_dir=None):
    df = pd.read_csv(csv_path)
    rel_paths = df["image_path"].tolist()
    image_paths = [os.path.join(input_dir, path) for path in rel_paths]
    ids = df["id"].astype(int).tolist()

    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # HDF5 output file
    output_name = os.path.basename(csv_path) + "_embeddings.h5"
    output_name = output_name.replace("reid_","").replace(".csv", "")
    output_file = os.path.join(output_dir, output_name)
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping extraction.")
        return

    model, emb_dim = load_model(model_name, cfg_file, cktp_file, device, repo_dir=repo_dir)

    num_images = len(image_paths)
    print(f"Processing {num_images} images from {csv_path}")

    # Only images that load are written, unused rows are cut at the end
    skipped = []
    written = 0
    with h5py.File(output_file, "w") as h5f:
        emb_dataset = h5f.create_dataset("embeddings", shape=(num_images, emb_dim),
                                         maxshape=(None, emb_dim), dtype=np.float32)
        ids_dataset = h5f.create_dataset("ids", shape=(num_images,), maxshape=(None,), dtype=np.int32)
        paths_dataset = h5f.create_dataset("paths", shape=(num_images,), maxshape=(None,),
                                           dtype=h5py.string_dtype())

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
                    img = Image.open(path).convert("RGB")
                    img = model.transform(img)
                    batch_images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Warning: Skipping corrupted image {path}: {e}")
                    skipped.append(path)
                    continue

            if not batch_images:
                print(f"Warning: No valid images in batch starting at {start}")
                continue

            # Update batch data to only include valid images.
            # Paths are stored as in the CSV (relative to --input)
            valid_batch_ids = [batch_ids[i] for i in valid_indices]
            valid_batch_paths = [rel_paths[start + i] for i in valid_indices]

            batch_tensor = torch.stack(batch_images).to(device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = model(batch_tensor)  # [B, emb_dim]
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)  # L2 normalize

            # Rows are packed one after another
            n = len(valid_indices)
            emb_dataset[written:written + n] = embeddings.cpu().numpy()
            ids_dataset[written:written + n] = valid_batch_ids
            paths_dataset[written:written + n] = valid_batch_paths
            written += n

        if written < num_images:
            for ds in (emb_dataset, ids_dataset, paths_dataset):
                ds.resize(written, axis=0)

    print(f"✅ Saved {written} embeddings to {output_file}")
    if skipped:
        print(f"⚠️  {len(skipped)} image(s) could not be read and are missing from the file. "
              f"Results are not comparable with the leaderboard until they are fixed. First ones:\n  - "
              + "\n  - ".join(skipped[:10]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for CHIRLA ReID benchmark and save to HDF5.")
    parser.add_argument("--csv", required=True,
            help="Path to metadata CSV (e.g., reid_long_term_gallery.csv)")
    parser.add_argument("--input", required=True,
            help="CHIRLA benchmark root; the CSV image paths are relative to it")
    parser.add_argument("--output", required=True,
            help="Directory to save embeddings.h5")
    parser.add_argument("--model", default="fastreid", choices=["fastreid", "differ", "sapiensid"],
            help="Re-ID model")
    parser.add_argument("--batch-size", type=int, default=32,
            help="Batch size for embedding extraction")
    parser.add_argument("--config",
            default="configs/Market1501/bagtricks_R101-ibn.yml",
            help="[fastreid, differ] Path to model configuration file")
    parser.add_argument("--cktp",
            default="checkpoints/market_bot_R101-ibn.pth",
            help="[fastreid, differ] Path to checkpoint file. "
                 "[sapiensid] Checkpoint folder with model.yaml and model.pth")
    parser.add_argument("--repo-dir", default=None,
            help="[differ, sapiensid] Local clone of the model's repository "
                 "(default: models/DIFFER or models/sapiensid next to this script)")
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
        device=args.device,
        repo_dir=args.repo_dir,
    )
