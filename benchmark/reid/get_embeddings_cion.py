# create_embeddings_cion.py
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import numpy as np
import timm
import h5py

# ----------------------------
# CION backbone + transforms
# ----------------------------
def _build_transform(img_size=(256, 128)):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def _load_cion_backbone(model_name: str, ckpt_path: str, device: str, img_size: Optional[tuple] = None):
    """
    Create a timm backbone and load CION weights.
    Uses num_classes=0 + global_pool='avg' to output feature vectors directly.
    If the checkpoint has pos_embed for a different grid, pass img_size=(H,W).
    Also tries to auto-resolve ViT variants (patch size / hybrid) by matching patch_embed shapes.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint first (to inspect shapes)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)

    # Basic key remap (common patterns)
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('backbone.'):
            nk = nk[len('backbone.'):]
        if nk.startswith('head') or nk.startswith('fc') or nk.startswith('classifier'):
            # skip task-specific heads
            continue
        new_state[nk] = v

    # Optionally adapt model_name for ViT based on patch_embed shape
    def _make(model_name_try: str):
        create_kwargs = dict(pretrained=False, num_classes=0, global_pool='avg')
        if img_size is not None:
            create_kwargs['img_size'] = img_size
        try:
            return timm.create_model(model_name_try, **create_kwargs), None
        except TypeError as e:
            # Some models may not accept img_size; retry without
            create_kwargs.pop('img_size', None)
            try:
                return timm.create_model(model_name_try, **create_kwargs), None
            except Exception as e2:
                return None, e2
        except Exception as e:
            return None, e

    target_model_name = model_name
    if model_name.startswith('vit_') and 'patch_embed.proj.weight' in new_state:
        in_ch_ck = new_state['patch_embed.proj.weight'].shape[1]
        k_ck = new_state['patch_embed.proj.weight'].shape[2]
        # Build candidate names to try
        base = model_name.split('_')[0]  # 'vit'
        size = model_name.split('_')[1]  # e.g., 'small'
        candidates = [
            f"vit_{size}_patch16_224",
            f"vit_{size}_patch8_224",
            f"vit_{size}_r26_s32_224",
            f"vit_{size}_r26_s16_224",
            "vit_base_r50_s16_224" if size != 'base' else "vit_base_patch16_224",
        ]
        chosen = None
        for name_try in candidates:
            m_try, err = _make(name_try)
            if m_try is None:
                continue
            try:
                w = m_try.patch_embed.proj.weight
                in_ch_m, k_m = w.shape[1], w.shape[2]
                if in_ch_m == in_ch_ck and k_m == k_ck:
                    chosen = name_try
                    break
            except Exception:
                # Some variants may not expose patch_embed in same way
                pass
        if chosen is not None and chosen != model_name:
            print(f"[info] Adapting ViT model from '{model_name}' -> '{chosen}' based on patch_embed shape match ({in_ch_ck}ch, k={k_ck})")
            target_model_name = chosen

    # Swin auto-resolution (window size / variant)
    if model_name.startswith('swin_'):
        def _get_ck_rpbt_len(state_dict):
            for k, v in state_dict.items():
                if 'relative_position_bias_table' in k and isinstance(v, torch.Tensor):
                    return v.shape[0]
            return None
        rpbt_len_ck = _get_ck_rpbt_len(new_state)

        # Try a set of candidate names likely to match checkpoints
        size_token = 'tiny' if 'tiny' in model_name else ('small' if 'small' in model_name else None)
        swin_candidates = []
        if size_token == 'tiny':
            swin_candidates = [
                'swin_tiny_patch4_window7_224',
                'swin_tiny_patch4_window12_384',
                'swinv2_tiny_window8_256',
            ]
        elif size_token == 'small':
            swin_candidates = [
                'swin_small_patch4_window7_224',
                'swin_small_patch4_window12_384',
                'swinv2_small_window8_256',
            ]
        # Always include original as first try
        if target_model_name not in swin_candidates:
            swin_candidates = [target_model_name] + swin_candidates

        best = None
        for name_try in swin_candidates:
            m_try, err = _make(name_try)
            if m_try is None:
                continue
            # Compare relative_position_bias_table length if available
            try:
                rpbt_mod = None
                for mod_k, mod_v in m_try.state_dict().items():
                    if 'relative_position_bias_table' in mod_k:
                        rpbt_mod = mod_v.shape[0]
                        break
                if rpbt_len_ck is not None and rpbt_mod is not None and rpbt_len_ck == rpbt_mod:
                    best = name_try
                    break
            except Exception:
                pass
        if best and best != target_model_name:
            print(f"[info] Adapting Swin model from '{target_model_name}' -> '{best}' based on relative_position_bias_table match")
            target_model_name = best

    # Finally, create the model
    model, err = _make(target_model_name)
    # If model isn't known in this timm version, try aliases (e.g., EdgeNeXt variants)
    if model is None:
        alias_map = {
            'edgenext_xsmall': ['edgenext_x_small', 'edgenext_xx_small', 'edgenext_small'],
        }
        tried = []
        if target_model_name in alias_map:
            for alt in alias_map[target_model_name]:
                m_try, err2 = _make(alt)
                tried.append(alt)
                if m_try is not None:
                    print(f"[info] Adapting model name '{target_model_name}' -> '{alt}' (alias fallback)")
                    model, err = m_try, None
                    target_model_name = alt
                    break
        if model is None:
            raise RuntimeError(
                f"Failed to create model '{target_model_name}': {err}. "
                + (f"Tried aliases: {tried}. " if tried else "")
                + "Consider updating timm (pip install -U timm)."
            )

    if model is None:
        raise RuntimeError(f"Failed to create model '{target_model_name}': {err}")

    model.eval().to(device)

    # Load weights with robustness to size mismatches (pos_embed, etc.)
    try:
        incompatible = model.load_state_dict(new_state, strict=False)
        missing = getattr(incompatible, 'missing_keys', []) if incompatible is not None else []
        unexpected = getattr(incompatible, 'unexpected_keys', []) if incompatible is not None else []
    except RuntimeError as e:
        # Handle common mismatches by pruning offending keys
        pruned = False
        prune_substrings = [
            'pos_embed',
            'patch_embed.proj.weight',
            'patch_embed.proj.bias',
            'relative_position_bias_table',
            'relative_position_index',
            'downsample.',
        ]
        for key in list(new_state.keys()):
            if any(s in key for s in prune_substrings):
                if key in str(e) or 'size mismatch' in str(e):
                    new_state.pop(key, None)
                    print(f"[warn] Dropped '{key}' from checkpoint due to size mismatch; using model's parameter.")
                    pruned = True
        if pruned:
            incompatible = model.load_state_dict(new_state, strict=False)
            missing = getattr(incompatible, 'missing_keys', []) if incompatible is not None else []
            unexpected = getattr(incompatible, 'unexpected_keys', []) if incompatible is not None else []
        else:
            raise

    if unexpected:
        print(f"[warn] Unexpected keys ignored: {len(unexpected)}")
    if missing:
        print(f"[warn] Missing keys (likely heads/not used): {len(missing)}")
    print(f"[ok] Loaded CION weights from {ckpt_path}")

    return model

# ----------------------------
# Dataset
# ----------------------------
class ReIDCSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, img_col_candidates: List[str], transform):
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform

        # Resolve image path column
        self.img_col = None
        for c in img_col_candidates:
            if c in self.df.columns:
                self.img_col = c
                break
        if self.img_col is None:
            raise ValueError(f"None of the image path columns found: {img_col_candidates}")

        # Build absolute paths
        abs_paths = []
        for rel in self.df[self.img_col].astype(str).tolist():
            p = rel if os.path.isabs(rel) else os.path.join(self.root_dir, rel)
            abs_paths.append(os.path.abspath(p))
        self.df['abs_path'] = abs_paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['abs_path']
        try:
            img = Image.open(path).convert('RGB')
            x = self.transform(img)
            return x, idx, path, True
        except Exception:
            # Return a marker for a failed sample; collate will filter it out
            return None, idx, path, False

# ----------------------------
# Helpers (CSV filtering)
# ----------------------------
def _filter_csv_files(csv_files, exclude_val=False, exclude_train=False):
    filtered = []
    for p in csv_files:
        name = os.path.basename(p)
        if exclude_val and name.endswith('_val.csv'):
            continue
        if exclude_train and name.endswith('_train.csv'):
            continue
        filtered.append(p)
    return filtered

def _collate_filter(batch):
    """Filter out failed samples from a batch and report their paths.
    Returns (xb, idxs, paths_ok, failed_paths). If no valid samples, xb is None.
    """
    xs, idxs, paths_ok, failed = [], [], [], []
    for x, idx, p, ok in batch:
        if ok and x is not None:
            xs.append(x)
            idxs.append(idx)
            paths_ok.append(p)
        else:
            failed.append(p)
    if len(xs) == 0:
        return None, [], [], failed
    return torch.stack(xs, dim=0), torch.as_tensor(idxs, dtype=torch.long), paths_ok, failed

# ----------------------------
# Public API (drop-in)
# ----------------------------
@torch.no_grad()
def extract_embeddings(
    csv_path: str,
    input_dir: str,
    output_dir: str,
    model_name: str,
    cfg_file: Optional[str] = None,   # unused; kept for signature compatibility
    cktp_file: Optional[str] = None,  # path to CION checkpoint (.pth)
    batch_size: int = 8,
    device: str = "cuda",
    img_h: int = 256,
    img_w: int = 128,
    csv_img_cols: Optional[List[str]] = None,
    skip_existing: bool = False,
):
    """
    Save embeddings for images listed in a metadata CSV, in FastReID-compatible HDF5 format.

    Writes an HDF5 file with datasets: 'embeddings' [N,D] float32, 'ids' [N] int32, 'paths' [N] string.
    Output filename mirrors get_embeddings_fastreid.py/create_embeddings.py: <csv_basename>_embeddings.h5
    with 'reid_' prefix removed and '.csv' stripped.
    """
    if cktp_file is None:
        raise ValueError("cktp_file is required and must point to a CION checkpoint (.pth)")

    # I/O setup
    csv_path = os.path.abspath(csv_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_name = Path(csv_path).name  # keep basename for naming
    # Build HDF5 output file name consistent with FastReID extractor
    h5_name = f"{csv_name}_embeddings.h5"
    h5_name = h5_name.replace("reid_", "").replace(".csv", "")
    h5_out = os.path.join(output_dir, h5_name)

    # Respect skip flag; otherwise overwrite
    if skip_existing and os.path.exists(h5_out):
        print(f"⏭️  Skipping (exists): {h5_out}")
        return

    print(f"[info] Reading metadata: {csv_path}")
    df = pd.read_csv(csv_path)

    # Prefer 'image_path' like FastReID, but support other common columns
    img_cols = csv_img_cols or ['image_path', 'path', 'img_path', 'image', 'filepath', 'filename']
    transform = _build_transform((img_h, img_w))

    dataset = ReIDCSVDataset(df, input_dir, img_cols, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate_filter,
    )

    # Load CION backbone
    print(f"[info] Loading backbone: {model_name}")
    model = _load_cion_backbone(model_name, cktp_file, device, img_size=(img_h, img_w))

    # Feature extraction (with robustness to corrupted files)
    features = {}  # idx -> np.ndarray
    failed_all = []  # list of corrupted file paths
    emb_dim = None

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for xb, idxs, paths_ok, failed in loader:
        # Log any failures from this batch
        if failed:
            failed_all.extend(failed)
        if xb is None or len(idxs) == 0:
            # Entire batch failed; continue
            continue

        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=amp_dtype, enabled=True):
            fb = model(xb)  # [B, D]
        fb = torch.nn.functional.normalize(fb, dim=1)
        fb_np = fb.cpu().float().numpy()

        if emb_dim is None:
            emb_dim = fb_np.shape[1]

        for i, idx in enumerate(idxs.tolist()):
            features[int(idx)] = fb_np[i]

    if emb_dim is None or len(features) == 0:
        print("❌ No valid embeddings were extracted (all images failed?)")
        if failed_all:
            print(f"First 10 corrupted files:\n  - " + "\n  - ".join(failed_all[:10]))
        return

    # Assemble full embedding matrix, filling zeros for failed images
    N = len(dataset)
    feats_full = np.zeros((N, emb_dim), dtype=np.float32)
    for idx, vec in features.items():
        if 0 <= idx < N:
            feats_full[idx] = vec

    # Prepare ids and paths arrays consistent with FastReID HDF5 outputs
    if 'id' not in df.columns:
        raise ValueError("CSV must contain an 'id' column for person IDs")
    ids = df['id'].astype(np.int32).to_numpy()
    paths = dataset.df['abs_path'].astype(str).to_list()

    # Write HDF5
    with h5py.File(h5_out, "w") as h5f:
        h5f.create_dataset("embeddings", data=feats_full, dtype=np.float32)
        h5f.create_dataset("ids", data=ids, dtype=np.int32)
        h5f.create_dataset("paths", data=paths, dtype=h5py.string_dtype())

    print(f"✅ Saved embeddings to {h5_out}  (shape={feats_full.shape})")

    if failed_all:
        sidecar = os.path.join(output_dir, h5_name + ".corrupted.txt")
        try:
            with open(sidecar, 'w') as f:
                for p in failed_all:
                    f.write(f"{p}\n")
            print(f"⚠️  {len(failed_all)} corrupted images skipped. Logged to: {sidecar}")
            print("  First 10 corrupted files:\n  - " + "\n  - ".join(failed_all[:10]))
        except Exception as e:
            print(f"[warn] Failed to write corrupted file list: {e}")

# ----------------------------
# Preset model mapping (key -> (timm_model_name, checkpoint_filename))
# ----------------------------
PRESETS = {
    # ConvNeXt
    "convnext_base": ("convnext_base", "ConvNext_Base_teacher.pth"),
    "convnext_small": ("convnext_small", "ConvNext_Small_teacher.pth"),
    "convnext_tiny": ("convnext_tiny", "ConvNext_Tiny_teacher.pth"),

    # EdgeNext
    "edgenext_base": ("edgenext_base", "EdgeNext_Base_teacher.pth"),
    "edgenext_small": ("edgenext_small", "EdgeNext_Small_teacher.pth"),
    "edgenext_xsmall": ("edgenext_xsmall", "EdgeNext_XSmall_teacher.pth"),

    # FastViT (names assume timm availability)
    "fastvit_s12": ("fastvit_s12", "FastViT_S12_teacher.pth"),
    "fastvit_sa12": ("fastvit_sa12", "FastViT_SA12_teacher.pth"),
    "fastvit_sa24": ("fastvit_sa24", "FastViT_SA24_teacher.pth"),

    # GhostNet
    "ghostnet_050": ("ghostnet_050", "GhostNet_0_5_teacher.pth"),
    "ghostnet_100": ("ghostnet_100", "GhostNet_1_0_teacher.pth"),
    "ghostnet_130": ("ghostnet_130", "GhostNet_1_3_teacher.pth"),

    # RepViT
    "repvit_m0_9": ("repvit_m0_9", "RepViT_m0_9_teacher.pth"),
    "repvit_m1_0": ("repvit_m1_0", "RepViT_m1_0_teacher.pth"),
    "repvit_m1_5": ("repvit_m1_5", "RepViT_m1_5_teacher.pth"),

    # ResNets (IBN variants mapped to base timm resnet as fallback)
    "resnet18": ("resnet18", "ResNet18_teacher.pth"),
    "resnet18_ibn": ("resnet18", "ResNet18_IBN_teacher.pth"),
    "resnet50": ("resnet50", "ResNet50_teacher.pth"),
    "resnet50_ibn": ("resnet50", "ResNet50_IBN_teacher.pth"),
    "resnet101": ("resnet101", "ResNet101_teacher.pth"),
    "resnet101_ibn": ("resnet101", "ResNet101_IBN_teacher.pth"),
    "resnet152": ("resnet152", "ResNet152_teacher.pth"),
    "resnet152_ibn": ("resnet152", "ResNet152_IBN_teacher.pth"),

    # Swin
    "swin_tiny": ("swin_tiny_patch4_window7_224", "Swin_Tiny_teacher.pth"),
    "swin_small": ("swin_small_patch4_window7_224", "Swin_Small_teacher.pth"),

    # ViT
    "vit_tiny": ("vit_tiny_patch16_224", "ViT_Tiny_teacher.pth"),
    "vit_small": ("vit_small_patch16_224", "ViT_Small_teacher.pth"),

    # VOLO
    "volo_d1": ("volo_d1_224", "VOLO_D1_teacher.pth"),
    "volo_d2": ("volo_d2_224", "VOLO_D2_teacher.pth"),
    "volo_d3": ("volo_d3_224", "VOLO_D3_teacher.pth"),
}

# Models to run with an additional higher input height (384x128)
RESNET_MULTI_SIZE_KEYS = {
    "resnet50", "resnet50_ibn", "resnet101", "resnet101_ibn", "resnet152", "resnet152_ibn"
}

# ----------------------------
# Optional CLI (debug/local)
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CION ReID - CSV embedding extractor")
    ap.add_argument("--csv", default=None, help="Path to a single metadata CSV. If omitted, process all CSVs in --metadata-dir")
    ap.add_argument("--input-dir",
        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/data/CHIRLA/benchmark",
        help="Root dir for relative image paths in CSV")
    ap.add_argument("--output-dir",
        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark",
        help="Root directory to save per-model *_embeddings.h5")
    ap.add_argument("--metadata-dir",
        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/metadata",
        help="Directory containing metadata CSV files (used when --csv is not provided)")

    # Model selection (like get_embeddings_fastreid)
    ap.add_argument("--models", nargs="+",
        choices=["all", "custom", *sorted(PRESETS.keys())],
        default=["all"],
        help="Which CION models to run. 'custom' uses --model/--ckpt; 'all' loads all presets present in --models-dir")
    ap.add_argument("--models-dir",
        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/models/cion",
        help="Directory containing CION checkpoints (.pth) for presets")

    # Custom single-model args (used when --models includes 'custom')
    ap.add_argument("--model", default="convnext_base",
                    help="timm model name matching the CION checkpoint (e.g. resnet50, resnet50d, resnet50_ibn, convnext_base)")
    ap.add_argument("--ckpt",
        default="/home/bdager/Dropbox/work/phd/rebuttal_2/CHIRLA/benchmark/models/ConvNext_Base_teacher.pth",
        help="Path to CION checkpoint .pth")

    # Runtime and filters
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--img-h", type=int, default=256)
    ap.add_argument("--img-w", type=int, default=128)
    ap.add_argument("--exclude-val", action="store_true", help="Exclude validation CSV files (*_val.csv)")
    ap.add_argument("--exclude-train", action="store_true", help="Exclude training CSV files (*_train.csv)")
    ap.add_argument("--skip", action="store_true", help="Skip CSVs already processed in the output directory")

    args = ap.parse_args()

    # Build list of CSVs to process
    csv_list = []
    if args.csv is not None and os.path.isfile(args.csv):
        csv_list = [os.path.abspath(args.csv)]
    else:
        if not os.path.isdir(args.metadata_dir):
            print(f"❌ Metadata directory not found: {args.metadata_dir}")
            sys.exit(1)
        all_csvs = [os.path.join(args.metadata_dir, f) for f in os.listdir(args.metadata_dir)
                    if f.endswith('.csv') and 'reid' in f]
        csv_list = _filter_csv_files(all_csvs, args.exclude_val, args.exclude_train)
        if not csv_list:
            print("❌ No CSV files to process after filtering")
            sys.exit(1)
        print(f"Found {len(csv_list)} CSV files to process:")
        for p in csv_list:
            print(f"  - {os.path.basename(p)}")

    # Resolve selected model configurations
    selected = []
    if "all" in args.models:
        for key, (timm_name, ck_name) in PRESETS.items():
            ck_path = os.path.join(args.models_dir, ck_name)
            if os.path.isfile(ck_path):
                selected.append((key, timm_name, ck_path))
            else:
                print(f"[warn] Preset '{key}' checkpoint not found: {ck_path} — skipping")
    else:
        for key in args.models:
            if key == "custom":
                selected.append(("custom", args.model, args.ckpt))
                continue
            if key not in PRESETS:
                print(f"[warn] Unknown preset '{key}' — skipping")
                continue
            timm_name, ck_name = PRESETS[key]
            ck_path = os.path.join(args.models_dir, ck_name)
            if os.path.isfile(ck_path):
                selected.append((key, timm_name, ck_path))
            else:
                print(f"[warn] Preset '{key}' checkpoint not found: {ck_path} — skipping")

    if not selected:
        print("[warn] No presets selected or found. Falling back to --model/--ckpt")
        selected = [("custom", args.model, args.ckpt)]

    print(f"Selected {len(selected)} model configuration(s):")
    for key, m, ck in selected:
        print(f"  - {key}: model={m}, ckpt={ck}")

    # Process
    ok, fail, skipped = 0, 0, 0
    for key, m, ck in selected:
        # Determine size variants: always base size; add 384x128 variant for specified ResNet models (if different)
        size_variants = [(args.img_h, args.img_w)]
        if key in RESNET_MULTI_SIZE_KEYS and (args.img_h, args.img_w) != (384, 128):
            size_variants.append((384, 128))

        for (cur_h, cur_w) in size_variants:
            size_suffix = f"_h{cur_h}" if (cur_h, cur_w) != (args.img_h, args.img_w) or len(size_variants) > 1 else ""
            model_out_dir = os.path.join(args.output_dir, f"CION/{key}{size_suffix}")
            os.makedirs(model_out_dir, exist_ok=True)
            print(f"\n{'='*60}\nProcessing model: {key} (input {cur_h}x{cur_w})\nOutput dir: {model_out_dir}\n{'='*60}")

            for csv_path in csv_list:
                # Compute expected output path for skipping check
                csv_base = os.path.basename(csv_path)
                h5_name = f"{csv_base}_embeddings.h5".replace("reid_", "").replace(".csv", "")
                h5_out = os.path.join(model_out_dir, h5_name)
                if args.skip and os.path.isfile(h5_out):
                    print(f"⏭️  Skipping {csv_base} for model {key}{size_suffix} — already exists: {h5_out}")
                    skipped += 1
                    continue
                try:
                    extract_embeddings(
                        csv_path=csv_path,
                        input_dir=args.input_dir,
                        output_dir=model_out_dir,
                        model_name=m,
                        cktp_file=ck,
                        batch_size=args.batch_size,
                        device=args.device,
                        img_h=cur_h,
                        img_w=cur_w,
                        skip_existing=args.skip,
                    )
                    ok += 1
                except Exception as e:
                    print(f"❌ Error processing {os.path.basename(csv_path)} with model {key}{size_suffix}: {e}")
                    fail += 1

    total = ok + fail + skipped
    if total:
        print(f"\nSummary: {ok} succeeded, {fail} failed, {skipped} skipped ({ok/total*100:.1f}% success)")
        print(f"Outputs written under: {args.output_dir}")
    else:
        print("No files processed")
