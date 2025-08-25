# inference_cion.py
import argparse, os, glob
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm
import numpy as np

# -------------------------
# Utils
# -------------------------
def build_transform(img_size=(256, 128)):
    # Common ReID eval transforms
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def load_backbone(model_name: str, ckpt_path: str, device: str):
    """
    Create a timm backbone and load CION weights.
    You may need to adapt key mapping if checkpoint was saved from a different repo wrapper.
    """
    # Create backbone; remove classifier to expose features
    model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
    model.eval().to(device)

    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)

        # Try common key patterns; adapt here if needed
        # Example: remove "module." or "backbone." prefixes
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            if nk.startswith('backbone.'):
                nk = nk[len('backbone.'):]
            # If the checkpoint has a classifier head that's not in our model, skip it
            if nk.startswith('head') or nk.startswith('fc') or nk.startswith('classifier'):
                continue
            new_state[nk] = v

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[warn] Missing keys (likely heads not present): {len(missing)}")
        if unexpected:
            print(f"[warn] Unexpected keys (ignored): {len(unexpected)}")
        print(f"[ok] Loaded weights from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return model

@torch.no_grad()
def extract_dir(model, image_dir: str, transform, device: str):
    paths = sorted([p for p in glob.glob(os.path.join(image_dir, '**', '*'), recursive=True)
                    if os.path.splitext(p)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])

    feats, names = [], []
    for p in paths:
        img = Image.open(p).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        f = model(x)  # (1, D)
        f = nn.functional.normalize(f, dim=1)  # unit L2 for cosine sim
        feats.append(f.cpu())
        names.append(p)
    if not feats:
        raise RuntimeError(f"No images found in {image_dir}")
    feats = torch.cat(feats, dim=0)  # (N, D)
    return feats, names

def topk_indices(query_feat, gallery_feats, k=5):
    # cosine similarity (features are L2-normalized)
    sims = (gallery_feats @ query_feat.T).squeeze(1)  # (G,)
    vals, idxs = torch.topk(sims, k=min(k, gallery_feats.shape[0]))
    return vals.cpu().numpy(), idxs.cpu().numpy()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="CION ReID Inference (feature extraction + retrieval)")
    ap.add_argument('--model', type=str, required=True,
                    help="timm model name matching the CION checkpoint (e.g. resnet50, resnet50d, resnet50_ibn, convnext_base)")
    ap.add_argument('--ckpt', type=str, required=True, help="Path to CION checkpoint (.pth)")
    ap.add_argument('--gallery_dir', type=str, required=True, help="Folder of gallery images")
    ap.add_argument('--query_dir', type=str, required=True, help="Folder of query images")
    ap.add_argument('--img_h', type=int, default=256)
    ap.add_argument('--img_w', type=int, default=128)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    transform = build_transform((args.img_h, args.img_w))
    device = args.device

    print(f"[info] Loading model: {args.model}")
    model = load_backbone(args.model, args.ckpt, device)

    print(f"[info] Extracting gallery from: {args.gallery_dir}")
    g_feats, g_names = extract_dir(model, args.gallery_dir, transform, device)

    print(f"[info] Extracting queries from: {args.query_dir}")
    q_feats, q_names = extract_dir(model, args.query_dir, transform, device)

    print("\n=== Retrieval Results (cosine, top-{}) ===".format(args.topk))
    for qi in range(q_feats.shape[0]):
        qf = q_feats[qi:qi+1]
        vals, idxs = topk_indices(qf, g_feats, k=args.topk)
        print(f"\nQuery: {q_names[qi]}")
        for rank, (sim, gi) in enumerate(zip(vals, idxs), start=1):
            print(f"  #{rank}: {g_names[gi]}   sim={sim:.4f}")

if __name__ == '__main__':
    main()
