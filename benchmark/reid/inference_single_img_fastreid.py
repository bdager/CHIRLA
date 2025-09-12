import torch
import cv2
import numpy as np
from PIL import Image
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.utils.visualizer import Visualizer
from fastreid.utils.logger import setup_logger
from fastreid.data.transforms import build_transforms
import argparse


# Setup logger
setup_logger(name="fastreid")

# --- Load configuration ---
cfg = get_cfg()
cfg.merge_from_file("configs/Market1501/bagtricks_R101-ibn.yml")
cfg.MODEL.WEIGHTS = "checkpoints/market_bot_R101-ibn.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# --- Load and preprocess image ---
img_path = "demo.png"
img = cv2.imread(img_path)
height, width = img.shape[:2]

# Convert BGR to RGB and then to PIL Image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# Build transforms for preprocessing
transforms = build_transforms(cfg, is_train=False)
img_tensor = transforms(img_pil)

# Add batch dimension
img_tensor = img_tensor.unsqueeze(0)

# --- Run inference ---
outputs = predictor(img_tensor)

# --- Feature vector ---
# The outputs are feature vectors directly, not instances
print("Output type:", type(outputs))
print("Output shape:", outputs.shape)
feats = outputs  # outputs are the feature vectors directly
print("Extracted feature shape:", feats.shape)
