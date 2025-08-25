import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os

class FastReIDModel:
    def __init__(self, 
            config_file="configs/Market1501/bagtricks_R101-ibn.yml",
            cktp="checkpoints/market_bot_R101-ibn.pth",
            model_dir="/home/bdager/Dropbox/work/phd/rebuttal_2/fast-reid"):
        # Change to fast-reid directory for proper relative paths
        os.chdir(model_dir)
        from fastreid.config import get_cfg
        from fastreid.engine import DefaultPredictor
        from fastreid.utils.visualizer import Visualizer
        from fastreid.utils.logger import setup_logger
        from fastreid.data.transforms import build_transforms

        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = cktp
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.transforms = build_transforms(self.cfg, is_train=False)
        setup_logger(name="fastreid")
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, img_tensor):
        """Make the model callable for batch inference"""
        outputs = self.predictor(img_tensor)
        return outputs
    
    def transform(self, img):
        img_tensor = self.transforms(img)
        return img_tensor #.unsqueeze(0)  # Add batch dimension

    def run_inference(self, img):
        outputs = self.predictor(img)
        feats = outputs  # outputs are the feature vectors directly
        return feats
