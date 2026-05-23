import torch
import cv2
import numpy as np

class DINODescriptor:
    def __init__(self):
        pass

    def extract(self, img):
        # Deterministic mock extraction: downsample image to 16x16x3 = 768 to act as a real descriptor baseline
        # img is usually a numpy array (H, W, 3) from cv2
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Ensure it's NHWC or HWC. The render_gaussians output is HWC
        img_small = cv2.resize(img.astype(np.uint8), (16, 16))
        vec = img_small.astype(np.float32).flatten()
        
        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return torch.from_numpy(vec)
