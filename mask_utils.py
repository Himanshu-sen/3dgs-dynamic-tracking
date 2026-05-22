import os

import torch


def detect_rasterizer(checkpoint):
    model = torch.load(checkpoint, map_location="cpu")
    if isinstance(model, tuple):
        return "original"
    if isinstance(model, dict) and "splats" in model:
        return "gsplat"
    raise ValueError("Unsupported checkpoint format; cannot detect rasterizer.")


def load_aligned_mask(mask_path, splat_count, device):
    mask = torch.load(mask_path).to(device=device).bool()
    if mask.shape[0] == splat_count:
        return mask

    root, ext = os.path.splitext(mask_path)
    full_mask_path = f"{root}_full{ext}"
    if os.path.exists(full_mask_path):
        full_mask = torch.load(full_mask_path).to(device=device).bool()
        if full_mask.shape[0] == splat_count:
            print(f"Loaded full-size mask from {full_mask_path}")
            return full_mask

    raise ValueError(
        "mask3d.pth and checkpoint have different Gaussian counts "
        f"({mask.shape[0]} vs {splat_count}). Use the matching checkpoint or full-size mask."
    )
