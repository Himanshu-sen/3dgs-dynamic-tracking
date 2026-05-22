from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import torch

class SIFTMatcher:
    def __init__(self, device):
        self.device = device
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def __call__(self, img0, img1):
        # img0, img1 are torch tensors C x H x W in [0, 1] on GPU.
        # Conver to numpy uint8
        img0_np = (img0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img1_np = (img1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray0 = cv2.cvtColor(img0_np, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_BGR2GRAY)
        
        kp0, des0 = self.sift.detectAndCompute(gray0, None)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        
        if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
            return {'num_inliers': 0, 'H': np.eye(3), 'inlier_kpts0': np.zeros((0,2)), 'inlier_kpts1': np.zeros((0,2))}
            
        matches = self.bf.knnMatch(des0, des1, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4:
            return {'num_inliers': 0, 'H': np.eye(3), 'inlier_kpts0': np.zeros((0,2)), 'inlier_kpts1': np.zeros((0,2))}
            
        pts0 = np.float32([ kp0[m.queryIdx].pt for m in good ])
        pts1 = np.float32([ kp1[m.trainIdx].pt for m in good ])
        
        return {
            'num_inliers': len(good),
            'H': np.eye(3),  # we don't strictly need H for now
            'inlier_kpts0': pts0,
            'inlier_kpts1': pts1
        }


@dataclass
class _LoFTRBackendInfo:
    name: str
    ckpt_path: str


def _parse_backend_and_ckpt(name: str) -> _LoFTRBackendInfo:
    """Parse matcher name and an optional checkpoint path.

    Supported forms:
    - "eloftr"                         (uses env/default ckpt location)
    - "eloftr:/abs/path/to/ckpt.ckpt"  (explicit path)
    - "loftr"                          (alias)
    """
    raw = (name or "").strip()
    if ":" in raw:
        base, ckpt = raw.split(":", 1)
        base = base.strip().lower()
        ckpt = ckpt.strip()
        return _LoFTRBackendInfo(base, ckpt)

    base = raw.lower()
    ckpt = os.environ.get("ELOFTR_CKPT", "").strip()
    if not ckpt:
        # Common local default (user can drop weights here)
        ckpt = str(Path(__file__).resolve().parent / "EfficientLoFTR" / "weights" / "eloftr_outdoor.ckpt")
    return _LoFTRBackendInfo(base, ckpt)


class EfficientLoFTRMatcher:
    """EfficientLoFTR wrapper.

    Expects img0/img1 as torch tensors CxHxW in [0,1] on GPU.
    Returns a dict with 'inlier_kpts0' and 'inlier_kpts1' (Nx2 float32) and 'num_inliers'.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        self.ckpt_path = ckpt_path
        self.max_side = int(os.environ.get("ELOFTR_MAX_SIDE", "832") or "832")

        amp_raw = str(os.environ.get("ELOFTR_AMP", "1")).strip().lower()
        self.use_amp = amp_raw not in {"0", "false", "no", "off"}
        amp_dtype_raw = str(os.environ.get("ELOFTR_AMP_DTYPE", "bf16")).strip().lower()
        if amp_dtype_raw in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            # bfloat16 is typically very fast on Ampere+ and is more numerically stable than fp16.
            self.amp_dtype = torch.bfloat16

        # Ensure local EfficientLoFTR package is importable
        repo_root = Path(__file__).resolve().parent
        eloftr_root = repo_root / "EfficientLoFTR"
        if not eloftr_root.exists():
            raise RuntimeError(f"EfficientLoFTR directory not found at: {eloftr_root}")
        sys.path.insert(0, str(eloftr_root))

        try:
            from copy import deepcopy
            import importlib

            loftr_pkg = importlib.import_module("src.loftr")
            LoFTR = getattr(loftr_pkg, "LoFTR")
            full_default_cfg = getattr(loftr_pkg, "full_default_cfg")
            reparameter = getattr(loftr_pkg, "reparameter")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import EfficientLoFTR modules. "
                "Make sure EfficientLoFTR dependencies are installed (see EfficientLoFTR/requirements.txt)."
            ) from e

        ckpt_path = str(ckpt_path)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                "EfficientLoFTR checkpoint not found.\n"
                f"Tried: {ckpt_path}\n"
                "Fix options:\n"
                "- Download a pretrained checkpoint (e.g. eloftr_outdoor.ckpt) into EfficientLoFTR/weights/, or\n"
                "- Pass an explicit path via --matcher 'eloftr:/path/to/eloftr_outdoor.ckpt', or\n"
                "- Set env var ELOFTR_CKPT=/path/to/eloftr_outdoor.ckpt"
            )

        _cfg = deepcopy(full_default_cfg)
        self._matcher = LoFTR(config=_cfg)

        # PyTorch 2.6+ defaults weights_only=True, but EfficientLoFTR ckpts are often
        # PyTorch-Lightning checkpoints that include extra metadata. We explicitly
        # load with weights_only=False to allow reading the state_dict.
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch
            state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._matcher.load_state_dict(state, strict=False)

        # Essential for good performance
        self._matcher = reparameter(self._matcher)
        self._matcher = self._matcher.eval().to(device)

    @staticmethod
    def _to_gray(img: "torch.Tensor") -> "torch.Tensor":
        # img: CxHxW, BGR in [0,1]
        if img.dim() != 3:
            raise ValueError("Expected img tensor CxHxW")
        if img.shape[0] == 1:
            return img
        if img.shape[0] != 3:
            raise ValueError("Expected 1 or 3 channels")
        b, g, r = img[0], img[1], img[2]
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return gray.unsqueeze(0)

    @staticmethod
    def _resize_for_loftr(gray: "torch.Tensor", max_side: int) -> tuple["torch.Tensor", float, float]:
        """Resize 1xHxW tensor so max(H,W) <= max_side and both dims divisible by 32.

        Returns: (resized_gray, sx, sy) where multiplying coords by (sx, sy)
        maps from resized -> original.
        """
        _, h, w = gray.shape
        new_h, new_w = h, w

        if max_side and max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
            new_h = max(32, int(round(h * scale)))
            new_w = max(32, int(round(w * scale)))

        # Make divisible by 32 (floor)
        new_h = max(32, (new_h // 32) * 32)
        new_w = max(32, (new_w // 32) * 32)
        if new_h == h and new_w == w:
            return gray, 1.0, 1.0

        g = gray.unsqueeze(0)  # 1x1xHxW
        g2 = torch.nn.functional.interpolate(g, size=(new_h, new_w), mode="bilinear", align_corners=False)
        sy = float(h) / float(new_h)
        sx = float(w) / float(new_w)
        return g2.squeeze(0), sx, sy

    def __call__(self, img0: "torch.Tensor", img1: "torch.Tensor") -> dict:
        g0 = self._to_gray(img0)
        g1 = self._to_gray(img1)
        g0, sx0, sy0 = self._resize_for_loftr(g0, self.max_side)
        g1, sx1, sy1 = self._resize_for_loftr(g1, self.max_side)

        batch = {
            "image0": g0.unsqueeze(0).to(self.device),  # 1x1xH0xW0
            "image1": g1.unsqueeze(0).to(self.device),  # 1x1xH1xW1
        }

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=bool(self.use_amp)):
                self._matcher(batch)
            mk0 = batch.get("mkpts0_f", None)
            mk1 = batch.get("mkpts1_f", None)

        if mk0 is None or mk1 is None or mk0.numel() == 0:
            return {
                "num_inliers": 0,
                "H": np.eye(3),
                "inlier_kpts0": np.zeros((0, 2), dtype=np.float32),
                "inlier_kpts1": np.zeros((0, 2), dtype=np.float32),
            }

        k0 = mk0.detach().cpu().numpy().astype(np.float32)
        k1 = mk1.detach().cpu().numpy().astype(np.float32)

        # Scale back to original input resolution
        k0[:, 0] *= sx0
        k0[:, 1] *= sy0
        k1[:, 0] *= sx1
        k1[:, 1] *= sy1

        return {
            "num_inliers": int(len(k0)),
            "H": np.eye(3),
            "inlier_kpts0": k0,
            "inlier_kpts1": k1,
        }

def get_matcher(name, device="cuda"):
    n = (name or "").lower().strip()

    # Explicit SIFT for debugging
    if n in {"sift", "opencv-sift"}:
        return SIFTMatcher(device)

    # LoFTR / EfficientLoFTR
    if "loftr" in n or "eloftr" in n:
        info = _parse_backend_and_ckpt(name)
        try:
            return EfficientLoFTRMatcher(info.ckpt_path, device=device)
        except Exception as e:
            raise RuntimeError(
                "Requested a LoFTR matcher but it could not be initialized. "
                "This usually means the EfficientLoFTR checkpoint or dependencies are missing.\n\n"
                f"Matcher name: {name}\n"
                f"Details: {e}"
            ) from e

    # Default: keep prior behavior
    warnings.warn(
        f"Unknown matcher '{name}', falling back to SIFTMatcher. "
        "Tip: use --matcher eloftr:/path/to/eloftr_outdoor.ckpt",
        RuntimeWarning,
    )
    return SIFTMatcher(device)
