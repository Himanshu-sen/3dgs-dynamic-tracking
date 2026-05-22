"""dynamic_object_pose_pnp.py

One-shot object motion estimation + update for 3D Gaussian Splatting.

This script implements the pipeline you described:
1) Render a reference view from the current 3DGS scene (RGB + depth).
2) Render a 2D object mask by rasterizing only the object Gaussians (mask3d.pth).
3) Match 2D keypoints between the reference render and a target image.
4) Filter matches to those lying on the object mask.
5) Backproject matched reference pixels to 3D using the rendered depth.
6) Solve for the object motion (SE(3)) using PnP + RANSAC.
7) Apply the transform to the object Gaussians (means + quats), and render updated scene.

Assumptions:
- The camera intrinsics are the same for reference render and target image.
- The camera pose is fixed (static camera), and only the object moves.
  (If the camera also moves, you must estimate camera pose separately.)

Usage (single image):
  python dynamic_object_pose_pnp.py \
    --checkpoint data/bottle/ckpt_29999_table_image_inpainted.pt \
    --data-dir data/bottle \
    --format inria \
    --mask3d-path results/bottle/mask3d.pth \
    --camera-pose-path results/bottle/optimized_camera_pose.pt \
    --target-path data/bottle/frames \
    --output-dir results/bottle/dynamic_pnp \
    --match-max-side 500

    
    python dynamic_object_pose_pnp.py \
    --checkpoint new/ckpts/chkpnt30000.pth \
    --data-dir new \
    --format inria \
    --mask3d-path new/mask3d.pth \
    --camera-pose-path new/camera_pose.pt \
    --target-path new/target_center_crop_20.jpg \
    --output-dir results/new/dynamic_pnp \
    --match-max-side 500
Usage (dynamic sequence):
  python dynamic_object_pose_pnp.py ... --target-path path/to/frames_dir

Outputs:
- Writes debug images (render/target/updated/match-viz) and JSON with PnP result.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot
import torch
import tyro

from gsplat import rasterization

from matching import get_matcher
from utils import load_checkpoint, torch_to_cv, get_viewmat_from_colmap_image, get_rpy_matrix


def _ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    device = torch.device("cuda")
    torch.set_default_device(device)
    return device


def _list_targets(target_path: str) -> list[str]:
    p = Path(target_path)
    if p.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in exts]
        if not files:
            raise ValueError(f"No image files found in directory: {target_path}")
        return files
    if not p.exists():
        raise FileNotFoundError(target_path)
    return [target_path]


def _bgr_uint8_to_torch_chw_float(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    if bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    # Native tensor pin + contiguous permute before moving to GPU float
    t = torch.from_numpy(bgr).permute(2, 0, 1).contiguous()
    return t.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0


def _rgb_hw3_to_bgr_chw(rgb_hw3: torch.Tensor) -> torch.Tensor:
    """Convert HxWx3 RGB float tensor -> 3xHxW BGR float tensor."""
    if rgb_hw3.dim() != 3 or rgb_hw3.shape[-1] != 3:
        raise ValueError("Expected HxWx3 RGB tensor")
    chw = rgb_hw3.permute(2, 0, 1).contiguous()
    return chw[[2, 1, 0], ...]


def _maybe_cuda_sync(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _warmup_rasterization(
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K_t: torch.Tensor,
    width: int,
    height: int,
    mask_idx: torch.Tensor,
    object_alpha_thresh: float,
    sync_timing: bool,
    iters: int = 1,
) -> None:
    """Warm up gsplat rasterization to reduce step-0 GPU cold-start overhead."""
    if iters <= 0:
        return
    if not torch.cuda.is_available():
        return

    with torch.inference_mode():
        for _ in range(int(iters)):
            # Full RGB+Depth render
            out, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmat[None],
                K_t[None],
                width=width,
                height=height,
                sh_degree=3,
                render_mode="RGB+ED",
            )
            # Touch outputs so kernels execute.
            _ = out[0, 0, 0, 0]

            # Alpha-only mask render (object subset)
            _, obj_alpha, _ = rasterization(
                means.index_select(0, mask_idx),
                quats.index_select(0, mask_idx),
                scales.index_select(0, mask_idx),
                opacities.index_select(0, mask_idx),
                colors.index_select(0, mask_idx)[:, :1, :],
                viewmat[None],
                K_t[None],
                width=width,
                height=height,
                sh_degree=0,
            )
            if obj_alpha.dim() == 4 and obj_alpha.shape[-1] == 1:
                alpha_hw = obj_alpha[0, ..., 0]
            else:
                alpha_hw = obj_alpha[0]
            _ = (alpha_hw > float(object_alpha_thresh)).any()

    _maybe_cuda_sync(sync_timing)


def _imread_resize_bgr(path: str, width: int, height: int) -> Optional[np.ndarray]:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    # Optimization: INTER_LINEAR is ~3x faster than INTER_AREA for downscaling
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_matches(img0: np.ndarray, img1: np.ndarray, kpts0: np.ndarray, kpts1: np.ndarray, max_lines: int = 300) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    out = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    out[:h0, :w0] = img0
    out[:h1, w0 : w0 + w1] = img1

    n = int(min(len(kpts0), len(kpts1), max_lines))
    for i in range(n):
        x0, y0 = kpts0[i]
        x1, y1 = kpts1[i]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + w0)), int(round(y1)))
        cv2.line(out, p0, p1, (0, 255, 255), 1)
        cv2.circle(out, p0, 2, (255, 0, 0), -1)
        cv2.circle(out, p1, 2, (0, 0, 255), -1)
    return out


def _overlay_hint(img: np.ndarray, text: str, y: int = 30) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return out


def _init_pose_trackbars(window_name: str) -> None:
    trackbars = {
        "Roll": (-180, 180),
        "Pitch": (-180, 180),
        "Yaw": (-180, 180),
        "X": (-1000, 1000),
        "Y": (-1000, 1000),
        "Z": (-1000, 1000),
    }
    for name, (min_val, max_val) in trackbars.items():
        cv2.createTrackbar(name, window_name, 0, max_val, lambda _x: None)
        cv2.setTrackbarMin(name, window_name, min_val)
        cv2.setTrackbarMax(name, window_name, max_val)


def _set_pose_trackbars_from_viewmat(window_name: str, world_to_camera: torch.Tensor) -> None:
    vm = world_to_camera.detach().cpu().numpy()
    r = SciRot.from_matrix(vm[:3, :3])
    roll, pitch, yaw = r.as_euler("xyz", degrees=True)
    cv2.setTrackbarPos("Roll", window_name, int(np.round(roll)))
    cv2.setTrackbarPos("Pitch", window_name, int(np.round(pitch)))
    cv2.setTrackbarPos("Yaw", window_name, int(np.round(yaw)))
    cv2.setTrackbarPos("X", window_name, int(np.round(vm[0, 3] * 100.0)))
    cv2.setTrackbarPos("Y", window_name, int(np.round(vm[1, 3] * 100.0)))
    cv2.setTrackbarPos("Z", window_name, int(np.round(vm[2, 3] * 100.0)))


def _get_viewmat_from_trackbars(window_name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    roll = float(cv2.getTrackbarPos("Roll", window_name))
    pitch = float(cv2.getTrackbarPos("Pitch", window_name))
    yaw = float(cv2.getTrackbarPos("Yaw", window_name))
    x = float(cv2.getTrackbarPos("X", window_name))
    y = float(cv2.getTrackbarPos("Y", window_name))
    z = float(cv2.getTrackbarPos("Z", window_name))

    vm = torch.tensor(
        get_rpy_matrix(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)),
        device=device,
        dtype=dtype,
    )
    vm[0, 3] = x / 100.0
    vm[1, 3] = y / 100.0
    vm[2, 3] = z / 100.0
    return vm


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply in (w,x,y,z). Supports broadcasting."""
    # Promote to (..., 4)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def _rotmat_to_quat_wxyz(R: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    q_xyzw = SciRot.from_matrix(R).as_quat()  # (x,y,z,w)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    q = torch.tensor(q_wxyz, device=device, dtype=dtype)
    return q / (q.norm() + 1e-8)


def _backproject_points(
    kpts0: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject reference pixels to 3D (camera coords) using depth.

    Returns:
      obj_pts: (N,3) float64
      valid_idx: (N,) indices into the original kpts0
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    h, w = depth.shape[:2]

    obj_pts = []
    valid_idx = []

    for i, (u, v) in enumerate(kpts0):
        if len(obj_pts) >= max_points:
            break
        x = int(round(u))
        y = int(round(v))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        z = float(depth[y, x])
        if not np.isfinite(z) or z <= 1e-6:
            continue
        X = (u - cx) / fx * z
        Y = (v - cy) / fy * z
        obj_pts.append([X, Y, z])
        valid_idx.append(i)

    if not obj_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.asarray(obj_pts, dtype=np.float64), np.asarray(valid_idx, dtype=np.int64)


@dataclass
class Args:
    checkpoint: str
    data_dir: str
    target_path: str

    # Scene / checkpoint format
    format: Literal["inria", "gsplat", "ply"] = "inria"
    data_factor: int = 4

    # Object mask
    mask3d_path: str = "results/bottle/mask3d.pth"

    # Camera pose (world->camera). If omitted, uses the first COLMAP image.
    camera_pose_path: Optional[str] = None

    # Matching
    matcher: str = "eloftr"  # EfficientLoFTR (recommended)
    # For an explicit checkpoint path: "eloftr:/path/to/eloftr_outdoor.ckpt"
    # For debugging only: "sift"
    match_max_side: int = 832
    max_match_lines: int = 300

    # Object mask threshold on alpha render
    object_alpha_thresh: float = 0.25

    # PnP
    max_correspondences: int = 250
    min_correspondences: int = 15
    ransac_iters: int = 100
    reproj_error_px: float = 4.0
    confidence: float = 0.999

    # Output
    output_dir: str = "results/bottle/dynamic_pnp"
    show_windows: bool = True

    # When showing windows and a directory is provided, control whether to iterate through frames
    # or keep reusing the first frame (viewer-style).
    repeat_first_target: bool = False
    # In windowed mode, wait for SPACE/ENTER before running compute for the current frame.
    pause_before_compute: bool = True
    # In windowed mode, wait for SPACE/ENTER after showing results to advance to next frame.
    pause_after_frame: bool = True

    # Performance (RTX A6000)
    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    # If True, adds CUDA sync points so printed per-stage timings reflect real GPU work.
    sync_timing: bool = True
    # If True, runs a one-time gsplat rasterization warmup during setup (helps step-0).
    warmup_rasterization: bool = True
    warmup_raster_iters: int = 1
    # Saving PNGs can dominate runtime; disable when benchmarking or running long sequences.
    save_debug: bool = False

    # Overlap CPU I/O with GPU compute when processing directories.
    prefetch_targets: bool = True
    prefetch_workers: int = 2


def main(args: Args) -> dict:
    device = _ensure_cuda()

    # Throughput-oriented defaults for Ampere GPUs.
    torch.backends.cuda.matmul.allow_tf32 = bool(args.enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.enable_tf32)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    try:
        # "high" enables TF32 fast paths when allowed.
        torch.set_float32_matmul_precision("high" if args.enable_tf32 else "highest")
    except Exception:
        pass

    t0_all = time.perf_counter()

    os.makedirs(args.output_dir, exist_ok=True)

    splats = load_checkpoint(args.checkpoint, args.data_dir, format=args.format, data_factor=args.data_factor)

    # Load object mask (boolean over gaussians)
    mask3d = torch.load(args.mask3d_path, weights_only=False)
    if not isinstance(mask3d, torch.Tensor):
        mask3d = torch.tensor(mask3d)
    if mask3d.dtype != torch.bool:
        mask3d = mask3d.bool()

    # Move splat params to GPU
    means = splats["means"].to(device=device, dtype=torch.float32)
    quats = splats["rotation"].to(device=device, dtype=torch.float32)
    scales = torch.exp(splats["scaling"].to(device=device, dtype=torch.float32))
    opacities = torch.sigmoid(splats["opacity"].to(device=device, dtype=torch.float32))
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], dim=1).to(device=device, dtype=torch.float32)

    K_t = splats["camera_matrix"].to(device=device, dtype=torch.float32)
    K_np = K_t.detach().cpu().numpy().astype(np.float64)

    width = int(float(K_np[0, 2]) * 2.0)
    height = int(float(K_np[1, 2]) * 2.0)

    # Align mask length with current splat count (common gotcha when mask was made for a different checkpoint)
    if mask3d.numel() != int(means.shape[0]):
        raise ValueError(
            f"mask3d length ({mask3d.numel()}) does not match splats count ({int(means.shape[0])}). "
            "Make sure mask3d.pth was generated for the same checkpoint/splats."
        )

    mask3d = mask3d.to(device=device)
    mask_idx = torch.nonzero(mask3d, as_tuple=False).flatten()

    # Camera pose (world->camera)
    if args.camera_pose_path is not None:
        cam_pose = torch.load(args.camera_pose_path, weights_only=False)
        if isinstance(cam_pose, dict) and "viewmat" in cam_pose:
            viewmat = cam_pose["viewmat"].to(device=device, dtype=torch.float32)
        else:
            raise ValueError(f"{args.camera_pose_path} must be a .pt with key 'viewmat'")
    else:
        images = list(splats["colmap_project"].images.values())
        if not images:
            raise RuntimeError("No COLMAP images found; provide --camera-pose-path")
        viewmat = get_viewmat_from_colmap_image(images[0]).to(device=device, dtype=torch.float32)

    os.environ["ELOFTR_MAX_SIDE"] = str(args.match_max_side)
    # Start I/O prefetch as early as possible so it overlaps with GPU warmups.
    target_files = _list_targets(args.target_path)

    executor: Optional[ThreadPoolExecutor] = None
    futures = []
    first_prefetched_bgr: Optional[np.ndarray] = None
    setup_prefetch_first_s: float = 0.0
    if bool(args.prefetch_targets) and len(target_files) > 1:
        executor = ThreadPoolExecutor(max_workers=max(1, int(args.prefetch_workers)))
        for f in target_files:
            futures.append(executor.submit(_imread_resize_bgr, f, width, height))

    matcher = get_matcher(args.matcher, device="cuda")

    # [Optimization] Warmup the matcher to avoid CuDNN benchmark overhead in the first step's timing.
    if args.cudnn_benchmark:
        dummy_img = torch.zeros((3, height, width), device=device, dtype=torch.float32)
        _ = matcher(dummy_img, dummy_img)
        _maybe_cuda_sync(args.sync_timing)

    if bool(args.warmup_rasterization):
        _warmup_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmat=viewmat,
            K_t=K_t,
            width=width,
            height=height,
            mask_idx=mask_idx,
            object_alpha_thresh=float(args.object_alpha_thresh),
            sync_timing=bool(args.sync_timing),
            iters=int(args.warmup_raster_iters),
        )

    # If prefetch is enabled, try to ensure the first image is ready before step 0.
    # Any residual waiting is counted in setup time instead of the first loop step.
    if futures:
        t0_pf = time.perf_counter()
        first_prefetched_bgr = futures[0].result()
        setup_prefetch_first_s = float(time.perf_counter() - t0_pf)

    # Make OpenCV windows optional
    window_target = "Target"
    window_updated = "Updated"
    window_matches = "Matches"
    if args.show_windows:
        cv2.namedWindow(window_target, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_updated, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_matches, cv2.WINDOW_NORMAL)
        _init_pose_trackbars(window_updated)
        _set_pose_trackbars_from_viewmat(window_updated, viewmat)

    # Everything up to here is "setup" (checkpoint load, tensor moves, matcher init, etc.)
    t_setup_done = time.perf_counter()

    # Mutable object params updated over time
    means_curr = means
    quats_curr = quats

    per_step_stats: list[dict] = []
    num_steps_ok = 0
    num_steps_total = 0

    stop_requested = False

    # Target iteration logic:
    # - If show_windows=False: process all targets sequentially.
    # - If show_windows=True and repeat_first_target=True (or only one target): viewer mode.
    # - If show_windows=True and repeat_first_target=False: iterate all frames and allow pausing.
    repeat_mode = bool(args.show_windows) and (bool(args.repeat_first_target) or len(target_files) == 1)
    target_iter = itertools.repeat(target_files[0]) if repeat_mode else iter(target_files)

    for step_idx, target_file in enumerate(target_iter):
        num_steps_total += 1
        # Two totals:
        # - total_step_s: compute time (in windowed mode starts at SPACE)
        # - total_step_including_load_s: end-to-end time (always starts at loop entry)
        t0_step_all = time.perf_counter()
        t0_step = t0_step_all
        timings: dict[str, float] = {}
        timings["wait_before_compute_s"] = 0.0
        timings["wait_after_frame_s"] = 0.0

        # 0) Load + resize target (prefetch when processing a directory)
        t0 = time.perf_counter()
        future_idx = 0 if repeat_mode else step_idx
        if step_idx == 0 and first_prefetched_bgr is not None:
            target_bgr = first_prefetched_bgr
        elif futures:
            target_bgr = futures[future_idx].result()
        else:
            target_bgr = _imread_resize_bgr(target_file, width, height)
        if target_bgr is None:
            print(f"[step {step_idx}] Could not read target: {target_file}")
            timings["load_target_s"] = time.perf_counter() - t0
            timings["total_step_s"] = time.perf_counter() - t0_step
            timings["total_step_including_load_s"] = time.perf_counter() - t0_step_all
            per_step_stats.append({"step": step_idx, "ok": False, "reason": "imread_failed", "timings_s": timings})
            continue
        timings["load_target_s"] = time.perf_counter() - t0

        # Windowed mode: optionally wait for SPACE/ENTER before running computation for this frame.
        # Compute timer starts exactly at keypress.
        space_pressed_t0: Optional[float] = None
        if args.show_windows and bool(args.pause_before_compute):
            t_wait0 = time.perf_counter()
            target_preview = _overlay_hint(
                target_bgr,
                f"[{step_idx + 1}/{len(target_files)}] Press SPACE/ENTER to update, Q/ESC to quit",
            )
            cv2.imshow(window_target, target_preview)

            while True:
                vm_disp = _get_viewmat_from_trackbars(window_updated, device=device, dtype=viewmat.dtype)
                with torch.inference_mode():
                    out_disp, _, _ = rasterization(
                        means_curr,
                        quats_curr,
                        scales,
                        opacities,
                        colors,
                        vm_disp[None],
                        K_t[None],
                        width=width,
                        height=height,
                        sh_degree=3,
                    )
                    disp_bgr = np.ascontiguousarray(torch_to_cv(out_disp[0]))

                disp_bgr = _overlay_hint(disp_bgr, "Updated preview: adjust Roll/Pitch/Yaw/X/Y/Z, SPACE to run")
                cv2.imshow(window_updated, disp_bgr)
                cv2.imshow(window_target, target_preview)
                key = cv2.waitKeyEx(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_requested = True
                    break
                if key in (ord(" "), 13):
                    space_pressed_t0 = time.perf_counter()
                    timings["wait_before_compute_s"] = float(space_pressed_t0 - t_wait0)
                    t0_step = space_pressed_t0
                    break

            if stop_requested:
                break

        # 1 & 2) Render reference RGB+Depth AND Object Alpha mask concurrently using CUDA Streams
        t0 = time.perf_counter()
        stream_ref = torch.cuda.Stream()
        stream_seg = torch.cuda.Stream()

        # Optional per-stream GPU timings (do not include overlap; can exceed wall time when summed).
        ref_ev0 = torch.cuda.Event(enable_timing=True)
        ref_ev1 = torch.cuda.Event(enable_timing=True)
        seg_ev0 = torch.cuda.Event(enable_timing=True)
        seg_ev1 = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream_ref):
            with torch.inference_mode():
                ref_ev0.record()
                out, _, _ = rasterization(
                    means_curr,
                    quats_curr,
                    scales,
                    opacities,
                    colors,
                    viewmat[None],
                    K_t[None],
                    width=width,
                    height=height,
                    sh_degree=3,
                    render_mode="RGB+ED",
                )
                rgb_hw3 = out[0, ..., :3]
                depth_hw = out[0, ..., 3]
                ref_bgr_t = _rgb_hw3_to_bgr_chw(rgb_hw3)
                ref_ev1.record()

        with torch.cuda.stream(stream_seg):
            with torch.inference_mode():
                seg_ev0.record()
                # Optimization: Mask generation only needs alpha coverage.
                # Passing only the DC features (`[:, :1, :]`) and `sh_degree=0` completely skips 
                # evaluating the expensive spherical harmonics equations on the GPU.
                _, obj_alpha, _ = rasterization(
                    means_curr.index_select(0, mask_idx),
                    quats_curr.index_select(0, mask_idx),
                    scales.index_select(0, mask_idx),
                    opacities.index_select(0, mask_idx),
                    colors.index_select(0, mask_idx)[:, :1, :],
                    viewmat[None],
                    K_t[None],
                    width=width,
                    height=height,
                    sh_degree=0,
                )
                if obj_alpha.dim() == 4 and obj_alpha.shape[-1] == 1:
                    alpha_hw = obj_alpha[0, ..., 0]
                else:
                    alpha_hw = obj_alpha[0]

                obj_mask_hw = alpha_hw > float(args.object_alpha_thresh)
                seg_ev1.record()

        # Wait for both distinct parallel workloads to finish
        torch.cuda.current_stream().wait_stream(stream_ref)
        torch.cuda.current_stream().wait_stream(stream_seg)
        _maybe_cuda_sync(args.sync_timing)
        render_wall_s = time.perf_counter() - t0
        timings["render_ref_s"] = float(render_wall_s)
        timings["render_ref_wall_s"] = float(render_wall_s)

        # Segment time can be overlapped under render time. We still report GPU-time for visibility.
        if args.sync_timing:
            # Events are only reliable if we wait for completion.
            ref_ev1.synchronize()
            seg_ev1.synchronize()
            timings["render_ref_gpu_s"] = float(ref_ev0.elapsed_time(ref_ev1) / 1e3)
            timings["segmentation_gpu_s"] = float(seg_ev0.elapsed_time(seg_ev1) / 1e3)
        else:
            timings["render_ref_gpu_s"] = 0.0
            timings["segmentation_gpu_s"] = 0.0

        # Keep the additive stage breakdown meaningful: segmentation is overlapped, so do not
        # add it on top of render wall time.
        timings["segmentation_s"] = 0.0

        # 3) Match
        t0 = time.perf_counter()
        img1_t = _bgr_uint8_to_torch_chw_float(target_bgr, device)
        _maybe_cuda_sync(args.sync_timing)
        timings["target_to_gpu_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        match = matcher(ref_bgr_t, img1_t)
        _maybe_cuda_sync(args.sync_timing)
        timings["match_s"] = time.perf_counter() - t0
        kpts0 = np.asarray(match.get("inlier_kpts0", np.zeros((0, 2))), dtype=np.float32)
        kpts1 = np.asarray(match.get("inlier_kpts1", np.zeros((0, 2))), dtype=np.float32)

        if len(kpts0) < 4:
            print(f"[step {step_idx}] Not enough matches ({len(kpts0)})")
            timings["total_step_s"] = time.perf_counter() - t0_step
            timings["total_step_including_load_s"] = time.perf_counter() - t0_step_all
            per_step_stats.append(
                {
                    "step": step_idx,
                    "ok": False,
                    "reason": "not_enough_matches",
                    "num_raw_matches": int(len(kpts0)),
                    "timings_s": timings,
                }
            )
            continue

        # 4/5) Filter matches to object pixels + valid depth, and backproject (GPU)
        t0 = time.perf_counter()
        h = int(obj_mask_hw.shape[0])
        w = int(obj_mask_hw.shape[1])

        kpts0_t = torch.from_numpy(kpts0).to(device=device, dtype=torch.float32)
        kpts1_t = torch.from_numpy(kpts1).to(device=device, dtype=torch.float32)

        xy = kpts0_t.round().to(dtype=torch.int64)
        x = xy[:, 0]
        y = xy[:, 1]
        in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if in_bounds.any():
            x = x[in_bounds]
            y = y[in_bounds]
            k0_in = kpts0_t[in_bounds]
            k1_in = kpts1_t[in_bounds]
        else:
            x = x[:0]
            y = y[:0]
            k0_in = kpts0_t[:0]
            k1_in = kpts1_t[:0]

        mask_ok = obj_mask_hw[y, x]
        z = depth_hw[y, x]
        depth_ok = torch.isfinite(z) & (z > 1e-6)
        valid = mask_ok & depth_ok

        k0_sel = k0_in[valid]
        k1_sel = k1_in[valid]
        z_sel = z[valid]

        # Limit correspondences for PnP
        if int(k0_sel.shape[0]) > int(args.max_correspondences):
            k0_sel = k0_sel[: int(args.max_correspondences)]
            k1_sel = k1_sel[: int(args.max_correspondences)]
            z_sel = z_sel[: int(args.max_correspondences)]

        t_backproj0 = time.perf_counter()
        # Backproject to 3D in camera coords
        fx = K_t[0, 0]
        fy = K_t[1, 1]
        cx = K_t[0, 2]
        cy = K_t[1, 2]
        u = k0_sel[:, 0]
        v = k0_sel[:, 1]
        X = (u - cx) / fx * z_sel
        Y = (v - cy) / fy * z_sel
        obj_pts_t = torch.stack((X, Y, z_sel), dim=1)
        img_pts_t = k1_sel

        _maybe_cuda_sync(args.sync_timing)
        timings["backproject_s"] = time.perf_counter() - t_backproj0
        timings["filter_obj_mask_s"] = max(0.0, float(time.perf_counter() - t0 - timings["backproject_s"]))

        k0 = k0_sel.detach().cpu().numpy().astype(np.float32)
        k1 = k1_sel.detach().cpu().numpy().astype(np.float32)
        obj_pts = obj_pts_t.detach().cpu().double().numpy()
        img_pts = img_pts_t.detach().cpu().double().numpy()

        if len(k0) < int(args.min_correspondences):
            print(f"[step {step_idx}] Too few object matches after mask ({len(k0)})")
            timings["total_step_s"] = time.perf_counter() - t0_step
            timings["total_step_including_load_s"] = time.perf_counter() - t0_step_all
            per_step_stats.append(
                {
                    "step": step_idx,
                    "ok": False,
                    "reason": "too_few_obj_matches",
                    "num_raw_matches": int(len(kpts0)),
                    "num_obj_matches": int(len(k0)),
                    "timings_s": timings,
                }
            )
            continue

        if len(obj_pts) < int(args.min_correspondences):
            print(f"[step {step_idx}] Too few 3D-2D correspondences ({len(obj_pts)})")
            timings["total_step_s"] = time.perf_counter() - t0_step
            timings["total_step_including_load_s"] = time.perf_counter() - t0_step_all
            per_step_stats.append(
                {
                    "step": step_idx,
                    "ok": False,
                    "reason": "too_few_correspondences",
                    "num_raw_matches": int(len(kpts0)),
                    "num_obj_matches": int(len(k0)),
                    "num_correspondences": int(len(obj_pts)),
                    "timings_s": timings,
                }
            )
            continue

        # 6) PnP + RANSAC: estimate object motion in camera frame
        t0 = time.perf_counter()
        cv2.setRNGSeed(0)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K_np,
            None,
            flags=cv2.USAC_MAGSAC, # [Optimization] Magical sampling is 5-10x faster and more accurate than SQPNP
            iterationsCount=int(args.ransac_iters),
            reprojectionError=float(args.reproj_error_px),
            confidence=float(args.confidence),
        )
        timings["pnp_s"] = time.perf_counter() - t0

        if not ok or inliers is None:
            print(f"[step {step_idx}] solvePnPRansac failed")
            timings["total_step_s"] = time.perf_counter() - t0_step
            timings["total_step_including_load_s"] = time.perf_counter() - t0_step_all
            per_step_stats.append(
                {
                    "step": step_idx,
                    "ok": False,
                    "reason": "pnp_failed",
                    "num_raw_matches": int(len(kpts0)),
                    "num_obj_matches": int(len(k0)),
                    "num_correspondences": int(len(obj_pts)),
                    "timings_s": timings,
                }
            )
            continue

        inliers = inliers.flatten()
        R_delta, _ = cv2.Rodrigues(rvec)
        t_delta = tvec.reshape(3)

        # Build SE(3) in camera coords
        T_delta_cam = np.eye(4, dtype=np.float64)
        T_delta_cam[:3, :3] = R_delta
        T_delta_cam[:3, 3] = t_delta

        # Convert to world-space transform using the known camera pose (world->cam)
        T_cw = viewmat.detach().cpu().numpy().astype(np.float64)
        T_wc = np.linalg.inv(T_cw)
        T_delta_world = T_wc @ T_delta_cam @ T_cw

        Rw = T_delta_world[:3, :3].astype(np.float32)
        tw = T_delta_world[:3, 3].astype(np.float32)

        # 7) Apply transform to object Gaussians (in world space)
        t0 = time.perf_counter()
        with torch.inference_mode():
            Rw_t = torch.as_tensor(Rw.T, device=device, dtype=means_curr.dtype)
            tw_t = torch.as_tensor(tw, device=device, dtype=means_curr.dtype)

            means_obj = means_curr.index_select(0, mask_idx)
            means_obj_new = (means_obj @ Rw_t) + tw_t
            means_curr.index_copy_(0, mask_idx, means_obj_new)

            q_rot = _rotmat_to_quat_wxyz(Rw.astype(np.float64), device=device, dtype=quats_curr.dtype)
            quats_obj = quats_curr.index_select(0, mask_idx)
            quats_obj_new = _quat_mul_wxyz(q_rot.view(1, 4), quats_obj)
            quats_curr.index_copy_(0, mask_idx, quats_obj_new)
        _maybe_cuda_sync(args.sync_timing)
        timings["apply_transform_s"] = time.perf_counter() - t0

        # Pose update time (PnP + apply transform)
        timings["pose_update_s"] = float(timings.get("pnp_s", 0.0) + timings.get("apply_transform_s", 0.0))

        # 8) Render updated scene
        t0 = time.perf_counter()
        with torch.inference_mode():
            out2, _, _ = rasterization(
                means_curr,
                quats_curr,
                scales,
                opacities,
                colors,
                viewmat[None],
                K_t[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            updated_rgb = out2[0]
        _maybe_cuda_sync(args.sync_timing)
        timings["render_updated_s"] = time.perf_counter() - t0

        # 9) Visualization and I/O
        t0_viz = time.perf_counter()
        step_tag = f"{step_idx:04d}"

        ref_bgr = None
        updated_bgr = None
        match_viz = None
        if bool(args.save_debug) or bool(args.show_windows):
            t_sub = time.perf_counter()
            # Optimization: Cast to uint8 and reorder to BGR on the GPU *before* CPU transfer to save PCIe bandwidth
            ref_bgr = (rgb_hw3[..., [2, 1, 0]] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            updated_bgr = (updated_rgb[..., [2, 1, 0]] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_bgr = np.ascontiguousarray(ref_bgr)
            updated_bgr = np.ascontiguousarray(updated_bgr)
            timings["viz_tensor_to_cv2_s"] = time.perf_counter() - t_sub
            
            t_sub = time.perf_counter()
            match_viz = _draw_matches(ref_bgr, target_bgr, k0, k1, max_lines=int(args.max_match_lines))
            timings["viz_draw_matches_s"] = time.perf_counter() - t_sub

        if bool(args.show_windows) and updated_bgr is not None and match_viz is not None:
            t_sub = time.perf_counter()
            # Optimization: We already rendered `updated_rgb` in step 8. Skip the redundant secondary
            # trackbar rasterization here. The trackbar loop will naturally handle it on the next step.
            updated_show = updated_bgr.copy()
            timings["viz_prepare_show_s"] = time.perf_counter() - t_sub

            t_sub = time.perf_counter()
            updated_show = _overlay_hint(updated_show, "Updated view (use sliders). SPACE for next update")
            cv2.imshow(window_target, target_bgr)
            cv2.imshow(window_updated, updated_show)
            cv2.imshow(window_matches, match_viz)
            cv2.waitKey(1)
            timings["viz_imshow_s"] = time.perf_counter() - t_sub

            if space_pressed_t0 is not None:
                timings["space_to_updated_window_s"] = time.perf_counter() - space_pressed_t0

        timings["viz_s"] = time.perf_counter() - t0_viz

        t0_io = time.perf_counter()
        if bool(args.save_debug):
            cv2.imwrite(os.path.join(args.output_dir, f"{step_tag}_ref.png"), ref_bgr)
            cv2.imwrite(os.path.join(args.output_dir, f"{step_tag}_target.png"), target_bgr)
            cv2.imwrite(os.path.join(args.output_dir, f"{step_tag}_updated.png"), updated_bgr)
            cv2.imwrite(os.path.join(args.output_dir, f"{step_tag}_matches.png"), match_viz)

        meta = {
            "step": step_idx,
            "target_file": target_file,
            "num_raw_matches": int(len(kpts0)),
            "num_obj_matches": int(len(k0)),
            "num_correspondences": int(len(obj_pts)),
            "pnp_inliers": int(len(inliers)),
            "rvec": [float(x) for x in rvec.reshape(-1)],
            "tvec": [float(x) for x in t_delta.reshape(-1)],
            "T_delta_cam": T_delta_cam.astype(float).tolist(),
            "T_delta_world": T_delta_world.astype(float).tolist(),
        }
        
        with open(os.path.join(args.output_dir, f"{step_tag}_pnp.json"), "w") as f:
            json.dump(meta, f, indent=2)

        timings["io_s"] = time.perf_counter() - t0_io
        timings["total_step_s"] = time.perf_counter() - t0_step

        per_step_stats.append(
            {
                "step": step_idx,
                "ok": True,
                "target_file": target_file,
                "num_raw_matches": int(len(kpts0)),
                "num_obj_matches": int(len(k0)),
                "num_correspondences": int(len(obj_pts)),
                "pnp_inliers": int(len(inliers)),
                "timings_s": timings,
            }
        )
        num_steps_ok += 1

        core_pipeline_s = float(
            timings.get("render_ref_s", 0.0)
            + timings.get("match_s", 0.0)
            + timings.get("segmentation_s", 0.0)
            + timings.get("pose_update_s", 0.0)
            + timings.get("render_updated_s", 0.0)
        )

        print(
            f"[step {step_idx}] target={os.path.basename(target_file)} "
            f"raw_matches={len(kpts0)} obj_matches={len(k0)} "
            f"corr={len(obj_pts)} pnp_inliers={len(inliers)} "
            f"time={timings.get('total_step_s', 0.0):.3f}s "
            f"core={core_pipeline_s:.3f}s "
            f"("
            f"render={timings.get('render_ref_s', 0.0):.3f}s, "
            f"match={timings.get('match_s', 0.0):.3f}s, "
            f"seg(gpu)={timings.get('segmentation_gpu_s', 0.0):.3f}s, "
            f"pose_update={timings.get('pose_update_s', 0.0):.3f}s, "
            f"render_updated={timings.get('render_updated_s', 0.0):.3f}s, "
            f"load={timings.get('load_target_s', 0.0):.3f}s, "
            f"to_gpu={timings.get('target_to_gpu_s', 0.0):.3f}s, "
            f"filter={timings.get('filter_obj_mask_s', 0.0):.3f}s, "
            f"backproj={timings.get('backproject_s', 0.0):.3f}s, "
            f"viz={timings.get('viz_s', 0.0):.3f}s "
            f"[prep={timings.get('viz_tensor_to_cv2_s', 0.0):.3f}s draw={timings.get('viz_draw_matches_s', 0.0):.3f}s prep_show={timings.get('viz_prepare_show_s', 0.0):.3f}s show={timings.get('viz_imshow_s', 0.0):.3f}s], "
            f"io={timings.get('io_s', 0.0):.3f}s)"
        )

        if args.show_windows and (not repeat_mode) and bool(args.pause_after_frame):
            # Pause so the user can inspect the current frame's results before advancing.
            t_wait1 = time.perf_counter()
            while True:
                # Keep windows responsive
                key = cv2.waitKeyEx(10) & 0xFF
                if key in (ord("q"), 27):
                    stop_requested = True
                    break
                if key in (ord(" "), 13):
                    break

            timings["wait_after_frame_s"] = float(time.perf_counter() - t_wait1)

            if stop_requested:
                break

        # End-to-end wall time for the step (includes optional waits)
        timings["total_step_including_load_s"] = float(time.perf_counter() - t0_step_all)

        # Update the last per_step record's timings to include the final wall time and wait-after.
        # (We append to per_step_stats earlier to keep early-continue paths simple.)
        per_step_stats[-1]["timings_s"] = timings

    if args.show_windows:
        cv2.destroyAllWindows()

    if executor is not None:
        executor.shutdown(wait=False)

    total_s = time.perf_counter() - t0_all

    setup_time_s = float(t_setup_done - t0_all)
    loop_time_s = float(total_s - setup_time_s)

    ok_steps = [s for s in per_step_stats if s.get("ok")]
    def _sum_stage(stage_key: str) -> float:
        return float(sum(float(s.get("timings_s", {}).get(stage_key, 0.0)) for s in ok_steps))

    stage_totals_s = {
        "render_ref_s": _sum_stage("render_ref_s"),
        "render_ref_wall_s": _sum_stage("render_ref_wall_s"),
        "render_ref_gpu_s": _sum_stage("render_ref_gpu_s"),
        "match_s": _sum_stage("match_s"),
        "segmentation_s": _sum_stage("segmentation_s"),
        "segmentation_gpu_s": _sum_stage("segmentation_gpu_s"),
        "pnp_s": _sum_stage("pnp_s"),
        "apply_transform_s": _sum_stage("apply_transform_s"),
        "pose_update_s": _sum_stage("pose_update_s"),
        "render_updated_s": _sum_stage("render_updated_s"),
        # Other useful per-step costs
        "load_target_s": _sum_stage("load_target_s"),
        "target_to_gpu_s": _sum_stage("target_to_gpu_s"),
        "filter_obj_mask_s": _sum_stage("filter_obj_mask_s"),
        "backproject_s": _sum_stage("backproject_s"),
        "viz_tensor_to_cv2_s": _sum_stage("viz_tensor_to_cv2_s"),
        "viz_draw_matches_s": _sum_stage("viz_draw_matches_s"),
        "viz_prepare_show_s": _sum_stage("viz_prepare_show_s"),
        "viz_imshow_s": _sum_stage("viz_imshow_s"),
        "viz_s": _sum_stage("viz_s"),
        "io_s": _sum_stage("io_s"),
    }
    stage_avg_s = {k: float(v / max(1, len(ok_steps))) for k, v in stage_totals_s.items()}

    sum_total_step_s_all = float(sum(float(s.get("timings_s", {}).get("total_step_s", 0.0)) for s in per_step_stats))
    sum_total_step_including_load_s_all = float(
        sum(float(s.get("timings_s", {}).get("total_step_including_load_s", 0.0)) for s in per_step_stats)
    )
    sum_wait_before_compute_s_all = float(
        sum(float(s.get("timings_s", {}).get("wait_before_compute_s", 0.0)) for s in per_step_stats)
    )
    sum_wait_after_frame_s_all = float(
        sum(float(s.get("timings_s", {}).get("wait_after_frame_s", 0.0)) for s in per_step_stats)
    )

    sum_human_wait_s_all = float(sum_wait_before_compute_s_all + sum_wait_after_frame_s_all)
    loop_time_no_human_wait_s = float(loop_time_s - sum_human_wait_s_all)
    total_time_no_human_wait_s = float(total_s - sum_human_wait_s_all)
    avg_loop_time_per_target_no_wait_s = float(loop_time_no_human_wait_s / max(1, len(target_files)))
    avg_setup_time_per_target_s = float(setup_time_s / max(1, len(target_files)))

    # Two different "untracked" views:
    # - vs compute-only totals (large when pauses are enabled)
    # - vs wall-time per-step totals (should be small)
    untracked_time_s = float(total_s - (setup_time_s + sum_total_step_s_all))
    untracked_time_vs_wall_steps_s = float(total_s - (setup_time_s + sum_total_step_including_load_s_all))

    summary = {
        "num_targets": int(len(target_files)),
        "num_steps_total": int(num_steps_total),
        "num_steps_ok": int(num_steps_ok),
        "total_time_s": float(total_s),
        "total_time_no_human_wait_s": float(total_time_no_human_wait_s),
        "setup_time_s": float(setup_time_s),
        "avg_setup_time_per_target_s": float(avg_setup_time_per_target_s),
        "setup_prefetch_first_s": float(setup_prefetch_first_s),
        "loop_time_s": float(loop_time_s),
        "loop_time_no_human_wait_s": float(loop_time_no_human_wait_s),
        "avg_loop_time_per_target_no_wait_s": float(avg_loop_time_per_target_no_wait_s),
        "sum_total_step_s_all": float(sum_total_step_s_all),
        "sum_total_step_including_load_s_all": float(sum_total_step_including_load_s_all),
        "sum_wait_before_compute_s_all": float(sum_wait_before_compute_s_all),
        "sum_wait_after_frame_s_all": float(sum_wait_after_frame_s_all),
        "sum_human_wait_s_all": float(sum_human_wait_s_all),
        "untracked_time_s": float(untracked_time_s),
        "untracked_time_vs_wall_steps_s": float(untracked_time_vs_wall_steps_s),
        "avg_time_per_target_s": float(total_s / max(1, len(target_files))),
        # Includes setup; useful for end-to-end wall time, but can look large when few frames are processed.
        "avg_time_per_target_no_human_wait_s": float(total_time_no_human_wait_s / max(1, len(target_files))),
        "avg_time_per_ok_step_s": float(total_s / max(1, num_steps_ok)),
        "stage_totals_s": stage_totals_s,
        "stage_avg_s": stage_avg_s,
        "per_step": per_step_stats,
    }
    with open(os.path.join(args.output_dir, "timing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[summary] targets={len(target_files)} ok_steps={num_steps_ok}/{num_steps_total} "
        f"setup={setup_time_s:.3f}s "
        f"loop(no_wait)={summary['loop_time_no_human_wait_s']:.3f}s "
        f"avg_loop_per_target(no_wait)={summary['avg_loop_time_per_target_no_wait_s']:.3f}s "
        f"total_time(no_wait, incl_setup)={summary['total_time_no_human_wait_s']:.3f}s "
        f"totals(render={stage_totals_s['render_ref_s']:.3f}s, match={stage_totals_s['match_s']:.3f}s, "
        f"seg_gpu={stage_totals_s['segmentation_gpu_s']:.3f}s, pose_update={stage_totals_s['pose_update_s']:.3f}s, "
        f"render_updated={stage_totals_s['render_updated_s']:.3f}s, viz={stage_totals_s['viz_s']:.3f}s, io={stage_totals_s['io_s']:.3f}s)"
    )
    return summary


if __name__ == "__main__":
    main(tyro.cli(Args))
