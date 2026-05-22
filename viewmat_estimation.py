# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
from dataclasses import dataclass
import os
import sys
import threading
import time
from typing import Literal, Optional
from collections import OrderedDict
import torch
from gsplat import rasterization

# OpenCV HighGUI can use a Qt backend; in some environments it can spam
# `QObject::moveToThread` warnings even when everything works. Suppress the
# most common Qt GUI-thread warning categories.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false;qt.core.qobject.*=false")


def _filter_process_stderr(substrings: list[str]) -> None:
    """Filter process-level stderr (FD 2) to drop known-noisy lines.

    Qt/OpenCV may print these warnings from native code directly to stderr,
    bypassing Python's `sys.stderr`. This redirects FD 2 through a pipe and
    forwards everything except lines containing any of `substrings`.
    """

    try:
        original_fd = os.dup(2)
        original = os.fdopen(original_fd, "w", buffering=1)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)

        def _pump() -> None:
            with os.fdopen(read_fd, "rb", buffering=0) as reader:
                buffer = b""
                while True:
                    chunk = reader.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        text = line.decode(errors="replace")
                        # Drop empty/whitespace-only lines to avoid terminal scrolling
                        # due to noisy native logging that sometimes leaves blank lines.
                        if text.strip() == "":
                            continue
                        if any(s in text for s in substrings):
                            continue
                        original.write(text + "\n")
                        original.flush()

        threading.Thread(target=_pump, daemon=True).start()
    except Exception:
        # If filtering fails for any reason, keep stderr unchanged.
        return


_filter_process_stderr(
    [
        "QObject::moveToThread",
        "Cannot move to target thread",
    ]
)
import cv2
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial import cKDTree
import pycolmap_scene_manager as pycolmap
import warnings
import faiss

import matching

from matching import get_matcher

matcher = get_matcher("superpoint-lg", device="cuda")
# matcher = get_matcher("eloftr", device="cuda")

import numpy as np
import json
import tyro

from utils import (
    get_rpy_matrix,
    get_viewmat_from_colmap_image,
    prune_by_gradients,
    torch_to_cv,
    load_checkpoint,
)

# Check if CUDA is available. Else raise an error.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Please install the correct version of PyTorch with CUDA support."
    )

device = torch.device("cuda")
torch.set_default_device("cuda")

 
@dataclass
class Args:
    checkpoint: str = "./data/bottle/ckpt_29999_rank0.pt" # Path to the 3DGS checkpoint file (.pth/.pt) to be visualized.
    data_dir: str = "./data/bottle"  # Path to the COLMAP project directory containing sparse reconstruction.
    format: Optional[Literal["inria", "gsplat"]] = "gsplat"  # Format of the checkpoint: 'inria' (original 3DGS) or 'gsplat'.
    rasterizer: Optional[Literal["inria", "gsplat"]] = None  # [Deprecated] Use --format instead. Provided for backward compatibility.
    data_factor: int = 1  # Downscaling factor for the renderings.



class Viewer:
    def __init__(self, splats):
        self.splats = None
        self.camera_matrix = None
        self.width = None
        self.height = None
        self.viewmat = None
        self.window_name = "Target"
        self.source_window_name = "Source"
        cv2.namedWindow(self.source_window_name, cv2.WINDOW_NORMAL)
        self.matching_window_name = "Matches"
        self._init_sliders()
        self._load_splats(splats)
        self.exact_viewmat = None
        self.last_tb_state = None
        self.matched_image = None
        self.target_file ="./data/bottle/camera_center_crop_20.jpg"
        self.target_img_cv = None
        # Match visualization runs an extra feature-matching pass; keep it OFF by default for speed.
        # Toggle with the 'm' key.
        self.show_match_viz = False
        self.match_downscale = 1
        # Local feature matcher (OpenCV SIFT) + small LRU caches.
        self._sift = cv2.SIFT_create()
        self._bf = cv2.BFMatcher()
        self._sift_cache: "OrderedDict[int, tuple[np.ndarray, Optional[np.ndarray]]]" = OrderedDict()
        self._sift_cache_max_items = 512
        # Small LRU cache to avoid repeated disk IO on candidate db images.
        self._db_img_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._db_cache_max_items = 512

    def _get_db_img_cached(self, idx: int, db_path: str) -> Optional[np.ndarray]:
        cached = self._db_img_cache.get(idx)
        if cached is not None:
            self._db_img_cache.move_to_end(idx)
            return cached
        db_img = cv2.imread(db_path)
        if db_img is None:
            return None
        db_img = cv2.resize(db_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._db_img_cache[idx] = db_img
        self._db_img_cache.move_to_end(idx)
        while len(self._db_img_cache) > self._db_cache_max_items:
            self._db_img_cache.popitem(last=False)
        return db_img

    def _sift_extract(self, bgr_uint8: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        gray = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2GRAY)
        kpts, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or kpts is None or len(kpts) < 4:
            return np.zeros((0, 2), dtype=np.float32), None
        pts = np.array([kp.pt for kp in kpts], dtype=np.float32)
        return pts, desc

    def _get_db_sift_cached(self, idx: int, db_small: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        cached = self._sift_cache.get(idx)
        if cached is not None:
            self._sift_cache.move_to_end(idx)
            return cached
        pts, desc = self._sift_extract(db_small)
        self._sift_cache[idx] = (pts, desc)
        self._sift_cache.move_to_end(idx)
        while len(self._sift_cache) > self._sift_cache_max_items:
            self._sift_cache.popitem(last=False)
        return pts, desc

    def _sift_match(self, pts0: np.ndarray, desc0: np.ndarray, pts1: np.ndarray, desc1: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        if desc0 is None or desc1 is None or len(pts0) < 4 or len(pts1) < 4:
            return 0, np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        matches = self._bf.knnMatch(desc0, desc1, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) < 4:
            return 0, np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        m0 = np.float32([pts0[m.queryIdx] for m in good])
        m1 = np.float32([pts1[m.trainIdx] for m in good])
        return int(len(good)), m0, m1

    def _load_splats(self, splats):
        K = splats["camera_matrix"].cuda()
        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)

        means = splats["means"].float()
        opacities = splats["opacity"]
        quats = splats["rotation"]
        scales = splats["scaling"].float()

        opacities = torch.sigmoid(opacities)
        scales = torch.exp(scales)
        colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

        self.splats = splats
        self.camera_matrix = K
        self.width = width
        self.height = height
        self.means = means
        self.opacities = opacities
        self.quats = quats
        self.scales = scales
        self.colors = colors
        self.global_descriptors = faiss.IndexFlatL2(768)

    def _init_sliders(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        trackbars = {
            "Roll": (-180, 180),
            "Pitch": (-180, 180),
            "Yaw": (-180, 180),
            "X": (-1000, 1000),
            "Y": (-1000, 1000),
            "Z": (-1000, 1000),
            "Scaling": (0, 100),
        }

        for name, (min_val, max_val) in trackbars.items():
            cv2.createTrackbar(name, self.window_name, 0, max_val, lambda x: None)
            cv2.setTrackbarMin(name, self.window_name, min_val)
            cv2.setTrackbarMax(name, self.window_name, max_val)

        cv2.setTrackbarPos(
            "Scaling", self.window_name, 100
        )  # Default value for scaling is 100

    def update_trackbars_from_viewmat(self, world_to_camera):
        # if torch tensor is passed, convert to numpy
        if isinstance(world_to_camera, torch.Tensor):
            world_to_camera = world_to_camera.cpu().numpy()
        r = scipyR.from_matrix(world_to_camera[:3, :3])
        roll, pitch, yaw = r.as_euler("xyz")
        cv2.setTrackbarPos("Roll", self.window_name, np.rad2deg(roll).astype(int))
        cv2.setTrackbarPos("Pitch", self.window_name, np.rad2deg(pitch).astype(int))
        cv2.setTrackbarPos("Yaw", self.window_name, np.rad2deg(yaw).astype(int))
        cv2.setTrackbarPos("X", self.window_name, int(world_to_camera[0, 3] * 100))
        cv2.setTrackbarPos("Y", self.window_name, int(world_to_camera[1, 3] * 100))
        cv2.setTrackbarPos("Z", self.window_name, int(world_to_camera[2, 3] * 100))

    def _get_viewmat_from_trackbars(self):
        roll = cv2.getTrackbarPos("Roll", self.window_name)
        pitch = cv2.getTrackbarPos("Pitch", self.window_name)
        yaw = cv2.getTrackbarPos("Yaw", self.window_name)
        x = cv2.getTrackbarPos("X", self.window_name)
        y = cv2.getTrackbarPos("Y", self.window_name)
        z = cv2.getTrackbarPos("Z", self.window_name)
        
        tb_state = (roll, pitch, yaw, x, y, z)

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )

        viewmat[0, 3] = x / 100.0
        viewmat[1, 3] = y / 100.0
        viewmat[2, 3] = z / 100.0

        return viewmat, tb_state

    def render_gaussians(self, viewmat, scaling, get_depth=False):

        output, _, _ = rasterization(
            self.means,
            self.quats,
            self.scales * scaling,
            self.opacities,
            self.colors,
            viewmat[None],
            self.camera_matrix[None],
            width=self.width,
            height=self.height,
            sh_degree=3,
            render_mode="RGB+ED"
        )
        output, depth = output[0,..., :3], output[0, ..., 3]
        if get_depth:
            return np.ascontiguousarray(torch_to_cv(output)), np.ascontiguousarray(depth.cpu().numpy())
        return np.ascontiguousarray(torch_to_cv(output))

    def run(self):
        """Run the interactive Gaussian Splat viewer loop once until exit."""
        show_anaglyph = False

        # Show the target image in the Source window at startup (not a random COLMAP view).
        self.target_img_cv = cv2.imread(self.target_file)
        if self.target_img_cv is None:
            print(f"Warning: {self.target_file} not found. Source window will remain empty until it exists.")
        else:
            self.target_img_cv = cv2.resize(
                self.target_img_cv, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            cv2.imshow(self.source_window_name, self.target_img_cv)

        colmap_project = self.splats["colmap_project"]
        colmap_dir = self.splats.get("colmap_dir")
        data_factor = int(self.splats.get("data_factor", 1))
        images_dir = None
        if colmap_dir is not None:
            candidate = os.path.join(colmap_dir, f"images_{data_factor}")
            images_dir = candidate if os.path.isdir(candidate) else os.path.join(colmap_dir, "images")

        from global_descriptors import DINODescriptor
        dino_descriptor = DINODescriptor()

        # Build an index over *real COLMAP images* (not renders), and precompute a fast mapping
        # from each image's 2D points -> 3D point IDs (via a KD-tree). This enables novel-view
        # localization by collecting 2D-3D correspondences and solving PnP.
        id_to_colmap_image = {}
        id_to_kdtree = {}
        id_to_p3d_ids = {}

        id_ = 0
        for image in colmap_project.images.values():
            if images_dir is None:
                continue
            img_path = os.path.join(images_dir, image.name)
            db_img = cv2.imread(img_path)
            if db_img is None:
                continue
            db_img = cv2.resize(db_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            desc = dino_descriptor.extract(db_img)
            self.global_descriptors.add(desc.unsqueeze(0).cpu().numpy())

            # 2D points in COLMAP are in the original image resolution.
            # We operate at viewer resolution, so scale them by data_factor.
            xy = (image.points2D / float(data_factor)).astype(np.float32)
            p3d_ids = image.point3D_ids
            valid = p3d_ids != colmap_project.INVALID_POINT3D
            xy_valid = xy[valid]
            p3d_valid = p3d_ids[valid]
            if len(xy_valid) > 0:
                id_to_kdtree[id_] = cKDTree(xy_valid)
                id_to_p3d_ids[id_] = p3d_valid

            id_to_colmap_image[id_] = image
            id_ += 1

            if id_ % 50 == 0:
                print(f"Indexed {id_} COLMAP images...")

        # exit()


        while True:
            scaling = cv2.getTrackbarPos("Scaling", self.window_name) / 100.0
            vm_tb, current_tb_state = self._get_viewmat_from_trackbars()
            
            # If trackbar changed manually, disable exact override
            if self.last_tb_state is not None and current_tb_state != self.last_tb_state:
                if self.exact_viewmat is not None:
                    print("Trackbars changed, clearing exact_viewmat!")
                self.exact_viewmat = None
                
            self.last_tb_state = current_tb_state
            
            viewmat = self.exact_viewmat if self.exact_viewmat is not None else vm_tb

            output_cv = self.render_gaussians(viewmat, scaling)

            cv2.putText(
                output_cv,
                "Press ENTER/SPACE to estimate pose",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if self.show_match_viz:
                cv2.putText(
                    output_cv,
                    "Match viz: ON (press 'm' to toggle)",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    output_cv,
                    "Match viz: OFF (press 'm' to toggle)",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            if show_anaglyph:
                offset_viewmat = viewmat.clone()
                offset_viewmat[0, 3] -= 0.05
                output_left = output_cv
                output_right = self.render_gaussians(offset_viewmat, scaling)
                output_left[..., :2] = 0
                output_right[..., -1] = 0
                output_cv = output_left + output_right

            cv2.imshow(self.window_name, output_cv)
            full_key = cv2.waitKeyEx(1)
            key = full_key & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord("3"):
                show_anaglyph = not show_anaglyph
            if key == ord("m"):
                self.show_match_viz = not self.show_match_viz
                print(f"Match visualization: {'ON' if self.show_match_viz else 'OFF'}")
            if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
                # Modify viewmat and sync UI
                delta = 0.1
                if key == ord("w"):
                    viewmat[2, 3] -= delta
                elif key == ord("s"):
                    viewmat[2, 3] += delta
                elif key == ord("a"):
                    viewmat[0, 3] += delta
                elif key == ord("d"):
                    viewmat[0, 3] -= delta
                self.update_trackbars_from_viewmat(viewmat)

            if key == ord("r"):
                # Reset the render pose to a random COLMAP camera (useful for debugging).
                images = list(self.splats["colmap_project"].images.values())
                if len(images) > 0:
                    image = images[int(np.random.randint(0, len(images)))]
                    new_vm = get_viewmat_from_colmap_image(image)
                    self.exact_viewmat = new_vm.clone()
                    self.update_trackbars_from_viewmat(new_vm)
                    _, self.last_tb_state = self._get_viewmat_from_trackbars()

            if key == 13 or key == ord(" "):
                t0 = time.perf_counter()
                print(f"\nEstimating pose for {self.target_file}...")
                if self.target_img_cv is None:
                    self.target_img_cv = cv2.imread(self.target_file)
                    if self.target_img_cv is None:
                        print(f"Error: {self.target_file} not found!")
                        continue
                    self.target_img_cv = cv2.resize(
                        self.target_img_cv,
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA,
                    )

                # Ensure Source window shows the actual target image at least once.
                cv2.imshow(self.source_window_name, self.target_img_cv)
                target_img = self.target_img_cv

                t_desc0 = time.perf_counter()
                desc_query = dino_descriptor.extract(target_img).unsqueeze(0).cpu().numpy()
                t_desc1 = time.perf_counter()
                # Increased to 15 to allow the system to search multiple potential nearby
                # views since "novel views" will break the simple 1-to-1 global descriptor
                k_search = 20
                t_faiss0 = time.perf_counter()
                D, I = self.global_descriptors.search(desc_query, k=k_search)
                t_faiss1 = time.perf_counter()
                print(f"Searching across {k_search} database images (novel-view localization)...")

                camera_matrix_np = self.camera_matrix.cpu().numpy().astype(np.float64)
                best_inliers = 0
                best_viewmat_target = None
                best_out_stack = None
                best_db_img = None
                best_debug_text = None

                # Speed knobs
                match_downscale = int(self.match_downscale)  # 2 => match at half-res (much faster), keypoints scaled back
                early_stop_inliers = 300 # stop searching once we have this many PnP inliers
                max_ransac_iters = 500

                if match_downscale > 1:
                    w_small = max(32, self.width // match_downscale)
                    h_small = max(32, self.height // match_downscale)
                    target_small = cv2.resize(
                        target_img, (w_small, h_small), interpolation=cv2.INTER_AREA
                    )
                else:
                    w_small, h_small = self.width, self.height
                    target_small = target_img

                img1_small = target_small / 255.0
                # Extract target local features once per query (major latency win).
                t_sift0 = time.perf_counter()
                pts1_small, desc1 = self._sift_extract(target_small)
                t_sift1 = time.perf_counter()
                if desc1 is None or len(pts1_small) < 4:
                    t1 = time.perf_counter()
                    print("-> Pose estimation failed (no target features).")
                    print(f"Timing: total={(t1 - t0) * 1000.0:.1f} ms")
                    continue

                nn_thresh_px = 4.0

                # Staged candidate evaluation for lower average latency.
                # If retrieval is good, we often succeed within the first few candidates.
                candidate_ids = list(I[0])
                stages = [min(5, len(candidate_ids)), min(15, len(candidate_ids)), len(candidate_ids)]
                tested_candidates = set()
                t_match_total = 0.0
                t_pnp_total = 0.0
                cand_checks = 0
                for stage_k in stages:
                    for attempt, idx in enumerate(candidate_ids[:stage_k]):
                        if idx in tested_candidates:
                            continue
                        tested_candidates.add(idx)

                        if idx not in id_to_colmap_image:
                            continue
                        if idx not in id_to_kdtree:
                            continue
                        if images_dir is None:
                            continue

                        colmap_img = id_to_colmap_image[idx]
                        db_path = os.path.join(images_dir, colmap_img.name)
                        db_img = self._get_db_img_cached(idx, db_path)
                        if db_img is None:
                            continue

                        if match_downscale > 1:
                            db_small = cv2.resize(
                                db_img, (w_small, h_small), interpolation=cv2.INTER_AREA
                            )
                        else:
                            db_small = db_img

                        try:
                            cand_checks += 1
                            t_cand_start = time.perf_counter()
                            # Local matching at reduced resolution for speed.
                            pts0_small, desc0 = self._get_db_sift_cached(idx, db_small)
                            num_inliers, inlier_kpts0, inlier_kpts1 = self._sift_match(
                                pts0_small, desc0, pts1_small, desc1
                            )
                            t_match_total += time.perf_counter() - t_cand_start
                            
                            if num_inliers < 15:
                                continue

                            inlier_kpts0 = np.asarray(inlier_kpts0, dtype=np.float32)
                            inlier_kpts1 = np.asarray(inlier_kpts1, dtype=np.float32)

                            # Scale keypoints back to viewer resolution.
                            if match_downscale > 1:
                                inlier_kpts0 *= float(match_downscale)
                                inlier_kpts1 *= float(match_downscale)

                            tree = id_to_kdtree[idx]
                            p3d_valid = id_to_p3d_ids[idx]
                            dists, nn_idx = tree.query(inlier_kpts0, k=1)
                            ok = dists < nn_thresh_px

                            # Build unique 2D-3D correspondences by point3D_id.
                            corr = {}
                            for i in np.where(ok)[0]:
                                pid = int(p3d_valid[nn_idx[i]])
                                dist = float(dists[i])
                                if pid not in colmap_project.point3D_id_to_point3D_idx:
                                    continue
                                prev = corr.get(pid)
                                if prev is None or dist < prev[0]:
                                    corr[pid] = (dist, inlier_kpts0[i], inlier_kpts1[i])

                            if len(corr) < 15:
                                continue

                            obj_pts = []
                            img_pts = []
                            kpts0_corr = []
                            kpts1_corr = []
                            for pid, (_dist, k0, k1) in corr.items():
                                pidx = colmap_project.point3D_id_to_point3D_idx[pid]
                                obj_pts.append(colmap_project.points3D[pidx])
                                img_pts.append(k1)
                                kpts0_corr.append(k0)
                                kpts1_corr.append(k1)

                            obj_pts = np.asarray(obj_pts, dtype=np.float64)
                            img_pts = np.asarray(img_pts, dtype=np.float64)
                            kpts0_corr = np.asarray(kpts0_corr, dtype=np.float64)
                            kpts1_corr = np.asarray(kpts1_corr, dtype=np.float64)

                            t_pnp_start = time.perf_counter()
                            cv2.setRNGSeed(0)
                            res, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
                                obj_pts,
                                img_pts,
                                camera_matrix_np,
                                None,
                                flags=cv2.SOLVEPNP_SQPNP,
                                iterationsCount=max_ransac_iters,
                                reprojectionError=4.0,
                                confidence=0.999,
                            )
                            t_pnp_total += time.perf_counter() - t_pnp_start

                            if not res or pnp_inliers is None:
                                continue

                            curr_inliers = int(len(pnp_inliers))
                            if curr_inliers <= best_inliers:
                                continue

                            R = cv2.Rodrigues(rvec)[0]
                            t = tvec.reshape(3)
                            T = np.eye(4, dtype=np.float32)
                            T[:3, :3] = R.astype(np.float32)
                            T[:3, 3] = t.astype(np.float32)

                            best_inliers = curr_inliers
                            best_viewmat_target = torch.from_numpy(T).to(device)
                            best_db_img = db_img
                            best_debug_text = (
                                f"cand={attempt+1}/{stage_k}  matches={int(num_inliers)}  pnp_inliers={curr_inliers}"
                            )

                            # Visualization: show matcher inliers + highlight PnP inliers.
                            out_stack = np.hstack((db_img, target_img))
                            for j in range(min(int(len(inlier_kpts0)), 250)):
                                x0, y0 = inlier_kpts0[j]
                                x1, y1 = inlier_kpts1[j]
                                cv2.line(
                                    out_stack,
                                    (int(x0), int(y0)),
                                    (int(x1 + self.width), int(y1)),
                                    (255, 0, 0),
                                    1,
                                )
                            inl = pnp_inliers.flatten()
                            for j in inl[: min(len(inl), 250)]:
                                x0, y0 = kpts0_corr[j]
                                x1, y1 = kpts1_corr[j]
                                cv2.line(
                                    out_stack,
                                    (int(x0), int(y0)),
                                    (int(x1 + self.width), int(y1)),
                                    (0, 255, 0),
                                    2,
                                )
                            if best_debug_text is not None:
                                cv2.putText(
                                    out_stack,
                                    best_debug_text,
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2,
                                )
                            best_out_stack = out_stack

                            if best_inliers >= early_stop_inliers:
                                break
                        except Exception:
                            continue

                        if best_inliers >= early_stop_inliers:
                            break

                    if best_viewmat_target is not None or best_inliers >= early_stop_inliers:
                        break

                if best_viewmat_target is not None:
                    t1 = time.perf_counter()
                    print(f"-> Target matched with {best_inliers} PnP inliers.")
                    print(
                        "Timing: "
                        f"canc_checked={cand_checks}, "
                        f"descriptor={(t_desc1 - t_desc0) * 1000.0:.1f} ms, "
                        f"faiss={(t_faiss1 - t_faiss0) * 1000.0:.1f} ms, "
                        f"sift_target={(t_sift1 - t_sift0) * 1000.0:.1f} ms, "
                        f"db_match={(t_match_total) * 1000.0:.1f} ms, "
                        f"pnp_total={(t_pnp_total) * 1000.0:.1f} ms, "
                        f"total={(t1 - t0) * 1000.0:.1f} ms"
                    )

                    self.exact_viewmat = best_viewmat_target.clone()
                    self.update_trackbars_from_viewmat(best_viewmat_target)
                    _, self.last_tb_state = self._get_viewmat_from_trackbars()  # save updated state immediately

                    torch.save({"viewmat": best_viewmat_target}, "camera_pose.pt")
                    print("Saved camera pose to camera_pose.pt")

                    # Keep the Source window pinned to the target image.
                    if self.target_img_cv is not None:
                        cv2.imshow(self.source_window_name, self.target_img_cv)

                    # Matching window should always show render <-> target side-by-side.
                    # When `m` is ON, overlay local feature correspondences.
                    render_best = self.render_gaussians(best_viewmat_target, scaling)
                    out_rt = np.hstack((render_best, target_img))

                    if self.show_match_viz:
                        try:
                            if match_downscale > 1:
                                render_small = cv2.resize(
                                    render_best,
                                    (w_small, h_small),
                                    interpolation=cv2.INTER_AREA,
                                )
                            else:
                                render_small = render_best

                            pts_r_small, desc_r = self._sift_extract(render_small)
                            rt_inliers, k0, k1 = self._sift_match(
                                pts_r_small, desc_r, pts1_small, desc1
                            )

                            if match_downscale > 1:
                                k0 *= float(match_downscale)
                                k1 *= float(match_downscale)

                            for j in range(min(int(len(k0)), 300)):
                                x0, y0 = k0[j]
                                x1, y1 = k1[j]
                                cv2.line(
                                    out_rt,
                                    (int(x0), int(y0)),
                                    (int(x1 + self.width), int(y1)),
                                    (0, 255, 255),
                                    1,
                                )
                            cv2.putText(
                                out_rt,
                                f"render<->target inliers={rt_inliers}  (press 'm' to toggle)",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 255),
                                2,
                            )
                        except Exception as e:
                            print(f"Warning: render<->target match viz failed: {e}")
                    else:
                        cv2.putText(
                            out_rt,
                            "render<->target (press 'm' to show matches)",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2,
                        )

                    best_out_stack = out_rt

                    cv2.namedWindow(self.matching_window_name, cv2.WINDOW_NORMAL)
                    if best_out_stack is not None:
                        cv2.imshow(self.matching_window_name, best_out_stack)
                    cv2.waitKey(1)
                else:
                    t1 = time.perf_counter()
                    print("-> Pose estimation failed (not enough 2D-3D correspondences).")
                    print(f"Timing: total={(t1 - t0) * 1000.0:.1f} ms")

        cv2.destroyAllWindows()


def main(args: Args):
    format = args.format or args.rasterizer
    if args.rasterizer:
        warnings.warn(
            "`rasterizer` is deprecated. Use `format` instead.", DeprecationWarning
        )
    if not format:
        raise ValueError("Must specify --format or the deprecated --rasterizer")

    splats = load_checkpoint(args.checkpoint, args.data_dir, format, args.data_factor)
    splats = prune_by_gradients(splats)

    viewer = Viewer(splats)
    viewer.run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
