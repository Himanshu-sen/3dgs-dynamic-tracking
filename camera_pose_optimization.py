"""
Camera pose optimization for 3D Gaussian Splatting scene alignment.
"""

from typing_extensions import Literal

import os
import time
import warnings

import cv2
import imageio
import numpy as np
import pycolmap_scene_manager as pycolmap
import torch
from gsplat import rasterization
import tyro

from utils_for_target_update import get_viewmat_from_colmap_image


device = torch.device("cuda:0")
CAMERA_OPT_WINDOW_NAME = "Camera Optimization: Target | Current | Delta"


def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d


def load_checkpoint(
    checkpoint: str,
    data_dir: str,
    rasterizer: Literal["inria", "original", "gsplat"] = "inria",
    data_factor: int = 1,
):
    colmap_project = pycolmap.SceneManager(f"{data_dir}/sparse/0")
    colmap_project.load_cameras()
    colmap_project.load_images()
    colmap_project.load_points3D()
    model = torch.load(checkpoint, weights_only=False)

    is_inria_like = isinstance(model, (tuple, list))
    is_gsplat_like = isinstance(model, dict) and "splats" in model

    if rasterizer == "gsplat" and is_inria_like and not is_gsplat_like:
        warnings.warn(
            "Checkpoint appears to be INRIA/original-style (tuple). "
            "Overriding rasterizer='gsplat' -> 'inria'.",
            RuntimeWarning,
        )
        rasterizer = "inria"
    elif rasterizer in ("inria", "original") and is_gsplat_like and not is_inria_like:
        warnings.warn(
            "Checkpoint appears to be gsplat-style (dict with 'splats'). "
            f"Overriding rasterizer='{rasterizer}' -> 'gsplat'.",
            RuntimeWarning,
        )
        rasterizer = "gsplat"

    if rasterizer == "inria" or rasterizer == "original":
        model_params, _ = model
        splats = {
            "active_sh_degree": model_params[0],
            "means": model_params[1],
            "features_dc": model_params[2],
            "features_rest": model_params[3],
            "scaling": model_params[4],
            "rotation": model_params[5],
            "opacity": model_params[6].squeeze(1),
        }
    elif rasterizer == "gsplat":
        model_params = model["splats"]
        splats = {
            "active_sh_degree": 3,
            "means": model_params["means"],
            "features_dc": model_params["sh0"],
            "features_rest": model_params["shN"],
            "scaling": model_params["scales"],
            "rotation": model_params["quats"],
            "opacity": model_params["opacities"],
        }
    else:
        raise ValueError("Invalid rasterizer")

    _detach_tensors_from_dict(splats)

    img_width, img_height = None, None
    for camera in colmap_project.cameras.values():
        camera_matrix = torch.tensor(
            [
                [camera.fx, 0, camera.cx],
                [0, camera.fy, camera.cy],
                [0, 0, 1],
            ]
        )
        if hasattr(camera, "width") and hasattr(camera, "height"):
            img_width, img_height = camera.width, camera.height
        break

    camera_matrix[:2, :3] /= data_factor

    splats["camera_matrix"] = camera_matrix
    if img_width is None or img_height is None:
        try:
            first_img = next(iter(colmap_project.images.values()))
            img_width = getattr(first_img, "width", int(camera_matrix[0, 2] * 2))
            img_height = getattr(first_img, "height", int(camera_matrix[1, 2] * 2))
        except StopIteration:
            img_width = int(camera_matrix[0, 2] * 2)
            img_height = int(camera_matrix[1, 2] * 2)

    splats["image_size"] = (
        int(img_width // max(1, data_factor)),
        int(img_height // max(1, data_factor)),
    )
    splats["colmap_project"] = colmap_project
    splats["colmap_dir"] = data_dir

    return splats


def rodrigues_rotation(rot_vec):
    """Convert axis-angle vector to 3x3 rotation matrix (Rodrigues formula)."""
    angle = (rot_vec.pow(2).sum() + 1e-12).sqrt()
    ax, ay, az = (rot_vec / angle).unbind()
    zero = torch.zeros_like(ax)
    K = torch.stack(
        [
            torch.stack([zero, -az, ay]),
            torch.stack([az, zero, -ax]),
            torch.stack([-ay, ax, zero]),
        ]
    )
    I = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)
    return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def torch_to_cv(tensor):
    if tensor.ndim == 3:
        if tensor.shape[0] == 3 and tensor.shape[2] != 3:
            tensor = tensor.permute(1, 2, 0)
    tensor = torch.clamp(tensor, 0, 1).cpu().numpy()
    return (tensor * 255).astype(np.uint8)[..., ::-1]


def make_delta_heatmap(delta_tensor: torch.Tensor):
    """Convert RGB delta tensor to a normalized heatmap (always visible)."""
    # delta_tensor: [H, W, 3] in [0, 1]
    delta_mag = torch.norm(delta_tensor, dim=-1)  # [H, W]

    # Robust normalization to avoid a flat/black image when absolute errors are tiny.
    q95 = torch.quantile(delta_mag, 0.95)
    denom = torch.clamp(q95, min=1e-8)
    delta_norm = torch.clamp(delta_mag / denom, 0.0, 1.0)

    delta_u8 = (delta_norm * 255.0).to(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(delta_u8, cv2.COLORMAP_INFERNO)

    # Add numeric stats so visibility is clear even when residuals are very small.
    mean_err = float(delta_mag.mean().item())
    max_err = float(delta_mag.max().item())
    cv2.putText(
        heatmap,
        f"mean {mean_err:.4f} max {max_err:.4f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        heatmap,
        f"mean {mean_err:.4f} max {max_err:.4f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
    )
    return heatmap


def resize_preview_for_window(image, scale: float = 0.5):
    """Resize only the live preview window; saved images stay full resolution."""
    if scale <= 0:
        scale = 0.5
    preview_width = max(1, int(image.shape[1] * scale))
    preview_height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (preview_width, preview_height), interpolation=cv2.INTER_AREA)


def draw_3d_axes(
    img_bgr,
    viewmat,
    K,
    width,
    height,
    origin=np.array([0, 0, 0]),
    axis_length=0.3,
):
    """Draw 3D coordinate axes on the image."""
    axes_3d = np.array(
        [
            origin,
            origin + np.array([axis_length, 0, 0]),
            origin + np.array([0, axis_length, 0]),
            origin + np.array([0, 0, axis_length]),
        ]
    )

    axes_3d_homo = np.concatenate([axes_3d, np.ones((4, 1))], axis=1)

    if isinstance(viewmat, torch.Tensor):
        viewmat = viewmat.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()

    axes_cam = (viewmat @ axes_3d_homo.T).T

    axes_2d = []
    for i in range(4):
        if axes_cam[i, 2] <= 0:
            return img_bgr
        point_2d = K @ axes_cam[i, :3]
        point_2d = point_2d[:2] / point_2d[2]
        axes_2d.append(point_2d)

    axes_2d = np.array(axes_2d).astype(int)
    origin_2d = tuple(axes_2d[0])

    cv2.line(img_bgr, origin_2d, tuple(axes_2d[1]), (0, 0, 255), 3)
    cv2.putText(img_bgr, "X", tuple(axes_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(img_bgr, origin_2d, tuple(axes_2d[2]), (0, 255, 0), 3)
    cv2.putText(img_bgr, "Y", tuple(axes_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.line(img_bgr, origin_2d, tuple(axes_2d[3]), (255, 0, 0), 3)
    cv2.putText(img_bgr, "Z", tuple(axes_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return img_bgr


def optimize_camera_pose(
    splats,
    K,
    width,
    height,
    camera_target_image_path: str,
    initial_view_idx: int = 10,
    initial_camera_pose_path: str = "",
    sh_degree_override: int = -1,
    camera_lr: float = 5,
    camera_iterations: int = 300,
    show_windows: bool = True,
    window_display_scale: float = 0.5,
    output_image_path: str = "results/person/optimized_object_render.png",
    camera_pose_output_path: str = "results/person/optimized_camera_pose.pt",
):
    if camera_target_image_path.strip() == "":
        raise ValueError("camera_target_image_path is required for camera optimization")

    camera_target_cv = cv2.imread(camera_target_image_path)
    if camera_target_cv is None:
        raise ValueError(f"Could not load camera target image from {camera_target_image_path}")
    camera_target_cv = cv2.cvtColor(camera_target_cv, cv2.COLOR_BGR2RGB)
    camera_target_cv = camera_target_cv.astype(np.float32) / 255.0
    camera_target_cv = cv2.resize(camera_target_cv, (width, height))
    camera_target_render = torch.from_numpy(camera_target_cv).to(device).unsqueeze(0)

    print("\n=== Step 1: Optimizing camera pose ===")
    print(f"Running camera pose optimization for {camera_iterations} iterations...")

    means_full = splats["means"].float().cuda()
    opacities_full = torch.sigmoid(splats["opacity"]).cuda()
    quats_full = splats["rotation"].cuda()
    scales_full = torch.exp(splats["scaling"].float()).cuda()
    colors_full = torch.cat([splats["features_dc"], splats["features_rest"]], 1).cuda()

    resolved_initial_view_idx = int(initial_view_idx)
    camera_pose_path = initial_camera_pose_path.strip() if initial_camera_pose_path else ""
    if camera_pose_path == "":
        camera_pose_path = os.path.join(splats.get("colmap_dir", ""), "camera_pose.pt")
    elif not os.path.isabs(camera_pose_path):
        camera_pose_path = os.path.abspath(camera_pose_path)

    initial_view_source = "colmap"
    if camera_pose_path and os.path.isfile(camera_pose_path):
        loaded_pose = torch.load(camera_pose_path, map_location=device, weights_only=False)
        if isinstance(loaded_pose, dict):
            if "viewmat" in loaded_pose and isinstance(loaded_pose["viewmat"], torch.Tensor):
                initial_viewmat = loaded_pose["viewmat"].float().to(device)
            else:
                raise ValueError(
                    f"Found camera pose file at {camera_pose_path}, but it does not contain a valid 'viewmat' tensor"
                )
        elif isinstance(loaded_pose, torch.Tensor):
            initial_viewmat = loaded_pose.float().to(device)
        else:
            raise ValueError(f"Unsupported camera pose format in {camera_pose_path}: {type(loaded_pose)}")

        if initial_viewmat.shape == (3, 4):
            initial_viewmat = torch.cat(
                [
                    initial_viewmat,
                    torch.tensor([[0, 0, 0, 1]], device=device, dtype=initial_viewmat.dtype),
                ],
                dim=0,
            )
        if initial_viewmat.shape != (4, 4):
            raise ValueError(
                f"Initial camera pose from {camera_pose_path} must be 4x4 (or 3x4), got shape {tuple(initial_viewmat.shape)}"
            )
        initial_view_source = "camera_pose_file"
        print(f"Using camera pose from file as STARTING POINT: {camera_pose_path}")
    else:
        colmap_images_dict = splats["colmap_project"].images
        image_ids_sorted = sorted(colmap_images_dict.keys())
        images = [colmap_images_dict[i] for i in image_ids_sorted]
        if len(images) == 0:
            raise ValueError("No COLMAP images found for selecting initial view")
        resolved_initial_view_idx = max(0, min(resolved_initial_view_idx, len(images) - 1))
        view_camera = images[resolved_initial_view_idx]
        initial_viewmat = get_viewmat_from_colmap_image(view_camera).cuda()
        print(
            f"camera_pose.pt not found at '{camera_pose_path}'. Using COLMAP view {resolved_initial_view_idx} as STARTING POINT (will be optimized)"
        )

    print(f"Initial view source: {initial_view_source}")

    R_cam_fixed = initial_viewmat[:3, :3].detach().clone()
    rot_delta = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    t_cam = initial_viewmat[:3, 3].detach().clone()
    t_cam.requires_grad = True

    opt_cam = torch.optim.Adam([rot_delta, t_cam], lr=camera_lr)
    sh_degree = (
        int(sh_degree_override)
        if sh_degree_override and sh_degree_override > 0
        else int(splats["active_sh_degree"])
        if "active_sh_degree" in splats
        else 3
    )

    t_cam_start = time.time()
    best_cam_loss = None
    best_rot_delta = None
    best_t_cam = None
    camera_frames = []

    if show_windows:
        cv2.namedWindow(CAMERA_OPT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    for i in range(camera_iterations):
        R_opt = R_cam_fixed @ rodrigues_rotation(rot_delta)
        viewmat_opt = torch.cat([R_opt, t_cam[:, None]], dim=1)
        viewmat_opt = torch.cat(
            [viewmat_opt, torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)],
            dim=0,
        )

        output_cam, _, _ = rasterization(
            means_full,
            quats_full,
            scales_full,
            opacities_full,
            colors_full,
            viewmat_opt[None],
            K[None],
            width=width,
            height=height,
            sh_degree=sh_degree,
        )

        cam_loss = torch.nn.functional.mse_loss(output_cam, camera_target_render)

        opt_cam.zero_grad()
        cam_loss.backward()
        opt_cam.step()

        if best_cam_loss is None or cam_loss.item() < best_cam_loss:
            best_cam_loss = cam_loss.item()
            best_rot_delta = rot_delta.detach().clone()
            best_t_cam = t_cam.detach().clone()

        if cam_loss.item() < 1e-5:
            print(f"Camera optimization: Loss below threshold at iteration {i}")
            break

        if i % 50 == 0:
            print(f"Camera optimization iter {i}: loss = {cam_loss.item():.6f}")

        if i % 5 == 0:
            render_with_axes = torch_to_cv(output_cam[0].detach()).copy()
            render_with_axes = draw_3d_axes(render_with_axes, viewmat_opt, K, width, height)

            target_with_axes = torch_to_cv(camera_target_render[0].detach()).copy()

            delta_cam = torch.abs(output_cam - camera_target_render).detach()
            delta_cv_cam = make_delta_heatmap(delta_cam[0])

            comparison = np.hstack([target_with_axes, render_with_axes, delta_cv_cam]).copy()
            cv2.putText(comparison, f"Iter {i} | Loss: {cam_loss.item():.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "Target", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(comparison, "Current", (width + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(comparison, "Delta", (width * 2 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            camera_frames.append(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))

        if show_windows:
            render_with_axes = torch_to_cv(output_cam[0].detach()).copy()
            render_with_axes = draw_3d_axes(render_with_axes, viewmat_opt, K, width, height)
            target_with_axes = torch_to_cv(camera_target_render[0].detach()).copy()
            delta_cam = torch.abs(output_cam - camera_target_render).detach()
            delta_cv_cam = make_delta_heatmap(delta_cam[0])
            comparison = np.hstack([target_with_axes, render_with_axes, delta_cv_cam])

            cv2.putText(comparison, f"Iter {i} | Loss: {cam_loss.item():.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            preview = resize_preview_for_window(comparison, window_display_scale)
            cv2.imshow(CAMERA_OPT_WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    t_cam_end = time.time()
    if best_rot_delta is None:
        best_rot_delta = torch.zeros(3, device=device, dtype=torch.float32)
    with torch.no_grad():
        best_R_cam = (R_cam_fixed @ rodrigues_rotation(best_rot_delta)).detach()

    print(f"Camera pose optimization completed in {t_cam_end - t_cam_start:.2f} seconds")
    print(f"Best camera pose loss: {best_cam_loss:.6f}")
    print(f"Optimized camera R matrix:\n{best_R_cam.cpu().numpy()}")
    print(f"Optimized camera t vector: {best_t_cam.cpu().numpy()}")

    if len(camera_frames) > 0:
        camera_gif_path = os.path.join(os.path.dirname(output_image_path) or ".", "camera_optimization.gif")
        imageio.mimsave(camera_gif_path, camera_frames, duration=0.1, loop=0)
        print(f"Saved camera optimization GIF to: {camera_gif_path}")

    viewmat = torch.cat([best_R_cam, best_t_cam[:, None]], 1)
    viewmat = torch.cat([viewmat, torch.tensor([[0, 0, 0, 1]], device=device)])
    print("\n*** USING OPTIMIZED CAMERA POSE ***\n")

    optimized_cam_render, _, _ = rasterization(
        means_full,
        quats_full,
        scales_full,
        opacities_full,
        colors_full,
        viewmat[None],
        K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
    )
    os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
    cv2.imwrite(
        os.path.join(os.path.dirname(output_image_path) or ".", "optimized_camera_render.png"),
        torch_to_cv(optimized_cam_render[0].detach()),
    )
    cv2.imwrite(
        os.path.join(os.path.dirname(output_image_path) or ".", "camera_target_image.png"),
        torch_to_cv(camera_target_render[0].detach()),
    )
    print(
        f"Saved optimized camera render to: {os.path.join(os.path.dirname(output_image_path) or '.', 'optimized_camera_render.png')}"
    )

    camera_pose_payload = {
        "viewmat": viewmat.detach().cpu(),
        "rotation_matrix": best_R_cam.detach().cpu(),
        "translation": best_t_cam.detach().cpu(),
        "sh_degree": int(sh_degree),
        "best_camera_loss": float(best_cam_loss),
        "initial_view_idx": int(resolved_initial_view_idx),
        "initial_view_source": initial_view_source,
        "initial_camera_pose_path": camera_pose_path,
        "camera_target_image_path": camera_target_image_path,
    }
    os.makedirs(os.path.dirname(camera_pose_output_path) or ".", exist_ok=True)
    torch.save(camera_pose_payload, camera_pose_output_path)
    print(f"Saved optimized camera pose to: {camera_pose_output_path}")

    return viewmat, sh_degree, best_cam_loss, best_R_cam, best_t_cam, camera_target_render


def main(
    checkpoint: str = "./data/person/chkpnt30000.pth",
    data_dir: str = "./data/person",
    camera_target_image_path: str = "./data/person/camera.jpeg",
    initial_view_idx: int = 3,
    initial_camera_pose_path: str = "./data/person/camera_pose.pt",
    rasterizer: Literal["inria", "original", "gsplat"] = "inria",
    data_factor: int = 4,
    output_image_path: str = "result/person/optimized_camera_render.png",
    show_windows: bool = True,
    window_display_scale: float = 0.5,
    sh_degree_override: int = -1,
    camera_lr: float = 0.05,
    camera_iterations: int = 300,
    camera_pose_output_path: str = "result/person/optimized_camera_pose.pt",
):
    print("Loading checkpoint...")
    splats = load_checkpoint(checkpoint, data_dir, rasterizer, data_factor)
    K = splats["camera_matrix"].cuda()
    width, height = splats.get("image_size", (int(K[0, 2] * 2), int(K[1, 2] * 2)))

    optimize_camera_pose(
        splats=splats,
        K=K,
        width=width,
        height=height,
        camera_target_image_path=camera_target_image_path,
        initial_view_idx=initial_view_idx,
        initial_camera_pose_path=initial_camera_pose_path,
        sh_degree_override=sh_degree_override,
        camera_lr=camera_lr,
        camera_iterations=camera_iterations,
        show_windows=show_windows,
        window_display_scale=window_display_scale,
        output_image_path=output_image_path,
        camera_pose_output_path=camera_pose_output_path,
    )

    if show_windows:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
