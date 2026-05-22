import os
from typing import Literal

import cv2
import torch
import tyro

from demof import load_checkpoint, get_viewmat_from_colmap_image, torch_to_cv
from draw_bbox import filter_splats_by_mask, get_oriented_bbox, render_splats_to_dir, sh_dc_to_rgb
from gsplat import rasterization
from mask_utils import detect_rasterizer, load_aligned_mask


def get_local_points(points, center, axes):
    return (points - center) @ axes


def estimate_table_plane_and_footprint(
    means,
    object_mask,
    bbox_center,
    bbox_axes,
    bbox_min,
    bbox_max,
    support_axis=0,
    support_side: Literal["auto", "min", "max"] = "auto",
    footprint_padding=0.04,
    search_depth=0.12,
):
    local_points = get_local_points(means, bbox_center, bbox_axes)
    non_object_mask = ~object_mask
    footprint_axes = [axis for axis in range(3) if axis != support_axis]

    in_footprint = torch.ones(means.shape[0], device=means.device, dtype=torch.bool)
    for axis in footprint_axes:
        in_footprint &= local_points[:, axis] >= (bbox_min[axis] - footprint_padding)
        in_footprint &= local_points[:, axis] <= (bbox_max[axis] + footprint_padding)

    min_side = (
        non_object_mask
        & in_footprint
        & (local_points[:, support_axis] < bbox_min[support_axis])
        & (local_points[:, support_axis] >= bbox_min[support_axis] - search_depth)
    )
    max_side = (
        non_object_mask
        & in_footprint
        & (local_points[:, support_axis] > bbox_max[support_axis])
        & (local_points[:, support_axis] <= bbox_max[support_axis] + search_depth)
    )

    if support_side == "auto":
        min_count = int(min_side.sum().item())
        max_count = int(max_side.sum().item())
        support_side = "min" if min_count >= max_count else "max"
        print(f"Auto table side counts: min={min_count}, max={max_count}; using {support_side}")

    plane_candidates = min_side if support_side == "min" else max_side
    if int(plane_candidates.sum().item()) == 0:
        raise ValueError(
            "Could not find table/support plane candidates. Increase --table-search-depth "
            "or --footprint-padding."
        )

    table_coord = torch.median(local_points[plane_candidates, support_axis])
    return local_points, table_coord, support_side, footprint_axes


def get_table_inpaint_masks(
    means,
    object_mask,
    local_points,
    table_coord,
    bbox_min,
    bbox_max,
    support_axis,
    support_side,
    footprint_axes,
    footprint_padding=0.04,
    table_thickness=0.025,
    reference_ring_padding=0.10,
):
    non_object_mask = ~object_mask

    in_footprint = torch.ones(means.shape[0], device=means.device, dtype=torch.bool)
    in_reference_outer = torch.ones(means.shape[0], device=means.device, dtype=torch.bool)
    for axis in footprint_axes:
        in_footprint &= local_points[:, axis] >= (bbox_min[axis] - footprint_padding)
        in_footprint &= local_points[:, axis] <= (bbox_max[axis] + footprint_padding)
        in_reference_outer &= local_points[:, axis] >= (bbox_min[axis] - footprint_padding - reference_ring_padding)
        in_reference_outer &= local_points[:, axis] <= (bbox_max[axis] + footprint_padding + reference_ring_padding)

    near_table = torch.abs(local_points[:, support_axis] - table_coord) <= table_thickness
    in_reference_ring = in_reference_outer & ~in_footprint

    under_object_table_mask = non_object_mask & in_footprint & near_table
    reference_table_mask = non_object_mask & in_reference_ring & near_table

    if int(reference_table_mask.sum().item()) == 0:
        print("Reference table ring is empty; using nearby non-object table candidates outside object mask.")
        reference_table_mask = non_object_mask & ~in_footprint & near_table

    if int(reference_table_mask.sum().item()) == 0:
        raise ValueError(
            "Could not find reference table Gaussians. Increase --table-thickness "
            "or --reference-ring-padding."
        )

    return under_object_table_mask, reference_table_mask


def get_luminance(colors):
    return 0.2126 * colors[..., 0] + 0.7152 * colors[..., 1] + 0.0722 * colors[..., 2]


def detect_table_shadow_gaussians(
    splats,
    object_mask,
    local_points,
    table_coord,
    bbox_min,
    bbox_max,
    support_axis,
    footprint_axes,
    reference_mask,
    under_object_table_mask,
    footprint_padding=0.04,
    shadow_search_padding=0.18,
    table_thickness=0.025,
    shadow_luminance_margin=0.08,
    knn_k=8,
    knn_power=2.0,
):
    non_object_mask = ~object_mask

    in_shadow_search = torch.ones(local_points.shape[0], device=local_points.device, dtype=torch.bool)
    for axis in footprint_axes:
        in_shadow_search &= local_points[:, axis] >= (
            bbox_min[axis] - footprint_padding - shadow_search_padding
        )
        in_shadow_search &= local_points[:, axis] <= (
            bbox_max[axis] + footprint_padding + shadow_search_padding
        )

    near_table = torch.abs(local_points[:, support_axis] - table_coord) <= table_thickness
    candidate_mask = (
        non_object_mask
        & in_shadow_search
        & near_table
        & ~reference_mask
        & ~under_object_table_mask
    )

    candidate_indices = torch.where(candidate_mask)[0]
    reference_indices = torch.where(reference_mask)[0]
    if candidate_indices.numel() == 0 or reference_indices.numel() == 0:
        return torch.zeros_like(object_mask), candidate_mask

    k = min(knn_k, reference_indices.numel())
    candidate_xy = local_points[candidate_indices][:, footprint_axes]
    reference_xy = local_points[reference_indices][:, footprint_axes]
    distances = torch.cdist(candidate_xy, reference_xy)
    knn_distances, knn_local_indices = torch.topk(distances, k=k, largest=False, dim=1)
    knn_indices = reference_indices[knn_local_indices]

    weights = 1.0 / torch.clamp(knn_distances, min=1e-6).pow(knn_power)
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)

    colors = sh_dc_to_rgb(splats["features_dc"])
    candidate_luminance = get_luminance(colors[candidate_indices])
    reference_luminance = get_luminance(colors[knn_indices])
    local_reference_luminance = (reference_luminance * weights).sum(dim=1)

    shadow_local = candidate_luminance < (local_reference_luminance - shadow_luminance_margin)
    shadow_mask = torch.zeros_like(object_mask)
    shadow_mask[candidate_indices[shadow_local]] = True

    print(f"Shadow search table candidates: {int(candidate_mask.sum().item())}")
    print(f"Shadow Gaussians detected: {int(shadow_mask.sum().item())}")
    return shadow_mask, candidate_mask


def inpaint_table_gaussians(
    splats,
    inpaint_mask,
    reference_mask,
    local_points,
    footprint_axes,
    knn_k=8,
    knn_power=2.0,
):
    inpainted = splats.copy()
    inpainted["features_dc"] = splats["features_dc"].clone()
    inpainted["features_rest"] = splats["features_rest"].clone()

    inpaint_indices = torch.where(inpaint_mask)[0]
    reference_indices = torch.where(reference_mask)[0]
    if inpaint_indices.numel() == 0:
        print("No under-object table Gaussians to inpaint.")
        return inpainted

    if reference_indices.numel() == 0:
        raise ValueError("No reference table Gaussians available for KNN inpainting.")

    k = min(knn_k, reference_indices.numel())
    inpaint_xy = local_points[inpaint_indices][:, footprint_axes]
    reference_xy = local_points[reference_indices][:, footprint_axes]
    distances = torch.cdist(inpaint_xy, reference_xy)
    knn_distances, knn_local_indices = torch.topk(distances, k=k, largest=False, dim=1)
    knn_indices = reference_indices[knn_local_indices]

    weights = 1.0 / torch.clamp(knn_distances, min=1e-6).pow(knn_power)
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)

    reference_dc = splats["features_dc"][knn_indices]
    reference_rest = splats["features_rest"][knn_indices]
    inpainted["features_dc"][inpaint_indices] = (reference_dc * weights[:, :, None, None]).sum(dim=1)
    inpainted["features_rest"][inpaint_indices] = (reference_rest * weights[:, :, None, None]).sum(dim=1)
    print(f"KNN table inpaint: k={k}, power={knn_power}")
    return inpainted


def render_table_plane_overlay(
    output_dir,
    splats,
    table_points,
    inpaint_points,
    shadow_search_points,
    bbox_center,
    bbox_axes,
    bbox_min,
    bbox_max,
    table_coord,
    support_axis,
    footprint_axes,
    footprint_padding=0.04,
    shadow_search_padding=0.18,
    point_radius=2,
    max_points=5000,
):
    os.makedirs(output_dir, exist_ok=True)

    means = splats["means"]
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]

    for image in sorted(splats["colmap_project"].images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image).cuda()
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=int(K[0, 2] * 2),
            height=int(K[1, 2] * 2),
        )

        img = torch_to_cv(output[0]).copy()
        img = draw_table_plane_region(
            img,
            bbox_center,
            bbox_axes,
            bbox_min,
            bbox_max,
            table_coord,
            support_axis,
            footprint_axes,
            footprint_padding,
            viewmat,
            K,
            color=(0, 255, 0),
            thickness=2,
        )
        img = draw_table_plane_region(
            img,
            bbox_center,
            bbox_axes,
            bbox_min,
            bbox_max,
            table_coord,
            support_axis,
            footprint_axes,
            footprint_padding + shadow_search_padding,
            viewmat,
            K,
            color=(255, 0, 255),
            thickness=1,
        )
        img = draw_points(img, table_points, viewmat, K, color=(255, 255, 0), point_radius=point_radius, max_points=max_points)
        img = draw_points(img, shadow_search_points, viewmat, K, color=(255, 0, 0), point_radius=point_radius, max_points=max_points)
        img = draw_points(img, inpaint_points, viewmat, K, color=(0, 0, 255), point_radius=point_radius + 1, max_points=max_points)
        cv2.imwrite(os.path.join(output_dir, image.name), img)


def draw_table_plane_region(
    img,
    bbox_center,
    bbox_axes,
    bbox_min,
    bbox_max,
    table_coord,
    support_axis,
    footprint_axes,
    padding,
    viewmat,
    K,
    color,
    thickness=2,
):
    local_corners = []
    ranges = []
    for axis in footprint_axes:
        ranges.append((bbox_min[axis] - padding, bbox_max[axis] + padding))

    for first in ranges[0]:
        for second in ranges[1]:
            local_point = torch.zeros(3, device=bbox_center.device, dtype=bbox_center.dtype)
            local_point[support_axis] = table_coord
            local_point[footprint_axes[0]] = first
            local_point[footprint_axes[1]] = second
            local_corners.append(local_point)

    corners_3d = torch.stack(local_corners) @ bbox_axes.T + bbox_center
    corners_3d_h = torch.cat(
        [corners_3d, torch.ones(corners_3d.shape[0], 1, device=corners_3d.device)],
        dim=1,
    )
    corners_cam_h = (viewmat @ corners_3d_h.T).T
    corners_cam = corners_cam_h[:, :3] / corners_cam_h[:, 3:]
    corners_2d_h = (K @ corners_cam.T).T
    corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:]

    points = corners_2d.detach().cpu().numpy().astype(int)
    edges = [(0, 1), (0, 2), (3, 1), (3, 2)]
    for i, j in edges:
        if corners_cam_h[i, 2] > 0.1 and corners_cam_h[j, 2] > 0.1:
            cv2.line(img, tuple(points[i]), tuple(points[j]), color, thickness)

    return img


def draw_points(img, points_3d, viewmat, K, color, point_radius=2, max_points=5000):
    if points_3d.numel() == 0:
        return img

    if points_3d.shape[0] > max_points:
        sample_ids = torch.randperm(points_3d.shape[0], device=points_3d.device)[:max_points]
        points_3d = points_3d[sample_ids]

    points_3d_h = torch.cat(
        [points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)],
        dim=1,
    )
    points_cam_h = (viewmat @ points_3d_h.T).T
    points_cam = points_cam_h[:, :3] / points_cam_h[:, 3:]
    points_2d_h = (K @ points_cam.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:]

    points_np = points_2d.detach().cpu().numpy().astype(int)
    depths = points_cam_h[:, 2].detach().cpu().numpy()
    height, width = img.shape[:2]

    for point, depth in zip(points_np, depths):
        x, y = int(point[0]), int(point[1])
        if depth <= 0.1 or x < 0 or x >= width or y < 0 or y >= height:
            continue
        cv2.circle(img, (x, y), point_radius, color, -1)

    return img


def main(
    data_dir: str = "./data/human",
    checkpoint: str = "./data/human/chkpnt30000.pth",
    mask_path: str = "./results/human/mask3d.pth",
    results_dir: str = "./results/human/table_inpaint",
    rasterizer: Literal["auto", "original", "gsplat"] = "auto",
    data_factor: int = 1,
    support_axis: int = 0,
    support_side: Literal["auto", "min", "max"] = "auto",
    footprint_padding: float = 0.05,
    table_search_depth: float = 0.12,
    table_thickness: float = 0.025,
    reference_ring_padding: float = 0.30,
    shadow_search_padding: float = 0.38,
    shadow_luminance_margin: float = 0.08,
    inpaint_shadow: bool = True,
    bbox_padding_x: float = 0.02,
    bbox_padding_y: float = 0.02,
    bbox_padding_z: float = 0.02,
    knn_k: int = 8,
    knn_power: float = 2.0,
    plane_overlay_dir: str = "",
    deleted_images_dir: str = "",
    inpainted_deleted_images_dir: str = "",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    torch.set_default_device("cuda")
    os.makedirs(results_dir, exist_ok=True)

    if plane_overlay_dir == "":
        plane_overlay_dir = os.path.join(results_dir, "table_plane_overlay")
    if deleted_images_dir == "":
        deleted_images_dir = os.path.join(results_dir, "deleted_images")
    if inpainted_deleted_images_dir == "":
        inpainted_deleted_images_dir = os.path.join(results_dir, "deleted_images_table_inpainted")

    if rasterizer == "auto":
        rasterizer = detect_rasterizer(checkpoint)
        print(f"Detected checkpoint rasterizer format: {rasterizer}")

    splats = load_checkpoint(checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor)
    means = splats["means"]
    mask = load_aligned_mask(mask_path, means.shape[0], means.device)

    if int(mask.sum().item()) == 0:
        raise ValueError("mask3d.pth is empty; cannot estimate object footprint.")

    _, bbox_center, bbox_axes, bbox_min, bbox_max = get_oriented_bbox(
        means[mask],
        padding=[bbox_padding_x, bbox_padding_y, bbox_padding_z],
    )

    local_points, table_coord, chosen_side, footprint_axes = estimate_table_plane_and_footprint(
        means,
        mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        support_axis=support_axis,
        support_side=support_side,
        footprint_padding=footprint_padding,
        search_depth=table_search_depth,
    )

    inpaint_mask, reference_mask = get_table_inpaint_masks(
        means,
        mask,
        local_points,
        table_coord,
        bbox_min,
        bbox_max,
        support_axis,
        chosen_side,
        footprint_axes,
        footprint_padding=footprint_padding,
        table_thickness=table_thickness,
        reference_ring_padding=reference_ring_padding,
    )

    if inpaint_shadow:
        shadow_mask, shadow_search_candidate_mask = detect_table_shadow_gaussians(
            splats,
            mask,
            local_points,
            table_coord,
            bbox_min,
            bbox_max,
            support_axis,
            footprint_axes,
            reference_mask,
            inpaint_mask,
            footprint_padding=footprint_padding,
            shadow_search_padding=shadow_search_padding,
            table_thickness=table_thickness,
            shadow_luminance_margin=shadow_luminance_margin,
            knn_k=knn_k,
            knn_power=knn_power,
        )
    else:
        shadow_mask = torch.zeros_like(mask)
        shadow_search_candidate_mask = torch.zeros_like(mask)

    combined_inpaint_mask = inpaint_mask | shadow_mask

    deleted_mask = ~mask
    deleted_splats = filter_splats_by_mask(splats, deleted_mask)
    inpainted_splats = inpaint_table_gaussians(
        splats,
        combined_inpaint_mask,
        reference_mask,
        local_points,
        footprint_axes,
        knn_k=knn_k,
        knn_power=knn_power,
    )
    inpainted_deleted_splats = filter_splats_by_mask(inpainted_splats, deleted_mask)

    print(f"Object mask Gaussians: {int(mask.sum().item())}")
    print(f"Table side: {chosen_side}, support axis: {support_axis}, table local coord: {table_coord.item():.6f}")
    print(f"Under-object table Gaussians recolored: {int(inpaint_mask.sum().item())}")
    print(f"Shadow search candidate Gaussians shown in overlay: {int(shadow_search_candidate_mask.sum().item())}")
    print(f"Shadow table Gaussians recolored: {int(shadow_mask.sum().item())}")
    print(f"Total table/shadow Gaussians recolored: {int(combined_inpaint_mask.sum().item())}")
    print(f"Reference table Gaussians: {int(reference_mask.sum().item())}")

    print(f"Rendering table plane overlay to {plane_overlay_dir}")
    render_table_plane_overlay(
        plane_overlay_dir,
        splats,
        means[reference_mask],
        means[combined_inpaint_mask],
        means[shadow_search_candidate_mask],
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        table_coord,
        support_axis,
        footprint_axes,
        footprint_padding=footprint_padding,
        shadow_search_padding=shadow_search_padding,
    )

    print(f"Rendering deleted images to {deleted_images_dir}")
    render_splats_to_dir(deleted_images_dir, deleted_splats)

    print(f"Rendering table-inpainted deleted images to {inpainted_deleted_images_dir}")
    render_splats_to_dir(inpainted_deleted_images_dir, inpainted_deleted_splats)


if __name__ == "__main__":
    tyro.cli(main)
