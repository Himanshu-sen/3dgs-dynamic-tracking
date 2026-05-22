import torch
import cv2
import os
import shutil
import tyro
from demof import load_checkpoint, get_viewmat_from_colmap_image, torch_to_cv
from gsplat import rasterization

def get_oriented_bbox(points, lower_quantile=0.01, upper_quantile=0.99, padding=0):
    padding = torch.as_tensor(padding, device=points.device, dtype=points.dtype)
    center = points.mean(dim=0)
    centered = points - center

    if points.shape[0] < 3:
        axes = torch.eye(3, device=points.device, dtype=points.dtype)
        min_bounds = torch.quantile(points, lower_quantile, dim=0)
        max_bounds = torch.quantile(points, upper_quantile, dim=0)
        min_bounds = min_bounds - padding
        max_bounds = max_bounds + padding
        corners = get_axis_aligned_bbox_corners(min_bounds, max_bounds)
        return corners, center, axes, min_bounds - center, max_bounds - center

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axes = vh.T

    # Ensure a right-handed local frame. The first column is the local x-axis,
    # aligned with the largest spread of object Gaussians.
    if torch.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    local_points = centered @ axes
    min_bounds = torch.quantile(local_points, lower_quantile, dim=0)
    max_bounds = torch.quantile(local_points, upper_quantile, dim=0)
    min_bounds = min_bounds - padding
    max_bounds = max_bounds + padding

    local_corners = get_axis_aligned_bbox_corners(min_bounds, max_bounds)
    corners = local_corners @ axes.T + center
    return corners, center, axes, min_bounds, max_bounds

def points_inside_oriented_bbox(points, center, axes, min_bounds, max_bounds):
    local_points = (points - center) @ axes
    return ((local_points >= min_bounds) & (local_points <= max_bounds)).all(dim=1)

def sh_dc_to_rgb(features_dc):
    dc = features_dc[:, 0, :] if features_dc.ndim == 3 else features_dc
    return torch.clamp(dc * 0.28209479177387814 + 0.5, 0.0, 1.0)

def prune_object_colored_gaussians_in_bbox(means, features_dc, keep_mask, object_mask,
                                           center, axes, min_bounds, max_bounds,
                                           color_similarity_threshold=0.12,
                                           color_pruning_mode="mean"):
    if color_similarity_threshold <= 0:
        print("Color pruning disabled.")
        return keep_mask, torch.zeros_like(keep_mask)

    inside_bbox = points_inside_oriented_bbox(means, center, axes, min_bounds, max_bounds)
    candidate_mask = keep_mask & inside_bbox

    if int(candidate_mask.sum().item()) == 0 or int(object_mask.sum().item()) == 0:
        return keep_mask, torch.zeros_like(keep_mask)

    colors = sh_dc_to_rgb(features_dc)
    object_colors = colors[object_mask]
    candidate_indices = torch.where(candidate_mask)[0]
    candidate_colors = colors[candidate_indices]

    if color_pruning_mode == "mean":
        object_reference_color = object_colors.mean(dim=0, keepdim=True)
        color_distances = torch.linalg.norm(candidate_colors - object_reference_color, dim=1)
    elif color_pruning_mode == "median":
        object_reference_color = torch.median(object_colors, dim=0).values[None]
        color_distances = torch.linalg.norm(candidate_colors - object_reference_color, dim=1)
    else:
        raise ValueError("color_pruning_mode must be one of: mean, median")

    remove_candidate_mask = color_distances <= color_similarity_threshold

    remove_indices = candidate_indices[remove_candidate_mask]
    color_prune_mask = torch.zeros_like(keep_mask)
    color_prune_mask[remove_indices] = True

    if color_distances.numel() > 0:
        stats = torch.quantile(
            color_distances,
            torch.tensor([0.0, 0.5, 0.9, 1.0], device=color_distances.device, dtype=color_distances.dtype),
        )
        print(
            f"Object-color {color_pruning_mode} distance inside bbox min/median/p90/max: "
            f"{stats[0].item():.6f}, {stats[1].item():.6f}, "
            f"{stats[2].item():.6f}, {stats[3].item():.6f}"
        )

    print(f"Color-pruned object-like Gaussians inside bbox: {int(color_prune_mask.sum().item())}")
    return keep_mask & ~color_prune_mask, color_prune_mask

def prune_low_opacity_gaussians_in_bbox(means, opacity, keep_mask,
                                        center, axes, min_bounds, max_bounds,
                                        opacity_threshold=0.05):
    if opacity_threshold <= 0:
        return keep_mask, torch.zeros_like(keep_mask)

    inside_bbox = points_inside_oriented_bbox(means, center, axes, min_bounds, max_bounds)
    opacities = torch.sigmoid(opacity)
    prune_mask = keep_mask & inside_bbox & (opacities < opacity_threshold)

    if int((keep_mask & inside_bbox).sum().item()) > 0:
        bbox_opacities = opacities[keep_mask & inside_bbox]
        stats = torch.quantile(
            bbox_opacities,
            torch.tensor([0.0, 0.5, 0.9, 1.0], device=bbox_opacities.device, dtype=bbox_opacities.dtype),
        )
        print(
            "Opacity inside bbox min/median/p90/max: "
            f"{stats[0].item():.6f}, {stats[1].item():.6f}, "
            f"{stats[2].item():.6f}, {stats[3].item():.6f}"
        )

    print(f"Low-opacity pruned Gaussians inside bbox: {int(prune_mask.sum().item())}")
    return keep_mask & ~prune_mask, prune_mask

def prune_low_opacity_gaussians(means, opacity, keep_mask, opacity_threshold=0.01):
    if opacity_threshold <= 0:
        return keep_mask, torch.zeros_like(keep_mask)

    opacities = torch.sigmoid(opacity)
    prune_mask = keep_mask & (opacities < opacity_threshold)

    if int(keep_mask.sum().item()) > 0:
        remaining_opacities = opacities[keep_mask]
        stats = torch.quantile(
            remaining_opacities,
            torch.tensor([0.0, 0.5, 0.9, 1.0], device=remaining_opacities.device, dtype=remaining_opacities.dtype),
        )
        print(
            "Global opacity min/median/p90/max: "
            f"{stats[0].item():.6f}, {stats[1].item():.6f}, "
            f"{stats[2].item():.6f}, {stats[3].item():.6f}"
        )

    print(f"Global low-opacity pruned Gaussians: {int(prune_mask.sum().item())}")
    return keep_mask & ~prune_mask, prune_mask

def prune_dark_gaussians_in_bbox(means, features_dc, keep_mask,
                                 center, axes, min_bounds, max_bounds,
                                 darkness_threshold=0.0):
    if darkness_threshold <= 0:
        return keep_mask, torch.zeros_like(keep_mask)

    inside_bbox = points_inside_oriented_bbox(means, center, axes, min_bounds, max_bounds)
    candidate_mask = keep_mask & inside_bbox

    if int(candidate_mask.sum().item()) == 0:
        return keep_mask, torch.zeros_like(keep_mask)

    colors = sh_dc_to_rgb(features_dc)
    luminance = (
        0.2126 * colors[:, 0]
        + 0.7152 * colors[:, 1]
        + 0.0722 * colors[:, 2]
    )
    prune_mask = candidate_mask & (luminance <= darkness_threshold)

    bbox_luminance = luminance[candidate_mask]
    stats = torch.quantile(
        bbox_luminance,
        torch.tensor([0.0, 0.5, 0.9, 1.0], device=bbox_luminance.device, dtype=bbox_luminance.dtype),
    )
    print(
        "Luminance inside bbox min/median/p90/max: "
        f"{stats[0].item():.6f}, {stats[1].item():.6f}, "
        f"{stats[2].item():.6f}, {stats[3].item():.6f}"
    )
    print(f"Dark-pruned Gaussians inside bbox: {int(prune_mask.sum().item())}")
    return keep_mask & ~prune_mask, prune_mask

def prune_sparse_gaussians_in_bbox(means, keep_mask, center, axes, min_bounds, max_bounds,
                                   sparse_neighbor_k=8,
                                   sparse_distance_quantile=0.9,
                                   sparse_distance_threshold=0.0):
    if sparse_neighbor_k <= 0:
        return keep_mask, torch.zeros_like(keep_mask)

    inside_bbox = points_inside_oriented_bbox(means, center, axes, min_bounds, max_bounds)
    candidate_mask = keep_mask & inside_bbox
    candidate_indices = torch.where(candidate_mask)[0]

    if candidate_indices.numel() <= sparse_neighbor_k:
        print("Sparse pruning skipped: not enough remaining Gaussians inside bbox.")
        return keep_mask, torch.zeros_like(keep_mask)

    local_points = (means[candidate_indices] - center) @ axes
    extent = torch.clamp(max_bounds - min_bounds, min=1e-6)
    normalized_points = (local_points - min_bounds) / extent

    distances = torch.cdist(normalized_points, normalized_points)
    kth_distances = torch.topk(
        distances,
        k=sparse_neighbor_k + 1,
        largest=False,
        dim=1,
    ).values[:, -1]

    if sparse_distance_threshold > 0:
        threshold = torch.tensor(
            sparse_distance_threshold,
            device=means.device,
            dtype=means.dtype,
        )
    else:
        threshold = torch.quantile(kth_distances, sparse_distance_quantile)

    sparse_local_mask = kth_distances > threshold
    sparse_indices = candidate_indices[sparse_local_mask]
    sparse_prune_mask = torch.zeros_like(keep_mask)
    sparse_prune_mask[sparse_indices] = True

    stats = torch.quantile(
        kth_distances,
        torch.tensor([0.0, 0.5, 0.9, 1.0], device=kth_distances.device, dtype=kth_distances.dtype),
    )
    print(
        f"kNN-{sparse_neighbor_k} distance inside bbox min/median/p90/max: "
        f"{stats[0].item():.6f}, {stats[1].item():.6f}, "
        f"{stats[2].item():.6f}, {stats[3].item():.6f}"
    )
    print(f"Sparse-pruned Gaussians inside bbox: {int(sparse_prune_mask.sum().item())}")
    return keep_mask & ~sparse_prune_mask, sparse_prune_mask

def prune_floating_gaussians(means, keep_mask,
                             floating_voxel_size=0.02,
                             floating_min_neighbors=8,
                             floating_lower_quantile=0.01,
                             floating_upper_quantile=0.99):
    if floating_voxel_size <= 0 or floating_min_neighbors <= 0:
        return keep_mask, torch.zeros_like(keep_mask)

    candidate_indices = torch.where(keep_mask)[0]
    if candidate_indices.numel() == 0:
        return keep_mask, torch.zeros_like(keep_mask)

    points = means[candidate_indices]
    lower = torch.quantile(points, floating_lower_quantile, dim=0)
    upper = torch.quantile(points, floating_upper_quantile, dim=0)
    extent = torch.clamp(upper - lower, min=1e-6)
    normalized_points = (points - lower) / extent
    voxel_coords = torch.floor(normalized_points / floating_voxel_size).to(torch.int64)

    coord_min = voxel_coords.min(dim=0).values
    voxel_coords = voxel_coords - coord_min
    coord_max = voxel_coords.max(dim=0).values + 1
    multipliers = torch.stack(
        [
            coord_max[1] * coord_max[2],
            coord_max[2],
            torch.ones((), device=means.device, dtype=torch.int64),
        ]
    )
    voxel_ids = (voxel_coords * multipliers).sum(dim=1)
    unique_ids, inverse, counts = torch.unique(
        voxel_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    offsets = torch.tensor(
        [
            [dx, dy, dz]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ],
        device=means.device,
        dtype=torch.int64,
    )
    neighbor_counts = torch.zeros(candidate_indices.shape[0], device=means.device, dtype=torch.int64)

    for offset in offsets:
        neighbor_coords = voxel_coords + offset
        valid = ((neighbor_coords >= 0) & (neighbor_coords < coord_max)).all(dim=1)
        if int(valid.sum().item()) == 0:
            continue

        neighbor_ids = (neighbor_coords[valid] * multipliers).sum(dim=1)
        positions = torch.searchsorted(unique_ids, neighbor_ids)
        in_range = positions < unique_ids.numel()
        found = torch.zeros_like(in_range)
        found[in_range] = unique_ids[positions[in_range]] == neighbor_ids[in_range]
        if int(found.sum().item()) == 0:
            continue

        valid_indices = torch.where(valid)[0]
        neighbor_counts[valid_indices[found]] += counts[positions[found]]

    neighbor_counts = neighbor_counts - counts[inverse]
    floating_local_mask = neighbor_counts < floating_min_neighbors
    floating_indices = candidate_indices[floating_local_mask]
    floating_prune_mask = torch.zeros_like(keep_mask)
    floating_prune_mask[floating_indices] = True

    if neighbor_counts.numel() > 0:
        stats = torch.quantile(
            neighbor_counts.float(),
            torch.tensor([0.0, 0.5, 0.9, 1.0], device=means.device),
        )
        print(
            "Global voxel-neighbor count min/median/p90/max: "
            f"{stats[0].item():.1f}, {stats[1].item():.1f}, "
            f"{stats[2].item():.1f}, {stats[3].item():.1f}"
        )

    print(f"Global floating pruned Gaussians: {int(floating_prune_mask.sum().item())}")
    return keep_mask & ~floating_prune_mask, floating_prune_mask

def count_remaining_gaussians_in_bbox(means, keep_mask, center, axes, min_bounds, max_bounds):
    inside_bbox = points_inside_oriented_bbox(means, center, axes, min_bounds, max_bounds)
    count = int((keep_mask & inside_bbox).sum().item())
    print(f"Remaining Gaussians inside bbox: {count}")
    return count

def kmeans(points, num_clusters, iterations=30):
    num_points = points.shape[0]
    num_clusters = min(num_clusters, num_points)
    init_ids = torch.randperm(num_points, device=points.device)[:num_clusters]
    centers = points[init_ids].clone()

    for _ in range(iterations):
        distances = torch.cdist(points, centers)
        labels = torch.argmin(distances, dim=1)

        new_centers = centers.clone()
        for cluster_id in range(num_clusters):
            cluster_points = points[labels == cluster_id]
            if cluster_points.shape[0] > 0:
                new_centers[cluster_id] = cluster_points.mean(dim=0)

        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers

    distances = torch.cdist(points, centers)
    labels = torch.argmin(distances, dim=1)
    return labels, centers

def colorize_bbox_clusters(means, colors_dc, colors_rest, keep_mask,
                           center, axes, min_bounds, max_bounds,
                           num_clusters=5, kmeans_iterations=30,
                           colorize_render=False):
    if num_clusters <= 0:
        return colors_dc, colors_rest, None, None, None

    inside_bbox = points_inside_oriented_bbox(means, center, axes, min_bounds, max_bounds)
    cluster_mask = keep_mask & inside_bbox
    cluster_indices = torch.where(cluster_mask)[0]

    if cluster_indices.numel() == 0:
        print("BBox clustering skipped: no remaining Gaussians inside bbox.")
        return colors_dc, colors_rest, None, None, None

    local_points = (means[cluster_indices] - center) @ axes
    extent = torch.clamp(max_bounds - min_bounds, min=1e-6)
    normalized_points = (local_points - min_bounds) / extent
    labels, _ = kmeans(normalized_points, num_clusters, iterations=kmeans_iterations)
    actual_clusters = int(labels.max().item()) + 1

    palette = torch.tensor(
        [
            [1.0, 0.1, 0.1],
            [0.1, 0.8, 1.0],
            [0.1, 1.0, 0.25],
            [1.0, 0.85, 0.1],
            [0.9, 0.1, 1.0],
            [1.0, 0.45, 0.05],
            [0.45, 0.2, 1.0],
            [0.0, 1.0, 0.8],
        ],
        device=means.device,
        dtype=means.dtype,
    )
    cluster_colors = palette[labels % palette.shape[0]]

    if colorize_render:
        render_colors_dc = colors_dc.clone()
        render_colors_rest = colors_rest.clone()
        render_colors_dc[cluster_indices, 0, :] = (cluster_colors - 0.5) / 0.28209479177387814
        render_colors_rest[cluster_indices] = 0
    else:
        render_colors_dc = colors_dc
        render_colors_rest = colors_rest

    counts = torch.bincount(labels, minlength=actual_clusters)
    print(f"Clustered remaining bbox Gaussians into {actual_clusters} clusters.")
    for cluster_id, count in enumerate(counts.tolist()):
        print(f"  cluster {cluster_id}: {count} Gaussians")

    return render_colors_dc, render_colors_rest, cluster_indices, labels, palette

def draw_cluster_overlay(img, means, cluster_indices, labels, palette, viewmat, K,
                         point_radius=3, max_points=2000):
    if cluster_indices is None or labels is None or palette is None:
        return img

    if cluster_indices.numel() == 0:
        return img

    if cluster_indices.numel() > max_points:
        sample_ids = torch.randperm(cluster_indices.numel(), device=cluster_indices.device)[:max_points]
        cluster_indices = cluster_indices[sample_ids]
        labels = labels[sample_ids]

    points_3d = means[cluster_indices]
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
    labels_np = labels.detach().cpu().numpy()
    palette_np = palette.detach().cpu().numpy()

    height, width = img.shape[:2]
    for point, depth, label in zip(points_np, depths, labels_np):
        x, y = int(point[0]), int(point[1])
        if depth <= 0.1 or x < 0 or x >= width or y < 0 or y >= height:
            continue

        rgb = palette_np[int(label) % palette_np.shape[0]]
        bgr = tuple(int(channel * 255) for channel in rgb[::-1])
        cv2.circle(img, (x, y), point_radius, bgr, -1)

    return img

def get_axis_aligned_bbox_corners(min_bounds, max_bounds):
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    return torch.stack([
        torch.stack([x_min, y_min, z_min]),
        torch.stack([x_min, y_min, z_max]),
        torch.stack([x_min, y_max, z_min]),
        torch.stack([x_min, y_max, z_max]),
        torch.stack([x_max, y_min, z_min]),
        torch.stack([x_max, y_min, z_max]),
        torch.stack([x_max, y_max, z_min]),
        torch.stack([x_max, y_max, z_max]),
    ])

def draw_3d_bbox(img, corners_3d, viewmat, K):
    
    # Transform to camera space
    # viewmat is 4x4 (world to camera, but wait... 
    # check demof.py get_viewmat_from_colmap_image, typically it is world to camera (R, t))
    # viewmat[:3, :3] = R, viewmat[:3, 3] = t
    # in demof.py: output, _, info = rasterization(..., viewmats=viewmat[None]...)
    # gsplat usually takes world to camera or camera to world? Yes, viewmat is W2C.
    
    corners_3d_h = torch.cat([corners_3d, torch.ones(8, 1, device=corners_3d.device)], dim=1)
    corners_cam_h = (viewmat @ corners_3d_h.T).T
    corners_cam = corners_cam_h[:, :3] / corners_cam_h[:, 3:]
    
    # Project to 2D
    # z > 0 check normally, but let's assume it's in front of camera
    corners_2d_h = (K @ corners_cam.T).T
    corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:]
    
    points = corners_2d.detach().cpu().numpy().astype(int)
    
    # 12 edges
    edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7)
    ]
    
    for (i, j) in edges:
        if corners_cam_h[i, 2] > 0.1 and corners_cam_h[j, 2] > 0.1: # simple near plane clipping check
            cv2.line(img, tuple(points[i]), tuple(points[j]), (0, 255, 0), 2)
            
    return img

def filter_splats_by_mask(splats, mask):
    filtered = splats.copy()
    for key in ["means", "features_dc", "features_rest", "scaling", "rotation", "opacity"]:
        filtered[key] = filtered[key][mask]
    return filtered

def render_splats_to_dir(output_dir, splats):
    os.makedirs(output_dir, exist_ok=True)

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
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
        cv2.imwrite(os.path.join(output_dir, image.name), torch_to_cv(output[0]))

def save_checkpoint_with_mask(checkpoint_path, keep_mask, output_path=None, backup=True):
    if output_path is None or output_path == "":
        root, ext = os.path.splitext(checkpoint_path)
        output_path = f"{root}_global_pruned{ext}"

    if backup and os.path.abspath(output_path) == os.path.abspath(checkpoint_path):
        backup_path = checkpoint_path + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(checkpoint_path, backup_path)
            print(f"Backed up original checkpoint to {backup_path}")

    model = torch.load(checkpoint_path)

    def filter_tensor(tensor):
        tensor_keep_mask = keep_mask.to(device=tensor.device)
        return tensor[tensor_keep_mask]

    if isinstance(model, tuple):
        model_params, extra = model
        params = list(model_params)
        for idx in [1, 2, 3, 4, 5, 6]:
            params[idx] = filter_tensor(params[idx])
        filtered_params = tuple(params) if isinstance(model_params, tuple) else params
        filtered_model = (filtered_params, extra)
    elif isinstance(model, dict) and "splats" in model:
        filtered_model = model.copy()
        filtered_splats = model["splats"].copy()
        for key in ["means", "sh0", "shN", "scales", "quats", "opacities"]:
            if key in filtered_splats:
                filtered_splats[key] = filter_tensor(filtered_splats[key])
        filtered_model["splats"] = filtered_splats
    else:
        raise ValueError("Unsupported checkpoint format for Gaussian deletion.")

    torch.save(filtered_model, output_path)
    print(f"Saved cleaned checkpoint to {output_path}")
    print(f"Checkpoint Gaussians kept: {int(keep_mask.sum().item())}/{keep_mask.numel()}")

def main(
    data_dir: str = "./data/toy",
    checkpoint: str = "./data/toy/chkpnt30000.pth",
    mask_path: str = "./results/toy/mask3d.pth",
    results_dir: str = "./results/toy/bbox_render",
    bbox_padding_x: float = 0.03,
    bbox_padding_y: float = 0.03,
    bbox_padding_z: float = 0.03,
    color_similarity_threshold: float = 0.5,
    color_pruning_mode: str = "mean",
    opacity_threshold: float = 0.01,
    sparse_neighbor_k: int = 8,
    sparse_distance_quantile: float = 0.9,
    sparse_distance_threshold: float = 0.09,
    darkness_threshold: float = 0.05,
    num_bbox_clusters: int = 3,
    kmeans_iterations: int = 30,
    colorize_cluster_render: bool = False,
    show_cluster_overlay: bool = True,
    cluster_point_radius: int = 3,
    max_cluster_overlay_points: int = 2000,
    extracted_results_dir: str = "",
    deleted_results_dir: str = "",
    final_results_dir: str = "",
    render_final_images: bool = True,
    render_extracted_with_original_mask: bool = True,
    save_updated_mask: bool = True,
    updated_mask_path: str = "./results/new/mask3d_bottle_preserved.pth",
    save_mask_aligned_to_cleaned_checkpoint: bool = True,
    save_cleaned_checkpoint: bool = True,
    cleaned_checkpoint_path: str = "./data/new/chkpnt30000_bottle_preserved.pth",
    backup_checkpoint: bool = True,
    delete_all_gaussians_in_bbox: bool = False,
    delete_low_opacity_gaussians_in_bbox: bool = True,
    delete_dark_gaussians_in_bbox: bool = False,
    delete_sparse_gaussians_in_bbox: bool = False,
    delete_color_pruned_gaussians_in_bbox: bool = False,
    prune_global_low_opacity: bool = False,
    global_opacity_threshold: float = 0.01,
    prune_floating_gaussians_after_bbox: bool = False,
    floating_voxel_size: float = 0.02,
    floating_min_neighbors: int = 8,
    floating_lower_quantile: float = 0.01,
    floating_upper_quantile: float = 0.99,
):
    torch.set_default_device('cuda')
    os.makedirs(results_dir, exist_ok=True)
    
    source_checkpoint = checkpoint
    splats = load_checkpoint(checkpoint, data_dir)
    means = splats["means"]
    mask = torch.load(mask_path).cuda()
    mask_root, mask_ext = os.path.splitext(mask_path)
    full_mask_path = f"{mask_root}_full{mask_ext}"

    if mask.shape[0] != means.shape[0] and checkpoint.endswith(".backup"):
        non_backup_checkpoint = checkpoint[: -len(".backup")]
        if os.path.exists(non_backup_checkpoint):
            print(
                "Mask/checkpoint Gaussian count mismatch "
                f"({mask.shape[0]} mask vs {means.shape[0]} checkpoint)."
            )
            print(f"Trying non-backup checkpoint with aligned mask: {non_backup_checkpoint}")
            candidate_splats = load_checkpoint(non_backup_checkpoint, data_dir)
            candidate_means = candidate_splats["means"]
            if mask.shape[0] == candidate_means.shape[0]:
                source_checkpoint = non_backup_checkpoint
                splats = candidate_splats
                means = candidate_means
                print(f"Using aligned checkpoint: {source_checkpoint}")

    if mask.shape[0] != means.shape[0] and os.path.exists(full_mask_path):
        print(
            "Mask/checkpoint Gaussian count mismatch "
            f"({mask.shape[0]} mask vs {means.shape[0]} checkpoint)."
        )
        print(f"Reloading full-size mask from {full_mask_path}")
        mask = torch.load(full_mask_path).cuda()

    if (mask.shape[0] != means.shape[0]) or (int(mask.sum().item()) == 0 and os.path.exists(full_mask_path)):
        backup_path = checkpoint + ".backup"
        if os.path.exists(backup_path):
            print(
                "Mask/checkpoint Gaussian count mismatch or empty aligned mask "
                f"({mask.shape[0]} mask vs {means.shape[0]} checkpoint, {int(mask.sum().item())} selected)."
            )
            print(f"Reloading original checkpoint from backup: {backup_path}")
            source_checkpoint = backup_path
            splats = load_checkpoint(source_checkpoint, data_dir)
            means = splats["means"]
            if os.path.exists(full_mask_path):
                print(f"Reloading full-size mask from {full_mask_path}")
                mask = torch.load(full_mask_path).cuda()

        if mask.shape[0] != means.shape[0]:
            raise ValueError(
                "mask3d.pth and checkpoint still have different Gaussian counts "
                f"({mask.shape[0]} vs {means.shape[0]}). "
                "Use the original checkpoint or regenerate mask3d.pth for this checkpoint."
            )
    
    object_means = means[mask]
    
    bbox_corners, bbox_center, bbox_axes, bbox_min, bbox_max = get_oriented_bbox(
        object_means,
        padding=[bbox_padding_x, bbox_padding_y, bbox_padding_z],
    )
    non_object_mask = ~mask
    keep_mask = non_object_mask.clone()
    keep_mask, color_prune_mask = prune_object_colored_gaussians_in_bbox(
        means,
        splats["features_dc"],
        keep_mask,
        mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        color_similarity_threshold=color_similarity_threshold,
        color_pruning_mode=color_pruning_mode,
    )
    _, low_opacity_prune_mask = prune_low_opacity_gaussians_in_bbox(
        means,
        splats["opacity"],
        non_object_mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        opacity_threshold=opacity_threshold,
    )
    keep_mask = keep_mask & ~low_opacity_prune_mask
    keep_mask, sparse_prune_mask = prune_sparse_gaussians_in_bbox(
        means,
        keep_mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        sparse_neighbor_k=sparse_neighbor_k,
        sparse_distance_quantile=sparse_distance_quantile,
        sparse_distance_threshold=sparse_distance_threshold,
    )
    keep_mask, dark_prune_mask = prune_dark_gaussians_in_bbox(
        means,
        splats["features_dc"],
        keep_mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        darkness_threshold=darkness_threshold,
    )
    remaining_bbox_count = count_remaining_gaussians_in_bbox(
        means,
        keep_mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
    )

    updated_mask = mask
    expanded_candidate_mask = mask | color_prune_mask | sparse_prune_mask
    inside_bbox = points_inside_oriented_bbox(means, bbox_center, bbox_axes, bbox_min, bbox_max)
    deleted_remaining_mask = keep_mask & inside_bbox
    print(f"Color-pruned non-mask candidates in bbox: {int(color_prune_mask.sum().item())}")
    print(f"Sparse-pruned non-mask candidates in bbox: {int(sparse_prune_mask.sum().item())}")
    print(f"Low-opacity non-mask Gaussians in bbox: {int(low_opacity_prune_mask.sum().item())}")
    print(f"Dark non-mask Gaussians in bbox: {int(dark_prune_mask.sum().item())}")
    print(f"Remaining non-mask Gaussians in bbox after candidate pruning: {int(deleted_remaining_mask.sum().item())}")
    print(f"Object mask Gaussian count: {int(updated_mask.sum().item())}")
    print(f"Expanded candidate mask Gaussian count: {int(expanded_candidate_mask.sum().item())}")

    if extracted_results_dir == "":
        extracted_results_dir = os.path.join(os.path.dirname(results_dir), "extracted_images_updated")
    if deleted_results_dir == "":
        deleted_results_dir = os.path.join(os.path.dirname(results_dir), "deleted_images_global_pruned")
    if final_results_dir == "":
        final_results_dir = os.path.join(os.path.dirname(results_dir), "final_render")

    if delete_all_gaussians_in_bbox:
        cleaned_render_mask = ~inside_bbox
    else:
        bbox_cleanup_prune_mask = torch.zeros_like(mask)
        if delete_low_opacity_gaussians_in_bbox:
            bbox_cleanup_prune_mask |= low_opacity_prune_mask
        if delete_dark_gaussians_in_bbox:
            bbox_cleanup_prune_mask |= dark_prune_mask
        if delete_sparse_gaussians_in_bbox:
            bbox_cleanup_prune_mask |= sparse_prune_mask
        if delete_color_pruned_gaussians_in_bbox:
            bbox_cleanup_prune_mask |= color_prune_mask

        cleaned_render_mask = ~bbox_cleanup_prune_mask
        print(f"Deleted selected non-mask Gaussians inside bbox: {int(bbox_cleanup_prune_mask.sum().item())}")

    if prune_global_low_opacity:
        cleaned_render_mask, global_low_opacity_prune_mask = prune_low_opacity_gaussians(
            means,
            splats["opacity"],
            cleaned_render_mask,
            opacity_threshold=global_opacity_threshold,
        )
    else:
        global_low_opacity_prune_mask = torch.zeros_like(cleaned_render_mask)

    if prune_floating_gaussians_after_bbox:
        cleaned_render_mask, floating_prune_mask = prune_floating_gaussians(
            means,
            cleaned_render_mask,
            floating_voxel_size=floating_voxel_size,
            floating_min_neighbors=floating_min_neighbors,
            floating_lower_quantile=floating_lower_quantile,
            floating_upper_quantile=floating_upper_quantile,
        )
    else:
        floating_prune_mask = torch.zeros_like(cleaned_render_mask)

    remaining_after_delete = int((cleaned_render_mask & inside_bbox).sum().item())
    print(f"Remaining Gaussians inside bbox for cleaned renders: {remaining_after_delete}")
    if delete_all_gaussians_in_bbox:
        print(f"Hard-deleted all Gaussians inside bbox: {int(inside_bbox.sum().item())}")
    print(f"Deleted global low-opacity Gaussians: {int(global_low_opacity_prune_mask.sum().item())}")
    print(f"Deleted global floating Gaussians: {int(floating_prune_mask.sum().item())}")
    print(f"Global-pruned environment Gaussians kept: {int(cleaned_render_mask.sum().item())}/{cleaned_render_mask.numel()}")

    if save_updated_mask:
        mask_output_path = updated_mask_path if updated_mask_path != "" else mask_path
        if save_mask_aligned_to_cleaned_checkpoint:
            mask_to_save = updated_mask[cleaned_render_mask]
            mask_root, mask_ext = os.path.splitext(mask_output_path)
            full_mask_path = f"{mask_root}_full{mask_ext}"
            torch.save(updated_mask.detach().cpu(), full_mask_path)
            print(
                "Saving mask3d aligned to cleaned checkpoint: "
                f"{mask_to_save.numel()} mask entries for {int(cleaned_render_mask.sum().item())} kept Gaussians"
            )
            print(f"Saved full-size mask3d copy to {full_mask_path}")
        else:
            mask_to_save = updated_mask
            print(f"Saving full-size mask3d: {mask_to_save.numel()} mask entries")

        torch.save(mask_to_save.detach().cpu(), mask_output_path)
        print(f"Saved updated mask3d to {mask_output_path}")

    extraction_mask = mask if render_extracted_with_original_mask else expanded_candidate_mask
    extracted_splats = filter_splats_by_mask(splats, extraction_mask)
    print(f"Extracted render Gaussians: {int(extraction_mask.sum().item())}")
    print(f"Rendering extracted images to {extracted_results_dir}")
    render_splats_to_dir(extracted_results_dir, extracted_splats)

    deleted_splats = filter_splats_by_mask(splats, cleaned_render_mask)
    print(f"Rendering deleted images after global environment pruning to {deleted_results_dir}")
    render_splats_to_dir(deleted_results_dir, deleted_splats)
    
    colmap_project = splats["colmap_project"]

    if save_cleaned_checkpoint:
        save_checkpoint_with_mask(
            source_checkpoint,
            cleaned_render_mask,
            output_path=cleaned_checkpoint_path,
            backup=backup_checkpoint,
        )

    if render_final_images:
        print(f"Rendering final images after all pruning to {final_results_dir}")
        render_splats_to_dir(final_results_dir, deleted_splats)

    render_means = means[cleaned_render_mask]
    (
        clustered_colors_dc,
        clustered_colors_rest,
        cluster_indices,
        cluster_labels,
        cluster_palette,
    ) = colorize_bbox_clusters(
        means,
        splats["features_dc"],
        splats["features_rest"],
        cleaned_render_mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        num_clusters=num_bbox_clusters,
        kmeans_iterations=kmeans_iterations,
        colorize_render=colorize_cluster_render,
    )
    colors_dc = clustered_colors_dc[cleaned_render_mask]
    colors_rest = clustered_colors_rest[cleaned_render_mask]
    opacity = splats["opacity"][cleaned_render_mask]
    scaling = splats["scaling"][cleaned_render_mask]
    rotation = splats["rotation"][cleaned_render_mask]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(opacity)
    scales = torch.exp(scaling)
    quats = rotation
    K = splats["camera_matrix"]
    
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image).cuda()
        
        output, _, _ = rasterization(
            render_means,
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
        
        img_np = torch_to_cv(output[0])
        
        # We need to copy img_np to avoid modifying the original array if needed, though it's new
        img_np = img_np.copy()

        if show_cluster_overlay:
            img_np = draw_cluster_overlay(
                img_np,
                means,
                cluster_indices,
                cluster_labels,
                cluster_palette,
                viewmat,
                K,
                point_radius=cluster_point_radius,
                max_points=max_cluster_overlay_points,
            )

        img_np = draw_3d_bbox(img_np, bbox_corners, viewmat, K)
        
        cv2.imwrite(os.path.join(results_dir, image.name), img_np)
        # print(f"Saved {image.name}")

if __name__ == "__main__":
    tyro.cli(main)
