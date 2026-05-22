import os
import shutil
from collections import deque
from typing import Literal

import cv2
import torch
import tyro

from demof import load_checkpoint, get_viewmat_from_colmap_image, torch_to_cv
from draw_bbox import (
    filter_splats_by_mask,
    get_oriented_bbox,
    points_inside_oriented_bbox,
    render_splats_to_dir,
    sh_dc_to_rgb,
)
from gsplat import rasterization
from mask_utils import detect_rasterizer, load_aligned_mask


def fill_voxel_holes(object_voxels, grid_size):
    voxels_cpu = object_voxels.detach().cpu().tolist()
    filled = set(tuple(v) for v in voxels_cpu)

    for fill_axis in range(3):
        columns = {}
        other_axes = [axis for axis in range(3) if axis != fill_axis]
        for voxel in voxels_cpu:
            key = (voxel[other_axes[0]], voxel[other_axes[1]])
            columns.setdefault(key, [voxel[fill_axis], voxel[fill_axis]])
            columns[key][0] = min(columns[key][0], voxel[fill_axis])
            columns[key][1] = max(columns[key][1], voxel[fill_axis])

        for key, (lower, upper) in columns.items():
            for value in range(lower, upper + 1):
                voxel = [0, 0, 0]
                voxel[fill_axis] = value
                voxel[other_axes[0]] = key[0]
                voxel[other_axes[1]] = key[1]
                filled.add(tuple(voxel))

    filled_voxels = torch.tensor(
        sorted(filled),
        device=object_voxels.device,
        dtype=object_voxels.dtype,
    )
    filled_voxels = torch.clamp(filled_voxels, min=0)
    filled_voxels = torch.minimum(filled_voxels, grid_size[None] - 1)
    return torch.unique(filled_voxels, dim=0)


def fill_voxel_shell_interior(shell_voxels, grid_size):
    grid_dims = tuple(int(v) for v in grid_size.detach().cpu().tolist())
    shell = set(tuple(v) for v in shell_voxels.detach().cpu().tolist())
    outside = set()
    queue = deque()

    def add_outside(voxel):
        if voxel in shell or voxel in outside:
            return
        outside.add(voxel)
        queue.append(voxel)

    nx, ny, nz = grid_dims
    for x in range(nx):
        for y in range(ny):
            add_outside((x, y, 0))
            add_outside((x, y, nz - 1))
    for x in range(nx):
        for z in range(nz):
            add_outside((x, 0, z))
            add_outside((x, ny - 1, z))
    for y in range(ny):
        for z in range(nz):
            add_outside((0, y, z))
            add_outside((nx - 1, y, z))

    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in neighbor_offsets:
            neighbor = (x + dx, y + dy, z + dz)
            if (
                neighbor[0] < 0 or neighbor[0] >= nx
                or neighbor[1] < 0 or neighbor[1] >= ny
                or neighbor[2] < 0 or neighbor[2] >= nz
            ):
                continue
            add_outside(neighbor)

    enclosed = [
        (x, y, z)
        for x in range(nx)
        for y in range(ny)
        for z in range(nz)
        if (x, y, z) not in outside
    ]
    if len(enclosed) == 0:
        return shell_voxels

    enclosed_voxels = torch.tensor(
        enclosed,
        device=shell_voxels.device,
        dtype=shell_voxels.dtype,
    )
    return torch.unique(enclosed_voxels, dim=0)


def build_voxel_surface_mesh(object_voxels, bbox_center, bbox_axes, bbox_min, bbox_max, voxel_size):
    voxels_cpu = [tuple(v) for v in object_voxels.detach().cpu().tolist()]
    voxel_set = set(voxels_cpu)
    extent = torch.clamp(bbox_max - bbox_min, min=1e-6)

    face_defs = [
        ((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
        ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
        ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
        ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
    ]

    vertex_ids = {}
    vertices_local = []
    faces = []

    for voxel in voxels_cpu:
        for neighbor_offset, corner_offsets in face_defs:
            neighbor = (
                voxel[0] + neighbor_offset[0],
                voxel[1] + neighbor_offset[1],
                voxel[2] + neighbor_offset[2],
            )
            if neighbor in voxel_set:
                continue

            face = []
            for corner_offset in corner_offsets:
                corner = (
                    voxel[0] + corner_offset[0],
                    voxel[1] + corner_offset[1],
                    voxel[2] + corner_offset[2],
                )
                vertex_id = vertex_ids.get(corner)
                if vertex_id is None:
                    vertex_id = len(vertices_local)
                    vertex_ids[corner] = vertex_id
                    vertices_local.append(corner)
                face.append(vertex_id)
            faces.append(face)

    if len(vertices_local) == 0:
        empty_vertices = torch.empty((0, 3), device=object_voxels.device, dtype=bbox_center.dtype)
        empty_faces = torch.empty((0, 4), device=object_voxels.device, dtype=torch.long)
        return empty_vertices, empty_faces

    vertices_grid = torch.tensor(vertices_local, device=object_voxels.device, dtype=bbox_center.dtype)
    vertices_normalized = torch.clamp(vertices_grid * voxel_size, min=0.0, max=1.0)
    vertices_local = vertices_normalized * extent + bbox_min
    vertices_world = vertices_local @ bbox_axes.T + bbox_center
    faces = torch.tensor(faces, device=object_voxels.device, dtype=torch.long)
    return vertices_world, faces


def save_quad_mesh_as_ply(path, vertices, faces):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    vertices_cpu = vertices.detach().cpu().tolist()
    faces_cpu = faces.detach().cpu().tolist()

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices_cpu)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces_cpu)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex in vertices_cpu:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces_cpu:
            f.write(f"4 {face[0]} {face[1]} {face[2]} {face[3]}\n")


def get_object_shape_region(
    means,
    object_mask,
    bbox_center,
    bbox_axes,
    bbox_min,
    bbox_max,
    voxel_size=0.025,
    dilation_voxels=10,
    fill_grid_holes=True,
    fill_mode: Literal["shell_flood", "axis_columns"] = "shell_flood",
):
    inside_bbox = points_inside_oriented_bbox(means, bbox_center, bbox_axes, bbox_min, bbox_max)
    local_points = (means - bbox_center) @ bbox_axes
    extent = torch.clamp(bbox_max - bbox_min, min=1e-6)
    normalized = (local_points - bbox_min) / extent

    grid_size = torch.ceil(torch.ones(3, device=means.device, dtype=means.dtype) / voxel_size).long() + 1
    grid_size = torch.clamp(grid_size, min=2)

    object_voxels = torch.floor(normalized[object_mask] / voxel_size).long()
    object_voxels = torch.clamp(object_voxels, min=0)
    object_voxels = torch.minimum(object_voxels, grid_size[None] - 1)
    object_voxels = torch.unique(object_voxels, dim=0)

    if dilation_voxels > 0:
        offsets_1d = torch.arange(-dilation_voxels, dilation_voxels + 1, device=means.device)
        offsets = torch.stack(torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"), dim=-1).reshape(-1, 3)
        object_voxels = object_voxels[:, None, :] + offsets[None, :, :]
        object_voxels = object_voxels.reshape(-1, 3)
        object_voxels = torch.clamp(object_voxels, min=0)
        object_voxels = torch.minimum(object_voxels, grid_size[None] - 1)
        object_voxels = torch.unique(object_voxels, dim=0)

    voxel_count_before_fill = object_voxels.shape[0]
    if fill_grid_holes:
        if fill_mode == "shell_flood":
            object_voxels = fill_voxel_shell_interior(object_voxels, grid_size)
        elif fill_mode == "axis_columns":
            object_voxels = fill_voxel_holes(object_voxels, grid_size)
        else:
            raise ValueError("fill_mode must be one of: shell_flood, axis_columns")

    multipliers = torch.tensor(
        [grid_size[1] * grid_size[2], grid_size[2], 1],
        device=means.device,
        dtype=torch.long,
    )
    object_keys = torch.sum(object_voxels * multipliers[None], dim=1)

    all_voxels = torch.floor(normalized / voxel_size).long()
    all_voxels = torch.clamp(all_voxels, min=0)
    all_voxels = torch.minimum(all_voxels, grid_size[None] - 1)
    all_keys = torch.sum(all_voxels * multipliers[None], dim=1)

    shape_region = inside_bbox & torch.isin(all_keys, object_keys)
    grid_centers_local = (object_voxels.to(dtype=means.dtype) + 0.5) * voxel_size * extent + bbox_min
    grid_centers_world = grid_centers_local @ bbox_axes.T + bbox_center
    mesh_vertices_world, mesh_faces = build_voxel_surface_mesh(
        object_voxels,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        voxel_size,
    )
    print(f"Object shape occupied voxels after dilation: {voxel_count_before_fill}")
    if fill_grid_holes:
        print(f"Object shape occupied voxels after {fill_mode} fill: {object_keys.numel()}")
    print(
        "Object shape surface mesh: "
        f"{mesh_vertices_world.shape[0]} vertices, {mesh_faces.shape[0]} quad faces"
    )
    return shape_region, inside_bbox, grid_centers_world, mesh_vertices_world, mesh_faces



def get_color_similar_candidates(
    features_dc,
    object_mask,
    candidate_mask,
    color_similarity_threshold,
    color_reference: Literal["mean", "median"] = "mean",
):
    if int(candidate_mask.sum().item()) == 0:
        return torch.zeros_like(object_mask)

    colors = sh_dc_to_rgb(features_dc)
    object_colors = colors[object_mask]

    if color_reference == "mean":
        reference_color = object_colors.mean(dim=0, keepdim=True)
    elif color_reference == "median":
        reference_color = torch.median(object_colors, dim=0).values[None]
    else:
        raise ValueError("color_reference must be one of: mean, median")

    candidate_indices = torch.where(candidate_mask)[0]
    color_distances = torch.linalg.norm(colors[candidate_indices] - reference_color, dim=1)
    similar_local = color_distances <= color_similarity_threshold

    similar_mask = torch.zeros_like(object_mask)
    similar_mask[candidate_indices[similar_local]] = True

    stats = torch.quantile(
        color_distances,
        torch.tensor([0.0, 0.5, 0.9, 1.0], device=color_distances.device, dtype=color_distances.dtype),
    )
    print(
        "Object-shape candidate color distance min/median/p90/max: "
        f"{stats[0].item():.6f}, {stats[1].item():.6f}, "
        f"{stats[2].item():.6f}, {stats[3].item():.6f}"
    )
    return similar_mask


def draw_projected_points(
    img,
    points_3d,
    viewmat,
    K,
    color=(0, 255, 255),
    point_radius=2,
    max_points=4000,
):
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


def draw_projected_mesh_edges(
    img,
    vertices_3d,
    faces,
    viewmat,
    K,
    color=(255, 0, 255),
    line_thickness=1,
    max_faces=8000,
):
    if vertices_3d.numel() == 0 or faces.numel() == 0:
        return img

    if faces.shape[0] > max_faces:
        sample_ids = torch.randperm(faces.shape[0], device=faces.device)[:max_faces]
        faces = faces[sample_ids]

    vertices_3d_h = torch.cat(
        [vertices_3d, torch.ones(vertices_3d.shape[0], 1, device=vertices_3d.device)],
        dim=1,
    )
    vertices_cam_h = (viewmat @ vertices_3d_h.T).T
    vertices_cam = vertices_cam_h[:, :3] / vertices_cam_h[:, 3:]
    vertices_2d_h = (K @ vertices_cam.T).T
    vertices_2d = vertices_2d_h[:, :2] / vertices_2d_h[:, 2:]

    points_np = vertices_2d.detach().cpu().numpy().astype(int)
    depths = vertices_cam_h[:, 2].detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    height, width = img.shape[:2]
    edges = set()
    for face in faces_np:
        for edge_id in range(4):
            a = int(face[edge_id])
            b = int(face[(edge_id + 1) % 4])
            if depths[a] <= 0.1 or depths[b] <= 0.1:
                continue

            ax, ay = int(points_np[a][0]), int(points_np[a][1])
            bx, by = int(points_np[b][0]), int(points_np[b][1])
            if (
                (ax < 0 and bx < 0)
                or (ax >= width and bx >= width)
                or (ay < 0 and by < 0)
                or (ay >= height and by >= height)
            ):
                continue

            edge = (min(a, b), max(a, b))
            if edge in edges:
                continue
            edges.add(edge)
            cv2.line(img, (ax, ay), (bx, by), color, line_thickness)

    return img


def render_splats_with_grid(
    output_dir,
    splats,
    grid_points,
    added_points=None,
    mesh_vertices=None,
    mesh_faces=None,
    render_mesh=True,
    point_radius=2,
    max_points=4000,
    max_mesh_faces=8000,
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
        img = draw_projected_points(
            img,
            grid_points,
            viewmat,
            K,
            color=(0, 255, 255),
            point_radius=point_radius,
            max_points=max_points,
        )
        if render_mesh and mesh_vertices is not None and mesh_faces is not None:
            img = draw_projected_mesh_edges(
                img,
                mesh_vertices,
                mesh_faces,
                viewmat,
                K,
                color=(255, 0, 255),
                line_thickness=max(1, point_radius),
                max_faces=max_mesh_faces,
            )
        if added_points is not None and added_points.numel() > 0:
            img = draw_projected_points(
                img,
                added_points,
                viewmat,
                K,
                color=(0, 0, 255),
                point_radius=point_radius + 1,
                max_points=max_points,
            )
        cv2.imwrite(os.path.join(output_dir, image.name), img)


def keep_largest_voxel_cluster(
    means,
    object_mask,
    bbox_center,
    bbox_axes,
    bbox_min,
    bbox_max,
    voxel_size=0.025,
    connectivity=26,
    min_cluster_keep_ratio=0.02,
):
    inside_bbox = points_inside_oriented_bbox(means, bbox_center, bbox_axes, bbox_min, bbox_max)
    object_indices = torch.where(object_mask & inside_bbox)[0]
    if object_indices.numel() == 0:
        raise ValueError("No object Gaussians inside bbox after scattered-cluster filtering.")

    local_points = (means[object_indices] - bbox_center) @ bbox_axes
    extent = torch.clamp(bbox_max - bbox_min, min=1e-6)
    normalized = (local_points - bbox_min) / extent

    grid_size = torch.ceil(torch.ones(3, device=means.device, dtype=means.dtype) / voxel_size).long() + 1
    grid_size = torch.clamp(grid_size, min=2)

    voxels = torch.floor(normalized / voxel_size).long()
    voxels = torch.clamp(voxels, min=0)
    voxels = torch.minimum(voxels, grid_size[None] - 1)
    unique_voxels, inverse = torch.unique(voxels, dim=0, return_inverse=True)

    voxel_list = [tuple(v) for v in unique_voxels.detach().cpu().tolist()]
    voxel_set = set(voxel_list)

    if connectivity == 6:
        offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
    elif connectivity == 26:
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    else:
        raise ValueError("cluster_connectivity must be 6 or 26")

    voxel_to_component = {}
    component_count = 0
    for voxel in voxel_list:
        if voxel in voxel_to_component:
            continue

        stack = [voxel]
        voxel_to_component[voxel] = component_count
        while stack:
            current = stack.pop()
            for offset in offsets:
                neighbor = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if neighbor in voxel_set and neighbor not in voxel_to_component:
                    voxel_to_component[neighbor] = component_count
                    stack.append(neighbor)

        component_count += 1

    component_ids = torch.tensor(
        [voxel_to_component[voxel] for voxel in voxel_list],
        device=means.device,
        dtype=torch.long,
    )
    gaussian_component_ids = component_ids[inverse]
    component_sizes = torch.bincount(gaussian_component_ids, minlength=component_count)
    largest_component_size = torch.max(component_sizes)
    keep_components = component_sizes >= (largest_component_size * min_cluster_keep_ratio)

    cleaned_mask = torch.zeros_like(object_mask)
    kept_indices = object_indices[keep_components[gaussian_component_ids]]
    cleaned_mask[kept_indices] = True
    scattered_mask = object_mask & ~cleaned_mask

    print(f"Object voxel clusters found: {component_count}")
    print(f"Kept object cluster Gaussians: {int(cleaned_mask.sum().item())}")
    print(f"Scattered object Gaussians excluded from shape seed only: {int(scattered_mask.sum().item())}")
    return cleaned_mask, scattered_mask


def main(
    data_dir: str = "./data/human",
    checkpoint: str = "./data/human/chkpnt30000.pth",
    mask_path: str = "./results/human/mask3d.pth",
    results_dir: str = "./results/human/object_shape_bbox",
    rasterizer: Literal["auto", "original", "gsplat"] = "auto",
    data_factor: int = 1,
    bbox_padding_x: float = 0.03,
    bbox_padding_y: float = 0.03,
    bbox_padding_z: float = 0.03,
    bbox_lower_quantile: float = 0.0,
    bbox_upper_quantile: float = 1.0,
    voxel_size: float = 0.025,
    dilation_voxels: int = 1,
    fill_grid_holes: bool = True,
    fill_mode: Literal["shell_flood", "axis_columns"] = "shell_flood",
    remove_scattered_gaussians: bool = True,
    cluster_voxel_size: float = 0.025,
    cluster_connectivity: int = 26,
    min_cluster_keep_ratio: float = 0.02,
    add_all_gaussians_in_shape: bool = True,
    color_similarity_threshold: float = 0.12,
    color_reference: Literal["mean", "median"] = "mean",
    updated_mask_path: str = "",
    backup_original_mask: bool = True,
    shape_images_dir: str = "",
    deleted_images_dir: str = "",
    extracted_images_dir: str = "",
    overlay_point_radius: int = 2,
    max_overlay_points: int = 4000,
    render_shape_mesh: bool = True,
    shape_mesh_path: str = "",
    max_mesh_overlay_faces: int = 8000,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    torch.set_default_device("cuda")
    os.makedirs(results_dir, exist_ok=True)

    if updated_mask_path == "":
        updated_mask_path = mask_path
    if shape_images_dir == "":
        shape_images_dir = os.path.join(results_dir, "object_shape_images")
    if deleted_images_dir == "":
        deleted_images_dir = os.path.join(results_dir, "deleted_images")
    if extracted_images_dir == "":
        extracted_images_dir = os.path.join(results_dir, "extracted_images")
    if shape_mesh_path == "":
        shape_mesh_path = os.path.join(results_dir, "object_shape_surface.ply")

    if rasterizer == "auto":
        rasterizer = detect_rasterizer(checkpoint)
        print(f"Detected checkpoint rasterizer format: {rasterizer}")

    splats = load_checkpoint(checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor)
    means = splats["means"]
    mask = load_aligned_mask(mask_path, means.shape[0], means.device)

    if int(mask.sum().item()) == 0:
        raise ValueError("mask3d.pth is empty; cannot create object-shaped bbox.")

    original_mask = mask.clone()
    original_mask_count = int(mask.sum().item())

    bbox_corners, bbox_center, bbox_axes, bbox_min, bbox_max = get_oriented_bbox(
        means[mask],
        lower_quantile=bbox_lower_quantile,
        upper_quantile=bbox_upper_quantile,
        padding=[bbox_padding_x, bbox_padding_y, bbox_padding_z],
    )
    del bbox_corners

    shape_seed_mask = mask
    scattered_mask = torch.zeros_like(mask)
    if remove_scattered_gaussians:
        shape_seed_mask, scattered_mask = keep_largest_voxel_cluster(
            means,
            mask,
            bbox_center,
            bbox_axes,
            bbox_min,
            bbox_max,
            voxel_size=cluster_voxel_size,
            connectivity=cluster_connectivity,
            min_cluster_keep_ratio=min_cluster_keep_ratio,
        )

    (
        shape_region,
        inside_bbox,
        grid_centers_world,
        mesh_vertices_world,
        mesh_faces,
    ) = get_object_shape_region(
        means,
        shape_seed_mask,
        bbox_center,
        bbox_axes,
        bbox_min,
        bbox_max,
        voxel_size=voxel_size,
        dilation_voxels=dilation_voxels,
        fill_grid_holes=fill_grid_holes,
        fill_mode=fill_mode,
    )

    candidate_mask = inside_bbox & ~original_mask
    shape_candidate_mask = shape_region & ~original_mask
    if add_all_gaussians_in_shape:
        added_mask = candidate_mask
        print("Adding all non-mask Gaussians inside the object bbox to mask3d.")
    else:
        added_mask = get_color_similar_candidates(
            splats["features_dc"],
            original_mask,
            candidate_mask,
            color_similarity_threshold=color_similarity_threshold,
            color_reference=color_reference,
        )

    updated_mask = original_mask | added_mask
    deleted_mask = ~updated_mask
    added_points = means[added_mask]

    print(f"Original mask Gaussians: {original_mask_count}")
    print(f"Shape seed Gaussians: {int(shape_seed_mask.sum().item())}")
    print(f"Scattered Gaussians kept as object: {int(scattered_mask.sum().item())}")
    print(f"Gaussians inside coarse bbox: {int(inside_bbox.sum().item())}")
    print(f"Gaussians inside object-shaped region: {int(shape_region.sum().item())}")
    print(f"Object-shaped candidate Gaussians: {int(shape_candidate_mask.sum().item())}")
    print(f"Bbox candidate Gaussians: {int(candidate_mask.sum().item())}")
    print(f"Gaussians added to mask: {int(added_mask.sum().item())}")
    print(f"Updated mask Gaussians: {int(updated_mask.sum().item())}")
    print(f"Deleted/environment Gaussians: {int(deleted_mask.sum().item())}")
    print(f"Saving object-shape multi-face surface mesh to {shape_mesh_path}")
    save_quad_mesh_as_ply(shape_mesh_path, mesh_vertices_world, mesh_faces)

    if backup_original_mask and os.path.abspath(updated_mask_path) == os.path.abspath(mask_path):
        backup_path = mask_path + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(mask_path, backup_path)
            print(f"Backed up original mask to {backup_path}")

    torch.save(updated_mask.detach().cpu(), updated_mask_path)
    print(f"Saved updated mask to {updated_mask_path}")

    print(f"Rendering object-shape grid images to {shape_images_dir}")
    render_splats_with_grid(
        shape_images_dir,
        splats,
        grid_centers_world,
        added_points=added_points,
        mesh_vertices=mesh_vertices_world,
        mesh_faces=mesh_faces,
        render_mesh=render_shape_mesh,
        point_radius=overlay_point_radius,
        max_points=max_overlay_points,
        max_mesh_faces=max_mesh_overlay_faces,
    )

    print(f"Rendering extracted images to {extracted_images_dir}")
    extracted_splats = filter_splats_by_mask(splats, updated_mask)
    render_splats_to_dir(extracted_images_dir, extracted_splats)

    print(f"Rendering deleted images to {deleted_images_dir}")
    deleted_splats = filter_splats_by_mask(splats, deleted_mask)
    render_splats_to_dir(deleted_images_dir, deleted_splats)


if __name__ == "__main__":
    tyro.cli(main)
