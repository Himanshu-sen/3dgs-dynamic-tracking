# Pipeline — Technical Reference

A stage-by-stage description of the dynamic object-tracking pipeline. The five
stages are orchestrated by [`main.sh`](../main.sh); each is also a standalone,
`tyro`-driven script (`python <script>.py --help`).

**Core assumption:** the camera is *static* and the *object moves*. The scene is
represented as 3D Gaussians; tracking means re-posing the subset of Gaussians
that belong to the object.

```
3DGS checkpoint + COLMAP ──▶ [1] segment ──▶ [2] shape ──▶ [3] inpaint
                                   │
real video frames ──▶ [4] camera align ──▶ [5] PnP track ──▶ per-frame SE(3)
```

---

## Stage 1 — Gradient-based 3D segmentation

**Script:** [`demof.py`](../demof.py) · **Output:** `results/<dataset>/mask3d.pth`

1. **Load** the 3DGS checkpoint (INRIA or gsplat format, auto-detected) and the
   COLMAP reconstruction.
2. **Gradient pruning** (`prune_by_gradients`) — every training view is
   rasterized and per-Gaussian colour gradients are accumulated. Gaussians that
   never influence any pixel (zero gradient) are pruned. `test_proper_pruning`
   asserts the pruned scene is pixel-identical to the original.
3. **2D mask** — YOLO-World detects the target object from a **text prompt** on
   the first view; **SAM 2**'s video predictor propagates that box into a mask
   across every view. Each mask's SAM logits are split into *positive*,
   *background*, and *ambiguous* pixels by thresholding.
4. **3D voting** — the 2D masks are lifted to 3D. A masked rasterization loss is
   back-propagated to per-Gaussian colours; the gradient magnitude each Gaussian
   receives is its *vote*. Three voting modes are available:
   - `gradient` — accumulate the gradient norm (opacity × transmittance);
   - `binary` — count any non-zero gradient;
   - `projection` — count projected 2D-centre hits inside the mask.
5. A Gaussian is kept if its `foreground_score = positive / (positive +
   negative)` clears a threshold and positive votes beat background votes.
   `keep_largest_voxel_components` then drops spatially scattered false
   positives.
6. The boolean 3D mask is **expanded back to the pre-pruning Gaussian count**
   and saved as `mask3d.pth`; voting statistics go to `vote_stats.pth`.

---

## Stage 2 — Object shape & bounding box

**Script:** [`object_shape_bbox.py`](../object_shape_bbox.py)
**Output:** `results/<dataset>/object_shape_bbox/`, updated `mask3d.pth`

1. Compute an **oriented bounding box** of the masked Gaussians.
2. `keep_largest_voxel_cluster` removes scattered Gaussians from the shape seed.
3. `get_object_shape_region` voxelizes the object inside the OBB, **dilates** the
   occupied voxels, and **fills interior holes** (`shell_flood` flood-fill or
   `axis_columns`). A watertight surface mesh is exported as
   `object_shape_surface.ply`.
4. Gaussians lying inside the recovered shape (or, optionally, only those whose
   colour is similar to the object) are **added back** to the mask — recovering
   parts the gradient vote missed.
5. The updated mask overwrites `mask3d.pth` (the previous one is backed up to
   `mask3d.pth.backup`).

---

## Stage 3 — Background / table inpainting

**Script:** [`table_inpaint_under_object.py`](../table_inpaint_under_object.py)
**Output:** `results/<dataset>/table_inpaint/`

So the object can be removed without leaving a hole:

1. `estimate_table_plane_and_footprint` finds the **support surface** beneath the
   object (the table) along a chosen axis.
2. `get_table_inpaint_masks` separates the table Gaussians **directly under** the
   object from a **reference ring** of visible table around it.
3. `detect_table_shadow_gaussians` (optional) finds the object's cast shadow by
   comparing local luminance to the reference ring.
4. `inpaint_table_gaussians` recolours the under-object (and shadow) Gaussians by
   a **KNN-weighted blend** of the reference-ring colours.
5. Renders of the background with the object removed — before and after
   inpainting — are written for inspection.

---

## Stage 4 — Camera pose optimization

**Script:** [`camera_pose_optimization.py`](../camera_pose_optimization.py)
**Output:** `results/<dataset>/optimized_camera_pose.pt`

The virtual camera is aligned to the **real static camera**:

1. Load the target image (`camera.jpg`) and an initial pose (from
   `camera_pose.pt` if present, otherwise a COLMAP view).
2. Parameterize the pose as a fixed rotation × a **Rodrigues rotation delta**
   plus a **translation** vector; both are optimized with Adam.
3. Minimize the MSE between the rasterized render and the target image.
4. Save the best `viewmat` (and a comparison GIF / images).

> ⚠️ **The alignment image must match the viewpoint the tracking `frames/` were
> captured from.** Stage 5 assumes this pose is the fixed camera; a mismatch is
> later misread as object motion.

---

## Stage 5 — Dynamic object pose (PnP)

**Script:** [`dynamic_object_pose_pnp.py`](../dynamic_object_pose_pnp.py)
**Output:** `results/<dataset>/dynamic_pnp/`

For each frame of the video:

1. **Render reference** — rasterize the current scene from the fixed camera in
   `RGB+ED` mode to obtain an RGB image and an expected-depth map.
2. **Render object mask** — rasterize only the object Gaussians (`mask3d.pth`)
   and threshold the alpha channel.
3. **Match** — EfficientLoFTR produces semi-dense keypoint correspondences
   between the reference render and the target frame.
4. **Filter** — keep matches that fall on the object mask and have valid depth.
5. **Back-project** — lift the reference keypoints to 3D camera-space points
   using the rendered depth and intrinsics.
6. **Solve PnP** — `cv2.solvePnPRansac` (USAC-MAGSAC) estimates the object's
   SE(3) motion `T_delta_cam` in the camera frame.
7. **To world space** — convert with the known camera pose:
   `T_delta_world = T_wc · T_delta_cam · T_cw`.
8. **Apply** — transform the object Gaussians' means and quaternions. Updates are
   **cumulative**, so the object is tracked through the whole sequence.
9. **Render & log** — write the updated render, match visualization, and the
   estimated pose (`NNNN_pnp.json`).

### Outputs per frame

| File | Contents |
|------|----------|
| `NNNN_target.png` | the real video frame |
| `NNNN_ref.png` | reference render before the update |
| `NNNN_updated.png` | render after applying the estimated pose |
| `NNNN_matches.png` | keypoint correspondences |
| `NNNN_pnp.json` | `rvec`, `tvec`, `T_delta_cam`, `T_delta_world`, inlier count |
| `timing_summary.json` | per-stage timing breakdown |

---

## Checkpoint formats

| Format | Filename pattern | Structure |
|--------|------------------|-----------|
| INRIA / original 3DGS | `chkpnt*.pth`, `checkpoint*.pth` | tuple of parameter tensors |
| gsplat | `ckpt_*_rank0.pt` | dict with a `"splats"` key |
| PLY | `*.ply` | standard Gaussian-splat PLY (loaded by `utils.load_checkpoint`) |

The loaders **auto-detect** the real format from the file contents and correct a
mismatched `--rasterizer` / `--format` argument.
