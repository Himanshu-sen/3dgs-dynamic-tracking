# Installation & Usage Guide

This guide covers installing the environment, preparing a dataset, and running
the end-to-end pipeline.

- [1. Prerequisites](#1-prerequisites)
- [2. Installation](#2-installation)
- [3. What gets installed](#3-what-gets-installed)
- [4. Dataset preparation](#4-dataset-preparation)
- [5. Running the pipeline](#5-running-the-pipeline)
- [6. Pipeline parameters](#6-pipeline-parameters)
- [7. Outputs](#7-outputs)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **OS** | Linux (tested on Ubuntu) |
| **GPU** | NVIDIA GPU with CUDA — **required** (the rasterizer, SAM 2 and the matcher are GPU-only) |
| **NVIDIA driver** | Recent enough for CUDA 12.x |
| **CUDA toolkit** | `nvcc` recommended so `gsplat` can compile its CUDA kernels |
| **Conda** | Miniconda or Anaconda |
| **git** | With submodule support |
| **Disk** | ≈ 8–10 GB for the Conda environment + model weights |
| **Internet** | Needed during installation to download packages and weights |

---

## 2. Installation

Clone the repository **with submodules** (the EfficientLoFTR matcher is a
submodule), then run the installer:

```bash
git clone --recurse-submodules https://github.com/Himanshu-sen/3dgs-dynamic-tracking.git
cd 3dgs-dynamic-tracking
bash installation.sh
```

`installation.sh` is **idempotent** (safe to re-run) and performs eight steps:

1. Pre-flight checks — `git`, `conda`, NVIDIA driver, `nvcc`.
2. Verifies/restores `draw_bbox.py` (needed by pipeline Steps 2 & 3).
3. Initializes the **EfficientLoFTR** submodule and applies an inference patch.
4. Creates a Conda environment (`3dgs`, Python 3.11).
5. Installs PyTorch 2.5.1 (CUDA build).
6. Installs all pipeline dependencies + exposes the SAM 2 config.
7. Downloads model weights (SAM 2, EfficientLoFTR, YOLO-World + CLIP).
8. Verifies the installation by importing every dependency.

It can be customized with environment variables:

```bash
ENV_NAME=myenv TORCH_CUDA=cu121 bash installation.sh
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENV_NAME` | `3dgs` | Conda environment name |
| `PYTHON_VERSION` | `3.11` | Python version |
| `TORCH_CUDA` | `cu124` | PyTorch CUDA build (`cu121`, `cu124`, …) |
| `TORCH_VERSION` | `2.5.1` | PyTorch version |
| `GSPLAT_VERSION` | `1.3.0` | gsplat version |

After it finishes:

```bash
conda activate 3dgs
```

> The pipeline scripts call `python` directly — **always `conda activate 3dgs`
> before running `main.sh`.**

---

## 3. What gets installed

**Python libraries** (into the Conda environment):

`torch` · `torchvision` · `gsplat` · `ultralytics` (YOLO-World) · `sam2`
(Segment Anything 2) · `pycolmap-scene-manager` · `opencv-python` · `numpy` ·
`scipy` · `imageio` · `tyro` · `plyfile` · `einops` · `kornia` · `yacs` ·
`loguru` · `ninja` · `gdown`.

**Model weights** (downloaded automatically):

| Weight | Location | Size |
|--------|----------|------|
| SAM 2 (Hiera-Large) | `checkpoints/sam2_hiera_large.pt` | ~900 MB |
| EfficientLoFTR (outdoor) | `EfficientLoFTR/weights/eloftr_outdoor.ckpt` | ~193 MB |
| YOLO-World v2 (small) | `yolov8s-worldv2.pt` | ~26 MB |
| CLIP text encoder | (ultralytics cache) | ~350 MB |

**Not** installed: your **scene data** — it is your own capture and must be
placed under `data/<dataset>/` (see next section).

---

## 4. Dataset preparation

📦 **Download the sample datasets:** [**Google Drive**](https://drive.google.com/file/d/1eN68qAjqv5BhF6cfiISJVnUkrYIBHeq-/view?usp=sharing)

From the command line (`gdown` is installed by `installation.sh`):

```bash
gdown 1eN68qAjqv5BhF6cfiISJVnUkrYIBHeq-
# then extract the archive so each scene ends up under data/<dataset>/
```

Each scene lives in its own folder under `data/`:

```
data/<dataset>/
├── sparse/0/            COLMAP reconstruction: cameras.bin, images.bin, points3D.bin
├── images/              the images used to train the 3DGS reconstruction
├── chkpnt30000.pth      3DGS checkpoint — INRIA format        ┐ one of
│   ckpt_*_rank0.pt      3DGS checkpoint — gsplat format       ┘ these
├── camera.jpg           a photo from the static camera viewpoint
└── frames/              ordered video frames of the moving object
```

Notes:

- **Checkpoint format is auto-detected.** Files named `ckpt_*_rank0.pt` are
  treated as gsplat checkpoints; `chkpnt*.pth` / `checkpoint*.pth` as INRIA
  checkpoints. The loaders also self-correct if the format is mismatched.
- **`camera.jpg` must be the same static viewpoint the `frames/` were shot
  from.** Step 4 aligns the virtual camera to this image; Step 5 then assumes
  the camera does not move. Using the first video frame as `camera.jpg` is the
  safest choice.
- The COLMAP reconstruction and the 3DGS checkpoint must correspond to the
  **same** set of images.

---

## 5. Running the pipeline

```bash
conda activate 3dgs
bash main.sh
```

`main.sh` first asks for a few shared settings:

| Prompt | Default | Meaning |
|--------|---------|---------|
| Dataset name | `toy` | Folder name under `data/` and `results/` |
| Data factor | `1` | Downscale factor for image resolution |
| Segmentation prompt | dataset name | Text prompt for YOLO-World detection (Step 1) |

It then **auto-detects** the checkpoint file and rasterizer, prints the resolved
paths, and runs the five steps. Before each step you are asked **`[Y/n/q]`** —
run it, skip it, or quit. This lets you re-run a single stage without repeating
the whole pipeline.

The five steps:

1. **Gradient-based 3D segmentation** → `results/<dataset>/mask3d.pth`
2. **Object shape & bounding box** → `results/<dataset>/object_shape_bbox/`
3. **Table inpainting under object** → `results/<dataset>/table_inpaint/`
4. **Camera pose optimization** → `results/<dataset>/optimized_camera_pose.pt`
5. **Dynamic object pose (PnP)** → `results/<dataset>/dynamic_pnp/`

Each script can also be run **standalone** with `--help` to see all options,
e.g. `python dynamic_object_pose_pnp.py --help`.

---

## 6. Pipeline parameters

The per-step parameters are defined as variables near the top of
[`main.sh`](main.sh) — edit them there. The most useful ones:

**Step 1 — segmentation**
- `DEMOF_VOTING_METHOD` — `gradient` | `binary` | `projection`
- `DEMOF_FOREGROUND_SCORE_THRESHOLD` — higher = stricter object selection
- `DEMOF_REMOVE_SCATTERED_COMPONENTS` — drop disconnected false positives

**Step 2 — shape**
- `SHAPE_VOXEL_SIZE` — voxel resolution for the shape grid
- `SHAPE_FILL_MODE` — `shell_flood` | `axis_columns`

**Step 3 — inpainting**
- `INPAINT_SUPPORT_AXIS` / `INPAINT_SUPPORT_SIDE` — which axis/side is the table
- `INPAINT_INPAINT_SHADOW` — also recolor the object's shadow

**Step 4 — camera**
- `CAM_TARGET_IMAGE` — the image to align the camera to
- `CAM_LR`, `CAM_ITERATIONS` — optimization learning rate / iterations

**Step 5 — tracking**
- `PNP_MATCHER` — `eloftr` (recommended) or `sift`
- `PNP_MAX_CORRESPONDENCES`, `PNP_RANSAC_ITERS`, `PNP_REPROJ_ERROR_PX`
- `PNP_OBJECT_ALPHA_THRESH` — alpha cutoff for the object mask render

---

## 7. Outputs

All results are written under `results/<dataset>/`:

```
results/<dataset>/
├── mask3d.pth                  3D object mask (boolean over Gaussians)
├── images/                     re-rendered training views
├── extracted_images/           object-only renders
├── deleted_images/             background-only renders
├── object_shape_bbox/          shape grid / mesh overlays
├── table_inpaint/              inpainted-background renders
├── optimized_camera_pose.pt    aligned camera pose
└── dynamic_pnp/                per-frame tracking results:
    ├── NNNN_target.png           the real frame
    ├── NNNN_ref.png              reference render
    ├── NNNN_updated.png          render after applying the estimated pose
    ├── NNNN_matches.png          keypoint correspondences
    ├── NNNN_pnp.json             estimated SE(3) motion
    └── timing_summary.json       per-stage timings
```

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'draw_bbox'`**
`draw_bbox.py` is missing. Restore it: `git checkout HEAD -- draw_bbox.py`
(the installer also does this automatically).

**`EfficientLoFTR checkpoint not found`**
`EfficientLoFTR/weights/eloftr_outdoor.ckpt` is missing — re-run
`installation.sh`, or download it manually into that folder.

**`Please download the checkpoint sam2_hiera_large.pt`**
`checkpoints/sam2_hiera_large.pt` is missing — re-run `installation.sh`.

**SAM 2 `MissingConfigException` / cannot find `sam2_hiera_l.yaml`**
The SAM 2 config was not exposed. Re-run `installation.sh` (Step 6b), which
copies the config to the SAM 2 package root.

**`mask3d length (...) does not match splats count (...)`**
`mask3d.pth` was generated for a different checkpoint. Re-run Step 1, or pass a
checkpoint with the same number of Gaussians.

**Tracking result is in the wrong pose**
Step 5 assumes a **static camera**. Make sure `camera.jpg` (Step 4's alignment
target) is the **same viewpoint** as the `frames/`. If they differ, the camera
parallax is mistaken for object motion. Using the first video frame as
`camera.jpg` avoids this.

**`CUDA is required for this script`**
No GPU is visible. Check `nvidia-smi` and that `torch.cuda.is_available()` is
`True` inside the `3dgs` environment.

**`gsplat` build errors**
Install a CUDA toolkit so `nvcc` is on `PATH`
(`conda install -n 3dgs -c nvidia cuda-toolkit`), then re-run the installer.
