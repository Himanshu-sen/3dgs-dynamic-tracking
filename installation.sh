#!/usr/bin/env bash
# =============================================================================
#  installation.sh  —  One-shot environment setup for the 3DGS dynamic-tracking
#                      pipeline (run once after `git clone`, then run main.sh).
#
#  What it does:
#    1. Pre-flight checks (git, conda, NVIDIA driver, nvcc).
#    2. Restores draw_bbox.py if missing (required by main.sh Steps 2 & 3).
#    3. Initialises the EfficientLoFTR git submodule + applies an inference patch.
#    4. Creates a dedicated conda env and installs every Python dependency.
#    5. Downloads all model weights:
#         - checkpoints/sam2_hiera_large.pt              (~900 MB)
#         - EfficientLoFTR/weights/eloftr_outdoor.ckpt   (~193 MB)
#         - yolov8s-worldv2.pt                           (~26 MB, via ultralytics)
#    6. Verifies the install by importing every pipeline dependency.
#
#  Usage:
#    bash installation.sh
#    conda activate 3dgs
#    bash main.sh
#
#  NOTE: This installs *code + model weights* only. The scene data
#        (data/<dataset>/ : COLMAP sparse/0, images, the 3DGS checkpoint,
#        camera.jpg, frames/) is your own capture and cannot be downloaded —
#        place it under data/<dataset>/ before running main.sh.
# =============================================================================

set -eo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration (override via environment variables if needed)
# ─────────────────────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-3dgs}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"          # cu124 matches the system nvcc (12.4)
GSPLAT_VERSION="${GSPLAT_VERSION:-1.3.0}"

# ─────────────────────────────────────────────────────────────────────────────
#  Pretty output
# ─────────────────────────────────────────────────────────────────────────────
BOLD="\033[1m"; CYAN="\033[1;36m"; GREEN="\033[1;32m"
YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"
step()  { echo -e "\n${CYAN}━━━ $* ━━━${RESET}"; }
ok()    { echo -e "${GREEN}✔ $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠ $*${RESET}"; }
die()   { echo -e "${RED}✘ $*${RESET}"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   3DGS Dynamic Tracking — Environment Installer               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  Repo        : ${SCRIPT_DIR}"
echo "  Conda env   : ${ENV_NAME}  (Python ${PYTHON_VERSION})"
echo "  PyTorch     : ${TORCH_VERSION} (${TORCH_CUDA}) + torchvision ${TORCHVISION_VERSION}"
echo "  gsplat      : ${GSPLAT_VERSION}"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
step "Step 1/8  Pre-flight checks"

command -v git  >/dev/null 2>&1 || die "git not found. Install git first."
command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda/Anaconda first."
[ -f "${SCRIPT_DIR}/main.sh" ] && [ -f "${SCRIPT_DIR}/demof.py" ] \
    || die "Run this script from the repository root (main.sh / demof.py not found)."

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
else
    warn "nvidia-smi not found — the pipeline REQUIRES an NVIDIA GPU with CUDA."
fi
if command -v nvcc >/dev/null 2>&1; then
    ok "nvcc found: $(nvcc --version | tail -1)"
else
    warn "nvcc (CUDA toolkit) not found. gsplat may fail to compile its CUDA kernels."
    warn "Install it with:  conda install -n ${ENV_NAME} -c nvidia cuda-toolkit"
fi
ok "Pre-flight checks passed."

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Restore draw_bbox.py  (imported by Steps 2 & 3 of main.sh)
# ─────────────────────────────────────────────────────────────────────────────
step "Step 2/8  Verifying draw_bbox.py"

if [ -f "${SCRIPT_DIR}/draw_bbox.py" ]; then
    ok "draw_bbox.py present."
else
    warn "draw_bbox.py missing — object_shape_bbox.py / table_inpaint_under_object.py need it."
    if git -C "${SCRIPT_DIR}" cat-file -e HEAD:draw_bbox.py 2>/dev/null; then
        git -C "${SCRIPT_DIR}" checkout HEAD -- draw_bbox.py
        ok "Restored draw_bbox.py from git history."
    else
        die "draw_bbox.py is not in the repository or git history. main.sh Steps 2 & 3 cannot run.
   Restore it from a backup, or unstage its deletion before pushing to GitHub:
       git restore --staged draw_bbox.py && git checkout -- draw_bbox.py"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — EfficientLoFTR submodule + inference patch
# ─────────────────────────────────────────────────────────────────────────────
step "Step 3/8  EfficientLoFTR submodule"

git -C "${SCRIPT_DIR}" submodule update --init --recursive
[ -d "${SCRIPT_DIR}/EfficientLoFTR/src/loftr" ] \
    || die "EfficientLoFTR submodule not initialised correctly."
ok "EfficientLoFTR submodule ready."

# Patch src/utils/misc.py so the matcher imports without pytorch-lightning installed.
python - <<'PYEOF'
import sys
p = "EfficientLoFTR/src/utils/misc.py"
try:
    s = open(p).read()
except FileNotFoundError:
    print("  misc.py not found — skipping patch."); sys.exit(0)
if "_RankZeroOnlyFallback" in s:
    print("  misc.py already patched."); sys.exit(0)
old = "from pytorch_lightning.utilities import rank_zero_only"
new = ("try:\n"
       "    from pytorch_lightning.utilities import rank_zero_only  # type: ignore\n"
       "except Exception:  # inference without pytorch-lightning\n"
       "    class _RankZeroOnlyFallback:\n"
       "        rank = 0\n"
       "    rank_zero_only = _RankZeroOnlyFallback()")
if old not in s:
    print("  WARNING: expected import line not found — skipping patch."); sys.exit(0)
open(p, "w").write(s.replace(old, new, 1))
print("  Patched misc.py (pytorch-lightning import is now optional).")
PYEOF
ok "EfficientLoFTR inference patch applied."

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Create conda environment
# ─────────────────────────────────────────────────────────────────────────────
step "Step 4/8  Conda environment '${ENV_NAME}'"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    ok "Conda env '${ENV_NAME}' already exists — reusing it."
else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    ok "Created conda env '${ENV_NAME}'."
fi

conda activate "${ENV_NAME}"
ok "Activated env: $(python --version 2>&1)  ($(which python))"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Install PyTorch (CUDA build)
# ─────────────────────────────────────────────────────────────────────────────
step "Step 5/8  Installing PyTorch ${TORCH_VERSION} (${TORCH_CUDA})"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

python - <<'PYEOF'
import torch
print(f"  torch {torch.__version__} | CUDA build {torch.version.cuda} | "
      f"GPU available: {torch.cuda.is_available()}")
PYEOF
ok "PyTorch installed."

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Install pipeline Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "Step 6/8  Installing pipeline dependencies"

# Build tooling (ninja speeds up gsplat's CUDA kernel compilation).
python -m pip install ninja

# Core pipeline packages (demof / utils / camera / pnp / shape / inpaint).
python -m pip install \
    "gsplat==${GSPLAT_VERSION}" \
    ultralytics \
    tyro \
    imageio \
    opencv-python \
    numpy \
    scipy \
    plyfile \
    typing_extensions \
    pycolmap-scene-manager \
    gdown

# Segment Anything 2 — used by demof.py for 2D mask propagation.
python -m pip install "git+https://github.com/facebookresearch/sam2.git"

# EfficientLoFTR inference dependencies (matcher used by dynamic_object_pose_pnp.py).
python -m pip install einops kornia yacs loguru

ok "Python dependencies installed."

# Expose the SAM2 'sam2_hiera_l.yaml' config under the bare name demof.py expects.
step "Step 6b   Exposing SAM2 config (sam2_hiera_l.yaml)"
python - <<'PYEOF'
import os, sam2, shutil, glob
pkg = os.path.dirname(sam2.__file__)
target = os.path.join(pkg, "sam2_hiera_l.yaml")
if os.path.isfile(target):
    print("  sam2_hiera_l.yaml already discoverable.")
else:
    found = glob.glob(os.path.join(pkg, "**", "sam2_hiera_l.yaml"), recursive=True)
    if not found:
        print("  WARNING: sam2_hiera_l.yaml not found inside the sam2 package.")
    else:
        shutil.copyfile(found[0], target)
        print(f"  Copied {os.path.relpath(found[0], pkg)} -> sam2_hiera_l.yaml (package root).")
PYEOF
ok "SAM2 config ready."

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Download model weights
# ─────────────────────────────────────────────────────────────────────────────
step "Step 7/8  Downloading model weights"

# fetch <url> <output> [min_bytes]
fetch() {
    local url="$1" out="$2" min="${3:-100000}"
    if [ -f "${out}" ] && [ "$(stat -c%s "${out}" 2>/dev/null || echo 0)" -ge "${min}" ]; then
        ok "$(basename "${out}") already present ($(du -h "${out}" | cut -f1)) — skipping."
        return 0
    fi
    mkdir -p "$(dirname "${out}")"
    echo "  Downloading $(basename "${out}") ..."
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${out}" "${url}" || return 1
    else
        curl -L -C - -o "${out}" "${url}" || return 1
    fi
    [ "$(stat -c%s "${out}" 2>/dev/null || echo 0)" -ge "${min}" ] || return 1
}

# --- 7a. SAM 2 checkpoint (Meta, ~900 MB) ------------------------------------
SAM2_CKPT="${SCRIPT_DIR}/checkpoints/sam2_hiera_large.pt"
if fetch "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" \
         "${SAM2_CKPT}" 800000000; then
    ok "SAM2 checkpoint ready: checkpoints/sam2_hiera_large.pt"
else
    die "Failed to download sam2_hiera_large.pt.
   Download it manually to: ${SAM2_CKPT}
   URL: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
fi

# --- 7b. EfficientLoFTR weights (~193 MB) ------------------------------------
ELOFTR_CKPT="${SCRIPT_DIR}/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
if fetch "https://huggingface.co/Realcat/imcui_checkpoints/resolve/main/eloftr/eloftr_outdoor.ckpt?download=true" \
         "${ELOFTR_CKPT}" 150000000; then
    ok "EfficientLoFTR weights ready: EfficientLoFTR/weights/eloftr_outdoor.ckpt"
else
    warn "HuggingFace mirror failed — trying the official Google Drive folder via gdown ..."
    mkdir -p "$(dirname "${ELOFTR_CKPT}")"
    if gdown --folder "https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q" \
             -O "${SCRIPT_DIR}/EfficientLoFTR/weights" 2>/dev/null \
       && [ -f "${ELOFTR_CKPT}" ]; then
        ok "EfficientLoFTR weights downloaded via Google Drive."
    else
        FOUND="$(find "${SCRIPT_DIR}/EfficientLoFTR/weights" -name 'eloftr_outdoor.ckpt' 2>/dev/null | head -1)"
        if [ -n "${FOUND}" ] && [ "${FOUND}" != "${ELOFTR_CKPT}" ]; then
            mv "${FOUND}" "${ELOFTR_CKPT}"; ok "EfficientLoFTR weights located."
        else
            die "Failed to download eloftr_outdoor.ckpt.
   Download it manually to: ${ELOFTR_CKPT}
   From: https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q"
        fi
    fi
fi

# --- 7c. YOLO-World weights + CLIP text encoder ------------------------------
# YOLOWorld("yolov8s-worldv2.pt")  -> downloads the detector (~26 MB).
# .set_classes([...])              -> makes ultralytics install the CLIP package
#                                     and download the CLIP text-encoder weights,
#                                     which demof.py needs for text prompts.
echo "  Fetching yolov8s-worldv2.pt + CLIP text encoder via ultralytics ..."
if ( cd "${SCRIPT_DIR}" && python -c "
from ultralytics import YOLOWorld
m = YOLOWorld('yolov8s-worldv2.pt')
m.set_classes(['object'])
print('  YOLO-World + CLIP ready.')
" ); then
    ok "YOLO-World detector and CLIP text encoder ready."
else
    warn "Could not pre-fetch YOLO-World/CLIP — ultralytics will download them"
    warn "automatically on the first main.sh run (requires internet at that time)."
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — Verify the installation
# ─────────────────────────────────────────────────────────────────────────────
step "Step 8/8  Verifying installation"

cd "${SCRIPT_DIR}"
VERIFY_OK=1
python - <<'PYEOF' || VERIFY_OK=0
import importlib, os, sys

print("  Importing pipeline dependencies ...")
mods = ["torch", "torchvision", "cv2", "numpy", "scipy", "imageio", "tyro",
        "gsplat", "ultralytics", "plyfile", "pycolmap_scene_manager",
        "sam2.build_sam", "einops", "kornia", "yacs", "loguru"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"    ok  {m}")
    except Exception as e:
        print(f"    FAIL {m}: {e}")
        bad.append(m)

import torch
print(f"  CUDA available to torch: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("    WARNING: torch cannot see a CUDA GPU — the pipeline needs one.")

# EfficientLoFTR matcher module must be importable from the repo's wrapper.
sys.path.insert(0, os.path.join(os.getcwd(), "EfficientLoFTR"))
try:
    importlib.import_module("src.loftr")
    print("    ok  EfficientLoFTR (src.loftr)")
except Exception as e:
    print(f"    FAIL EfficientLoFTR (src.loftr): {e}")
    bad.append("src.loftr")

# Required weight files.
for f, n in [("checkpoints/sam2_hiera_large.pt", 8e8),
             ("EfficientLoFTR/weights/eloftr_outdoor.ckpt", 1.5e8)]:
    sz = os.path.getsize(f) if os.path.exists(f) else 0
    print(f"    {'ok ' if sz >= n else 'FAIL'} {f} ({sz/1e6:.0f} MB)")
    if sz < n:
        bad.append(f)

if bad:
    print("\n  Problems detected with: " + ", ".join(bad))
    sys.exit(1)
print("\n  All pipeline dependencies and weights verified.")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [ "${VERIFY_OK}" -eq 1 ]; then
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${GREEN}║   Installation complete.                                      ║${RESET}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "  Next steps:"
    echo -e "    ${BOLD}conda activate ${ENV_NAME}${RESET}"
    echo -e "    ${BOLD}bash main.sh${RESET}"
    echo ""
    echo -e "  ${YELLOW}Before running main.sh, place your scene data under data/<dataset>/:${RESET}"
    echo -e "    data/<dataset>/sparse/0/           COLMAP reconstruction"
    echo -e "    data/<dataset>/images/             training images"
    echo -e "    data/<dataset>/chkpnt30000.pth  OR  ckpt_*_rank0.pt   3DGS checkpoint"
    echo -e "    data/<dataset>/camera.jpg          camera-alignment image (Step 4)"
    echo -e "    data/<dataset>/frames/             target frames for tracking (Step 5)"
else
    echo -e "${BOLD}${RED}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${RED}║   Installation finished WITH ERRORS — see the log above.      ║${RESET}"
    echo -e "${BOLD}${RED}╚══════════════════════════════════════════════════════════════╝${RESET}"
    exit 1
fi
