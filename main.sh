#!/usr/bin/env bash
# =============================================================================
#  main.sh  —  End-to-end 3DGS pipeline
#
#  Pipeline order:
#    Step 1 │ demof.py                      – gradient-based 3D segmentation
#    Step 2 │ object_shape_bbox.py          – object shape / bounding-box extraction
#    Step 3 │ table_inpaint_under_object.py – inpaint table surface under object
#    Step 4 │ camera_pose_optimization.py   – camera pose optimisation
#    Step 5 │ dynamic_object_pose_pnp.py    – dynamic object pose via PnP + EfficientLoFTR
#
#  Usage:
#    bash main.sh
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
BOLD="\033[1m"
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
DIM="\033[2m"
RESET="\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_DATASET="toy"
_DEFAULT_DATA_FACTOR=1

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: prompt for a value (stderr label, stdout value for $() capture)
# ─────────────────────────────────────────────────────────────────────────────
ask_config() {
    local label="$1"
    local default="$2"
    local result
    echo -en "  ${BOLD}${label}${RESET} ${DIM}[${default}]${RESET}: " >&2
    read -r result </dev/tty
    echo "${result:-${default}}"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: convert "True"/"False" variable to tyro --flag / --no-flag
#  Usage: bool_flag "True" "flag-name"  →  outputs "--flag-name"
#         bool_flag "False" "flag-name" →  outputs "--no-flag-name"
# ─────────────────────────────────────────────────────────────────────────────
bool_flag() {
    local val="${1,,}"   # lowercase
    local flag="$2"
    if [[ "${val}" == "true" ]]; then
        echo "--${flag}"
    else
        echo "--no-${flag}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: ask user to run / skip a step.  Returns 0=run, 1=skip
# ─────────────────────────────────────────────────────────────────────────────
ask_step() {
    local step_num="$1"
    local step_name="$2"
    local script="$3"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  Step ${step_num}: ${step_name}${RESET}"
    echo -e "${DIM}  Script : ${script}${RESET}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    while true; do
        echo -en "  ${BOLD}Run this step? [Y/n/q(quit)]:${RESET} "
        read -r ans </dev/tty
        case "${ans,,}" in
            "" | y | yes) return 0 ;;
            n | no)       return 1 ;;
            q | quit)
                echo -e "${YELLOW}Pipeline aborted by user.${RESET}"
                exit 0
                ;;
            *)
                echo -e "  ${RED}Please enter Y (run), N (skip), or Q (quit).${RESET}"
                ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run a step from an args array, print command, capture exit code
# ─────────────────────────────────────────────────────────────────────────────
run_step() {
    local script="$1"; shift
    local -n _args="$1"   # nameref to caller's array

    echo -e "\n${GREEN}▶ Running: python ${script} ${_args[*]}${RESET}\n"

    exit_code=0
    python "${script}" "${_args[@]}" || exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
        echo -e "${GREEN}✔  Step completed successfully.${RESET}"
    else
        echo -e "${RED}✘  Step FAILED (exit code ${exit_code}). Check output above.${RESET}"
    fi
}

print_skipped() { echo -e "${YELLOW}⏭  Skipped.${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
#  Banner + collect shared config from user
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║   3DGS Gradient Segmentation — End-to-End Pipeline          ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${DIM}Press Enter to accept the default value shown in [brackets].${RESET}"
echo ""

DATASET=$(ask_config    "Dataset name (e.g. toy, bottle, new)" "${_DEFAULT_DATASET}")
DATA_FACTOR=$(ask_config "Data factor (int)"                  "${_DEFAULT_DATA_FACTOR}")
DEMOF_PROMPT=$(ask_config "Segmentation prompt"               "${DATASET}")

# ── Derive fixed paths from dataset name ─────────────────────────────────────
DATA_DIR="./data/${DATASET}"
RESULTS_DIR="./results/${DATASET}"

# ── Auto-detect checkpoint file and rasterizer ───────────────────────────────
# gsplat checkpoints are named ckpt_*_rank0.pt; INRIA/original ones are chkpnt*.pth / checkpoint*.pth
if   ls "${DATA_DIR}"/ckpt_*_rank0.pt 2>/dev/null | head -1 | grep -q .; then
    CHECKPOINT=$(ls "${DATA_DIR}"/ckpt_*_rank0.pt | tail -1)
    RASTERIZER="gsplat"
elif [ -f "${DATA_DIR}/chkpnt30000.pth" ]; then
    CHECKPOINT="${DATA_DIR}/chkpnt30000.pth"
    RASTERIZER="inria"
elif ls "${DATA_DIR}"/checkpoint*.pth 2>/dev/null | head -1 | grep -q .; then
    CHECKPOINT=$(ls "${DATA_DIR}"/checkpoint*.pth | tail -1)
    RASTERIZER="inria"
else
    echo -e "${RED}ERROR: No checkpoint found in ${DATA_DIR}${RESET}"
    exit 1
fi

echo ""
echo -e "  ${BOLD}Running with:${RESET}"
echo -e "  ${BOLD}DATASET    :${RESET} ${DATASET}"
echo -e "  ${BOLD}DATA_DIR   :${RESET} ${DATA_DIR}"
echo -e "  ${BOLD}CHECKPOINT :${RESET} ${CHECKPOINT}"
echo -e "  ${BOLD}RESULTS    :${RESET} ${RESULTS_DIR}"
echo -e "  ${BOLD}RASTERIZER :${RESET} ${RASTERIZER}  ${DIM}(auto-detected)${RESET}"
echo -e "  ${BOLD}DATA_FACTOR:${RESET} ${DATA_FACTOR}"
echo -e "  ${BOLD}PROMPT     :${RESET} ${DEMOF_PROMPT}"
echo ""
echo -e "  ${DIM}You will be asked before each step — answer Y/N/Q.${RESET}"

# ─────────────────────────────────────────────────────────────────────────────
#  Per-step parameters (tune as needed)
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: demof.py ──────────────────────────────────────────────────────────
DEMOF_SHOW_VISUAL_FEEDBACK="True"
DEMOF_MASK_INTERVAL=1
DEMOF_VOTING_METHOD="gradient"        # gradient | binary | projection
DEMOF_POSITIVE_MASK_THRESHOLD=0.0
DEMOF_NEGATIVE_MASK_THRESHOLD=-0.8
DEMOF_FOREGROUND_SCORE_THRESHOLD=0.55
DEMOF_BACKGROUND_SCORE_THRESHOLD=0.02
DEMOF_POSITIVE_MASK_DILATION=0
DEMOF_BACKGROUND_EXCLUSION_DILATION=32
DEMOF_REMOVE_SCATTERED_COMPONENTS="True"
DEMOF_COMPONENT_VOXEL_SIZE=0.025
DEMOF_MIN_COMPONENT_KEEP_RATIO=0.05

# ── Step 2: object_shape_bbox.py ─────────────────────────────────────────────
SHAPE_MASK_PATH="${RESULTS_DIR}/mask3d.pth"
SHAPE_RESULTS_DIR="${RESULTS_DIR}/object_shape_bbox"
SHAPE_BBOX_PADDING_X=0.03
SHAPE_BBOX_PADDING_Y=0.03
SHAPE_BBOX_PADDING_Z=0.03
SHAPE_VOXEL_SIZE=0.025
SHAPE_DILATION_VOXELS=1
SHAPE_FILL_GRID_HOLES="True"
SHAPE_FILL_MODE="shell_flood"         # shell_flood | axis_columns
SHAPE_REMOVE_SCATTERED_GAUSSIANS="True"
SHAPE_ADD_ALL_GAUSSIANS_IN_SHAPE="True"
SHAPE_COLOR_SIMILARITY_THRESHOLD=0.12
SHAPE_RENDER_SHAPE_MESH="True"

# ── Step 3: table_inpaint_under_object.py ────────────────────────────────────
INPAINT_MASK_PATH="${RESULTS_DIR}/mask3d.pth"
INPAINT_RESULTS_DIR="${RESULTS_DIR}/table_inpaint"
INPAINT_SUPPORT_AXIS=0
INPAINT_SUPPORT_SIDE="auto"           # auto | min | max
INPAINT_FOOTPRINT_PADDING=0.05
INPAINT_TABLE_SEARCH_DEPTH=0.12
INPAINT_TABLE_THICKNESS=0.025
INPAINT_INPAINT_SHADOW="True"
INPAINT_KNN_K=8

# ── Step 4: camera_pose_optimization.py ──────────────────────────────────────
CAM_TARGET_IMAGE="${DATA_DIR}/camera.jpg"
CAM_INITIAL_VIEW_IDX=3
CAM_INITIAL_POSE_PATH="${DATA_DIR}/camera_pose.pt"
CAM_OUTPUT_IMAGE="${RESULTS_DIR}/optimized_camera_render.png"
CAM_POSE_OUTPUT="${RESULTS_DIR}/optimized_camera_pose.pt"
CAM_SHOW_WINDOWS="True"
CAM_WINDOW_DISPLAY_SCALE=0.5
CAM_LR=0.05
CAM_ITERATIONS=300

# ── Step 5: dynamic_object_pose_pnp.py ───────────────────────────────────────
PNP_TARGET_PATH="${DATA_DIR}/frames"
PNP_FORMAT="${RASTERIZER}"
PNP_MASK3D_PATH="${RESULTS_DIR}/mask3d.pth"
PNP_CAMERA_POSE_PATH="${RESULTS_DIR}/optimized_camera_pose.pt"
PNP_OUTPUT_DIR="${RESULTS_DIR}/dynamic_pnp"
PNP_MATCHER="eloftr"
PNP_OBJECT_ALPHA_THRESH=0.25
PNP_MAX_CORRESPONDENCES=250
PNP_RANSAC_ITERS=100
PNP_REPROJ_ERROR_PX=4.0
PNP_SHOW_WINDOWS="True"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — demof.py
# ─────────────────────────────────────────────────────────────────────────────
if ask_step 1 "Gradient-based 3D Segmentation" "demof.py"; then
    STEP1_ARGS=(
        --data-dir           "${DATA_DIR}"
        --checkpoint         "${CHECKPOINT}"
        --prompt             "${DEMOF_PROMPT}"
        --results-dir        "${RESULTS_DIR}"
        $(bool_flag "${DEMOF_SHOW_VISUAL_FEEDBACK}"        "show-visual-feedback")
        --rasterizer         "${RASTERIZER}"
        --data-factor        "${DATA_FACTOR}"
        --mask-interval      "${DEMOF_MASK_INTERVAL}"
        --voting-method      "${DEMOF_VOTING_METHOD}"
        --positive-mask-threshold         "${DEMOF_POSITIVE_MASK_THRESHOLD}"
        --negative-mask-threshold         "${DEMOF_NEGATIVE_MASK_THRESHOLD}"
        --foreground-score-threshold      "${DEMOF_FOREGROUND_SCORE_THRESHOLD}"
        --background-score-threshold      "${DEMOF_BACKGROUND_SCORE_THRESHOLD}"
        --positive-mask-dilation          "${DEMOF_POSITIVE_MASK_DILATION}"
        --background-exclusion-dilation   "${DEMOF_BACKGROUND_EXCLUSION_DILATION}"
        $(bool_flag "${DEMOF_REMOVE_SCATTERED_COMPONENTS}" "remove-scattered-components")
        --component-voxel-size            "${DEMOF_COMPONENT_VOXEL_SIZE}"
        --min-component-keep-ratio        "${DEMOF_MIN_COMPONENT_KEEP_RATIO}"
    )
    run_step "demof.py" STEP1_ARGS
else
    print_skipped
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — object_shape_bbox.py
# ─────────────────────────────────────────────────────────────────────────────
if ask_step 2 "Object Shape & Bounding-Box Extraction" "object_shape_bbox.py"; then
    STEP2_ARGS=(
        --data-dir      "${DATA_DIR}"
        --checkpoint    "${CHECKPOINT}"
        --mask-path     "${SHAPE_MASK_PATH}"
        --results-dir   "${SHAPE_RESULTS_DIR}"
        --rasterizer    "auto"
        --data-factor   "${DATA_FACTOR}"
        --bbox-padding-x "${SHAPE_BBOX_PADDING_X}"
        --bbox-padding-y "${SHAPE_BBOX_PADDING_Y}"
        --bbox-padding-z "${SHAPE_BBOX_PADDING_Z}"
        --voxel-size     "${SHAPE_VOXEL_SIZE}"
        --dilation-voxels "${SHAPE_DILATION_VOXELS}"
        $(bool_flag "${SHAPE_FILL_GRID_HOLES}"             "fill-grid-holes")
        --fill-mode      "${SHAPE_FILL_MODE}"
        $(bool_flag "${SHAPE_REMOVE_SCATTERED_GAUSSIANS}"  "remove-scattered-gaussians")
        $(bool_flag "${SHAPE_ADD_ALL_GAUSSIANS_IN_SHAPE}"  "add-all-gaussians-in-shape")
        --color-similarity-threshold "${SHAPE_COLOR_SIMILARITY_THRESHOLD}"
        $(bool_flag "${SHAPE_RENDER_SHAPE_MESH}"           "render-shape-mesh")
    )
    run_step "object_shape_bbox.py" STEP2_ARGS
else
    print_skipped
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — table_inpaint_under_object.py
# ─────────────────────────────────────────────────────────────────────────────
if ask_step 3 "Table Inpainting Under Object" "table_inpaint_under_object.py"; then
    STEP3_ARGS=(
        --data-dir          "${DATA_DIR}"
        --checkpoint        "${CHECKPOINT}"
        --mask-path         "${INPAINT_MASK_PATH}"
        --results-dir       "${INPAINT_RESULTS_DIR}"
        --rasterizer        "auto"
        --data-factor       "${DATA_FACTOR}"
        --support-axis      "${INPAINT_SUPPORT_AXIS}"
        --support-side      "${INPAINT_SUPPORT_SIDE}"
        --footprint-padding "${INPAINT_FOOTPRINT_PADDING}"
        --table-search-depth "${INPAINT_TABLE_SEARCH_DEPTH}"
        --table-thickness   "${INPAINT_TABLE_THICKNESS}"
        $(bool_flag "${INPAINT_INPAINT_SHADOW}" "inpaint-shadow")
        --knn-k             "${INPAINT_KNN_K}"
    )
    run_step "table_inpaint_under_object.py" STEP3_ARGS
else
    print_skipped
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — camera_pose_optimization.py
# ─────────────────────────────────────────────────────────────────────────────
if ask_step 4 "Camera Pose Optimisation" "camera_pose_optimization.py"; then
    STEP4_ARGS=(
        --checkpoint                "${CHECKPOINT}"
        --data-dir                  "${DATA_DIR}"
        --camera-target-image-path  "${CAM_TARGET_IMAGE}"
        --initial-view-idx          "${CAM_INITIAL_VIEW_IDX}"
        --initial-camera-pose-path  "${CAM_INITIAL_POSE_PATH}"
        --rasterizer                "${RASTERIZER}"
        --data-factor               "${DATA_FACTOR}"
        --output-image-path         "${CAM_OUTPUT_IMAGE}"
        --camera-pose-output-path   "${CAM_POSE_OUTPUT}"
        $(bool_flag "${CAM_SHOW_WINDOWS}" "show-windows")
        --window-display-scale      "${CAM_WINDOW_DISPLAY_SCALE}"
        --camera-lr                 "${CAM_LR}"
        --camera-iterations         "${CAM_ITERATIONS}"
    )
    run_step "camera_pose_optimization.py" STEP4_ARGS
else
    print_skipped
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — dynamic_object_pose_pnp.py
# ─────────────────────────────────────────────────────────────────────────────
if ask_step 5 "Dynamic Object Pose Estimation (PnP)" "dynamic_object_pose_pnp.py"; then
    STEP5_ARGS=(
        --checkpoint          "${CHECKPOINT}"
        --data-dir            "${DATA_DIR}"
        --target-path         "${PNP_TARGET_PATH}"
        --format              "${PNP_FORMAT}"
        --data-factor         "${DATA_FACTOR}"
        --mask3d-path         "${PNP_MASK3D_PATH}"
        --camera-pose-path    "${PNP_CAMERA_POSE_PATH}"
        --output-dir          "${PNP_OUTPUT_DIR}"
        --matcher             "${PNP_MATCHER}"
        --object-alpha-thresh "${PNP_OBJECT_ALPHA_THRESH}"
        --max-correspondences "${PNP_MAX_CORRESPONDENCES}"
        --ransac-iters        "${PNP_RANSAC_ITERS}"
        --reproj-error-px     "${PNP_REPROJ_ERROR_PX}"
        $(bool_flag "${PNP_SHOW_WINDOWS}" "show-windows")
    )
    run_step "dynamic_object_pose_pnp.py" STEP5_ARGS
else
    print_skipped
fi

# ─────────────────────────────────────────────────────────────────────────────
#  Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║   Pipeline finished.                                         ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
