#!/bin/bash
#SBATCH -J yolo_roi_pipe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00

# Fixed-CV3 YOLO detector + AE ROI classifier pipeline.
#
# 提交：
#   cd GBCNET_Plus && SKIP_PIP_INSTALL=1 sbatch run_yolo_roi_pipeline.slurm
#
# 可选环境变量：
#   SPLIT=test|val|train
#   BOX_SELECTION=largest_area|highest_conf|all
#   OUTPUT_DIR=outputs/fixed_cv3/yolo_ae_two_step

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
CLASSIFIER_HEAD="${CLASSIFIER_HEAD:-mlp}"
SPLIT="${SPLIT:-test}"
CV_SPLIT_FILE="${CV_SPLIT_FILE:-${PROJECT_ROOT}/GBCU/cv_splits/cv3_seed42_binary.json}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-outputs/roi_repr/binary/ae_cv3_fixed}"
YOLO_CV_ROOT="${YOLO_CV_ROOT:-outputs/yolov11_onestep_cv3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fixed_cv3/yolo_ae_two_step}"
BOX_SELECTION="${BOX_SELECTION:-largest_area}"
MAX_DETECTIONS="${MAX_DETECTIONS:-5}"
SAVE_VIS="${SAVE_VIS:-0}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

MYENV_PYTHON="${HOME}/.conda/envs/myenv/bin/python"
if [ -x "${MYENV_PYTHON}" ]; then
  PYTHON_BIN="${MYENV_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

VIS_FLAG="--no-save_vis"
if [ "${SAVE_VIS}" = "1" ]; then
  VIS_FLAG="--save_vis"
fi

"${PYTHON_BIN}" -m roi_repr.eval_yolo_roi_pipeline \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --classifier_head "${CLASSIFIER_HEAD}" \
  --split "${SPLIT}" \
  --all_folds \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --classifier_root "${CLASSIFIER_ROOT}" \
  --yolo_cv_root "${YOLO_CV_ROOT}" \
  --n_cv_folds 3 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --eval_score_threshold 0.5 \
  --nms_iou_threshold 0.5 \
  --max_detections "${MAX_DETECTIONS}" \
  --box_selection "${BOX_SELECTION}" \
  --no_box_fallback full_image \
  ${VIS_FLAG}
