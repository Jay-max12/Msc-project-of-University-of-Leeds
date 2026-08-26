#!/bin/bash
#SBATCH -J oracle_roi_abl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-all}"
TASK_MODE="${TASK_MODE:-binary}"
SPLIT="${SPLIT:-train}"
YOLO_CKPT="${YOLO_CKPT:-outputs/yolov11_onestep_run1/weights/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/roi_repr/binary/oracle_roi_ablation_single_roi}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# source ~/.bashrc
# conda activate myenv

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

python -m roi_repr.eval_oracle_roi_ablation \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --split "${SPLIT}" \
  --all_folds \
  --yolo_ckpt "${YOLO_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_score_threshold 0.5 \
  --nms_iou_threshold 0.5 \
  --max_detections 5 \
  --no_box_fallback full_image
