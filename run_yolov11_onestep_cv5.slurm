#!/bin/bash
#SBATCH -J yolo_cv5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

# YOLOv11 one-step with 5-fold CV (detect_cls, fliplr=0.5 only)
#
# Submit:
#   SKIP_PIP_INSTALL=1 sbatch run_yolov11_onestep_cv5.slurm

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/GBCU}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/yolov11_onestep_cv5}"
MODEL="${MODEL:-yolo11n.pt}"
N_CV_FOLDS="${N_CV_FOLDS:-5}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMGSZ="${IMGSZ:-800}"
LR="${LR:-0.005}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
EVAL_SCORE_THRESHOLD="${EVAL_SCORE_THRESHOLD:-0.5}"
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.5}"
MAX_DETECTIONS="${MAX_DETECTIONS:-5}"
FALLBACK_CLASS="${FALLBACK_CLASS:-0}"
SEED="${SEED:-42}"

MYENV_PYTHON="${HOME}/.conda/envs/myenv/bin/python"
if [ -x "${MYENV_PYTHON}" ]; then
  PYTHON_BIN="${MYENV_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install ultralytics
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

echo "=== YOLO one-step ${N_CV_FOLDS}-fold CV ==="
echo "python=${PYTHON_BIN}"
echo "output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" train_yolov11_onestep.py \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --imgsz "${IMGSZ}" \
  --lr "${LR}" \
  --momentum "${MOMENTUM}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --eval_score_threshold "${EVAL_SCORE_THRESHOLD}" \
  --nms_iou_threshold "${NMS_IOU_THRESHOLD}" \
  --max_detections "${MAX_DETECTIONS}" \
  --fallback_class "${FALLBACK_CLASS}" \
  --n_cv_folds "${N_CV_FOLDS}" \
  --seed "${SEED}"

echo "=== Done: ${OUTPUT_DIR} ==="
