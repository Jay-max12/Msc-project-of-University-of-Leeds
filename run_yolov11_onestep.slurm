#!/bin/bash
#SBATCH -J yolov11_onestep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

# source ~/.bashrc
# conda activate <your_env>

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/GBCU}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/yolov11_onestep_run1}"
MODEL="${MODEL:-yolo11n.pt}"

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

# 虚拟环境里若已装好 ultralytics / numpy / pandas / matplotlib，可提交前设 SKIP_PIP_INSTALL=1 跳过 pip。
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install ultralytics
fi

python train_yolov11_onestep.py \
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
  --seed "${SEED}"
