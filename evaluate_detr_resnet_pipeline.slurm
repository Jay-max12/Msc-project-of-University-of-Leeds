#!/bin/bash
#SBATCH -J gb_pipe_detr
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${PROJECT_ROOT}/GBCU"
CLS_CKPT="${PROJECT_ROOT}/outputs/resnet50_roi_cls_run1/checkpoints/best_resnet50.pth"

DETECTOR_TYPE="detr"
DET_CKPT="${PROJECT_ROOT}/outputs/detr_roi_run1/checkpoints/best_recall_then_miou.pth"
DETECTOR_MODEL_NAME="facebook/detr-resnet-50"
OUT_DIR="${PROJECT_ROOT}/outputs/pipeline_eval_detr"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# Use your own environment here.
# source ~/.bashrc
# conda activate <your_env>

echo "[pipeline] detector=${DETECTOR_TYPE} ckpt=${DET_CKPT} out=${OUT_DIR}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision transformers

python -m evaluate_fasterrcnn_resnet_pipeline \
  --dataset_root "${DATA_ROOT}" \
  --detector_type "${DETECTOR_TYPE}" \
  --detector_ckpt "${DET_CKPT}" \
  --detector_model_name "${DETECTOR_MODEL_NAME}" \
  --detr_use_paper_preprocess \
  --prep_shortest_edge 800 \
  --prep_longest_edge 1333 \
  --prep_center_crop_h 800 \
  --prep_center_crop_w 800 \
  --classifier_ckpt "${CLS_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --eval_score_threshold 0.5 \
  --nms_iou_threshold 0.5 \
  --max_detections 5
