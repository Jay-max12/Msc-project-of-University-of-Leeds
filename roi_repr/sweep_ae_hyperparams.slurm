#!/bin/bash
#SBATCH -J ae_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --array=0-8

# AE hyperparameter sweep (9 runs):
#   embedding_dim in {64, 128, 256}
#   lambda_cls    in {1.0, 2.0, 3.0}
#   lambda_recon  = 0.3 (fixed)
#
# Submit:
#   cd GBCNET_Plus
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_sweep.sh
#
# Morning summary:
#   python roi_repr/summarize_ae_sweep.py

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
LAMBDA_RECON="${LAMBDA_RECON:-0.3}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_sweep_night}"

EMBEDDING_DIMS=(64 128 256)
LAMBDA_CLS_VALUES=(1.0 2.0 3.0)

IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
EMB_IDX=$((IDX / 3))
CLS_IDX=$((IDX % 3))
EMBEDDING_DIM="${EMBEDDING_DIMS[$EMB_IDX]}"
LAMBDA_CLS="${LAMBDA_CLS_VALUES[$CLS_IDX]}"

CLS_TAG="${LAMBDA_CLS//./p}"
OUTPUT_DIR="${SWEEP_ROOT}/emb${EMBEDDING_DIM}_cls${CLS_TAG}_recon0p3"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/${SWEEP_ROOT}"
cd "${PROJECT_ROOT}"

# Prefer explicit env python (avoids `set -u` + source bashrc failures on login nodes).
MYENV_PYTHON="${HOME}/.conda/envs/myenv/bin/python"
if [ -x "${MYENV_PYTHON}" ]; then
  PYTHON_BIN="${MYENV_PYTHON}"
elif [ -n "${PYTHON_BIN:-}" ] && [ -x "${PYTHON_BIN}" ]; then
  :
else
  if [ -f "${HOME}/.bashrc" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
    set -u
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate myenv
  fi
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

echo "=== AE sweep task ${IDX} ==="
echo "python=${PYTHON_BIN}"
echo "embedding_dim=${EMBEDDING_DIM} lambda_cls=${LAMBDA_CLS} lambda_recon=${LAMBDA_RECON}"
echo "output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" -m roi_repr.train \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --lambda_cls "${LAMBDA_CLS}" \
  --lambda_recon "${LAMBDA_RECON}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42

echo "=== Done: ${OUTPUT_DIR} ==="
