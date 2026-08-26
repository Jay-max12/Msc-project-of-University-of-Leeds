#!/bin/bash
#SBATCH -J ae_recon
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --array=0-3

# AE reconstruction-weight sweep (4 runs):
#   embedding_dim = 128 (fixed)
#   lambda_cls    = 1.0 (fixed)
#   lambda_recon  in {0.05, 0.1, 0.5, 1.0}
#
# Submit:
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_recon_sweep.sh

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_recon_sweep}"

LAMBDA_RECON_VALUES=(0.05 0.1 0.5 1.0)

IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LAMBDA_RECON="${LAMBDA_RECON_VALUES[$IDX]}"

case "${LAMBDA_RECON}" in
  0.05) RECON_TAG="0p05" ;;
  0.1)  RECON_TAG="0p1" ;;
  0.5)  RECON_TAG="0p5" ;;
  1.0)  RECON_TAG="1p0" ;;
  *)    echo "Unknown lambda_recon: ${LAMBDA_RECON}"; exit 1 ;;
esac

CLS_TAG="${LAMBDA_CLS//./p}"
OUTPUT_DIR="${SWEEP_ROOT}/emb${EMBEDDING_DIM}_cls${CLS_TAG}_recon${RECON_TAG}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/${SWEEP_ROOT}"
cd "${PROJECT_ROOT}"

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

echo "=== AE recon sweep task ${IDX} ==="
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
