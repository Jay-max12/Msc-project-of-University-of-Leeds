#!/usr/bin/env bash
# Generate fixed CV3 split manifest for GBCU (binary stratify, seed=42).
#
# Output: GBCU/cv_splits/cv3_seed42_binary.json
#
# Usage:
#   cd GBCNET_Plus
#   bash roi_repr/generate_cv_splits.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/GBCU}"
PYTHON_BIN="${HOME}/.conda/envs/myenv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python)"
fi

"${PYTHON_BIN}" -m roi_repr.data.cv_splits \
  --dataset_root "${DATASET_ROOT}" \
  --n_cv_folds 3 \
  --seed 42 \
  --stratify binary
