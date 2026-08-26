#!/usr/bin/env bash
# Submit AE improvement experiments 4/5/6 as a Slurm array job.
#
# Baseline (exp 0): emb=128, cls=1.0, recon=0.3, lr=0.005
#   4) malignant class weight = 1.5
#   5) backbone lr=0.001, head/decoder lr=0.005
#   6) freeze backbone for first 5 epochs
#
# Usage:
#   cd GBCNET_Plus
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_improve_round2.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-1}"
export SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_improve_round2}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
export LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
export LAMBDA_RECON="${LAMBDA_RECON:-0.3}"

JOB_ID="$(sbatch --parsable roi_repr/sweep_ae_improve_round2.slurm)"
echo "Submitted Slurm array job: ${JOB_ID}"
echo "Tasks:"
echo "  [0] exp4_mal_weight1p5"
echo "  [1] exp5_lr_bb0p001_hd0p005"
echo "  [2] exp6_freeze_bb5"
echo "Outputs: ${SWEEP_ROOT}/"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ae_imp2_${JOB_ID}_*.out"
echo ""
echo "After completion:"
echo "  python roi_repr/summarize_ae_sweep.py --sweep_root ${SWEEP_ROOT} --sort_by test_select_mean"
