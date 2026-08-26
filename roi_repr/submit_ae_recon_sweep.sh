#!/usr/bin/env bash
# Submit 4 AE runs (lambda_recon sweep) as a Slurm array job.
#
# Fixed: embedding_dim=128, lambda_cls=1.0
# Sweep: lambda_recon in {0.05, 0.1, 0.5, 1.0}
#
# Usage:
#   cd GBCNET_Plus
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_recon_sweep.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-1}"
export SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_recon_sweep}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
export LAMBDA_CLS="${LAMBDA_CLS:-1.0}"

JOB_ID="$(sbatch --parsable roi_repr/sweep_ae_recon_weight.slurm)"
echo "Submitted Slurm array job: ${JOB_ID}"
echo "Tasks: 4 (emb=128, lambda_cls=1.0, lambda_recon=0.05,0.1,0.5,1.0)"
echo "Outputs: ${SWEEP_ROOT}/emb128_cls1p0_recon<weight>/"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ae_recon_${JOB_ID}_*.out"
echo ""
echo "After completion:"
echo "  python roi_repr/summarize_ae_sweep.py --sweep_root ${SWEEP_ROOT} --sort_by test_select_mean"
