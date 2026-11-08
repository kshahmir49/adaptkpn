#!/usr/bin/env bash
#SBATCH --job-name=adaptkpn-ct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=adaptkpn-ct-%A_%a.log
#SBATCH --error=adaptkpn-ct-%A_%a.err

# Uncomment and customize these lines for your cluster
##SBATCH --partition=<GPU_PARTITION>
##SBATCH --account=<PROJECT_ACCOUNT>
# module purge
# module load <PYTHON_OR_CONDA_MODULE>
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate <YOUR_ENVIRONMENT>

set -euo pipefail
: "${MANIFEST:?set MANIFEST}"
: "${NORMALIZATION:?set NORMALIZATION to fixed4095 or minmax}"
OUT_DIR="${OUT_DIR:-outputs/ct}"
BATCH_SIZE="${BATCH_SIZE:-1}"

python scripts/ct/run.py \
  --manifest "$MANIFEST" \
  --normalization "$NORMALIZATION" \
  --batch-index "${SLURM_ARRAY_TASK_ID}" \
  --batch-size "$BATCH_SIZE" \
  --out-dir "$OUT_DIR" \
  --save-images
