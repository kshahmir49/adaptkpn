#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?usage: $0 manifest.csv fixed4095|minmax}"
NORMALIZATION="${2:?usage: $0 manifest.csv fixed4095|minmax}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
OUT_DIR="${OUT_DIR:-outputs/ct}"
LAMBDA_EDGE="${LAMBDA_EDGE:-350}"
N=$(python - "$MANIFEST" <<'PY'
import pandas as pd, sys
print(len(pd.read_csv(sys.argv[1])))
PY
)
TASKS=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
MANIFEST="$MANIFEST" NORMALIZATION="$NORMALIZATION" BATCH_SIZE="$BATCH_SIZE" OUT_DIR="$OUT_DIR" LAMBDA_EDGE="$LAMBDA_EDGE" \
  sbatch --array="0-$((TASKS-1))%${MAX_PARALLEL}" slurm/ct_array.sh
