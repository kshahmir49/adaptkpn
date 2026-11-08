#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?usage: $0 manifest.csv}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
OUT_DIR="${OUT_DIR:-outputs/fmd}"
N=$(python - "$MANIFEST" <<'PY'
import pandas as pd, sys
print(len(pd.read_csv(sys.argv[1])))
PY
)
TASKS=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
MANIFEST="$MANIFEST" BATCH_SIZE="$BATCH_SIZE" OUT_DIR="$OUT_DIR" \
  sbatch --array="0-$((TASKS-1))%${MAX_PARALLEL}" slurm/fmd_array.sh
