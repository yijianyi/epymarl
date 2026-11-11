#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TASK_BANK:?Set TASK_BANK to the task bank JSON}"
: "${RESULT_ROOT:?Set RESULT_ROOT to a fresh output directory}"
: "${GPUS:?Set GPUS to allocated external GPU IDs}"
: "${TASKS:?Set TASKS to comma-separated task IDs}"
: "${MEDIUM_STEPS:?Set MEDIUM_STEPS to the first-stage step budget}"
: "${ADVANCED_STEPS:?Set ADVANCED_STEPS to the additional second-stage steps}"
batch_id=${BATCH_ID:-medium-advanced-s1}
options=()
if [[ -f "$RESULT_ROOT/_batches/hp/$batch_id/batch.json" ]]; then
    options+=(--resume)
fi
if [[ -n "${EVAL_CORES:-}" ]]; then
    options+=(--eval-cpu-affinity "$EVAL_CORES")
fi
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
python src/tabx_pipeline.py batch-train \
    --bank "$TASK_BANK" --tasks "$TASKS" --root "$RESULT_ROOT" \
    --rule hp --batch-id "$batch_id" --seeds "${SEEDS:-1}" \
    --medium-steps "$MEDIUM_STEPS" --advanced-steps "$ADVANCED_STEPS" \
    --gpus "$GPUS" --jobs-per-gpu "${JOBS_PER_GPU:-1}" \
    --eval-workers "${EVAL_WORKERS:-2}" --eval-threads "${EVAL_THREADS:-1}" \
    "${options[@]}"
