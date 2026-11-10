#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${TASK_BANK:?Set TASK_BANK to the full merged task bank on the server}"
: "${RESULT_ROOT:?Set RESULT_ROOT to an output directory on the server}"
: "${GPUS:?Set GPUS to allocated external GPU IDs, e.g. 2,3}"
batch_id=${BATCH_ID:-pilot32-1500k-s1}
pilot=artifacts/tabx/pilot-v2.json
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

for rule in hp loss; do
    options=()
    if [[ -f "$RESULT_ROOT/_batches/$rule/$batch_id/batch.json" ]]; then
        options+=(--resume)
    fi
    if [[ -n "${EVAL_CORES:-}" ]]; then
        options+=(--eval-cpu-affinity "$EVAL_CORES")
    fi
    python src/tabx_pipeline.py batch-train \
        --bank "$TASK_BANK" --pilot "$pilot" --root "$RESULT_ROOT" \
        --rule "$rule" --batch-id "$batch_id" --seeds "${SEEDS:-1}" \
        --gpus "$GPUS" --jobs-per-gpu "${JOBS_PER_GPU:-1}" \
        --eval-workers "${EVAL_WORKERS:-2}" --eval-threads "${EVAL_THREADS:-1}" \
        "${options[@]}"
done

python scripts/tabx_pilot_samples.py --root "$RESULT_ROOT" --bank "$TASK_BANK" \
    --batch-id "$batch_id" --episodes "${SAMPLE_EPISODES:-8}" --workers "${SAMPLE_WORKERS:-1}"
