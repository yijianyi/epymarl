"""Train selected TABX tasks on four GPUs with MAPPO and HP adjudication."""

import argparse
import os
from pathlib import Path

from tabx_pipeline import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Comma-separated TABX task IDs")
    parser.add_argument("--jobs-per-gpu", type=int, default=1,
                        help="Maximum concurrent training processes per GPU (default: 1)")
    options = parser.parse_args()

    for name in ("TASK_BANK", "RESULT_ROOT", "GPUS"):
        if not os.environ.get(name):
            raise SystemExit(f"Set {name} before running this command.")

    gpus = os.environ["GPUS"]
    if len(gpus.split(",")) != 4:
        raise SystemExit("GPUS must list four GPU IDs, for example 0,1,2,3.")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    output = Path(os.environ["RESULT_ROOT"]).expanduser().resolve()
    batch_id = "mappo32"
    args = [
        "batch-train", "--alg", "mappo_tabx", "--rule", "hp", "--skip-bank-hash-check",
        "--bank", os.environ["TASK_BANK"], "--root", str(output), "--tasks", options.tasks,
        "--batch-id", batch_id, "--gpus", gpus, "--eval-workers", "4",
        "--jobs-per-gpu", str(options.jobs_per_gpu),
    ]
    if (output / "_batches" / "hp" / batch_id / "batch.json").is_file():
        args.append("--resume")
    main(args)
