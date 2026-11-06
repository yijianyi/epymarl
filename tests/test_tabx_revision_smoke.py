"""Opt-in real two-rule smoke: TABX_RUN_SMOKE=1 TABX_BANK=... TABX_ROOT=... pytest ..."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tabx_selection import select_tiers
from tabx_storage import completed_checkpoints, read_json


@pytest.mark.skipif(os.environ.get("TABX_RUN_SMOKE") != "1" or not os.environ.get("TABX_BANK") or not os.environ.get("TABX_ROOT"),
                    reason="Opt in with TABX_RUN_SMOKE=1, TABX_BANK and TABX_ROOT")
def test_two_rule_training_seeds_and_completed_selection(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / "output"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")
    records, audit = {}, {}
    for rule in ("hp", "loss"):
        run_id = f"pair-{rule}"
        run = root / rule / "23" / "7" / run_id
        command = [sys.executable, str(repository / "src/main.py"), "--config=qmix_tabx", "--env-config=tabx", "with",
                   'env_args.key="23"', f"env_args.task_bank={json.dumps(environment['TABX_BANK'])}",
                   f"env_args.tabx_root={json.dumps(environment['TABX_ROOT'])}", f"env_args.timeout_rule={rule}",
                   f"tabx_output_root={json.dumps(str(root))}", f"tabx_run_id={run_id}", "seed=7", "use_cuda=False",
                   "t_max=80", "batch_size=2", "buffer_size=4", "env_args.max_episode_steps=8",
                   "save_model_interval=32", "tabx_screen_episodes=2"]
        saw_in_progress = False
        with (tmp_path / f"{rule}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=repository, env=environment, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            try:
                while process.poll() is None:
                    if not saw_in_progress and (run / "checkpoints/32/manifest.json").exists() and not (run / "training_complete.json").exists():
                        result = select_tiers(root, run, 2, 2, 2)
                        assert result["status"] == "training_in_progress"
                        assert not (run / "selection").exists()
                        saw_in_progress = True
                    if time.monotonic() >= deadline:
                        pytest.fail(f"{rule} training smoke timed out")
                    time.sleep(0.02)
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
        assert process.returncode == 0, (tmp_path / f"{rule}.log").read_text()[-4000:]
        assert saw_in_progress, "The smoke must exercise select while training is still running"
        rows = [json.loads(line) for line in (run / "training_episodes.jsonl").read_text().splitlines()]
        records[rule] = rows
        assert [row["episode_index"] for row in rows] == list(range(len(rows)))
        complete, checksum, snapshots = completed_checkpoints(run)
        steps = [entry["t_env"] for entry in complete["checkpoints"]]
        assert steps == [0, 32, 64, 80]
        assert read_json(run / "training_summary.json")["learner_updates"] > 0
        result = select_tiers(root, run, 2, 2, 2)
        assert result["status"] == "pending"
        state = read_json(result["state"])
        assert state["training_complete_sha256"] == checksum
        assert state["snapshots"] == [str(path.relative_to(root)) for path in snapshots]
        audit[rule] = dict(training_in_progress_observed=True, checkpoints=steps, training_complete_sha256=checksum,
                           episode_seeds=[row["episode_seed"] for row in rows])
    assert len(records["hp"]) >= 10 and len(records["loss"]) >= 10
    assert [row["episode_seed"] for row in records["hp"][:10]] == [row["episode_seed"] for row in records["loss"][:10]]
    from tabx_storage import atomic_json
    atomic_json(tmp_path / "acceptance.json", audit)
