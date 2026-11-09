"""Opt-in official two-rule smoke: TABX_RUN_SMOKE=1 TABX_BANK=... pytest ..."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tabx_selection import select_tiers
from tabx_storage import completed_checkpoints, read_json
from tabx_reporting import read_csv


@pytest.mark.skipif(os.environ.get("TABX_RUN_SMOKE") != "1" or not os.environ.get("TABX_BANK"),
                    reason="Opt in with TABX_RUN_SMOKE=1 and TABX_BANK")
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
                   f"env_args.timeout_rule={rule}",
                   f"tabx_output_root={json.dumps(str(root))}", f"tabx_run_id={run_id}", "seed=7", "use_cuda=False",
                   f"tabx_batch_id={run_id}", "learner_log_interval=16",
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
        from components.epsilon_schedules import DecayThenFlatSchedule
        config = read_json(run / "config.json")
        schedule = DecayThenFlatSchedule(config["epsilon_start"], config["epsilon_finish"],
                                        config["epsilon_anneal_time"], decay="linear")
        assert [row["epsilon"] for row in rows] == [schedule.eval(row["t_env"] - row["length"]) for row in rows]
        metrics = read_csv(run / "train_metrics.csv")
        assert [int(row["t_env"]) for row in metrics] == [16, 32, 48, 64, 80]
        assert [int(row["learner_updates"]) for row in metrics] == [1, 3, 5, 7, 9]
        assert all(float(row["grad_norm_clip"]) == 10 for row in metrics)
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
        worker_command = [sys.executable, str(repository / "src/tabx_pipeline.py"), "eval-worker", "--root", str(root),
                          "--bank", environment["TABX_BANK"], "--rule", rule, "--batch-id", run_id, "--workers", "1", "--once"]
        with (tmp_path / f"{rule}-worker.log").open("w") as log:
            subprocess.run(worker_command, check=True, env=environment, cwd=repository, stdout=log, stderr=subprocess.STDOUT)
        evaluated = [read_json(path) for path in (run / "evaluations").glob("*.json")]
        assert len(evaluated) == 5
        summaries = {row["request_id"]: row for row in read_csv(run / "eval_summary.csv")}
        assert len(summaries) == 5
        for evaluation in evaluated:
            from tabx_storage import EvalQueue
            key = EvalQueue.request_id(evaluation["request"])
            row = summaries[key]
            wins = sum(episode["is_win"] for episode in evaluation["episodes"])
            assert int(row["wins"]) == wins
            assert float(row["win_rate_pct"]) == 100 * wins / len(evaluation["episodes"])
            assert float(row["mean_return"]) == pytest.approx(sum(ep["return_value"] for ep in evaluation["episodes"]) / len(evaluation["episodes"]))
            assert int(row["checkpoint_step"]) == int(Path(evaluation["request"]["snapshot"]).name)
            assert Path(evaluation["provenance"]["imports"]["src.tabx"]).is_relative_to(repository / "third_party/TABX")
            assert evaluation["provenance"]["tabx"]["submodule_commit"]
            assert evaluation["provenance"]["tabx"]["parameter_json_sha256"]
            assert evaluation["native_layout"]["native_shape"] == [2, 3, 0]
            assert evaluation["worker"]["cuda_visible_devices"] == "" and evaluation["worker"]["jax_backend"] == "cpu"
        from tabx_pipeline import build_rollout, rollout, configure_cpu
        from tabx_dataset import EpisodeRecorder, write_shard, verify_shard, read_episode, read_episode_batch
        import numpy as np
        configure_cpu()
        runner = build_rollout(snapshots[-1], environment["TABX_BANK"], rule)
        episodes = []
        try:
            for seed in (9001, 9002):
                recorder = EpisodeRecorder()
                rollout(runner, seed, recorder)
                episodes.append(recorder.episode)
            metadata = dict(runner.env.get_layout_info(), timeout_rule=rule,
                            provenance=evaluated[0]["provenance"], purpose="migration_smoke_not_qualified_tier")
            shard = write_shard(run / "validation" / "native-roundtrip.h5", episodes, metadata)
            verified = verify_shard(shard)
            assert verified["episodes"] == 2 and verified["frames"] == verified["steps"] + 2
            batch, lengths = read_episode_batch(shard)
            for i, length in enumerate(lengths):
                episode = read_episode(shard, i)
                for field in ("obs", "state", "avail_actions"):
                    np.testing.assert_array_equal(batch[field][i, :length + 1].numpy(), episode["frames"][field])
                for field in ("actions", "reward", "terminated"):
                    np.testing.assert_array_equal(batch[field][i, :length].numpy(), episode["transitions"][field])
            audit[rule].update(cpu_evaluated_requests=len(evaluated), hdf5_roundtrip=dict(
                episodes=verified["episodes"], steps=verified["steps"], frames=verified["frames"],
                layout_version=verified["metadata"]["layout_version"], native_shape=verified["metadata"]["native_shape"]))
        finally:
            runner.close_env()
    assert len(records["hp"]) >= 10 and len(records["loss"]) >= 10
    assert [row["episode_seed"] for row in records["hp"][:10]] == [row["episode_seed"] for row in records["loss"][:10]]
    from tabx_storage import atomic_json
    atomic_json(tmp_path / "acceptance.json", audit)


@pytest.mark.skipif(os.environ.get("TABX_RUN_SMOKE") != "1" or not os.environ.get("TABX_BANK"),
                    reason="Opt in with TABX_RUN_SMOKE=1 and TABX_BANK")
def test_batch_real_cpu_training_evaluation_and_resume(tmp_path, monkeypatch):
    """Exercise the whole scheduler with real QMIX on CPU; GPU hardware is mocked."""
    import tabx_scheduler
    from tabx_pipeline import parser
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tabx_scheduler, "probe_gpus", lambda gpus: {gpu: dict(name="CPU-test-only") for gpu in gpus})
    original = tabx_scheduler.BatchScheduler.training_command
    def cpu_command(self, job, attempt):
        command = original(self, job, attempt)
        return ["use_cuda=False" if part == "use_cuda=True" else
                "tabx_require_cuda=False" if part == "tabx_require_cuda=True" else part for part in command]
    monkeypatch.setattr(tabx_scheduler.BatchScheduler, "training_command", cpu_command)
    arguments = ["batch-train", "--bank", os.environ["TABX_BANK"], "--root", str(tmp_path / "output"),
                 "--tasks", "23,651,647", "--gpus", "0,1", "--jobs-per-gpu", "1", "--eval-workers", "2",
                 "--eval-threads", "1", "--rule", "loss", "--batch-id", "real-cpu-test", "--poll-seconds", "0.1"]
    for setting in ("t_max=24", "batch_size=2", "buffer_size=4", "env_args.max_episode_steps=8",
                    "save_model_interval=8", "tabx_screen_episodes=2", "tabx_max_pending_evals=1", "learner_log_interval=8"):
        arguments.extend(["--set", setting])
    assert tabx_scheduler.batch_train(parser().parse_args(arguments)) == 0
    batch = tmp_path / "output/_batches/loss/real-cpu-test"
    before = read_json(batch / "state.json")
    assert before["status"] == "complete"
    assert not list((tmp_path / "output").rglob("eval_queue.sqlite3"))
    for job in before["jobs"]:
        assert job["train_status"] == job["eval_status"] == "complete"
        assert job["eval_done"] == job["eval_expected"] == 4 and job["eval_remaining"] == 0
        run = tmp_path / "output" / job["attempts"][-1]["run"]
        assert completed_checkpoints(run)[0]["learner_updates"] > 0
        assert read_json(run / "manifest.json")["batch_id"] == "real-cpu-test"
        from tabx_reporting import TRAIN_METRICS_VERSION
        assert read_json(run / "manifest.json")["training_metrics_version"] == TRAIN_METRICS_VERSION
        assert len(read_csv(run / "eval_summary.csv")) == 4
        assert len(read_csv(run / "train_metrics.csv")) == 2
        assert (run / "train.log").is_file()
        assert not list(run.glob("*.png"))
    assert tabx_scheduler.batch_train(parser().parse_args(arguments + ["--resume"])) == 0
    after = read_json(batch / "state.json")
    assert [len(job["attempts"]) for job in after["jobs"]] == [1, 1, 1]
    assert not list((tmp_path / "output").rglob("eval_queue.sqlite3"))
    # After the scheduler exits, a standalone CPU pool must also admit new
    # selection phases from the same managed queue.
    run = tmp_path / "output" / after["jobs"][0]["attempts"][-1]["run"]
    assert select_tiers(tmp_path / "output", run, 2, 2, 2)["status"] == "pending"
    command = [sys.executable, str(tabx_scheduler.REPOSITORY / "src/tabx_pipeline.py"), "eval-worker",
        "--root", str(tmp_path / "output"), "--bank", os.environ["TABX_BANK"], "--rule", "loss",
        "--batch-id", "real-cpu-test", "--workers", "2", "--once"]
    with (tmp_path / "standalone-worker.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1"))
    assert "baseline" in {row["phase"] for row in read_csv(run / "eval_summary.csv")}
