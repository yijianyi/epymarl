"""Scale checks separate orchestration cost from real GPU model capacity."""
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabx_pipeline import parser
from tabx_reporting import TRAIN_FIELDS, TrainingLogReader, append_train_metrics, read_csv
import tabx_scheduler as scheduling
from tabx_storage import atomic_json


class IdleProcess:
    pid = None
    returncode = None

    def poll(self):
        return None


def make_logical_batch(path, count=8000, active=24):
    """Build already-recovered in-memory jobs; no fake checkpoint files needed."""
    root = Path(path) / "output"
    directory = root / "_batches/loss/scale"
    directory.mkdir(parents=True)
    args = parser().parse_args(["batch-train", "--bank", "unused", "--root", str(root), "--rule", "loss",
        "--tasks", "23", "--gpus", "0,1", "--jobs-per-gpu", "12", "--eval-workers", "4", "--batch-id", "scale"])
    args.gpus = ["0", "1"]
    jobs = []
    for index in range(count):
        running = index < active
        key = f"already-evaluated-{index}"
        attempt = dict(run_id=f"run{index}", run=f"loss/{index}/1/run{index}", log=f"logs/run{index}.log",
            gpu=str(index % 2), started_at=time.time(), pid=None, process_start=None,
            status="running" if running else "complete", train_status="running" if running else "complete",
            inventory_verified=not running, eval_expected=None if running else 1,
            screen_requests={} if running else {"0": key},
            evaluations={} if running else {key: dict(status="complete", valid=True)})
        jobs.append(dict(job_id=f"{index:06d}", task_id=str(index), train_seed=1, attempts=[attempt], needs_training=False))
    state = dict(status="running", jobs=jobs)
    manager = scheduling.BatchScheduler(args, directory, dict(training_config={"tabx_screen_episodes": 2}), state)
    manager.authorized = True
    manager.log = (directory / "batch.log").open("a")
    manager.initializing = False
    for job in jobs:
        attempt = job["attempts"][-1]
        manager._track_attempt(job, attempt)
        if attempt["train_status"] == "running":
            manager.running[attempt["run_id"]] = (IdleProcess(), io.StringIO(), job)
        else:
            manager.records.update(attempt["evaluations"])
    manager.eval_counts = Counter(record["status"] for record in manager.records.values())
    manager.initialize_job_cache()
    manager.dirty_runs = set(manager.running)
    manager.refresh_reports(force=True)
    return manager


def idle_poll(manager):
    manager.poll_training()
    manager.fill_slots()
    manager.evaluations()
    manager.refresh_reports()


def test_8000_jobs_24_active_100_idle_polls_have_no_batch_work(tmp_path, monkeypatch):
    manager = make_logical_batch(tmp_path)
    calls = dict(status=0, history=0, csv=0)
    original_status = scheduling.derive_job_status
    original_open = Path.open
    original_csv = scheduling.write_csv
    def status(attempt):
        calls["status"] += 1
        return original_status(attempt)
    def opening(path, *args, **kwargs):
        if any(part in str(path) for part in ("/checkpoints/", "/evaluations/", "/evaluation_requests/")):
            calls["history"] += 1
        return original_open(path, *args, **kwargs)
    def csv(*args, **kwargs):
        calls["csv"] += 1
        return original_csv(*args, **kwargs)
    monkeypatch.setattr(scheduling, "derive_job_status", status)
    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(scheduling, "write_csv", csv)
    try:
        for _ in range(100):
            idle_poll(manager)
        manager.progress()
        assert calls == dict(status=0, history=0, csv=0)
        assert len(manager.running) == 24 and manager.gpu_in_use == {"0": 12, "1": 12}
    finally:
        manager.log.close()


def test_one_evaluation_updates_one_job_and_duplicate_is_free(tmp_path, monkeypatch):
    manager = make_logical_batch(tmp_path)
    calls = []
    original = scheduling.derive_job_status
    monkeypatch.setattr(scheduling, "derive_job_status", lambda attempt: (calls.append(attempt["run_id"]), original(attempt))[1])
    request = dict(run_id="run0", task_id="0", train_seed=1, rule="loss", snapshot="loss/0/1/run0/checkpoints/0",
                   phase="screen", action_selection="greedy", seeds=[7])
    report = dict(request=request, episodes=[dict(seed=7, is_win=1, end_reason=1, return_value=1)])
    before = manager.eval_counts["complete"]
    try:
        manager._adopt(request, "complete", report)
        manager.refresh_reports()
        assert calls == ["run0"]
        assert manager.eval_counts["complete"] == before + 1
        calls.clear()
        manager._adopt(request, "complete", report)
        manager.refresh_reports()
        assert calls == [] and manager.eval_counts["complete"] == before + 1
        assert len(list((manager.directory / "jobs").glob("*.json"))) == 1
    finally:
        manager.log.close()


def test_periodic_snapshot_is_separate_from_progress_and_job_events(tmp_path, monkeypatch):
    manager = make_logical_batch(tmp_path, count=800)
    calls = []
    original = scheduling.write_csv
    monkeypatch.setattr(scheduling, "write_csv", lambda *a, **k: (calls.append(a[0]), original(*a, **k))[1])
    try:
        manager.progress()
        manager.refresh_reports()
        assert calls == []
        manager.last_snapshot -= 301
        manager.refresh_reports()
        manager.refresh_reports()
        assert len(calls) == 1
        assert json.loads((manager.directory / "state.json").read_text())["snapshot_time_utc"]
    finally:
        manager.log.close()


def test_incremental_training_reader_preserves_partial_lines_and_rotation(tmp_path):
    view = {}
    reader = TrainingLogReader(tmp_path)
    row = dict(zip(TRAIN_FIELDS, [16, 1, 4, 2, 4, 10, 5, 7, .01]))
    append_train_metrics(tmp_path / "train_metrics.csv", row)
    episodes = tmp_path / "training_episodes.jsonl"
    episodes.write_text('{"t_env":16,"end_reason":2}\n{"t_env":2')
    rows, changed = reader.update(view)
    assert changed and len(rows) == 1 and view["t_env"] == 16
    metric_offset = reader.metrics.offset
    episode_offset = reader.episodes.offset
    assert reader.update(view) == ([], False)
    assert reader.metrics.offset == metric_offset and reader.episodes.offset == episode_offset
    with episodes.open("a") as stream:
        stream.write('4,"end_reason":1}\n')
    assert reader.update(view)[1] and view["t_env"] == 24
    append_train_metrics(tmp_path / "train_metrics.csv", dict(row, t_env=24, learner_updates=2))
    assert [entry["learner_updates"] for entry in reader.update(view)[0]] == ["2"]
    rotated = tmp_path / "replacement.csv"
    append_train_metrics(rotated, dict(row, t_env=32, learner_updates=3))
    rotated.replace(tmp_path / "train_metrics.csv")
    assert [entry["learner_updates"] for entry in reader.update(view)[0]] == ["3"]


class ConstantMAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(5.0))

    def init_hidden(self, batch_size):
        pass

    def forward(self, batch, t):
        return self.weight.expand(batch.batch_size, 10, 8)


def test_q_means_include_padding_and_preserve_loss_gradient_and_update(tmp_path):
    from learners.q_learner import QLearner
    values = {
        "reward": torch.tensor([[[7.], [7.], [999.], [999.], [0.]]]),
        "actions": torch.zeros(1, 5, 10, 1, dtype=torch.long),
        "terminated": torch.tensor([[[0], [1], [0], [0], [0]]]),
        "filled": torch.tensor([[[1], [1], [1], [0], [0]]]),
        "avail_actions": torch.ones(1, 5, 10, 8),
    }
    class Batch:
        batch_size, max_seq_length = 1, 5
        def __getitem__(self, key):
            return values[key]
    logged = {}
    logger = NS(log_stat=lambda name, value, step: logged.update({name: value}))
    args = NS(n_agents=10, mixer=None, lr=.01, learner_log_interval=1, use_cuda=False,
        standardise_returns=False, standardise_rewards=False, common_reward=True, double_q=True,
        gamma=0, grad_norm_clip=10, target_update_interval_or_tau=200, env="tabx", tabx_run_dir=str(tmp_path))
    mac = ConstantMAC()
    learner = QLearner(mac, {}, logger, args)
    baseline = torch.nn.Parameter(torch.tensor(5.0))
    optimiser = torch.optim.Adam([baseline], lr=.01)
    mask = torch.tensor([1., 1., 0., 0.]).reshape(1, 4, 1).expand(1, 4, 10)
    old_loss = (((baseline.expand(1, 4, 10) - 7) * mask)**2).sum() / mask.sum()
    old_loss.backward()
    torch.nn.utils.clip_grad_norm_([baseline], 10)
    optimiser.step()
    learner.train(Batch(), t_env=16, episode_num=1)
    assert logged["q_taken_mean"] == 5 and logged["target_mean"] == 7
    assert logged["loss"] == old_loss.item() == 4
    torch.testing.assert_close(mac.weight.grad, baseline.grad)
    torch.testing.assert_close(mac.weight, baseline)
    row = read_csv(tmp_path / "train_metrics.csv")[0]
    assert float(row["q_taken_mean"]) == 5 and float(row["target_mean"]) == 7


def test_prefetch_is_bounded_fair_and_restores_every_registered_id(tmp_path):
    from tabx_storage import EvalQueue
    from tabx_dispatcher import EvalDispatcher
    queue = EvalQueue(tmp_path)
    dispatcher = EvalDispatcher(queue, "loss", "batch", workers=2, prefetch=1)
    requests = []
    for run_id in ("A", "B", "C"):
        run = f"loss/{run_id}/1/{run_id}"
        for step in (0, 8, 16):
            snapshot = f"{run}/checkpoints/{step}"
            digest = f"weights-{run_id}-{step}"
            atomic_json(tmp_path / snapshot / "manifest.json",
                        dict(t_env=step, run_id=run_id, timeout_rule="loss", agent_sha256=digest))
            request = dict(rule="loss", batch_id="batch", run=run, run_id=run_id, snapshot=snapshot,
                           checkpoint_hash=digest, phase="screen", action_selection="greedy", seeds=[1, 2])
            requests.append(request)
            queue.enqueue(request)
    assert queue.active_keys("loss", "batch") == {}
    dispatcher.restore()
    dispatcher.poll()
    ready = queue.active_keys("loss", "batch")
    assert len(ready) == 2
    assert len({queue.info(key)["request"]["run_id"] for key in ready}) == 2
    claim = queue.claim("loss", batch_id="batch")
    key, owner, request = claim
    queue.record(key, owner, dict(seed=1, return_value=1, is_win=1, end_reason=1))
    executions = {(key, 1)}
    queue.release_batch("loss", "batch")
    dispatcher.close()
    # Change worker count/prefetch on recovery without losing queued requests.
    dispatcher = EvalDispatcher(queue, "loss", "batch", workers=1, prefetch=1)
    dispatcher.restore()
    expected = {queue.request_id(request) for request in requests}
    completed = set()
    try:
        for _ in range(100):
            dispatcher.poll()
            active = queue.active_keys("loss", "batch")
            assert sum(status == "pending" for status in active.values()) <= 1
            assert sum(status == "running" for status in active.values()) <= 1
            claim = queue.claim("loss", batch_id="batch")
            if claim is None:
                if not active and dispatcher.counts["registered"] == 0:
                    break
                continue
            key, owner, request = claim
            existing = {row["seed"] for row in queue.results(key)}
            for seed in request["seeds"]:
                if seed not in existing:
                    assert (key, seed) not in executions
                    executions.add((key, seed))
                    queue.record(key, owner, dict(seed=seed, return_value=1, is_win=1, end_reason=1))
            queue.publish_result(key, owner, dict(request=request, episodes=queue.results(key)))
            queue.finish(key, owner)
            completed.add(key)
        assert completed == expected and len(executions) == 18
        assert sum(len((tmp_path / request["run"] / "evaluation_requests/requests.jsonl").read_text().splitlines())
                   for request in requests[::3]) >= 9
        assert not list(tmp_path.rglob("eval_queue.sqlite3"))
    finally:
        dispatcher.close()


def run_scale_benchmark(output, polls=100):
    """Measure 800/8000 logical jobs with the same 24 real, idle CPU children."""
    import platform
    import statistics
    import subprocess
    import tempfile
    results = []
    with tempfile.TemporaryDirectory(prefix="tabx-scale-") as temp:
        for count in (800, 8000):
            manager = make_logical_batch(Path(temp) / str(count), count=count)
            children = []
            try:
                for run_id, (_, stream, job) in list(manager.running.items()):
                    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    children.append(child)
                    manager.running[run_id] = (child, stream, job)
                time.sleep(.5)
                idle_poll(manager)
                def wchar():
                    return int(next(line.split()[1] for line in Path("/proc/self/io").read_text().splitlines()
                                    if line.startswith("wchar:")))
                status_calls = 0
                original = scheduling.derive_job_status
                def counted(attempt):
                    nonlocal status_calls
                    status_calls += 1
                    return original(attempt)
                scheduling.derive_job_status = counted
                samples = []
                written_before, cpu_before, wall_before = wchar(), time.process_time(), time.perf_counter()
                try:
                    for _ in range(polls):
                        start = time.perf_counter()
                        idle_poll(manager)
                        samples.append(time.perf_counter() - start)
                finally:
                    scheduling.derive_job_status = original
                wall = time.perf_counter() - wall_before
                cpu = time.process_time() - cpu_before
                written = wchar() - written_before
                start, snapshot_before = time.perf_counter(), wchar()
                manager.snapshot(force=True)
                snapshot_seconds = time.perf_counter() - start
                snapshot_bytes = wchar() - snapshot_before
                results.append(dict(logical_jobs=count, active_trainers=24, polls=polls,
                    median_poll_ms=1000 * statistics.median(samples), p95_poll_ms=1000 * sorted(samples)[int(.95*(len(samples)-1))],
                    scheduler_cpu_ms=1000*cpu, estimated_cpu_pct_at_1s_poll=100*cpu/polls,
                    idle_write_bytes=written, idle_bytes_per_minute=written*60/max(wall, 1e-9),
                    completed_status_recomputations=status_calls,
                    snapshot_seconds=snapshot_seconds, snapshot_write_bytes=snapshot_bytes))
            finally:
                for child in children:
                    scheduling.stop_process(child)
                manager.log.close()
    ratio = results[1]["median_poll_ms"] / results[0]["median_poll_ms"]
    report = dict(schema_version="tabx-scheduler-scale-1", python=platform.python_version(),
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_hash=scheduling.code_hash(scheduling.REPOSITORY), code=scheduling.git_version(scheduling.REPOSITORY),
        hardware_gpu_test=False, tasks_are_logical_jobs=True, results=results, median_cost_ratio_8000_to_800=ratio,
        passed=ratio < 5 and all(item["idle_write_bytes"] == item["completed_status_recomputations"] == 0 for item in results))
    atomic_json(output, report)
    return report


def test_retry_announces_registration_to_an_existing_dispatcher(tmp_path):
    from tabx_storage import EvalQueue
    from tabx_dispatcher import EvalDispatcher
    queue = EvalQueue(tmp_path)
    dispatcher = EvalDispatcher(queue, "loss", "batch", workers=1, prefetch=1)
    request = dict(rule="loss", batch_id="batch", checkpoint_hash="weights", seeds=[7])
    key = queue.enqueue(request)
    try:
        dispatcher.restore()
        dispatcher.poll()
        _, owner, _ = queue.claim("loss", batch_id="batch")
        queue.finish(key, owner, "worker failed")
        queue.retry_failed("loss", "batch")
        dispatcher.poll()
        assert queue.active_keys("loss", "batch") == {key: "pending"}
    finally:
        dispatcher.close()


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", default="artifacts/tabx/scheduler-scale-validation.json")
    cli.add_argument("--polls", type=int, default=100)
    opts = cli.parse_args()
    result = run_scale_benchmark(opts.output, opts.polls)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
