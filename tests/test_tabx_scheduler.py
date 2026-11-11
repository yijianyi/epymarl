"""Batch orchestration tests use real subprocesses with lightweight producers.

GPU allocation is simulated here; official QMIX/CPU rollouts are separately
covered by the opt-in test_tabx_revision_smoke.py integration test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest

from tabx_pipeline import parser
from tabx_reporting import evaluation_summary, read_csv, refresh_eval_summary
import tabx_scheduler as scheduler
from tabx_storage import EvalQueue, atomic_json, completed_checkpoints, read_json
from test_tabx_official import tiny_task


def test_file_queue_scoping_stable_ids_and_partial_retry(tmp_path):
    payload = dict(rule="hp", seeds=[1], checkpoint_hash="old")
    original_id = EvalQueue.request_id(payload)
    queue = EvalQueue(tmp_path)
    assert queue.enqueue(payload) == original_id
    first = queue.enqueue(dict(payload, batch_id="A"), 1)
    additional = queue.enqueue(dict(payload, batch_id="A", seeds=[2]), 1)
    assert additional is not None
    second = queue.enqueue(dict(payload, batch_id="B"), 1)
    assert queue.claim("loss", batch_id="A") is None
    assert queue.claim("hp", batch_id="missing") is None
    key, owner, _ = queue.claim("hp", batch_id="B")
    assert key == second
    queue.record(key, owner, dict(seed=1, return_value=3))
    queue.finish(key, owner, "interrupted")
    assert queue.retry_failed("hp", "A") == 0
    assert queue.retry_failed("hp", "B") == 1
    assert queue.results(second) == [dict(seed=1, return_value=3)]
    assert queue.claim("hp", batch_id="A")[0] in {first, additional}
    assert queue.release_batch("hp", "B") == 0
    assert queue.release_batch("hp", "A") == 1
    assert set(queue.statuses([original_id, first, second])) == {original_id, first, second}
    assert queue.claim("hp")[0] == original_id  # Legacy workers retain their old behavior.
    assert queue.claim("hp") is None  # An unbatched worker cannot steal batch requests.


@pytest.mark.parametrize("gpus,environment", [
    ("0,0", {}), ("0,00", {}), ("0,", {}), ("-1", {}),
    ("0,2", {"CUDA_VISIBLE_DEVICES": "0,1"}), ("0", {"CUDA_VISIBLE_DEVICES": ""}),
    ("0", {"NVIDIA_VISIBLE_DEVICES": "GPU-allocated"}),
])
def test_reject_duplicate_and_unallocated_gpus(gpus, environment):
    with pytest.raises(ValueError):
        scheduler.gpu_ids(gpus, environment)


def test_gpu_probe_and_training_never_fall_back_to_cpu(monkeypatch):
    assert scheduler.gpu_ids("2,3", {"CUDA_VISIBLE_DEVICES": "2,3", "NVIDIA_VISIBLE_DEVICES": "all"}) == ["2", "3"]
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **kw: NS(returncode=1, stdout="CUDA unavailable"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        scheduler.probe_gpus(["2"])
    import run
    monkeypatch.setattr(run.th.cuda, "is_available", lambda: False)
    config = dict(use_cuda=True, tabx_require_cuda=True, test_nepisode=1, batch_size_run=1)
    with pytest.raises(RuntimeError, match="requires an available training GPU"):
        run.args_sanity_check(config, NS(warning=lambda *a: None))
    assert config["use_cuda"] is True


@pytest.mark.parametrize("setting", ["use_cuda=False", "env_args.key=23", "tabx_batch_id=other", "checkpoint_path=old",
    "save_model=False", "tabx_async_eval=False", "t_max=0", "lr=-1", "batch_size=513", "unknown=1"])
def test_reject_overrides_that_break_batch_contract(setting):
    with pytest.raises(ValueError):
        scheduler.training_config([setting])


def test_evaluation_statistics_and_checkpoint_step(tmp_path):
    root, run = tmp_path, tmp_path / "hp/23/1/run"
    snapshot = run / "checkpoints/25048"
    request = dict(seeds=[1, 2, 3], phase="screen", action_selection="greedy", run_id="run",
                   checkpoint_hash="weights", snapshot=str(snapshot.relative_to(root)))
    report = dict(request=request, episodes=[dict(seed=1, return_value=-1, is_win=0, end_reason=2),
        dict(seed=2, return_value=1, is_win=1, end_reason=1), dict(seed=3, return_value=3, is_win=1, end_reason=2)])
    row = evaluation_summary("abc", report, 25048)
    assert row["wins"] == 2 and row["win_rate_pct"] == pytest.approx(200 / 3)
    assert row["mean_return"] == 1 and row["return_std"] == 2
    assert row["elimination_wins"] == row["timeout_wins"] == 1 and row["pure_timeouts"] == 2
    atomic_json(snapshot / "manifest.json", dict(t_env=25048, run_id="run", agent_sha256="weights"))
    atomic_json(run / "evaluations/abc.json", report)
    refresh_eval_summary(run)
    refresh_eval_summary(run)
    rows = read_csv(run / "eval_summary.csv")
    assert len(rows) == 1 and rows[0]["checkpoint_step"] == "25048"
    assert not list(run.glob("*.png"))


FAKE_TRAIN = r'''
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,sys.argv[1])
from tabx_storage import SnapshotPublisher,atomic_json
from tabx_reporting import append_train_metrics
from envs.tabx_layout import LAYOUT_VERSION,SIMULATOR_CONTRACT
job=json.loads(sys.argv[2]); run=Path(job['run']); run.mkdir(parents=True)
atomic_json(run/'manifest.json',dict(task_id=job['task_id'],train_seed=job['seed'],timeout_rule=job['rule'],
    bank_hash=job['bank_hash'],task_hash=job['task_hash'],batch_id=job['batch_id'],run_id=job['run_id'],master_seed=0,
    provenance={},native_shape=[1,1,0]))
atomic_json(run/'config.json',dict(env_args=dict(timeout_rule=job['rule'])))
(run/'train.log').symlink_to(os.path.relpath(job['log'],run))
atomic_json(run/'assigned.json',dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],jax=os.environ['JAX_PLATFORMS'],pid=os.getpid()))
if job['fail']:
    time.sleep(.2); sys.exit(5)
args=NS(tabx_output_root=job['root'],tabx_run_dir=str(run),save_model_interval=8,tabx_screen_episodes=2,
    tabx_max_pending_evals=1,tabx_batch_id=job['batch_id'])
class Learner:
    training_steps=0
    def save_models(self,path):
        for name in ('agent.th','mixer.th','opt.th'):
            (Path(path)/name).write_bytes(str(self.training_steps).encode())
learner=Learner(); publisher=SnapshotPublisher(args,None)
publisher.publish(learner,0,force_eval=True)
for step in (8,16):
    time.sleep(.2); learner.training_steps+=1
    append_train_metrics(run/'train_metrics.csv',dict(t_env=step,learner_updates=learner.training_steps,loss=3.8,
        td_error_abs=1.2,grad_norm_before_clip=18.4,grad_norm_clip=10,q_taken_mean=.2,target_mean=.1,lr=.00005))
    with (run/'training_episodes.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(t_env=step,end_reason=2,epsilon=.99))+'\n')
    publisher.publish(learner,step)
publisher.complete(learner,16,dict(t_env=16,learner_updates=2,elapsed_seconds=.4,peak_cuda_bytes=123))
'''


FAKE_WORKER = r'''
import json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tabx_storage import EvalQueue,atomic_json,read_json,resolve_under
queue=EvalQueue(sys.argv[2])
while True:
    claim=queue.claim(sys.argv[3],batch_id=sys.argv[4])
    if claim is None:
        time.sleep(.02); continue
    key,owner,request=claim
    time.sleep(.25)
    finished={row['seed'] for row in queue.results(key)}
    for seed in request['seeds']:
        if seed not in finished:
            queue.record(key,owner,dict(seed=seed,return_value=-.42,is_win=0,end_reason=2))
            with (queue.root/'executions.jsonl').open('a') as stream:
                stream.write(json.dumps([key,seed])+'\n')
    run=resolve_under(queue.root,request['run'])
    try:
        queue.publish_result(key,owner,dict(request=request,episodes=queue.results(key),opponent_difficulty=request.get("opponent_difficulty","medium"),evaluated_at=time.time()))
        queue.finish(key,owner)
    except RuntimeError:
        # Match the real worker: reconciliation may adopt the final JSON before
        # this worker publishes its completion notification and release its lease.
        if queue.completed_result(key) is None:
            raise
'''


@pytest.fixture
def batch_fixture(monkeypatch, tmp_path):
    tasks = []
    for key in ("23", "651", "647"):
        task = tiny_task()
        task["task_id"] = key
        tasks.append(task)
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=tasks))
    root = tmp_path / "output"
    arguments = ["batch-train", "--bank", str(bank), "--root", str(root), "--tasks", "23,651,647",
                 "--gpus", "0,1", "--eval-workers", "2", "--batch-id", "test", "--rule", "loss",
                 "--poll-seconds", "0.02", "--set", "tabx_max_pending_evals=1", "--set", "tabx_screen_episodes=2"]
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(scheduler, "probe_gpus", lambda gpus: {gpu: dict(name="simulated") for gpu in gpus})
    failures = set()
    original_command = scheduler.BatchScheduler.training_command
    def command(self, job, attempt):
        actual = original_command(self, job, attempt)
        assert f'env_args.key="{job["task_id"]}"' in actual
        assert "tabx_require_cuda=True" in actual and "use_cuda=True" in actual
        data = dict(run=str(self.root / attempt["run"]), log=str(self.root / attempt["log"]), root=str(self.root),
                    run_id=attempt["run_id"], task_id=job["task_id"], seed=job["train_seed"], rule=self.args.rule,
                    bank_hash=self.manifest["bank_hash"], task_hash=self.manifest["task_hashes"][job["task_id"]],
                    batch_id=self.args.batch_id, fail=job["task_id"] in failures)
        return [sys.executable, "-c", FAKE_TRAIN, str(scheduler.REPOSITORY / "src"), json.dumps(data)]
    def start_workers(self):
        # Test a real child process without initializing CUDA/JAX.
        assert (self.root / "tabx_semantics.json").is_file()
        self.worker_log = (self.directory / "eval-workers.log").open("a", encoding="utf-8")
        self.worker = subprocess.Popen([sys.executable, "-c", FAKE_WORKER, str(scheduler.REPOSITORY / "src"),
            str(self.root), self.args.rule, self.args.batch_id], start_new_session=True,
            stdout=self.worker_log, stderr=subprocess.STDOUT)
        self.state["worker"] = dict(pid=self.worker.pid, process_start=scheduler.process_identity(self.worker.pid))
    monkeypatch.setattr(scheduler.BatchScheduler, "training_command", command)
    monkeypatch.setattr(scheduler.BatchScheduler, "start_workers", start_workers)
    return NS(root=root, bank=bank, argv=arguments, failures=failures, production_command=original_command,
              directory=root / "_batches/loss/test", monkeypatch=monkeypatch)


def test_two_slots_refill_backfill_complete_and_idempotent_resume(batch_fixture):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    attempts = [job["attempts"][-1] for job in state["jobs"]]
    assert [row["gpu"] for row in attempts[:2]] == ["0", "1"]
    assert attempts[1]["started_at"] < attempts[0]["ended_at"]
    assert attempts[2]["started_at"] >= min(row["ended_at"] for row in attempts[:2])
    for job, attempt in zip(state["jobs"], attempts):
        run = f.root / attempt["run"]
        assert completed_checkpoints(run)[0]["final_t_env"] == 16
        assigned = read_json(run / "assigned.json")
        assert assigned["gpu"] == attempt["gpu"] and assigned["jax"] == "cpu"
        assert (run / "train.log").read_text().startswith("[START]")
        assert len(read_csv(run / "eval_summary.csv")) == 3  # Every skipped snapshot was backfilled.
        assert not scheduler.process_alive(attempt)
    assert len(list(EvalQueue(f.root).directory("loss", "test").glob("archive/complete/*.json"))) == 9
    assert not list(f.root.rglob("eval_queue.sqlite3"))
    summary = read_csv(f.directory / "batch_summary.csv")
    assert len(summary) == 3 and {row["status"] for row in summary} == {"complete"}
    text = (f.directory / "batch.log").read_text()
    assert "[TRAIN] rule=loss" in text and "checkpoint_step=8" in text and "win_rate=0.0%" in text
    # Resource allocation may change; a finished model must not train again.
    resumed = parser().parse_args(f.argv + ["--resume", "--jobs-per-gpu", "2", "--eval-workers", "1", "--gpus", "1"])
    assert scheduler.batch_train(resumed) == 0
    assert [job["attempts"] for job in read_json(f.directory / "state.json")["jobs"]] == [job["attempts"] for job in state["jobs"]]
    with pytest.raises(ValueError, match="Resume requires identical"):
        scheduler.batch_train(parser().parse_args(f.argv + ["--resume", "--set", "t_max=123"]))


def test_failed_model_does_not_block_others_and_resume_retries_only_failure(batch_fixture):
    f = batch_fixture
    f.failures.add("651")
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    before = read_json(f.directory / "state.json")
    assert [job["status"] for job in before["jobs"]] == ["complete", "failed", "complete"]
    failed_log = f.root / before["jobs"][1]["attempts"][0]["log"]
    assert failed_log.is_file()
    f.failures.clear()
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    after = read_json(f.directory / "state.json")
    assert [len(job["attempts"]) for job in after["jobs"]] == [1, 2, 1]
    assert after["jobs"][1]["attempts"][0]["exit_code"] == 5 and failed_log.is_file()


def test_interruption_reaps_owned_children_and_restarts_as_new_attempt(batch_fixture, monkeypatch):
    f = batch_fixture
    original_fill = scheduler.BatchScheduler.fill_slots
    def stop_after_launch(self):
        original_fill(self)
        self.stopping = True
    monkeypatch.setattr(scheduler.BatchScheduler, "fill_slots", stop_after_launch)
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 130
    before = read_json(f.directory / "state.json")
    assert before["status"] == "interrupted"
    for job in before["jobs"]:
        for attempt in job["attempts"]:
            assert not scheduler.process_alive(attempt)
    monkeypatch.setattr(scheduler.BatchScheduler, "fill_slots", original_fill)
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    after = read_json(f.directory / "state.json")
    assert [len(job["attempts"]) for job in after["jobs"]] == [2, 2, 1]
    assert after["jobs"][0]["attempts"][0]["run_id"] != after["jobs"][0]["attempts"][1]["run_id"]


def test_task_sources_locking_and_hash_validation(batch_fixture, tmp_path):
    f = batch_fixture
    args = parser().parse_args(f.argv)
    _, tasks, digest, _ = scheduler.task_selection(args)
    pilot = tmp_path / "pilot.json"
    atomic_json(pilot, dict(bank_hash=digest, pilot=tasks, splits=dict(train=tasks)))
    args.tasks, args.pilot = None, str(pilot)
    assert scheduler.task_selection(args)[1] == tasks
    atomic_json(pilot, dict(bank_hash="wrong", pilot=tasks, splits=dict(train=tasks)))
    with pytest.raises(ValueError, match="hash mismatch"):
        scheduler.task_selection(args)
    listing = tmp_path / "tasks.txt"
    listing.write_text("23\n651,647\n")
    args.pilot, args.task_list = None, str(listing)
    assert scheduler.task_selection(args)[1] == tasks
    listing.write_text("23\n23")
    with pytest.raises(ValueError, match="repeat"):
        scheduler.task_selection(args)
    with scheduler.batch_lock(f.directory):
        with pytest.raises(RuntimeError, match="Another scheduler"):
            scheduler.batch_train(parser().parse_args(f.argv))


def test_zero_exit_without_completion_inventory_is_failed(batch_fixture, monkeypatch):
    f = batch_fixture
    monkeypatch.setattr(scheduler.BatchScheduler, "training_command", lambda *a: [sys.executable, "-c", "pass"])
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    state = read_json(f.directory / "state.json")
    assert all(job["status"] == "failed" for job in state["jobs"])
    assert all("Missing training_complete.json" in job["attempts"][-1]["error"] for job in state["jobs"])


def test_worker_failure_finishes_training_and_returns_failure_without_hanging(batch_fixture, monkeypatch):
    f = batch_fixture
    def dead_worker(self):
        self.worker = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], start_new_session=True)
    monkeypatch.setattr(scheduler.BatchScheduler, "start_workers", dead_worker)
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    state = read_json(f.directory / "state.json")
    assert "code 3" in state["worker_error"]
    assert all(completed_checkpoints(f.root / job["attempts"][-1]["run"]) is not None for job in state["jobs"])


def test_preflight_failure_is_logged_and_launches_no_training(batch_fixture, monkeypatch):
    f = batch_fixture
    def unavailable(gpus):
        raise RuntimeError("GPU 0 unavailable")
    monkeypatch.setattr(scheduler, "probe_gpus", unavailable)
    with pytest.raises(RuntimeError, match="GPU 0 unavailable"):
        scheduler.batch_train(parser().parse_args(f.argv))
    state = read_json(f.directory / "state.json")
    assert state["status"] == "failed" and all(not job["attempts"] for job in state["jobs"])
    assert "GPU 0 unavailable" in (f.directory / "batch.log").read_text()


def test_reject_overlapping_task_sources(batch_fixture):
    with pytest.raises(SystemExit):
        parser().parse_args(batch_fixture.argv + ["--pilot", "pilot.json"])


def test_multiple_jobs_per_gpu_use_multiple_slots(batch_fixture):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--jobs-per-gpu", "2"])) == 0
    jobs = read_json(f.directory / "state.json")["jobs"]
    attempts = [job["attempts"][-1] for job in jobs]
    assert [attempt["gpu"] for attempt in attempts] == ["0", "1", "0"]
    assert max(attempt["started_at"] for attempt in attempts) < min(attempt["ended_at"] for attempt in attempts)


def test_resume_refuses_a_still_live_old_worker(batch_fixture):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    state["worker"] = dict(pid=os.getpid(), process_start=scheduler.process_identity(os.getpid()))
    atomic_json(f.directory / "state.json", state)
    original_state = (f.directory / "state.json").read_bytes()
    original_csv = (f.directory / "batch_summary.csv").read_bytes()
    with pytest.raises(RuntimeError, match="Previous batch worker PID"):
        scheduler.batch_train(parser().parse_args(f.argv + ["--resume"]))
    assert (f.directory / "state.json").read_bytes() == original_state
    assert (f.directory / "batch_summary.csv").read_bytes() == original_csv


def test_resume_uses_per_job_files_newer_than_batch_snapshot(batch_fixture):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    old = dict(state, jobs=[dict(job_id=job["job_id"], task_id=job["task_id"], train_seed=job["train_seed"],
                                status="pending", attempts=[]) for job in state["jobs"]])
    atomic_json(f.directory / "state.json", old)
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    recovered = read_json(f.directory / "state.json")
    assert all(job["status"] == "complete" and len(job["attempts"]) == 1 for job in recovered["jobs"])


def test_late_evaluation_of_previous_attempt_is_announced(batch_fixture, capsys):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    job = state["jobs"][0]
    old = job["attempts"][-1]
    run = f.root / old["run"]
    report = read_json(next((run / "evaluations").glob("*.json")))
    report["request"]["phase"] = "confirm"
    key = EvalQueue.request_id(report["request"])
    atomic_json(run / "evaluations" / f"{key}.json", report)
    refresh_eval_summary(run)
    job["attempts"].append(dict(old, run_id="replacement", run=str(Path(old["run"]).parent / "replacement")))
    capsys.readouterr()
    monitor = scheduler.BatchScheduler(parser().parse_args(f.argv), f.directory, read_json(f.directory / "batch.json"), state)
    try:
        monitor.prepare_resume()
        monitor.refresh_reports({})
    finally:
        monitor.log.close()
    text = capsys.readouterr().out
    assert "phase=confirm" in text and f"run_id={old['run_id']}" in text


def test_completed_requests_have_zero_history_access_in_100_idle_polls(batch_fixture, monkeypatch):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    monitor = scheduler.BatchScheduler(parser().parse_args(f.argv + ["--resume"]), f.directory,
        read_json(f.directory / "batch.json"), read_json(f.directory / "state.json"))
    try:
        monitor.prepare_resume()
        monitor.evaluations()
        monitor.refresh_reports(force=True)
        assert len(monitor.records) == 9 and all(row["valid"] for row in monitor.records.values())
        accesses, enqueues = [], []
        original_open = Path.open
        original_enqueue = monitor.queue.enqueue
        def watch_open(path, *args, **kwargs):
            text = str(path)
            if path.suffix == ".json" and any(part in text for part in ("/evaluations/", "/checkpoints/", "/evaluation_requests/", "/_eval_requests/")):
                accesses.append(text)
            return original_open(path, *args, **kwargs)
        def watch_enqueue(*args, **kwargs):
            enqueues.append(args)
            return original_enqueue(*args, **kwargs)
        monkeypatch.setattr(Path, "open", watch_open)
        monkeypatch.setattr(monitor.queue, "enqueue", watch_enqueue)
        for _ in range(100):
            monitor.evaluations()
            monitor.refresh_reports()
        assert enqueues == [] and accesses == []
    finally:
        monitor.log.close()


def test_two_workers_racing_one_request_execute_it_once(tmp_path):
    queue = EvalQueue(tmp_path)
    payload = dict(rule="hp", seeds=[7], checkpoint_hash="weights")
    key = queue.enqueue(payload)
    script = r'''
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tabx_storage import EvalQueue
root=Path(sys.argv[2]); queue=EvalQueue(root)
(root/f'ready-{os.getpid()}').touch()
while not (root/'go').exists(): time.sleep(.01)
claim=queue.claim('hp')
if claim is not None:
    key,owner,request=claim
    with (root/'executions').open('a') as stream: stream.write(f'{os.getpid()}\n')
    time.sleep(.2)
    queue.record(key,owner,dict(seed=7,return_value=1))
    queue.finish(key,owner)
'''
    processes = [subprocess.Popen([sys.executable, "-c", script, str(scheduler.REPOSITORY / "src"), str(tmp_path)])
                 for _ in range(2)]
    try:
        deadline = time.monotonic() + 10
        while len(list(tmp_path.glob("ready-*"))) < 2:
            assert time.monotonic() < deadline
            time.sleep(.01)
        (tmp_path / "go").touch()
        assert all(process.wait(timeout=10) == 0 for process in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert len((tmp_path / "executions").read_text().splitlines()) == 1
    assert queue.status(key)["status"] == "complete"
    assert queue.results(key) == [dict(seed=7, return_value=1)]
    assert not list(tmp_path.rglob("eval_queue.sqlite3"))


def test_publication_is_idempotent_while_running_and_after_acknowledgement(tmp_path):
    queue = EvalQueue(tmp_path)
    request = dict(rule="loss", seeds=[1], checkpoint_hash="weights", batch_id="batch")
    key = queue.enqueue(request)
    _, owner, _ = queue.claim("loss", batch_id="batch")
    for _ in range(10):
        assert queue.enqueue(request) == key
    assert not list((queue.directory("loss", "batch") / "pending").glob("*.json"))
    queue.record(key, owner, dict(seed=1, return_value=1))
    queue.finish(key, owner)
    queue.acknowledge(queue.notifications("loss", "batch")[0])
    for _ in range(10):
        assert queue.enqueue(request) == key
    assert queue.notifications("loss", "batch") == []
    assert queue.active_keys("loss", "batch") == {}


def test_partial_20_of_32_seeds_and_final_json_before_notification_recover(tmp_path):
    queue = EvalQueue(tmp_path)
    request = dict(rule="loss", seeds=list(range(32)), checkpoint_hash="weights", batch_id="batch")
    key = queue.enqueue(request)
    _, owner, _ = queue.claim("loss", batch_id="batch")
    for seed in range(20):
        queue.record(key, owner, dict(seed=seed, return_value=1))
    queue.release_batch("loss", "batch")
    recovered = EvalQueue(tmp_path)
    assert recovered.recover(request)["status"] == "pending"
    _, next_owner, _ = recovered.claim("loss", batch_id="batch")
    existing = {row["seed"] for row in recovered.results(key)}
    missing = set(request["seeds"]) - existing
    assert existing == set(range(20)) and len(missing) == 12
    for seed in sorted(missing):
        recovered.record(key, next_owner, dict(seed=seed, return_value=1))
    recovered.publish_result(key, next_owner, dict(request=request, episodes=recovered.results(key)))
    # Crash before finish(): the final JSON is authoritative on the next resume.
    recovered.release_batch("loss", "batch")
    result = EvalQueue(tmp_path).recover(request)
    assert result["status"] == "complete" and len(result["report"]["episodes"]) == 32
    assert len(EvalQueue(tmp_path).notifications("loss", "batch")) == 1


def test_exact_screen_ids_are_required_and_completed_model_survives_pool_exit():
    attempt = dict(train_status="complete", inventory_verified=True, eval_expected=3,
                   screen_requests={"0": "s0", "1000": "s1", "2000": "s2"}, eval_interrupted=True,
                   evaluations={"s0": dict(status="complete", valid=True),
                                "confirm1000": dict(status="complete", valid=True)})
    derived = scheduler.derive_job_status(attempt)
    assert derived["eval_done"] == 1 and derived["eval_remaining"] == 2
    assert derived["status"] == "evaluation_interrupted"
    attempt["evaluations"].update(s1=dict(status="complete", valid=True), s2=dict(status="complete", valid=True))
    assert scheduler.derive_job_status(attempt)["status"] == "complete"
    attempt.update(train_status="running", inventory_verified=False, eval_expected=None)
    assert scheduler.derive_job_status(attempt)["status"] == "running"


FAULT_WORKER = r'''
import json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tabx_storage import EvalQueue
queue=EvalQueue(sys.argv[2]); deadline=time.monotonic()+20
while len(list((queue.root/'loss').glob('*/*/*/training_complete.json'))) < 3:
    if time.monotonic()>deadline: raise RuntimeError('training never sealed')
    time.sleep(.02)
def record(key,owner,seed):
    queue.record(key,owner,dict(seed=seed,return_value=-.42,is_win=0,end_reason=2))
    with (queue.root/'executions.jsonl').open('a') as stream: stream.write(json.dumps([key,seed])+'\n')
completed=set()
while len(completed)<4:
    if time.monotonic()>deadline: raise RuntimeError('target requests never admitted')
    for key,status in queue.active_keys('loss','test').items():
        request=queue.info(key)['request']; step=int(Path(request['snapshot']).name)
        if status=='pending' and (request['task_id']=='651' or (request['task_id']=='23' and step==0)):
            claim=queue.claim('loss',batch_id='test',request_id=key)
            if claim is None: continue
            key,owner,request=claim
            for seed in request['seeds']: record(key,owner,seed)
            queue.publish_result(key,owner,dict(request=request,episodes=queue.results(key),opponent_difficulty=request.get("opponent_difficulty","medium"),evaluated_at=time.time()))
            queue.finish(key,owner); completed.add(key)
    time.sleep(.02)
while True:
    if time.monotonic()>deadline: raise RuntimeError('partial target never admitted')
    target=next((key for key,status in queue.active_keys('loss','test').items()
        if status=='pending' and queue.info(key)['request']['task_id']=='23'
        and int(Path(queue.info(key)['request']['snapshot']).name)==8),None)
    if target is not None:
        key,owner,request=queue.claim('loss',batch_id='test',request_id=target)
        record(key,owner,request['seeds'][0]); break
    time.sleep(.02)
raise SystemExit(3)
'''


def test_interrupted_evaluation_state_logs_and_resume_only_missing_seeds(batch_fixture, monkeypatch):
    f = batch_fixture
    normal_workers = scheduler.BatchScheduler.start_workers
    def fault_workers(self):
        self.worker = subprocess.Popen([sys.executable, "-c", FAULT_WORKER, str(scheduler.REPOSITORY / "src"),
            str(self.root)], start_new_session=True)
        self.state["worker"] = dict(pid=self.worker.pid, process_start=scheduler.process_identity(self.worker.pid))
    monkeypatch.setattr(scheduler.BatchScheduler, "start_workers", fault_workers)
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    before = read_json(f.directory / "state.json")
    jobs = {job["task_id"]: job for job in before["jobs"]}
    row = jobs["23"]
    assert row["train_status"] == "complete" and row["eval_status"] == "interrupted"
    assert (row["status"], row["eval_done"], row["eval_expected"], row["eval_remaining"]) == ("evaluation_interrupted", 1, 3, 2)
    assert jobs["651"]["status"] == "complete"
    csv_rows = {row["task_id"]: row for row in read_csv(f.directory / "batch_summary.csv")}
    for key, job in jobs.items():
        for field in ("status", "train_status", "eval_status", "eval_done", "eval_expected", "eval_remaining"):
            assert csv_rows[key][field] == str(job[field])
            assert job["attempts"][-1][field] == job[field]
    assert "eval_done=1 eval_expected=3 eval_remaining=2" in (f.directory / "batch.log").read_text()
    executions_before = [tuple(json.loads(line)) for line in (f.root / "executions.jsonl").read_text().splitlines()]
    assert len(executions_before) == 9  # Four full requests and one partial seed.
    monkeypatch.setattr(scheduler.BatchScheduler, "start_workers", normal_workers)
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    after = read_json(f.directory / "state.json")
    assert all(job["status"] == "complete" and len(job["attempts"]) == 1 for job in after["jobs"])
    executions = [tuple(json.loads(line)) for line in (f.root / "executions.jsonl").read_text().splitlines()]
    assert len(executions) == len(set(executions)) == 18  # Reused the partial seed; no repeat execution.
    assert not list(f.root.rglob("eval_queue.sqlite3"))


def _watch_eval_summary_writes(monkeypatch):
    import tabx_reporting
    writes = []
    original = tabx_reporting.write_csv
    def write(path, *args, **kwargs):
        if Path(path).name == "eval_summary.csv":
            writes.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(tabx_reporting, "write_csv", write)
    return writes


def test_resume_writes_zero_correct_csvs_repairs_one_run_once_then_writes_zero(batch_fixture, monkeypatch):
    import tabx_reporting
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    paths = [f.root / job["attempts"][-1]["run"] / "eval_summary.csv" for job in state["jobs"]]
    assert sum(len(read_csv(path)) for path in paths) == 9
    stamps = {path: path.stat().st_mtime_ns for path in paths}
    writes = _watch_eval_summary_writes(monkeypatch)
    reads = []
    original_open = Path.open
    def open_csv(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.name == "eval_summary.csv" and "r" in mode:
            reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_csv)
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    assert writes == []
    assert sorted(reads) == sorted(paths)
    assert {path: path.stat().st_mtime_ns for path in paths} == stamps

    # One CSV simultaneously loses two rows and has an incorrect win rate.
    target = paths[0]
    row = dict(read_csv(target)[0], win_rate_pct="91.0")
    tabx_reporting.write_csv(target, tabx_reporting.EVAL_FIELDS, [row])
    writes.clear()
    reads.clear()
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    assert writes == [target]
    assert sorted(reads) == sorted(paths)
    assert {path: path.stat().st_mtime_ns for path in paths[1:]} == {path: stamps[path] for path in paths[1:]}
    repaired = read_csv(target)
    assert len(repaired) == 3
    for row in repaired:
        report = read_json(target.parent / "evaluations" / f"{row['request_id']}.json")
        episodes = report["episodes"]
        assert float(row["win_rate_pct"]) == 100 * sum(ep["is_win"] for ep in episodes) / len(episodes)
        assert float(row["mean_return"]) == pytest.approx(sum(ep["return_value"] for ep in episodes) / len(episodes))

    writes.clear()
    reads.clear()
    stamps = {path: path.stat().st_mtime_ns for path in paths}
    assert scheduler.batch_train(parser().parse_args(f.argv + ["--resume"])) == 0
    assert writes == [] and {path: path.stat().st_mtime_ns for path in paths} == stamps
    assert sorted(reads) == sorted(paths)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "header", "value", "order"])
def test_reconcile_eval_summary_rewrites_bad_csv_once(damage, tmp_path, monkeypatch):
    import tabx_reporting as reporting
    run = tmp_path / "run"
    run.mkdir()
    report = dict(request=dict(seeds=[1, 2], phase="screen", action_selection="greedy"),
                  episodes=[dict(seed=1, is_win=1, end_reason=1, return_value=1),
                            dict(seed=2, is_win=0, end_reason=2, return_value=-1)])
    rows = {key: evaluation_summary(key, report, step) for key, step in (("a", 0), ("b", 8))}
    path = run / "eval_summary.csv"
    reporting.write_csv(path, reporting.EVAL_FIELDS, rows.values())
    if damage == "missing":
        path.unlink()
    elif damage == "duplicate":
        reporting.write_csv(path, reporting.EVAL_FIELDS, [rows["a"], rows["a"], rows["b"]])
    elif damage == "header":
        reporting.write_csv(path, list(reversed(reporting.EVAL_FIELDS)), rows.values())
    elif damage == "value":
        reporting.write_csv(path, reporting.EVAL_FIELDS, [dict(rows["a"], mean_return=91), rows["b"]])
    elif damage == "order":
        reporting.write_csv(path, reporting.EVAL_FIELDS, list(reversed(list(rows.values()))))
    writes = _watch_eval_summary_writes(monkeypatch)
    assert reporting.reconcile_eval_summary(run, rows) == "written"
    assert writes == [path]
    stamp = path.stat().st_mtime_ns
    writes.clear()
    assert reporting.reconcile_eval_summary(run, rows) == "unchanged"
    assert writes == [] and path.stat().st_mtime_ns == stamp


def test_reconcile_empty_results_does_not_create_empty_csv(tmp_path, monkeypatch):
    from tabx_reporting import reconcile_eval_summary
    run = tmp_path / "no-run-yet"
    writes = _watch_eval_summary_writes(monkeypatch)
    assert reconcile_eval_summary(run, {}) == "unchanged"
    run.mkdir()
    assert reconcile_eval_summary(run, {}) == "unchanged"
    assert writes == [] and not (run / "eval_summary.csv").exists()


def test_recover_and_retry_failed_forward_defer_summary(tmp_path, monkeypatch):
    queue = EvalQueue(tmp_path)
    request = dict(rule="loss", seeds=[7], checkpoint_hash="weights")
    key = queue.enqueue(request)
    _, owner, _ = queue.claim("loss")
    queue.record(key, owner, dict(seed=7, return_value=1))
    report = dict(request=request, episodes=queue.results(key))
    queue.publish_result(key, owner, report)
    queue.finish(key, owner, "crashed before completion notification")
    summaries = []
    monkeypatch.setattr(queue, "_summary", lambda *args: summaries.append(args))
    assert queue.retry_failed("loss", defer_summary=True) == 1
    assert summaries == [] and queue.status(key)["status"] == "complete"
    assert queue.recover(request, defer_summary=True)["report"] == report
    assert summaries == []
    assert queue.recover(request)["report"] == report
    assert len(summaries) == 1


def _finish_synthetic_request(queue, request):
    key = queue.enqueue(request)
    queue.expose(key)
    _, owner, _ = queue.claim(request["rule"], batch_id=request.get("batch_id"), request_id=key)
    for index, seed in enumerate(request["seeds"]):
        queue.record(key, owner, dict(seed=seed, return_value=float(index), is_win=int(index > 0),
                                     end_reason=1 if index > 0 else 2))
    queue.publish_result(key, owner, dict(request=request, episodes=queue.results(key)))
    queue.finish(key, owner)
    return key


def test_resume_keeps_all_phases_excludes_partial_and_worker_still_updates_csv(batch_fixture, monkeypatch, capsys):
    from tabx_storage import evaluation_request
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    run = f.root / state["jobs"][0]["attempts"][-1]["run"]
    queue = EvalQueue(f.root)
    complete_ids = {row["request_id"] for row in read_csv(run / "eval_summary.csv")}
    for phase, step, count, random_policy in (("confirm", 8, 4, False), ("baseline", 0, 2, True), ("final", 8, 2, False)):
        request = evaluation_request(f.root, run, run / "checkpoints" / str(step), phase, count, random_policy=random_policy)
        complete_ids.add(_finish_synthetic_request(queue, request))
    partial_request = evaluation_request(f.root, run, run / "checkpoints/16", "screen", 2)
    partial_key = queue.request_id(partial_request)
    complete_ids.remove(partial_key)
    queue.result_path(partial_key).unlink()
    info = queue.info(partial_key)
    partial_dir = queue._evaluation_dir(info) / "partial" / partial_key
    (partial_dir / f"{partial_request['seeds'][1]}.json").unlink()

    writes = _watch_eval_summary_writes(monkeypatch)
    resume_state = read_json(f.directory / "state.json")
    # This request represents an evaluation that never reached a completion
    # announcement; the completed fixture above only supplies its checkpoint.
    resume_state["jobs"][0]["attempts"][-1]["evaluations"][partial_key]["announced"] = False
    monitor = scheduler.BatchScheduler(parser().parse_args(f.argv + ["--resume"]), f.directory,
        read_json(f.directory / "batch.json"), resume_state)
    result_reads = {}
    original_read = monitor.queue._read
    def read_result(path):
        if Path(path).parent.name == "evaluations" and Path(path).suffix == ".json":
            result_reads[Path(path)] = result_reads.get(Path(path), 0) + 1
        return original_read(path)
    monkeypatch.setattr(monitor.queue, "_read", read_result)
    try:
        monitor.prepare_resume()
        assert len(result_reads) == 11 and set(result_reads.values()) == {1}
        assert writes == [run / "eval_summary.csv"]
        rows = read_csv(run / "eval_summary.csv")
        assert {row["request_id"] for row in rows} == complete_ids
        assert {row["phase"] for row in rows} == {"screen", "confirm", "baseline", "final"}
        assert sum(row["checkpoint_step"] == "8" for row in rows) == 3
        assert monitor.jobs[0]["eval_done"] == 2 and monitor.jobs[0]["eval_expected"] == 3
        assert [row["seed"] for row in monitor.queue.results(partial_key)] == partial_request["seeds"][:1]
        monitor.evaluations()
        _, owner, _ = monitor.queue.claim("loss", batch_id="test", request_id=partial_key)
        seed = partial_request["seeds"][1]
        monitor.queue.record(partial_key, owner, dict(seed=seed, return_value=1, is_win=1, end_reason=1))
        monitor.queue.publish_result(partial_key, owner, dict(request=partial_request, episodes=monitor.queue.results(partial_key)))
        writes.clear()
        monitor.queue.finish(partial_key, owner)
        assert writes == [run / "eval_summary.csv"]
        assert {row["request_id"] for row in read_csv(run / "eval_summary.csv")} == complete_ids | {partial_key}
        capsys.readouterr()
        monitor.evaluations()
        monitor.refresh_reports()
        assert "[EVAL]" in capsys.readouterr().out and monitor.jobs[0]["status"] == "complete"
    finally:
        if monitor.dispatcher is not None:
            monitor.dispatcher.close()
        if monitor.log is not None:
            monitor.log.close()


def test_failed_run_validation_does_not_publish_partial_csv(batch_fixture, monkeypatch):
    f = batch_fixture
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / "state.json")
    run = f.root / state["jobs"][0]["attempts"][-1]["run"]
    target = run / "eval_summary.csv"
    contents, stamp = target.read_bytes(), target.stat().st_mtime_ns
    original = EvalQueue.recover
    def interrupted(self, payload, *args, **kwargs):
        if payload["run_id"] == run.name and Path(payload["snapshot"]).name == "8":
            raise RuntimeError("injected run validation failure")
        return original(self, payload, *args, **kwargs)
    monkeypatch.setattr(EvalQueue, "recover", interrupted)
    writes = _watch_eval_summary_writes(monkeypatch)
    with pytest.raises(RuntimeError, match="injected run validation failure"):
        scheduler.batch_train(parser().parse_args(f.argv + ["--resume"]))
    assert writes == [] and target.read_bytes() == contents and target.stat().st_mtime_ns == stamp


def test_lightweight_worker_survives_reconciliation_before_notification(batch_fixture, monkeypatch):
    # Force the result-published-before-finish window instead of relying on timing.
    monkeypatch.setitem(globals(), 'FAKE_WORKER', FAKE_WORKER.replace(
        '        queue.finish(key,owner)',
        '        queue.recover(request)\n        queue.finish(key,owner)'))
    assert scheduler.batch_train(parser().parse_args(batch_fixture.argv)) == 0
