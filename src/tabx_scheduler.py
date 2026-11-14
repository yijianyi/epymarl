"""One-terminal TABX GPU scheduling, batch-scoped CPU evaluation and text logs."""
import ast
from collections import Counter, deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

from tabx_reporting import eval_line, train_line, write_csv
from tabx_config import configure_bc
from tabx_storage import (EvalQueue, atomic_json, code_hash, completed_checkpoints, ensure_output_contract,
                          evaluation_request, git_version, read_json, resolve_under, runtime_versions, safe_component)


REPOSITORY = Path(__file__).resolve().parents[1]
BATCH_FIELDS = ("snapshot_time_utc", "task_id", "train_seed", "gpu", "status", "train_status", "eval_status", "eval_expected", "eval_done",
                "eval_pending", "eval_running", "eval_failed", "eval_remaining", "run_id", "attempt", "pid", "exit_code", "t_env",
                "learner_updates", "elapsed_seconds", "peak_cuda_bytes", "latest_eval_step", "latest_eval_phase",
                "latest_eval_episodes", "latest_eval_wins", "latest_eval_win_rate_pct", "latest_eval_mean_return",
                "latest_eval_return_std", "log_path", "error")
PROTECTED = {"env", "runner", "batch_size_run", "learner", "mixer", "mac", "agent", "use_rnn", "use_cuda",
             "device", "seed", "tabx_require_cuda", "tabx_batch_id", "tabx_assigned_gpu", "tabx_train_log",
             "tabx_run_id", "tabx_run_dir", "tabx_output_root", "local_results_path", "checkpoint_path",
             "evaluate", "save_replay", "env_args.key", "env_args.seed", "env_args.task_bank",
             "env_args.timeout_rule", "env_args.expected_bank_hash", "env_args.tabx_root"}


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            _merge(target.setdefault(key, {}), value)
        else:
            target[key] = value
    return target


def training_config(overrides):
    """Match main.py's default -> env -> algorithm -> CLI merge, without torch."""
    import yaml
    config = {}
    for name in ("default.yaml", "envs/tabx.yaml", "algs/qmix_tabx.yaml"):
        with (REPOSITORY / "src/config" / name).open(encoding="utf-8") as stream:
            _merge(config, yaml.safe_load(stream))
    canonical = {}
    for setting in overrides:
        key, separator, raw = setting.partition("=")
        if not separator or not key or not raw:
            raise ValueError(f"Expected --set KEY=VALUE, got {setting!r}")
        if key in PROTECTED or any(part.startswith("tabx_") for part in key.split(".")[1:]):
            raise ValueError(f"--set {key} is managed by batch-train")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError(f"Unknown training option: {key}")
            target = target[part]
        if parts[-1] not in target or isinstance(target[parts[-1]], dict):
            raise ValueError(f"Unknown or non-scalar training option: {key}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = yaml.safe_load(raw)
        previous = target[parts[-1]]
        if previous is not None:
            if isinstance(previous, bool) and not isinstance(value, bool):
                raise ValueError(f"{key} requires a boolean")
            if isinstance(previous, (int, float)) and not isinstance(previous, bool):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{key} requires a number")
                if isinstance(previous, int) and (not isinstance(value, int) or
                        (value < 0 if key == "tabx_dagger_rounds" else value <= 0)):
                    raise ValueError(f"{key} requires a positive integer")
        target[parts[-1]] = value
        canonical[key] = value
    configure_bc(config)
    if not config["save_model"] or not config["tabx_async_eval"]:
        raise ValueError("Batch training requires save_model=True and tabx_async_eval=True")
    if config["tabx_screen_episodes"] < 2:
        raise ValueError("Require tabx_screen_episodes>=2")
    if config["buffer_size"] < config["batch_size"]:
        raise ValueError("buffer_size must be at least batch_size")
    if config["lr"] <= 0 or config["grad_norm_clip"] <= 0:
        raise ValueError("lr and grad_norm_clip must be positive")
    config.update(use_cuda=True, tabx_require_cuda=True)
    # Identity and deployment are supplied independently, not part of hyperparameters.
    for key in ("seed", "tabx_batch_id", "tabx_assigned_gpu", "tabx_train_log", "tabx_run_id",
                "tabx_output_root", "local_results_path"):
        config.pop(key, None)
    for key in ("key", "task_bank", "timeout_rule", "expected_bank_hash", "tabx_root"):
        config["env_args"].pop(key, None)
    return config, canonical


def _split_unique(raw, label):
    values = [part.strip() for part in raw.split(",")]
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{label} must be a nonempty list without duplicates")
    return values


def gpu_ids(raw, environment=None):
    environment = os.environ if environment is None else environment
    def canonical(value):
        return str(int(value)) if value.isdigit() else value
    values = [canonical(value) for value in _split_unique(raw, "gpus")]
    if len(set(values)) != len(values) or any(not re.fullmatch(r"\d+|GPU-[\w-]+|MIG-[\w./-]+", value) for value in values):
        raise ValueError("GPU IDs must be unique external indices or GPU/MIG UUIDs")
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if name not in environment or (name == "NVIDIA_VISIBLE_DEVICES" and environment[name] == "all"):
            continue
        allowed = {canonical(item.strip()) for item in environment[name].split(",") if item.strip()}
        if not set(values).issubset(allowed):
            raise ValueError(f"Requested GPUs {values} exceed {name}={environment[name]!r}; use external IDs from that allocation")
    return values


def probe_gpus(gpus):
    """Probe in fresh interpreters: the scheduler never initializes torch/CUDA/JAX."""
    code = ("import json,torch; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
            "assert torch.cuda.device_count()==1, 'expected one visible GPU'; "
            "x=torch.zeros(1,device='cuda'); torch.cuda.synchronize(); "
            "print(json.dumps(dict(name=torch.cuda.get_device_name(0), "
            "total_memory=torch.cuda.get_device_properties(0).total_memory)))")
    devices = {}
    for gpu in gpus:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cpu")
        try:
            result = subprocess.run([sys.executable, "-c", code], env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Cannot validate training GPU {gpu}: {exc}") from exc
        if result.returncode:
            raise RuntimeError(f"Training GPU {gpu} is unavailable; no CPU fallback allowed:\n{result.stdout[-4000:]}")
        devices[gpu] = json.loads(result.stdout.strip().splitlines()[-1])
    return devices


def task_selection(args):
    from envs.tabx_tasks import index_bank, task_bank_path, task_hash, validate_task_structure
    bank = task_bank_path(args.bank)
    _, tasks, digest = index_bank(bank)
    if args.bank_sha256 and digest != args.bank_sha256:
        raise ValueError("Task bank SHA-256 mismatch")
    if args.tasks is not None:
        selected = _split_unique(args.tasks, "tasks")
    elif args.pilot is not None:
        pilot = read_json(args.pilot)
        if pilot["bank_hash"] != digest:
            raise ValueError("Pilot task-bank hash mismatch")
        selected = pilot["pilot"]
        if not set(map(str, selected)).issubset(set(map(str, pilot["splits"]["train"]))):
            raise ValueError("Pilot contains tasks outside its frozen training split")
    else:
        path = Path(args.task_list)
        text = path.read_text(encoding="utf-8")
        selected = json.loads(text) if path.suffix.lower() == ".json" else re.split(r"[\s,]+", text.strip())
    if not isinstance(selected, list) or not selected:
        raise ValueError("Task source must contain a nonempty list")
    selected = [str(value) for value in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("Task IDs must not repeat")
    hashes = {}
    for key in selected:
        safe_component(key)
        if key not in tasks:
            raise ValueError(f"Task {key!r} is missing from the specified task bank")
        validate_task_structure(tasks[key])
        hashes[key] = task_hash(tasks[key])
    return bank, selected, digest, hashes


@contextmanager
def batch_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".scheduler.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another scheduler owns batch {directory.name}") from exc
        yield


def process_identity(pid):
    try:
        # comm in /proc/PID/stat may itself contain spaces or parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(record):
    return bool(record.get("pid") and record.get("process_start") and
                process_identity(record["pid"]) == record["process_start"])


def stop_process(process):
    """Own process groups only, including an eval supervisor's grandchildren."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _sacred(value):
    # A single argv element, not a shell command. Quotes preserve string task IDs.
    return repr(value) if isinstance(value, (bool, type(None))) else json.dumps(value)


def derive_job_status(attempt):
    """One definition for state.json, CSV and terminal; only exact screen IDs count."""
    train = attempt.get("train_status", "pending")
    expected = attempt.get("eval_expected")
    requests = set(attempt.get("screen_requests", {}).values())
    records = attempt.get("evaluations", {})
    done = sum(records.get(key, {}).get("valid", False) and records[key].get("status") == "complete" for key in requests)
    counts = Counter(records.get(key, {}).get("status", "pending") for key in requests
                     if not (records.get(key, {}).get("valid", False) and records[key].get("status") == "complete"))
    if expected is not None:
        counts["pending"] += max(0, expected - len(requests))
    sealed = train == "complete" and attempt.get("inventory_verified", False)
    if sealed and expected is not None and expected > 0 and len(requests) == expected == done:
        evaluation = "complete"
    elif counts["failed"]:
        evaluation = "failed"
    elif attempt.get("eval_interrupted", False):
        evaluation = "interrupted"
    elif counts["running"]:
        evaluation = "running"
    else:
        evaluation = "pending"
    if train != "complete":
        status = train
    elif evaluation == "complete":
        status = "complete"
    elif evaluation in ("failed", "interrupted"):
        status = f"evaluation_{evaluation}"
    else:
        status = "evaluating"
    return dict(train_status=train, eval_status=evaluation, eval_expected=expected, eval_done=done,
                eval_pending=counts["pending"] + counts["registered"], eval_running=counts["running"], eval_failed=counts["failed"],
                eval_remaining=None if expected is None else max(0, expected - done), status=status)


class BatchScheduler:
    def __init__(self, args, directory, manifest, state):
        self.args, self.directory, self.manifest, self.state = args, directory, manifest, state
        self.root = Path(args.root).expanduser().resolve()
        self.queue = EvalQueue(self.root)
        self.jobs = state["jobs"]
        self.jobs_by_id = {}
        for index, job in enumerate(self.jobs):
            job.setdefault("job_id", f"{index:06d}")
            self.jobs_by_id[job["job_id"]] = job
        self.running, self.attempts, self.requests, self.records = {}, {}, {}, {}
        self.worker = self.worker_log = self.worker_error = None
        self.dispatcher = None
        self.stopping = self.authorized = False
        self.log = None
        self.views, self.log_readers, self.inventories = {}, {}, {}
        self.reported_metrics = set(state.get("reported_metrics", []))
        self.reported_evals = set(state.get("reported_evals", []))
        self.reported_failures, self.model_states = set(), {}
        self.dirty_jobs, self.dirty_job_files, self.dirty_runs = set(), set(), set()
        self.job_events, self.acks = {}, []
        self.train_counts, self.model_counts, self.eval_counts = Counter(), Counter(), Counter()
        self.status_cache, self.summary_rows = {}, {}
        self.pending_train, self.free_slots, self.gpu_in_use = deque(), deque(), Counter()
        self.last_snapshot = time.monotonic()
        self.snapshot_interval = getattr(args, "snapshot_interval", 300.0)
        self.eval_prefetch = getattr(args, "eval_prefetch", 4)
        self.pending_ids = set()
        self.initializing = True

    def emit(self, text):
        print(text, flush=True)
        if self.log is not None:
            self.log.write(text + "\n")

    def save(self, event=None, job=None):
        """Runtime metadata is small; whole-batch state is written only by snapshot()."""
        if not self.authorized:
            return
        self.state["updated_at"] = time.time()
        if job is not None:
            self.mark_job(job, event)
        elif event:
            self._append_event(dict(time=time.time(), event=event, worker=self.state.get("worker", {})))
        self._write_runtime()

    def _track_attempt(self, job, attempt):
        self.attempts[attempt["run_id"]] = (job, attempt)
        attempt.setdefault("train_status", {"trained": "complete", "starting": "running"}.get(attempt["status"], attempt["status"]))
        attempt.setdefault("screen_requests", {})
        attempt.setdefault("evaluations", {})
        attempt.setdefault("eval_expected", None)
        attempt.setdefault("inventory_verified", False)
        view = dict(attempt.get("training_view", {}), evaluations={}, latest_evaluation={})
        self.views.setdefault(attempt["run_id"], view)
        self.dirty_runs.add(attempt["run_id"])

    def _check_run(self, job, run):
        saved = read_json(run / "manifest.json")
        expected = dict(task_id=job["task_id"], train_seed=job["train_seed"], timeout_rule=self.args.rule,
                        batch_id=self.args.batch_id, bank_hash=self.manifest["bank_hash"],
                        task_hash=self.manifest["task_hashes"][job["task_id"]])
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"Batch run identity differs: {key} in {run}")

    def _adopt(self, request, status, report=None, error=None):
        from tabx_reporting import evaluation_summary
        key = self.queue.request_id(request)
        run_id = request["run_id"]
        if run_id not in self.attempts:
            raise ValueError(f"Request belongs to an unknown run in this batch: {run_id}")
        job, attempt = self.attempts[run_id]
        step = int(Path(request["snapshot"]).name)
        self.requests[key] = request
        record = attempt["evaluations"].setdefault(key, dict(valid=False))
        before = dict(record)
        previous_status = record.get("status")
        if report is not None and record.get("valid") and previous_status == status == "complete":
            return key
        record.update(status=status, phase=request["phase"], checkpoint_step=step, error=error)
        if request["phase"] == "screen" and not attempt["inventory_verified"]:
            attempt["screen_requests"][str(step)] = key
        if report is not None:
            record["valid"] = True
            row = evaluation_summary(key, report, step)
            record["summary"] = row
            view = self.views[run_id]
            latest = view.get("latest_evaluation", {})
            rank = lambda item: (int(item["checkpoint_step"]), item["phase"], item["request_id"])
            if not latest or rank(row) >= rank(latest):
                view["latest_evaluation"] = row
            if not record.get("announced") and key not in self.reported_evals:
                self.emit(eval_line(self.args.rule, job["task_id"], job["train_seed"], row) + f" run_id={run_id}")
            record["announced"] = True
        elif status != "complete":
            record["valid"] = False
        if previous_status != status:
            if previous_status is not None:
                self.eval_counts[previous_status] -= 1
            self.eval_counts[status] += 1
        self.records[key] = record
        self.pending_ids.add(key)
        if self.dispatcher is not None:
            self.dispatcher.observe(key, status)
        if record != before:
            self.mark_job(job, "evaluation_" + status)
        return key

    def reconcile_run(self, job, attempt, inventory, retry_failed=False, *, defer_summary=False):
        """Exactly once per completed run per invocation; never called by idle polls."""
        run = resolve_under(self.root, attempt["run"])
        count = int(self.manifest["training_config"]["tabx_screen_episodes"])
        requests = [evaluation_request(self.root, run, snapshot, "screen", count) for snapshot in inventory[2]]
        attempt.update(train_status="complete", inventory_verified=True, eval_expected=len(requests),
                       screen_requests={Path(request["snapshot"]).name: self.queue.request_id(request) for request in requests})
        self.inventories[attempt["run_id"]] = inventory
        rows_by_request_id = {}
        for request in requests:
            key = self.queue.request_id(request)
            cached = attempt["evaluations"].get(key)
            if cached and cached.get("valid") and cached["status"] == "complete" and cached.get("summary"):
                rows_by_request_id[key] = cached["summary"]
                continue
            recovered = self.queue.recover(request, retry_failed=retry_failed, defer_summary=defer_summary)
            self._adopt(request, recovered["status"], recovered["report"], recovered.get("error"))
            if recovered["report"] is not None:
                rows_by_request_id[key] = attempt["evaluations"][key]["summary"]
            # Retain the original per-run request references for inspection.
            step = Path(request["snapshot"]).name
            reference = run / "evaluation_requests" / f"{step}.json"
            if not reference.exists() or read_json(reference).get("request_id") != key:
                atomic_json(reference, dict(status="queued", request_id=key, checkpoint_step=int(step)))
        self.mark_job(job, "training_sealed")
        self.dirty_runs.add(attempt["run_id"])
        return rows_by_request_id

    def prepare_resume(self):
        runtime_path = self.directory / "runtime.json"
        runtime = read_json(runtime_path) if runtime_path.exists() else {}
        if process_alive(runtime.get("worker", {})):
            raise RuntimeError(f"Previous batch worker PID {runtime['worker']['pid']} is still running")
        for trainer in runtime.get("active_trainers", []):
            if process_alive(trainer):
                raise RuntimeError(f"Previous training PID {trainer['pid']} is still running; stop it before resuming")
        if process_alive(self.state.get("worker", {})):
            raise RuntimeError(f"Previous batch worker PID {self.state['worker']['pid']} is still running")
        for job in self.jobs:
            for attempt in job["attempts"]:
                if process_alive(attempt):
                    raise RuntimeError(f"Previous training PID {attempt['pid']} is still running; stop it before resuming")
        from tabx_dispatcher import EvalDispatcher
        self.dispatcher = EvalDispatcher(self.queue, self.args.rule, self.args.batch_id,
            self.args.eval_workers, self.eval_prefetch, on_request=self._adopt,
            known_status=lambda key: self.records.get(key, {}).get("status"))
        self.authorized = True
        self.log = (self.directory / "batch.log").open("a", encoding="utf-8", buffering=1)
        if self.args.resume:
            self.queue.release_batch(self.args.rule, self.args.batch_id)
            self.queue.retry_failed(self.args.rule, self.args.batch_id, defer_summary=True)
        for job in self.jobs:
            job["needs_training"] = not job["attempts"]
            for attempt in job["attempts"]:
                rows_by_request_id = {}
                # Disk is authoritative. Stored counters may precede a final JSON.
                self.reported_evals.update(key for key, value in attempt.get("evaluations", {}).items() if value.get("announced"))
                attempt.update(screen_requests={}, evaluations={}, eval_expected=None, inventory_verified=False,
                               eval_interrupted=False)
                self._track_attempt(job, attempt)
                run = resolve_under(self.root, attempt["run"])
                current = attempt is job["attempts"][-1]
                inventory = None
                if current:
                    try:
                        inventory = completed_checkpoints(run)
                    except (ValueError, KeyError, OSError) as error:
                        attempt.update(train_status="failed", error=f"Invalid training completion: {error}")
                if inventory is not None:
                    self._check_run(job, run)
                    rows_by_request_id.update(self.reconcile_run(
                        job, attempt, inventory, retry_failed=self.args.resume, defer_summary=True))
                    job["needs_training"] = False
                else:
                    if current:
                        if attempt.get("train_status") != "failed":
                            attempt["train_status"] = "interrupted"
                        job["needs_training"] = True
                    # Only startup/resume inspects old request references.
                    for path in (run / "evaluation_requests").glob("*.json"):
                        reference = read_json(path)
                        if not reference.get("request_id"):
                            continue
                        info = self.queue.info(reference["request_id"])
                        request = (info["request"] if info else evaluation_request(self.root, run,
                                   run / "checkpoints" / path.stem, "screen",
                                   int(self.manifest["training_config"]["tabx_screen_episodes"])))
                        if self.queue.request_id(request) != reference["request_id"]:
                            raise ValueError("Saved evaluation request ID differs from the snapshot")
                        recovered = self.queue.recover(request, retry_failed=self.args.resume, defer_summary=True)
                        self._adopt(request, recovered["status"], recovered["report"], recovered.get("error"))
                        if recovered["report"] is not None:
                            rows_by_request_id[reference["request_id"]] = attempt["evaluations"][reference["request_id"]]["summary"]
                # Recover results published before their notification/state update.
                for path in (run / "evaluations").glob("*.json"):
                    if self.records.get(path.stem, {}).get("valid"):
                        rows_by_request_id[path.stem] = self.records[path.stem]["summary"]
                        continue
                    info = self.queue.info(path.stem)
                    request = info["request"] if info else read_json(path)["request"]
                    if request["run_id"] != attempt["run_id"] or self.queue.request_id(request) != path.stem:
                        raise ValueError("Saved evaluation JSON identity mismatch")
                    recovered = self.queue.recover(request, retry_failed=self.args.resume, defer_summary=True)
                    self._adopt(request, recovered["status"], recovered["report"], recovered.get("error"))
                    if recovered["report"] is not None:
                        rows_by_request_id[path.stem] = attempt["evaluations"][path.stem]["summary"]
                # All checkpoint and extra-phase reports for this run have now
                # validated. A failed check above cannot publish a partial CSV.
                from tabx_reporting import reconcile_eval_summary
                reconcile_eval_summary(run, rows_by_request_id)
                del rows_by_request_id
            if job["attempts"]:
                self.mark_job(job, "recovered")
        self.state.update(status="running", worker={})
        for key in ("ended_at", "error", "worker_error", "failed_requests"):
            self.state.pop(key, None)
        self.initialize_job_cache()
        self.dispatcher.restore()
        self.save("resume" if self.args.resume else "start")

    def start_workers(self):
        command = [sys.executable, str(REPOSITORY / "src/tabx_pipeline.py"), "eval-worker",
                   "--bank", self.manifest["bank"], "--root", str(self.root), "--rule", self.args.rule,
                   "--batch-id", self.args.batch_id, "--workers", str(self.args.eval_workers),
                   "--external-dispatch",
                   "--threads", str(self.args.eval_threads), "--poll-seconds", str(self.args.poll_seconds)]
        if self.args.eval_cpu_affinity:
            command.extend(["--cpu-affinity", self.args.eval_cpu_affinity])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
        self.worker_log = (self.directory / "eval-workers.log").open("a", encoding="utf-8", buffering=1)
        self.worker = subprocess.Popen(command, cwd=REPOSITORY, env=environment, stdout=self.worker_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.state["worker"] = dict(pid=self.worker.pid, process_start=process_identity(self.worker.pid))
        self.save("workers_started")
        self.emit(f"[WORKER] rule={self.args.rule} batch={self.args.batch_id} pid={self.worker.pid} workers={self.args.eval_workers}")

    def training_command(self, job, attempt):
        values = dict(self.manifest["overrides"])
        values.update({"env_args.key": job["task_id"], "env_args.task_bank": self.manifest["bank"],
                       "env_args.expected_bank_hash": self.manifest["bank_hash"], "env_args.timeout_rule": self.args.rule,
                       "seed": job["train_seed"], "tabx_output_root": str(self.root), "tabx_run_id": attempt["run_id"],
                       "tabx_batch_id": self.args.batch_id, "tabx_assigned_gpu": attempt["gpu"],
                       "tabx_train_log": attempt["log"], "tabx_require_cuda": True, "use_cuda": True})
        return [sys.executable, str(REPOSITORY / "src/main.py"), "--config=qmix_tabx", "--env-config=tabx", "with",
                *[f"{key}={_sacred(value)}" for key, value in sorted(values.items())]]

    def launch(self, job, gpu):
        number = len(job["attempts"]) + 1
        run_id = f"{self.args.batch_id[:64]}-t{job['task_id'][:32]}-s{job['train_seed']}-a{number}-{uuid.uuid4().hex[:8]}"
        log = self.directory / "logs" / run_id / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        run = self.root / self.args.rule / job["task_id"] / str(job["train_seed"]) / run_id
        attempt = dict(run_id=run_id, run=str(run.relative_to(self.root)), log=str(log.relative_to(self.root)),
                       gpu=gpu, status="running", train_status="running", started_at=time.time(), pid=None,
                       process_start=None, exit_code=None, error=None, eval_interrupted=bool(self.worker_error))
        job["attempts"].append(attempt)
        job["needs_training"] = False
        self._track_attempt(job, attempt)
        self.save("starting", job)
        command = self.training_command(job, attempt)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
        stream = log.open("a", encoding="utf-8", buffering=1)
        stream.write(f"[START] task={job['task_id']} seed={job['train_seed']} gpu={gpu} run_id={run_id}\n")
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        try:
            process = subprocess.Popen(command, cwd=REPOSITORY, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            stream.write(f"[FAILED] {error}\n")
            stream.close()
            attempt.update(train_status="failed", error=str(error), ended_at=time.time())
            self.save("launch_failed", job)
            self.emit(f"[FAILED] task={job['task_id']} seed={job['train_seed']} error={error} log={log}")
            return
        attempt.update(pid=process.pid, process_start=process_identity(process.pid))
        stream.write(f"[PID] {process.pid}\n")
        self.running[run_id] = (process, stream, job)
        self.save("started", job)
        self.emit(f"[START] rule={self.args.rule} task={job['task_id']} seed={job['train_seed']} gpu={gpu} "
                  f"pid={process.pid} run_id={run_id} log={log}")

    def poll_training(self):
        changed = False
        for run_id, (process, stream, job) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del self.running[run_id]
            attempt = job["attempts"][-1]
            self.gpu_in_use[attempt["gpu"]] -= 1
            self.free_slots.append(attempt["gpu"])
            attempt.update(exit_code=code, ended_at=time.time())
            try:
                if code != 0:
                    raise RuntimeError(f"Training exited with code {code}")
                run = resolve_under(self.root, attempt["run"])
                inventory = completed_checkpoints(run)
                if inventory is None:
                    raise ValueError("Missing training_complete.json; process exit is insufficient")
                self._check_run(job, run)
                self.reconcile_run(job, attempt, inventory)
                self.emit(f"[DONE] rule={self.args.rule} task={job['task_id']} seed={job['train_seed']} "
                          f"gpu={attempt['gpu']} step={inventory[0]['final_t_env']} run_id={run_id} train_status=complete")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                attempt.update(train_status="failed", error=str(error))
                self.emit(f"[FAILED] rule={self.args.rule} task={job['task_id']} seed={job['train_seed']} "
                          f"error={error} log={self.root / attempt['log']}")
            self.dirty_runs.add(run_id)
            self.save("training_finished", job)
            self.fill_slots()
            changed = True
        return changed

    def fill_slots(self):
        while self.pending_train and self.free_slots and not self.stopping:
            job = self.jobs_by_id[self.pending_train.popleft()]
            gpu = self.free_slots.popleft()
            self.launch(job, gpu)
            if job["attempts"][-1]["run_id"] in self.running:
                self.gpu_in_use[gpu] += 1
            else:
                self.free_slots.append(gpu)

    def evaluations(self):
        # Only active work and unconsumed notifications. No checkpoint traversal.
        if self.dispatcher is not None:
            self.dispatcher.poll()
        for key, status in self.queue.active_keys(self.args.rule, self.args.batch_id).items():
            if self.records.get(key, {}).get("valid"):
                continue
            if key not in self.requests:
                info = self.queue.info(key)
                if info is None:
                    raise ValueError("Active evaluation lost its request index")
                self._adopt(info["request"], status)
            elif self.records[key]["status"] != status:
                self._adopt(self.requests[key], status)
        for notice in self.queue.notifications(self.args.rule, self.args.batch_id):
            key, request = notice["request_id"], notice["request"]
            cached = self.records.get(key, {})
            report = None
            if notice["status"] == "complete" and not cached.get("valid"):
                try:
                    report = self.queue.completed_result(key)
                    if report is None:
                        raise ValueError("Completion notification has no final evaluation JSON")
                except (ValueError, KeyError, OSError) as error:
                    notice = self.queue.invalidate(key, str(error))
            if not cached.get("valid") or notice["status"] != "complete":
                self._adopt(request, notice["status"], report, notice.get("error"))
            token = (key, notice["version"])
            if notice["status"] == "failed" and token not in self.reported_failures:
                self.emit(f"[EVAL-FAILED] rule={self.args.rule} task={request['task_id']} seed={request['train_seed']} "
                          f"checkpoint_step={Path(request['snapshot']).name} request={key} error={notice['error']}")
                self.reported_failures.add(token)
            self.acks.append(notice)
        return self.records

    def recover_outstanding(self):
        """On shutdown only, close the final-JSON-before-notification crash window."""
        for key, record in list(self.records.items()):
            if record.get("valid"):
                continue
            request = self.requests[key]
            recovered = self.queue.recover(request)
            self._adopt(request, recovered["status"], recovered["report"], recovered.get("error"))

    def run_view(self, job, attempt):
        from tabx_reporting import TrainingLogReader
        run_id = attempt["run_id"]
        reader = self.log_readers.get(run_id)
        if reader is None:
            reader = TrainingLogReader(resolve_under(self.root, attempt["run"]))
            self.log_readers[run_id] = reader
        view = self.views[run_id]
        metrics, changed = reader.update(view)
        if changed:
            self.mark_job(job, status_changed=False)
        for metric in metrics:
            update = int(metric["learner_updates"])
            legacy_key = f"{run_id}:{update}"
            if update > attempt.get("reported_update", 0) and legacy_key not in self.reported_metrics:
                self.emit(train_line(self.args.rule, job["task_id"], job["train_seed"], attempt["gpu"], metric)
                          + f" run_id={run_id}")
            attempt["reported_update"] = max(update, attempt.get("reported_update", 0))
        attempt["training_view"] = {key: view[key] for key in (
            "t_env", "learner_updates", "elapsed_seconds", "peak_cuda_bytes", "peak_rss_kib", "last_end_reason", "latest_metric")
            if key in view}
        if run_id not in self.running:
            self.log_readers.pop(run_id, None)
        return view

    def refresh_reports(self, statuses=None, force=False):
        for run_id in set(self.running) | self.dirty_runs:
            job, attempt = self.attempts[run_id]
            self.run_view(job, attempt)
        self.dirty_runs.clear()
        for job_id in self.dirty_jobs | self.dirty_job_files:
            self._flush_job(job_id)
        self.dirty_jobs.clear()
        self.dirty_job_files.clear()
        for notice in self.acks:
            self.queue.acknowledge(notice)
        self.acks.clear()
        self.snapshot(force)

    def progress(self, statuses=None):
        gpus = self.args.gpus if isinstance(self.args.gpus, list) else self.args.gpus.split(",")
        gpu = ",".join(f"{key}:{self.gpu_in_use[key]}/{self.args.jobs_per_gpu}" for key in gpus)
        self.emit(f"[BATCH] rule={self.args.rule} gpu_slots={gpu} "
                  f"train_running={self.train_counts['running']} train_done={self.train_counts['complete']} "
                  f"train_pending={len(self.pending_train)} train_failed={self.train_counts['failed']} "
                  f"eval_registered_waiting={self.eval_counts['registered']} eval_ready={self.eval_counts['pending']} "
                  f"eval_running={self.eval_counts['running']} eval_done={self.eval_counts['complete']} "
                  f"eval_failed={self.eval_counts['failed']}")

    def interrupt_evaluations(self):
        # A pool interruption is a batch-wide event, not an idle-loop operation.
        for job in self.jobs:
            if job["attempts"] and job.get("status") != "complete":
                job["attempts"][-1]["eval_interrupted"] = True
                self.mark_job(job, "evaluation_interrupted")

    def log_remaining(self):
        for job in self.jobs:
            if job["status"] != "complete":
                self.emit(f"[INCOMPLETE] task={job['task_id']} seed={job['train_seed']} "
                          f"train_status={job['train_status']} eval_status={job['eval_status']} status={job['status']} "
                          f"eval_done={job['eval_done']} eval_expected={job['eval_expected']} eval_remaining={job['eval_remaining']}")

    def run(self):
        previous_handlers = {}
        def stop(signum, frame):
            self.stopping = True
        try:
            self.prepare_resume()
            devices = probe_gpus(self.args.gpus) if self.pending_train else {}
            self.state.setdefault("deployments", []).append(dict(time=time.time(), gpus=self.args.gpus, devices=devices,
                bank=self.manifest["bank"], jobs_per_gpu=self.args.jobs_per_gpu, eval_workers=self.args.eval_workers,
                eval_threads=self.args.eval_threads, eval_prefetch=self.eval_prefetch, scheduler_pid=os.getpid()))
            self.save("deployment")
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, stop)
            self.start_workers()
            last_progress = 0.0
            while not self.stopping:
                changed = self.poll_training()
                worker_died = self.worker and self.worker.poll() is not None and self.worker_error is None
                if worker_died:
                    self.worker_error = f"CPU worker supervisor exited with code {self.worker.returncode}; see eval-workers.log"
                    self.emit(f"[WORKER-FAILED] {self.worker_error}")
                    stop_process(self.worker)
                    self.queue.release_batch(self.args.rule, self.args.batch_id)
                    self.interrupt_evaluations()
                self.fill_slots()
                self.evaluations()
                if worker_died:
                    self.recover_outstanding()
                self.refresh_reports()
                now = time.monotonic()
                if changed or now - last_progress >= self.args.status_interval:
                    self.progress()
                    last_progress = now
                training_done = not self.pending_train and not self.running
                eval_done = not (self.eval_counts["registered"] + self.eval_counts["pending"] + self.eval_counts["running"])
                if training_done and (eval_done or self.worker_error):
                    self.verify_counts()
                    failed = bool(self.worker_error) or self.model_counts["complete"] != len(self.jobs)
                    self.state.update(status="failed" if failed else "complete", ended_at=time.time(),
                        failed_requests=[key for key, value in self.records.items() if value["status"] == "failed"],
                        worker_error=self.worker_error)
                    self.refresh_reports(force=True)
                    self.progress()
                    self.log_remaining()
                    self.emit(f"[FINISH] batch={self.args.batch_id} status={self.state['status']} summary={self.directory / 'batch_summary.csv'}")
                    return int(failed)
                time.sleep(self.args.poll_seconds)
            self.emit(f"[INTERRUPTED] batch={self.args.batch_id}; stopping owned child processes")
            self.state["status"] = "interrupted"
            return 130
        except BaseException as error:
            if self.authorized:
                self.state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
                self.emit(f"[ERROR] {error}")
            else:
                print(f"[REFUSED] {error}", file=sys.stderr, flush=True)
            raise
        finally:
            # A refused takeover must leave the original state and CSV untouched.
            if self.authorized:
                try:
                    for process, stream, job in list(self.running.values()):
                        stop_process(process)
                        stream.close()
                        attempt = job["attempts"][-1]
                        self.gpu_in_use[attempt["gpu"]] -= 1
                        attempt.update(train_status="interrupted", exit_code=process.returncode, ended_at=time.time())
                        inventory = completed_checkpoints(resolve_under(self.root, attempt["run"]))
                        if inventory is not None:
                            self._check_run(job, resolve_under(self.root, attempt["run"]))
                            self.reconcile_run(job, attempt, inventory)
                        self.dirty_runs.add(attempt["run_id"])
                        self.mark_job(job, "training_interrupted")
                    self.running.clear()
                    if self.worker is not None:
                        stop_process(self.worker)
                        self.queue.release_batch(self.args.rule, self.args.batch_id)
                        self.state["worker"] = {}
                        self.evaluations()
                        self.recover_outstanding()
                        if self.stopping or self.worker_error or self.state["status"] != "complete":
                            self.interrupt_evaluations()
                    if self.initializing:
                        self.initialize_job_cache()
                    self.refresh_reports(force=True)
                    self._write_runtime()
                    if self.stopping:
                        self.log_remaining()
                finally:
                    if self.worker_log:
                        self.worker_log.close()
                    if self.dispatcher is not None:
                        self.dispatcher.close()
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                    if self.log is not None:
                        self.log.close()

    def mark_job(self, job, event=None, status_changed=True):
        job_id = job["job_id"]
        if status_changed:
            self.dirty_jobs.add(job_id)
        self.dirty_job_files.add(job_id)
        if event:
            self.job_events.setdefault(job_id, set()).add(event)

    def _write_runtime(self):
        if not self.authorized:
            return
        atomic_json(self.directory / "runtime.json", dict(
            status=self.state["status"], worker=self.state.get("worker", {}), scheduler_pid=os.getpid(),
            updated_at=time.time(), active_trainers=[dict(job_id=job["job_id"], **{
                key: job["attempts"][-1].get(key) for key in ("run_id", "run", "log", "gpu", "pid", "process_start", "started_at")})
                for _, _, job in self.running.values()]))

    def _append_event(self, event):
        with (self.directory / "jobs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def initialize_job_cache(self):
        """Startup/recovery only. Idle loops never call this full-batch pass."""
        self.train_counts.clear()
        self.model_counts.clear()
        self.status_cache.clear()
        self.summary_rows.clear()
        self.pending_train.clear()
        for job in self.jobs:
            attempt = job["attempts"][-1] if job["attempts"] else {}
            derived = derive_job_status(attempt)
            if attempt:
                attempt.update(derived)
            job.update(derived)
            self.train_counts[derived["train_status"]] += 1
            self.model_counts[derived["status"]] += 1
            self.status_cache[job["job_id"]] = (derived["train_status"], derived["status"])
            self.model_states[job["job_id"]] = tuple(derived.values())
            self.summary_rows[job["job_id"]] = self._summary_row(job)
            if job.get("needs_training", not job["attempts"]):
                self.pending_train.append(job["job_id"])
        self.gpu_in_use = Counter(job["attempts"][-1]["gpu"] for _, _, job in self.running.values())
        gpus = self.args.gpus if isinstance(self.args.gpus, list) else self.args.gpus.split(",")
        self.free_slots = deque(gpu for slot in range(self.args.jobs_per_gpu) for gpu in gpus
                                if slot >= self.gpu_in_use[gpu])
        self.initializing = False

    def _summary_row(self, job):
        attempt = job["attempts"][-1] if job["attempts"] else {}
        view = self.views.get(attempt.get("run_id"), {})
        latest = view.get("latest_evaluation", {})
        elapsed = view.get("elapsed_seconds")
        if elapsed is None and attempt:
            elapsed = attempt.get("ended_at", time.time()) - attempt["started_at"]
        return dict(task_id=job["task_id"], train_seed=job["train_seed"], gpu=attempt.get("gpu"),
            **{key: job.get(key) for key in ("status", "train_status", "eval_status", "eval_expected", "eval_done",
                    "eval_pending", "eval_running", "eval_failed", "eval_remaining")},
            run_id=attempt.get("run_id"), attempt=len(job["attempts"]), pid=attempt.get("pid"),
            exit_code=attempt.get("exit_code"), t_env=view.get("t_env", 0), learner_updates=view.get("learner_updates", 0),
            elapsed_seconds=elapsed, peak_cuda_bytes=view.get("peak_cuda_bytes"),
            latest_eval_step=latest.get("checkpoint_step"), latest_eval_phase=latest.get("phase"),
            latest_eval_episodes=latest.get("episodes"), latest_eval_wins=latest.get("wins"),
            latest_eval_win_rate_pct=latest.get("win_rate_pct"), latest_eval_mean_return=latest.get("mean_return"),
            latest_eval_return_std=latest.get("return_std"),
            log_path=str(self.root / attempt["log"]) if attempt else None, error=attempt.get("error"))

    def _flush_job(self, job_id):
        job = self.jobs_by_id[job_id]
        attempt = job["attempts"][-1] if job["attempts"] else {}
        if job_id in self.dirty_jobs:
            derived = derive_job_status(attempt)
            previous_train, previous_model = self.status_cache[job_id]
            if previous_train != derived["train_status"]:
                self.train_counts[previous_train] -= 1
                self.train_counts[derived["train_status"]] += 1
            if previous_model != derived["status"]:
                self.model_counts[previous_model] -= 1
                self.model_counts[derived["status"]] += 1
            self.status_cache[job_id] = (derived["train_status"], derived["status"])
            signature = tuple(derived.values())
            if self.model_states.get(job_id) != signature:
                self.emit(f"[MODEL] task={job['task_id']} seed={job['train_seed']} train_status={derived['train_status']} "
                          f"eval_status={derived['eval_status']} status={derived['status']} eval_done={derived['eval_done']} "
                          f"eval_expected={derived['eval_expected']} eval_remaining={derived['eval_remaining']}")
            self.model_states[job_id] = signature
            if attempt:
                attempt.update(derived)
            job.update(derived)
        self.summary_rows[job_id] = self._summary_row(job)
        atomic_json(self.directory / "jobs" / f"{job_id}.json", job)
        events = sorted(self.job_events.pop(job_id, {"log_update"}))
        self._append_event(dict(time=time.time(), job_id=job_id, run_id=attempt.get("run_id"), events=events,
            **{key: job.get(key) for key in ("status", "train_status", "eval_status", "eval_done", "eval_expected", "eval_remaining")},
            t_env=self.summary_rows[job_id]["t_env"]))

    def snapshot(self, force=False):
        if not self.authorized:
            return False
        now = time.monotonic()
        if not force and now - self.last_snapshot < self.snapshot_interval:
            return False
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.state.update(snapshot_time_utc=stamp, schema_version="tabx-batch-state-2",
            counters=dict(train=dict(self.train_counts), model=dict(self.model_counts), evaluation=dict(self.eval_counts)))
        self.state.pop("reported_metrics", None)
        self.state.pop("reported_evals", None)
        write_csv(self.directory / "batch_summary.csv", BATCH_FIELDS,
                  (dict(row, snapshot_time_utc=stamp) for row in self.summary_rows.values()))
        atomic_json(self.directory / "state.json", self.state)
        self.last_snapshot = now
        return True

    def verify_counts(self):
        # Final verification is allowed to be O(batch size), unlike the hot path.
        trains, models, evaluations = Counter(), Counter(), Counter()
        for job in self.jobs:
            derived = derive_job_status(job["attempts"][-1] if job["attempts"] else {})
            if any(job.get(key) != value for key, value in derived.items()):
                raise RuntimeError("Job cache differs from its exact checkpoint completion status")
            trains[derived["train_status"]] += 1
            models[derived["status"]] += 1
        evaluations.update(record["status"] for record in self.records.values())
        if +trains != +self.train_counts or +models != +self.model_counts or +evaluations != +self.eval_counts:
            raise RuntimeError("Incremental batch counters failed final reconciliation")



def batch_train(args):
    for name in ("jobs_per_gpu", "eval_workers", "eval_threads", "eval_prefetch", "poll_seconds", "status_interval", "snapshot_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    safe_component(args.batch_id)
    args.gpus = gpu_ids(args.gpus)
    seeds = _split_unique(args.seeds, "seeds")
    if any(not seed.isdigit() or not 0 <= int(seed) < 2**32 for seed in seeds):
        raise ValueError("Training seeds must be integers in [0,2**32)")
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must not repeat")
    bank, tasks, bank_hash, hashes = task_selection(args)
    if args.tabx_root is not None:
        from envs.tabx_tasks import official_tabx_root
        if Path(args.tabx_root).expanduser().resolve() != official_tabx_root().resolve():
            raise ValueError("Only the pinned official submodule may be used")
    config, overrides = training_config(args.overrides)
    identity = dict(rule=args.rule, tasks=tasks, seeds=seeds, bank_hash=bank_hash, task_hashes=hashes,
                    training_config=config, source_hash=code_hash(REPOSITORY))
    root = Path(args.root).expanduser().resolve()
    # This MUST precede EvalQueue and worker startup in a new output root.
    ensure_output_contract(root)
    directory = root / "_batches" / args.rule / args.batch_id
    with batch_lock(directory):
        path = directory / "batch.json"
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"Batch already exists: {directory}; use --resume")
            manifest = read_json(path)
            if manifest["identity"] != identity:
                raise ValueError("Resume requires identical tasks, seeds, rule, training hyperparameters, task bank and source code")
            state = read_json(directory / "state.json")
            runtime_path = directory / "runtime.json"
            runtime = read_json(runtime_path) if runtime_path.exists() else {}
            for previous in (state.get("worker", {}), runtime.get("worker", {})):
                if process_alive(previous):
                    raise RuntimeError(f"Previous batch worker PID {previous['pid']} is still running")
            for trainer in runtime.get("active_trainers", []):
                if process_alive(trainer):
                    raise RuntimeError(f"Previous training PID {trainer['pid']} is still running; stop it before resuming")
            # Per-job files are newer than the five-minute whole-batch snapshot.
            for index, job in enumerate(state["jobs"]):
                job_id = job.get("job_id", f"{index:06d}")
                job_path = directory / "jobs" / f"{job_id}.json"
                if job_path.exists():
                    state["jobs"][index] = read_json(job_path)
            runtime_path = directory / "runtime.json"
            if runtime_path.exists():
                runtime_jobs = {job.get("job_id", f"{i:06d}"): job for i, job in enumerate(state["jobs"])}
                for saved in read_json(runtime_path).get("active_trainers", []):
                    job = runtime_jobs[saved["job_id"]]
                    if not any(attempt["run_id"] == saved["run_id"] for attempt in job["attempts"]):
                        job["attempts"].append(dict(saved, status="running", train_status="running",
                                                    exit_code=None, error=None))
            expected = [(task, seed) for task in tasks for seed in seeds]
            if [(job["task_id"], job["train_seed"]) for job in state["jobs"]] != expected:
                raise ValueError("Batch job state differs from its frozen task/seed list")
            manifest["bank"] = str(bank)
        else:
            if args.resume:
                raise FileNotFoundError(f"Cannot resume a batch that has no batch.json: {directory}")
            manifest = dict(schema_version="tabx-batch-1", batch_id=args.batch_id, created_at=time.time(),
                            identity=identity, bank=str(bank), bank_hash=bank_hash, task_hashes=hashes,
                            tasks=tasks, seeds=seeds, rule=args.rule, training_config=config, overrides=overrides,
                            code=git_version(REPOSITORY), source_hash=identity["source_hash"], versions=runtime_versions())
            state = dict(status="new", jobs=[dict(task_id=task, train_seed=seed, status="pending", attempts=[])
                                             for task in tasks for seed in seeds])
            atomic_json(path, manifest)
            atomic_json(directory / "state.json", state)
        scheduler = BatchScheduler(args, directory, manifest, state)
        return scheduler.run()
