"""Release per-job dependencies and share GPU slots across the four MAPPO schedulers."""
from collections import Counter, deque
import os
from pathlib import Path
import time
import uuid

from tabx_expert_shards import demonstration_directory, read_expert_manifest, verified_expert_shards
from tabx_gpu_pool import SharedGpuPool
from tabx_medium_collection import collection_status_path
from tabx_online_start import (WEIGHTS, read_step0_record, resolve_source, step0_record_path,
                               write_step0_index)
from tabx_scheduler import BatchScheduler, _sacred, process_alive
from tabx_storage import atomic_json, read_json, simulator_identity


def _read_status(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}


class MediumDemoScheduler(BatchScheduler):
    def __init__(self, args, directory, manifest, state):
        super().__init__(args, directory, manifest, state)
        self.config = manifest["training_config"]
        self.demo_results = {}
        self.demo_identity = None

    def task_identity(self, task):
        return dict(task_id=task, task_hash=self.manifest["task_hashes"][task], bank_hash=self.manifest["bank_hash"])

    def demo_ready(self, task):
        from tabx_dataset import SCHEMA_VERSION, verify_shard

        if task in self.demo_results:
            return self.demo_results[task]
        directory = demonstration_directory(self.config["tabx_demo_root"], self.args.rule, task, "medium")
        frozen = self.manifest["identity"]["demo_manifest_sha256"]
        expected = frozen.get(task)
        status_path = collection_status_path(self.config["tabx_demo_root"], self.args.rule, task)
        status = None
        if expected is None:
            status = _read_status(status_path)
            if status.get("collection_id") != self.manifest["identity"]["collection_id"]:
                return False
            for key, value in dict(self.task_identity(task), rule=self.args.rule,
                                   episode_limit=self.config["env_args"]["max_episode_steps"]).items():
                if status.get(key) != value:
                    raise ValueError(f"Collection identity mismatch for {key}: {status_path}")
            for key in ("teacher", "opponent"):
                if status.get(key, "medium") != "medium":
                    raise ValueError(f"Collection identity mismatch for {key}: {status_path}")
            if status["status"] == "failed":
                raise ValueError(f"Demonstration collection failed: {status['error']} ({status.get('log')})")
            if status["status"] != "ready":
                if status["status"] != "interrupted" and not process_alive(status["owner"]):
                    raise ValueError(f"Demonstration collector exited before publishing {status_path}")
                return False
            expected = status["manifest_sha256"]
            if not expected:
                raise ValueError(f"Ready collection record is missing its manifest SHA-256: {status_path}")
        try:
            manifest, digest = read_expert_manifest(directory, expected)
        except (FileNotFoundError, ValueError):
            if status is not None and _read_status(status_path) != status:
                return False
            raise
        metadata = manifest["metadata"]
        fields = dict(self.task_identity(task), timeout_rule=self.args.rule,
                      episode_limit=self.config["env_args"]["max_episode_steps"], teacher="medium",
                      opponent="medium", label_teacher="medium", label_epsilon=0.0,
                      collection_phase="expert_collect")
        for key, value in fields.items():
            if metadata.get(key) != value:
                raise ValueError(f"Demonstration metadata mismatch for {key}: {directory}")
        if self.demo_identity is None:
            self.demo_identity = simulator_identity()
        official = metadata.get("provenance", {}).get("tabx", {})
        if any(official.get(key) != value for key, value in self.demo_identity.items()):
            raise ValueError(f"Demonstration simulator differs from the current official medium: {directory}")
        if status and (manifest["episodes"] != status["actual_episodes"]
                       or status["actual_episodes"] < status["requested_episodes"]):
            raise ValueError(f"Demonstration episode counts differ from the ready record: {directory}")
        if status and "seed" in status and metadata.get("master_seed") != status["seed"]:
            raise ValueError(f"Demonstration seed differs from the ready record: {directory}")
        shard_metadata = {key: value for key, value in metadata.items() if key != "provenance"}
        seeds = set()
        for path in verified_expert_shards(directory, manifest):
            shard = verify_shard(path, shard_metadata, verify_trajectories=False)
            if shard["schema_version"] != SCHEMA_VERSION or shard["label_source"] != "deterministic_teacher":
                raise ValueError(f"Expected v3 demonstration with deterministic teacher labels: {path}")
            if seeds.intersection(shard["seeds"]):
                raise ValueError(f"Duplicate demonstration episode seeds across shards: {path}")
            seeds.update(shard["seeds"])
        if not seeds or len(seeds) != manifest["episodes"]:
            raise ValueError(f"Demonstration manifest episode count differs from the shard index: {directory}")
        if status is not None:
            frozen[task] = digest
            atomic_json(self.directory / "batch.json", self.manifest)
            self.emit(f"[PIN-DEMO] task={task} sha256={digest}")
        self.demo_results[task] = True
        return True

    def set_dependency(self, job, status, error=None):
        if (job.get("dependency_status") != status or job.get("dependency_error") != error
                or job.get("scheduler_attempt_id") != self.scheduler_attempt_id):
            if self.scheduler_attempt_id is not None:
                job["scheduler_attempt_id"] = self.scheduler_attempt_id
            job["dependency_status"] = status
            job["dependency_error"] = error
            self.mark_job(job, status or "gpu_admitted")
            if error:
                group = f" group={job['group']}" if "group" in job else ""
                self.emit(f"[DEPENDENCY-FAILED] task={job['task_id']} seed={job['train_seed']}{group} error={error}")

class MediumScheduler(MediumDemoScheduler):
    def __init__(self, args, directory, manifest, state):
        super().__init__(args, directory, manifest, state)
        self.group = args.medium_group
        self.pool = SharedGpuPool(self.root, args.rule, args.gpus, args.jobs_per_gpu, self.group, args.batch_id)
        self.sources = {}
        self.source_batch = None
        self.historical_source_failures = {}
        self.lease = None

    def run(self):
        runtime_path = self.directory / "runtime.json"
        preparing = None
        if self.group == "B":
            self.scheduler_attempt_id = uuid.uuid4().hex
            preparing = dict(_read_status(runtime_path), status="preparing", scheduler_pid=os.getpid(),
                             scheduler_start=self.process_start, scheduler_attempt_id=self.scheduler_attempt_id,
                             updated_at=time.time())
            # Keep the previous worker and trainer inventory available during recovery.
            atomic_json(runtime_path, preparing)
        elif self.group in ("C", "D") and self.args.resume:
            self.remember_source_failures()
        registered = False
        try:
            self.pool.register()
            registered = True
            return super().run()
        except (Exception, KeyboardInterrupt) as error:
            if preparing is not None and not self.authorized:
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                atomic_json(runtime_path, dict(preparing, status=status, error=str(error), updated_at=time.time()))
            raise
        finally:
            if registered:
                self.pool.unregister()

    def prepare_resume(self):
        runtime_path = self.directory / "runtime.json"
        runtime = _read_status(runtime_path)
        old_processes = [self.state.get("worker", {}), runtime.get("worker", {}),
                         *runtime.get("active_trainers", [])]
        old_processes.extend(attempt for job in self.jobs for attempt in job["attempts"])
        announced = False
        while list(self.pool.active_attempts()) or any(process_alive(record) for record in old_processes):
            if not announced:
                self.emit(f"[RECOVER] batch={self.args.batch_id} waiting for existing child processes to exit")
                announced = True
            time.sleep(self.args.poll_seconds)
        for job in self.jobs:
            job.pop("dependency_status", None)
            job.pop("dependency_error", None)
        super().prepare_resume()
        if self.args.resume:
            for job in self.jobs:
                if not job["needs_training"]:
                    try:
                        self.check_dependency(job)
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        self.set_dependency(job, "dependency_failed", str(error))

    def source_job_path(self, source, job):
        if self.source_batch is None:
            self.source_batch = _read_status(source / "batch.json") or None
        if self.source_batch is None:
            return None
        batch = self.source_batch
        task, seed = job["task_id"], job["train_seed"]
        if (batch["identity"].get("medium_group") != "B" or batch["bank_hash"] != self.manifest["bank_hash"]
                or task not in batch["tasks"] or seed not in batch["seeds"]):
            raise ValueError(f"Source B batch does not provide task {task}, seed {seed}: {source}")
        index = batch["tasks"].index(task) * len(batch["seeds"]) + batch["seeds"].index(seed)
        return source / "jobs" / f"{index:06d}.json"

    @staticmethod
    def source_failed(runtime, job_path):
        status = runtime.get("status")
        if status in ("complete", "failed"):
            return True
        attempt = runtime.get("scheduler_attempt_id")
        if status == "running" and attempt is not None:
            job = _read_status(job_path)
            return job.get("scheduler_attempt_id") == attempt and job.get("status") in ("failed", "dependency_failed")
        return False

    def remember_source_failures(self):
        source = Path(self.config["tabx_online_source"])
        runtime = _read_status(source / "runtime.json")
        if runtime.get("status") not in ("running", "complete", "failed"):
            return
        attempt = runtime.get("scheduler_attempt_id")
        for job in self.jobs:
            if step0_record_path(source, job["task_id"], job["train_seed"]).is_file():
                continue
            try:
                path = self.source_job_path(source, job)
            except ValueError:
                # The dependency check will report an incompatible task without stopping other jobs.
                continue
            if path is not None and self.source_failed(runtime, path):
                self.historical_source_failures[job["job_id"]] = attempt

    def source_ready(self, job, source):
        key = job["job_id"]
        if key in self.sources:
            return True
        task, seed = job["task_id"], job["train_seed"]
        task_manifest = self.task_identity(task)
        record_path = step0_record_path(source, task, seed)
        if not record_path.is_file():
            if job.get("source_step0_sha256"):
                raise ValueError(f"Previously verified step-0 record is missing: {record_path}")
            if self.group != "B":
                state_path = self.source_job_path(source, job)
                if state_path is not None:
                    runtime_path = source / "runtime.json"
                    runtime = _read_status(runtime_path)
                    attempt = runtime.get("scheduler_attempt_id")
                    historical = key in self.historical_source_failures and self.historical_source_failures[key] == attempt
                    if not historical and self.source_failed(runtime, state_path):
                        latest = _read_status(runtime_path)
                        if (latest.get("scheduler_attempt_id") == attempt and self.source_failed(latest, state_path)
                                and not record_path.is_file()):
                            job["failed_source_attempt_id"] = attempt
                            raise ValueError(f"Source B attempt {attempt} failed before sealing step 0: "
                                             f"{task}/{seed} ({source})")
            if not record_path.is_file():
                return False
        b_manifest = read_json(source / "batch.json")
        record, digest = read_step0_record(source, task_manifest, self.args.rule, seed,
                                           job.get("source_step0_sha256"))
        frozen = b_manifest["identity"]["demo_manifest_sha256"].get(task)
        if not frozen or record["entry"]["dataset_manifest_sha256"] != frozen:
            raise ValueError(f"B demonstration SHA-256 differs from step-0 record: {task}/{seed}")
        config = dict(self.config, tabx_online_source=str(source), tabx_source_step0_sha256=digest)
        resolve_source(config, task_manifest, self.args.rule, seed)
        job["source_step0_sha256"] = digest
        self.sources[key] = record["entry"]
        self.mark_job(job, "step0_verified", status_changed=False)
        return True

    def check_dependency(self, job):
        if self.group == "B":
            task = job["task_id"]
            if not self.demo_ready(task):
                return "waiting_demo"
            if job["attempts"]:
                ready = self.source_ready(job, self.directory)
                if not ready and not job["needs_training"]:
                    raise ValueError("Completed B job is missing its step-0 record")
        if self.group in ("C", "D"):
            if not self.source_ready(job, Path(self.config["tabx_online_source"])):
                if not job["needs_training"]:
                    raise ValueError("Completed job is missing its source B step-0 record")
                return "waiting_step0"
        return None

    def fill_slots(self):
        ready, waiting = deque(), deque()
        for job_id in self.pending_train:
            job = self.jobs_by_id[job_id]
            try:
                status = self.check_dependency(job)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                self.set_dependency(job, "dependency_failed", str(error))
                continue
            self.set_dependency(job, status or "waiting_gpu")
            (waiting if status else ready).append(job_id)
        self.pool.ready(bool(ready) and not self.stopping)
        while ready and not self.stopping:
            lease = self.pool.acquire(more_ready=len(ready) > 1)
            if lease is None:
                break
            gpu, _, fd = lease
            self.lease = lease
            job = self.jobs_by_id[ready.popleft()]
            self.set_dependency(job, None)
            try:
                self.launch(job, gpu)
                if job["attempts"][-1]["run_id"] in self.running:
                    self.gpu_in_use[gpu] += 1
            finally:
                os.close(fd)
                self.lease = None
        self.pending_train = ready + waiting

    def training_command(self, job, attempt):
        command = super().training_command(job, attempt)
        if self.group == "B":
            command.append("tabx_publish_bc_step0=True")
            if job["job_id"] in self.sources:
                command.append(f"tabx_online_source={_sacred(str(self.directory))}")
        if job.get("source_step0_sha256"):
            command.append(f"tabx_source_step0_sha256={_sacred(job['source_step0_sha256'])}")
        return command

    def training_pass_fds(self, job, attempt):
        _, slot, fd = self.lease
        self._flush_job(job["job_id"])
        self.pool.record_launch(slot, job, attempt)
        return (fd,)

    def progress(self, statuses=None):
        counts = Counter(self.jobs_by_id[job_id].get("dependency_status") for job_id in self.pending_train)
        self.emit(f"[GROUP {self.group}] waiting_demo={counts['waiting_demo']} "
                  f"waiting_step0={counts['waiting_step0']} waiting_gpu={counts['waiting_gpu']} "
                  f"running={len(self.running)} complete={self.model_counts['complete']} "
                  f"dependency_failed={self.model_counts['dependency_failed']}")
        super().progress(statuses)

    def seal_initial_weights(self, failed):
        if self.group != "B":
            return
        entries = {}
        for job in self.jobs:
            self.sources.pop(job["job_id"], None)
            if not self.source_ready(job, self.directory):
                continue
            task, seed = job["task_id"], job["train_seed"]
            entry = self.sources[job["job_id"]]
            if job["train_status"] == "complete":
                run = self.root / job["attempts"][-1]["run"]
                snapshot = read_json(run / "checkpoints/0/manifest.json")
                if any(snapshot[key] != entry[key] for key in WEIGHTS.values()):
                    raise ValueError(f"B online attempt started from different step-0 weights: {task}/{seed}")
            entries[f"{task}/{seed}"] = entry
        write_step0_index(self.directory, self.manifest, entries, extend=True)
