"""Schedule B, C and D jobs from a shared, sealed BC checkpoint."""
from collections import deque
import os
import time
import uuid

from tabx_gpu_pool import SharedGpuPool
from tabx_medium_scheduler import MediumDemoScheduler, _read_status
from tabx_online_start import WEIGHTS, read_step0_record, resolve_source, step0_record_path, write_step0_index
from tabx_scheduler import _sacred, process_alive
from tabx_storage import read_json

GROUPS = ("B", "C", "D")


def coach_jobs(tasks, seeds):
    return [dict(task_id=task, train_seed=seed, group=group)
            for task in tasks for seed in seeds for group in GROUPS]


class CoachScheduler(MediumDemoScheduler):
    def __init__(self, args, directory, manifest, state):
        super().__init__(args, directory, manifest, state)
        self.pool = SharedGpuPool(self.root, args.rule, args.gpus, args.jobs_per_gpu, "coach", args.batch_id)
        self.b_jobs = {(job["task_id"], job["train_seed"]): job for job in self.jobs if job["group"] == "B"}
        self.lease = None
        self.sources = {}

    def run(self):
        self.scheduler_attempt_id = uuid.uuid4().hex
        self.pool.register()
        try:
            code = super().run()
        finally:
            self.pool.unregister()
        if code == 130:
            return code
        from tabx_coach_selection import select_coaches
        selection_code = select_coaches(self)
        return 130 if selection_code == 130 else max(code, selection_code)

    def prepare_resume(self):
        runtime = _read_status(self.directory / "runtime.json")
        old = [self.state.get("worker", {}), runtime.get("worker", {}), *runtime.get("active_trainers", [])]
        old.extend(attempt for job in self.jobs for attempt in job["attempts"])
        if list(self.pool.active_attempts()) or any(process_alive(record) for record in old):
            self.emit(f"[RECOVER] batch={self.args.batch_id} waiting for previous workers and GPU locks")
            while list(self.pool.active_attempts()) or any(process_alive(record) for record in old):
                time.sleep(self.args.poll_seconds)
        for job in self.jobs:
            job.pop("dependency_status", None)
            job.pop("dependency_error", None)
        super().prepare_resume()
        self.pending_train = deque(sorted(self.pending_train, key=lambda job_id: self.jobs_by_id[job_id]["group"] != "B"))

    def source_ready(self, job, *, refresh=False):
        task, seed = job["task_id"], job["train_seed"]
        key = (task, seed)
        if key not in self.sources or refresh:
            path = step0_record_path(self.directory, task, seed)
            if not path.is_file():
                if key in self.sources or job.get("source_step0_sha256"):
                    raise ValueError(f"Previously verified step-0 record is missing: {path}")
                return False
            expected = self.sources[key][0] if key in self.sources else job.get("source_step0_sha256")
            record, digest = read_step0_record(self.directory, self.task_identity(task), self.args.rule, seed, expected)
            entry = record["entry"]
            frozen = self.manifest["identity"]["demo_manifest_sha256"].get(task)
            if not frozen or entry["dataset_manifest_sha256"] != frozen:
                raise ValueError(f"Step-0 demonstration SHA-256 differs from the batch: {task}/{seed}")
            config = dict(self.training_parameters(job)[0], tabx_online_source=str(self.directory),
                          tabx_source_step0_sha256=digest)
            resolve_source(config, self.task_identity(task), self.args.rule, seed)
            self.sources[key] = digest, entry
        digest, _ = self.sources[key]
        if job.get("source_step0_sha256", digest) != digest:
            raise ValueError(f"Step-0 record SHA-256 differs from the previous attempt: {task}/{seed}")
        job["source_step0_sha256"] = digest
        self.mark_job(job, "step0_verified", status_changed=False)
        return True

    def _check_run(self, job, run):
        super()._check_run(job, run)
        task, seed = job["task_id"], job["train_seed"]
        if not self.source_ready(job):
            raise ValueError(f"Completed {job['group']} run lacks its shared step 0: {run}")
        entry = self.sources[task, seed][1]
        weights = read_json(run / "checkpoints" / "0" / "manifest.json")
        if any(weights.get(key) != entry[key] for key in WEIGHTS.values()):
            raise ValueError(f"{job['group']} started from different BC weights: {run}")

    def check_dependency(self, job):
        task, seed = job["task_id"], job["train_seed"]
        if job["group"] == "B":
            if not self.demo_ready(task):
                return "waiting_demo"
            self.source_ready(job)
            return None
        if self.source_ready(job):
            return None
        b = self.b_jobs[task, seed]
        attempt = b["attempts"][-1] if b["attempts"] else {}
        if b["needs_training"] or attempt.get("run_id") in self.running:
            return "waiting_step0"
        error = b.get("dependency_error") or attempt.get("error") or "B exited without a sealed checkpoint"
        raise ValueError(f"B failed before sealing step 0: {error}")

    def launch_with_lease(self, job, lease):
        self.lease = lease
        gpu, _, fd = lease
        try:
            self.set_dependency(job, None)
            self.launch(job, gpu)
            if job["attempts"][-1]["run_id"] in self.running:
                self.gpu_in_use[gpu] += 1
        finally:
            os.close(fd)
            self.lease = None

    def launch_ready(self, ready):
        self.pool.ready(bool(ready) and not self.stopping)
        while ready and not self.stopping:
            lease = self.pool.acquire(more_ready=len(ready) > 1)
            if lease is None:
                return False
            self.launch_with_lease(self.jobs_by_id[ready.popleft()], lease)
        return not self.stopping

    def fill_slots(self):
        ready, blocked, waiting = deque(), deque(), deque()
        for job_id in self.pending_train:
            queue = ready if self.jobs_by_id[job_id].get("dependency_status") == "waiting_gpu" else blocked
            queue.append(job_id)

        can_acquire = self.launch_ready(ready)
        while blocked and not self.stopping:
            job_id = blocked.popleft()
            job = self.jobs_by_id[job_id]
            try:
                status = self.check_dependency(job)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                self.set_dependency(job, "dependency_failed", str(error))
                job["needs_training"] = False
                continue
            self.set_dependency(job, status or "waiting_gpu")
            if status:
                waiting.append(job_id)
                continue
            ready.append(job_id)
            if can_acquire:
                can_acquire = self.launch_ready(ready)
        if self.stopping:
            self.pool.ready(False)
        self.pending_train = ready + waiting + blocked

    def training_command(self, job, attempt):
        command = super().training_command(job, attempt)
        if job["group"] == "B":
            command.append("tabx_publish_bc_step0=True")
            if job.get("source_step0_sha256"):
                command.append(f"tabx_online_source={_sacred(str(self.directory))}")
        if job.get("source_step0_sha256"):
            command.append(f"tabx_source_step0_sha256={_sacred(job['source_step0_sha256'])}")
        return command

    def training_pass_fds(self, job, attempt):
        _, slot, fd = self.lease
        self._flush_job(job["job_id"])
        self.pool.record_launch(slot, job, attempt)
        return (fd,)

    def seal_initial_weights(self, failed):
        entries = {}
        for job in self.jobs:
            if job["group"] == "B" and self.source_ready(job, refresh=True):
                task, seed = job["task_id"], job["train_seed"]
                entries[f"{task}/{seed}"] = self.sources[task, seed][1]
        write_step0_index(self.directory, self.manifest, entries, extend=True)
