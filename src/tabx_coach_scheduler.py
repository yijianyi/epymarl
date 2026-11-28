"""Train ordered groups of B, C and D from a shared, sealed BC checkpoint."""
from collections import deque
from itertools import groupby
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
        if len(args.gpus) * args.jobs_per_gpu < len(GROUPS):
            raise ValueError("Coach training needs at least three GPU slots across all GPUs")
        self.pool = SharedGpuPool(self.root, args.rule, args.gpus, args.jobs_per_gpu, "coach", args.batch_id)
        self.bundles = deque()
        self.reserved = {}
        self.lease = None
        self.sources = {}

    def run(self):
        self.scheduler_attempt_id = uuid.uuid4().hex
        self.pool.register()
        try:
            code = super().run()
        finally:
            self.release_reservations()
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
        for _, jobs in groupby(self.jobs, key=lambda job: (job["task_id"], job["train_seed"])):
            bundle = {job["group"]: job for job in jobs}
            if any(job["needs_training"] for job in bundle.values()):
                self.bundles.append(bundle)

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

    def release_reservations(self):
        for _, _, fd in self.reserved.values():
            os.close(fd)
        self.reserved.clear()

    def launch_reserved(self, job):
        self.lease = self.reserved.pop(job["group"])
        gpu, _, fd = self.lease
        self.set_dependency(job, None)
        try:
            self.launch(job, gpu)
            if job["attempts"][-1]["run_id"] in self.running:
                self.gpu_in_use[gpu] += 1
        finally:
            os.close(fd)
            self.lease = None

    def fill_slots(self):
        while self.bundles and not self.stopping:
            bundle = self.bundles[0]
            b = bundle["B"]
            task = b["task_id"]
            waiting = [job for job in bundle.values() if job["needs_training"]]
            try:
                if not self.reserved:
                    if not self.demo_ready(task):
                        self.pool.ready(False)
                        for job in waiting:
                            self.set_dependency(job, "waiting_demo")
                        break
                    self.pool.ready(True)
                    leases = self.pool.acquire_many(len(waiting), more_ready=False)
                    if not leases:
                        for job in waiting:
                            self.set_dependency(job, "waiting_gpu")
                        break
                    self.reserved = {job["group"]: lease for job, lease in zip(waiting, leases)}
                    if b["needs_training"]:
                        self.source_ready(b)
                        self.launch_reserved(b)
                if self.reserved:
                    if not self.source_ready(b):
                        last_b = b["attempts"][-1]["run_id"]
                        if last_b not in self.running:
                            error = b["attempts"][-1].get("error") or "B exited without a sealed checkpoint"
                            raise ValueError(f"B failed before sealing step 0: {error}")
                        for group in self.reserved:
                            self.set_dependency(bundle[group], "waiting_step0")
                        break
                    for group in ("C", "D"):
                        if group in self.reserved:
                            self.source_ready(bundle[group])
                            self.launch_reserved(bundle[group])
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                for job in bundle.values():
                    if job["needs_training"]:
                        self.set_dependency(job, "dependency_failed", str(error))
                        job["needs_training"] = False
                self.release_reservations()
            self.bundles.popleft()
        if not self.bundles or self.stopping:
            self.pool.ready(False)
        self.pending_train = deque(job["job_id"] for bundle in self.bundles
                                   for job in bundle.values() if job["needs_training"])

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
